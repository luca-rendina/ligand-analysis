"""Compare two public-demo runs without interpreting the small demo as a benchmark.

Run inside ligand-chem, not host Python: python scripts/k8s_compare.py PODMAN K8S
"""

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem
from scipy import sparse


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_files(left, right, paths):
    for name in paths:
        if digest(left / name) != digest(right / name):
            raise AssertionError(f"{name}: differing SHA-256 checksums")


def selected_fields(left, right, name, fields, key):
    def indexed(root):
        result = {}
        for row in rows(root / name):
            identity = tuple(row[field] for field in key)
            if identity in result:
                raise AssertionError(f"{name}: duplicate identity {identity}")
            result[identity] = tuple(row[field] for field in fields)
        return result

    if indexed(left) != indexed(right):
        raise AssertionError(f"{name}: differing {fields}")


def compare_poses(left, right, group):
    relative = f"{group}/poses/selected_poses.sdf"
    def molecules(root):
        result = {}
        for molecule in Chem.SDMolSupplier(str(root / relative), removeHs=True):
            if molecule is None:
                raise AssertionError(f"{relative}: invalid SDF record")
            sample = molecule.GetProp("sample_id")
            if sample in result:
                raise AssertionError(f"{relative}: duplicate {sample}")
            result[sample] = molecule
        return result

    podman, k8s = molecules(left), molecules(right)
    if podman.keys() != k8s.keys():
        raise AssertionError(f"{relative}: sample IDs differ")
    maximum = 0.0
    for sample in podman:
        a, b = podman[sample], k8s[sample]
        if a.GetNumAtoms() != b.GetNumAtoms():
            raise AssertionError(f"{sample}: atom counts differ")
        xyz_a = a.GetConformer().GetPositions()
        xyz_b = b.GetConformer().GetPositions()
        matches = b.GetSubstructMatches(a, uniquify=False, maxMatches=10000)
        if not matches:
            raise AssertionError(f"{sample}: atom graph differs")
        rmsd = min(float(np.sqrt(np.mean(np.sum((xyz_a - xyz_b[list(match)]) ** 2, axis=1))))
                   for match in matches)
        maximum = max(maximum, rmsd)
        if rmsd > 0.10:
            raise AssertionError(f"{sample}: in-place symmetry-aware RMSD {rmsd:.4f} > 0.10 A")
    return maximum


def compare(left, right):
    for root in (left, right):
        report = json.loads((root / "report/report.json").read_text(encoding="utf-8"))
        if report["status"] != "success":
            raise AssertionError(f"{root}: report status is {report['status']}")
    for path in ("inputs/ligands.tsv", "inputs/prediction_ligands.tsv", "split/split.tsv"):
        equal_files(left, right, [path])
    for group in ("labelled", "prediction"):
        selected_fields(left, right, f"{group}/prepared/ligand_outcomes.tsv",
                        ("status", "reason", "parent_inchikey"), ("ligand_id",))
        selected_fields(left, right, f"{group}/poses/pose_selection.tsv",
                        ("status", "reason", "state_id", "pose_rank"), ("ligand_id",))
        selected_fields(left, right, f"{group}/features/plec/rows.tsv",
                        ("row", "sample_id", "state_id", "pose_rank"), ("sample_id",))
        for representation in ("plec", "morgan"):
            base = f"{group}/features/{representation}"
            equal_files(left, right, [f"{base}/rows.tsv"])
            a = sparse.load_npz(left / base / "features.npz")
            b = sparse.load_npz(right / base / "features.npz")
            if a.shape != b.shape or (a != b).nnz:
                raise AssertionError(f"{base}: feature matrices differ")
    scores = {}
    for group in ("labelled", "prediction"):
        docking_a = {path.name: path for path in (left / group / "docking").iterdir() if path.is_dir()}
        docking_b = {path.name: path for path in (right / group / "docking").iterdir() if path.is_dir()}
        if docking_a.keys() != docking_b.keys():
            raise AssertionError(f"{group}: docking task directories differ")
        docking_delta = 0.0
        for name, a in docking_a.items():
            b = docking_b[name]
            selected_fields(a, b, "docking_tasks.tsv", ("status", "reason", "n_poses"), ("state_id",))
            pose_a = {(row["state_id"], row["pose_rank"]): float(row["score"])
                      for row in rows(a / "docking_poses.tsv")}
            pose_b = {(row["state_id"], row["pose_rank"]): float(row["score"])
                      for row in rows(b / "docking_poses.tsv")}
            if pose_a.keys() != pose_b.keys():
                raise AssertionError(f"{group}/{name}: generated pose identities differ")
            docking_delta = max(docking_delta, max((abs(pose_a[key] - pose_b[key]) for key in pose_a),
                                                    default=0.0))
        if docking_delta > 0.01:
            raise AssertionError(f"{group}: maximum Vina score delta {docking_delta:.4f} > 0.01 kcal/mol")
        name = f"{group}/poses/pose_selection.tsv"
        a = {row["ligand_id"]: row for row in rows(left / name)}
        b = {row["ligand_id"]: row for row in rows(right / name)}
        delta = max((abs(float(a[key]["score"]) - float(b[key]["score"]))
                     for key in a if a[key]["status"] == "selected"), default=0.0)
        if delta > 0.01:
            raise AssertionError(f"{name}: maximum selected score delta {delta:.4f} > 0.01 kcal/mol")
        scores[group] = {"maximum_score_delta_kcal_mol": docking_delta,
                         "maximum_pose_rmsd_angstrom": compare_poses(left, right, group)}
    models = sorted(path.name for path in (left / "models").iterdir() if path.is_dir())
    if models != sorted(path.name for path in (right / "models").iterdir() if path.is_dir()):
        raise AssertionError("model sets differ")
    for model in models:
        a, b = left / "models" / model, right / "models" / model
        selected_fields(a, b, "evaluation/predictions.tsv",
                        ("partition", "true_label", "predicted_label", "predicted_class", "score_class1",
                         "score_type"), ("sample_id", "partition"))
        ma = json.loads((a / "evaluation/metrics.json").read_text(encoding="utf-8"))
        mb = json.loads((b / "evaluation/metrics.json").read_text(encoding="utf-8"))
        for partition in ("train", "test"):
            if ma["metrics"][partition]["confusion_counts"] != mb["metrics"][partition]["confusion_counts"]:
                raise AssertionError(f"{model}/{partition}: confusion counts differ")
        name = f"{model}.tsv"
        selected_fields(left / "prediction/predictions", right / "prediction/predictions",
                        name, ("predicted_label", "predicted_class", "score_class1", "score_type"),
                        ("sample_id",))
    return {"models": models, "comparisons": scores, "status": "equivalent"}


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: k8s_compare.py PODMAN_RESULTS K8S_RESULTS")
    left, right = (Path(path) for path in sys.argv[1:])
    result = compare(left, right)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
