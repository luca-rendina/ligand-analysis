"""Pose validation, class-blind pose selection and the reference-ligand redocking diagnostic.

Selection rule ``best_valid_score``: among the valid poses of all states of a parent ligand,
take the lowest Vina score; ties go to the lower state ID, then pose rank. A pose is valid when
Meeko's export reproduces the docked state (connectivity, charges and 3D stereochemistry), all
heavy atoms lie in the docking box (plus a tolerance) and no ligand-receptor heavy-atom pair is
closer than ``clash_distance``. Labels are never read.
"""

import collections
import io
import json
from pathlib import Path

import numpy as np

from .. import __version__
from ..fileio import write_json, write_text_atomic
from ..pdbfile import parse_atoms
from ..tables import read_table, write_table
from . import ChemistryError

OUTPUT_FILES = ("pose_checks.tsv", "pose_selection.tsv", "selected_poses.sdf", "pose_selection.json")
REDOCK_FILES = ("redocking.json", "redocking_poses.sdf")


def receptor_heavy_coordinates(receptor_dir):
    atoms = parse_atoms((Path(receptor_dir) / "receptor.pdb").read_text(encoding="utf-8"))
    return np.array([[atom.x, atom.y, atom.z] for atom in atoms if not atom.is_hydrogen])


def heavy_coordinates(mol, conf_id=-1):
    positions = mol.GetConformer(conf_id).GetPositions()
    return positions[[atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]]


def smiles_from_3d(mol, conf_id=-1):
    from rdkit import Chem

    copy = Chem.Mol(mol, confId=conf_id)
    Chem.AssignStereochemistryFrom3D(copy)
    return Chem.MolToSmiles(Chem.RemoveHs(copy))


def geometry(mol, conf_id, box, tolerance, receptor_xyz, clash_distance, reference_centroid):
    coords = heavy_coordinates(mol, conf_id)
    half = np.asarray(box["size"]) / 2 + tolerance
    in_box = bool(np.all(np.abs(coords - np.asarray(box["center"])) <= half))
    distances = np.linalg.norm(coords[:, None, :] - receptor_xyz[None, :, :], axis=2)
    return {"in_box": in_box, "min_distance": round(float(distances.min()), 3),
            "clashes": int((distances < clash_distance).sum()),
            "centroid_to_reference": round(float(np.linalg.norm(coords.mean(axis=0) - reference_centroid)), 3)}


def read_sdf(path, remove_hs=False):
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), removeHs=remove_hs)
    molecules = list(supplier)
    if any(mol is None for mol in molecules):
        raise ChemistryError(f"{path} has unreadable records")
    return molecules


def reference_ligand(receptor_dir):
    from rdkit import Chem

    mol = read_sdf(Path(receptor_dir) / "reference_ligand.sdf")[0]
    heavy = Chem.RemoveHs(mol)
    return heavy, heavy_coordinates(heavy).mean(axis=0)


def select_poses(ligand_rows, ligands_dir, docking_dirs, receptor_dir, cfg, output_dir):
    """Check every pose, select one per ligand, account for every input ligand."""
    from rdkit import Chem

    ligands_dir, receptor_dir, output_dir = Path(ligands_dir), Path(receptor_dir), Path(output_dir)
    structure = json.loads((receptor_dir / "receptor_structure.json").read_text(encoding="utf-8"))
    box = json.loads((receptor_dir / "box.json").read_text(encoding="utf-8"))
    receptor_xyz = receptor_heavy_coordinates(receptor_dir)
    _reference, reference_centroid = reference_ligand(receptor_dir)
    outcomes = {row["ligand_id"]: row for row in read_table(ligands_dir / "ligand_outcomes.tsv", "ligand_outcome")}
    states = {row["state_id"]: row for row in read_table(ligands_dir / "ligand_states.tsv", "ligand_state")}
    tasks, scores = {}, {}
    for directory in map(Path, docking_dirs):
        for task in read_table(directory / "docking_tasks.tsv", "docking_task"):
            if task["state_id"] in tasks:
                raise ChemistryError(f"state {task['state_id']} docked in more than one docking directory")
            tasks[task["state_id"]] = (directory, task)
        for pose in read_table(directory / "docking_poses.tsv", "docking_pose"):
            scores[(pose["state_id"], pose["pose_rank"])] = pose["score"]

    checks, selections, selected_records = [], [], []
    for row in ligand_rows:
        ligand_id = row["ligand_id"]
        selection = {"ligand_id": ligand_id, "receptor_id": structure["receptor_id"],
                     "structure_id": structure["structure_id"], "status": "rejected", "state_id": None,
                     "pose_rank": None, "score": None, "n_states": 0, "n_poses": 0, "n_valid_poses": 0,
                     "min_distance": None, "centroid_to_reference": None, "reason": None, "details": None}
        outcome = outcomes.get(ligand_id)
        if outcome is None or outcome["status"] != "prepared":
            selection.update(reason="not_prepared", details=outcome["reason"] if outcome else "no preparation outcome")
            selections.append(selection)
            continue
        ligand_states = sorted(state_id for state_id, state in states.items() if state["ligand_id"] == ligand_id)
        selection["n_states"] = len(ligand_states)
        docked = [tasks[state_id] for state_id in ligand_states
                  if state_id in tasks and tasks[state_id][1]["status"] == "docked"]
        if not docked:
            failures = [tasks[state_id][1]["reason"] for state_id in ligand_states if state_id in tasks]
            selection.update(reason="not_docked", details="; ".join(filter(None, failures)) or "no docking task")
            selections.append(selection)
            continue
        candidates, rejected = [], collections.Counter()
        for directory, task in docked:
            state = states[task["state_id"]]
            for mol in read_sdf(directory / task["poses_sdf"]):
                rank = int(mol.GetProp("pose_rank"))
                check = {"state_id": state["state_id"], "ligand_id": ligand_id, "pose_rank": rank,
                         "score": scores[(state["state_id"], rank)],
                         "identity_ok": smiles_from_3d(mol) == state["smiles"]}
                check.update(geometry(mol, -1, box, cfg["box_tolerance"], receptor_xyz, cfg["clash_distance"],
                                      reference_centroid))
                problems = [name for name, failed in (("identity", not check["identity_ok"]),
                                                      ("outside_box", not check["in_box"]),
                                                      ("clash", check["clashes"] > 0)) if failed]
                check.update(valid=not problems, reason="|".join(problems) or None)
                rejected.update(problems)
                checks.append(check)
                if not problems:
                    candidates.append((check["score"], state["state_id"], rank, mol, check, directory))
        selection["n_poses"] = sum(task["n_poses"] for _, task in docked)
        selection["n_valid_poses"] = len(candidates)
        if not candidates:
            selection.update(reason="no_valid_pose", details=", ".join(f"{name}: {count}"
                                                                         for name, count in sorted(rejected.items())))
            selections.append(selection)
            continue
        score, state_id, rank, mol, check, _directory = min(candidates, key=lambda item: item[:3])
        state_mol = read_sdf(ligands_dir / states[state_id]["sdf_file"])[0]
        atom_map = mol.GetSubstructMatch(state_mol)
        if len(atom_map) != state_mol.GetNumAtoms():
            raise ChemistryError(f"selected pose of {state_id} does not map onto the prepared state")
        selection.update(status="selected", state_id=state_id, pose_rank=rank, score=score,
                         min_distance=check["min_distance"], centroid_to_reference=check["centroid_to_reference"])
        selections.append(selection)
        record = Chem.Mol(mol)
        group_id = (outcome["parent_inchikey"] or outcome["input_inchikey"] or ligand_id)[:14]
        properties = {"sample_id": f"{ligand_id}@{structure['receptor_id']}", "ligand_id": ligand_id,
                      "receptor_id": structure["receptor_id"], "structure_id": structure["structure_id"],
                      "state_id": state_id, "pose_rank": rank, "score": score, "group_id": group_id,
                      "state_atom_map": ",".join(map(str, atom_map))}
        record.SetProp("_Name", properties["sample_id"])
        for name, value in properties.items():
            record.SetProp(name, str(value))
        selected_records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(output_dir / "pose_checks.tsv", checks, "pose_check")
    write_table(output_dir / "pose_selection.tsv", selections, "pose_selection")
    write_text_atomic(output_dir / "selected_poses.sdf", _sdf(selected_records))
    summary = {
        "report_type": "pose_selection",
        "package_version": __version__,
        "rule": cfg["rule"],
        "rule_description": __doc__.split("Selection rule ``best_valid_score``: ", 1)[1].strip(),
        "config": cfg,
        "structure_id": structure["structure_id"],
        "counts": {
            "ligands": len(selections),
            "selected": sum(1 for row in selections if row["status"] == "selected"),
            "rejected_by_reason": dict(sorted(collections.Counter(
                row["reason"] for row in selections if row["status"] == "rejected").items())),
            "poses_checked": len(checks),
            "invalid_poses_by_problem": dict(sorted(collections.Counter(
                problem for row in checks if row["reason"] for problem in row["reason"].split("|")).items())),
        },
        "outputs": list(OUTPUT_FILES),
    }
    write_json(output_dir / "pose_selection.json", summary)
    return summary


def _sdf(molecules):
    from rdkit import Chem

    stream = io.StringIO()
    writer = Chem.SDWriter(stream)
    for mol in molecules:
        writer.write(mol)
    writer.close()
    return stream.getvalue()


def rmsd_in_place(reference_heavy, pose, conf_id):
    """Symmetry-aware heavy-atom RMSD in the receptor frame (no superposition).

    The crystal ligand is the substructure query, so protonation differences of the docked
    state do not prevent the match.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    return float(rdMolAlign.CalcRMS(reference_heavy, Chem.RemoveHs(pose), refId=conf_id))


def redock_reference(receptor_dir, ligand_cfg, docking_cfg, redocking_cfg, pose_cfg, cpu, output_dir):
    """Prepare the reference ligand from its SMILES, dock it and compare poses with the crystal pose."""
    from .docking import dock_state, engine_version, export_poses, poses_sdf, read_box
    from .ligands import prepare_ligand

    receptor_dir, output_dir = Path(receptor_dir), Path(output_dir)
    structure = json.loads((receptor_dir / "receptor_structure.json").read_text(encoding="utf-8"))
    spec = structure["reference_ligand"]
    box = read_box(receptor_dir)
    receptor_xyz = receptor_heavy_coordinates(receptor_dir)
    crystal, crystal_centroid = reference_ligand(receptor_dir)
    row = {"ligand_id": f"reference:{spec['resname']}", "name": spec["name"], "smiles": spec["smiles"],
           "inchikey": spec["inchikey"]}
    outcome, states = prepare_ligand(row, ligand_cfg, output_dir / "prepared")
    if outcome["status"] != "prepared":
        raise ChemistryError(f"reference ligand preparation failed: {outcome['reason']}: {outcome['details']}")
    poses, sdf_parts = [], []
    for state in states:
        ligand_pdbqt = (output_dir / "prepared" / state["pdbqt_file"]).read_text(encoding="utf-8")
        poses_pdbqt, energies = dock_state(receptor_dir / "receptor.pdbqt", box, ligand_pdbqt, docking_cfg, cpu)
        mol = export_poses(poses_pdbqt)
        properties = []
        for rank, (conformer, energy) in enumerate(zip(mol.GetConformers(), energies, strict=True), start=1):
            item = {"state_id": state["state_id"], "pose_rank": rank, "score": float(energy[0]),
                    "rmsd": round(rmsd_in_place(crystal, mol, conformer.GetId()), 3),
                    "identity_ok": smiles_from_3d(mol, conformer.GetId()) == state["smiles"]}
            item.update(geometry(mol, conformer.GetId(), box, pose_cfg["box_tolerance"], receptor_xyz,
                                 pose_cfg["clash_distance"], crystal_centroid))
            poses.append(item)
            properties.append(item)
        sdf_parts.append(poses_sdf(mol, properties))
    top = min(poses, key=lambda item: (item["score"], item["state_id"], item["pose_rank"]))
    best = min(poses, key=lambda item: item["rmsd"])
    passed = top["rmsd"] <= redocking_cfg["rmsd_threshold"]
    report = {
        "report_type": "redocking",
        "package_version": __version__,
        "structure_id": structure["structure_id"],
        "reference_ligand": spec,
        "engine": engine_version(),
        "method": "reference ligand prepared from SMILES (no crystal coordinates), docked with the demo settings; "
                  "symmetry-aware heavy-atom RMSD to the crystal pose in the receptor frame without superposition",
        "rmsd_threshold": redocking_cfg["rmsd_threshold"],
        "passed": passed,
        "top_pose": top,
        "best_rmsd_pose": best,
        "poses": poses,
        "config": {"ligand_preparation": ligand_cfg, "docking": docking_cfg, "pose_selection": pose_cfg},
        "outputs": list(REDOCK_FILES),
    }
    write_text_atomic(output_dir / "redocking_poses.sdf", "".join(sdf_parts))
    write_json(output_dir / "redocking.json", report)
    return report
