"""Rigid-receptor AutoDock Vina docking of prepared ligand states; poses exported through Meeko.

Every requested state gets a task row (docked or failed). All poses are kept in PDBQT and in
SDF exported by Meeko, which rebuilds bond orders and hydrogens from the SMILES remarks written
at preparation instead of guessing them from PDBQT coordinates.
"""

import io
import json
from pathlib import Path
import time

from .. import __version__
from ..config import canonical_sha256
from ..fileio import write_json, write_text_atomic
from ..tables import read_table, write_table
from . import ChemistryError, safe_key

OUTPUT_FILES = ("docking_tasks.tsv", "docking_poses.tsv", "docking.json")


def read_box(receptor_dir):
    return json.loads((Path(receptor_dir) / "box.json").read_text(encoding="utf-8"))


def docking_config_sha256(cfg, box):
    return canonical_sha256({"docking": cfg, "box": {key: box[key] for key in ("center", "size", "spacing")}})


def engine_version():
    import vina

    return f"AutoDock Vina {vina.__version__}"


def dock_state(receptor_pdbqt, box, ligand_pdbqt, cfg, cpu):
    """Dock one state with a fresh Vina object (results do not depend on task order)."""
    from vina import Vina

    engine = Vina(sf_name=cfg["scoring"], cpu=cpu, seed=cfg["seed"], verbosity=0)
    engine.set_receptor(str(receptor_pdbqt))
    engine.set_ligand_from_string(ligand_pdbqt)
    engine.compute_vina_maps(center=box["center"], box_size=box["size"], spacing=box["spacing"])
    engine.dock(exhaustiveness=cfg["exhaustiveness"], n_poses=cfg["num_modes"], min_rmsd=cfg["min_rmsd"])
    poses = engine.poses(n_poses=cfg["num_modes"], energy_range=cfg["energy_range"])
    energies = engine.energies(n_poses=cfg["num_modes"], energy_range=cfg["energy_range"])
    return poses, energies


def export_poses(poses_pdbqt):
    """RDKit molecule with one conformer per pose, rebuilt by Meeko."""
    from meeko import PDBQTMolecule, RDKitMolCreate

    molecules = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(poses_pdbqt, is_dlg=False, skip_typing=True))
    if len(molecules) != 1 or molecules[0] is None:
        raise ChemistryError("Meeko could not rebuild the docked molecule from the PDBQT remarks")
    return molecules[0]


def poses_sdf(mol, properties_by_conformer):
    """SDF text with one record per conformer and its properties."""
    from rdkit import Chem

    stream = io.StringIO()
    writer = Chem.SDWriter(stream)
    for conformer, properties in zip(mol.GetConformers(), properties_by_conformer, strict=True):
        record = Chem.Mol(mol, confId=conformer.GetId())
        record.SetProp("_Name", str(properties.get("state_id", "")))
        for name, value in properties.items():
            record.SetProp(name, str(value))
        writer.write(record)
    writer.close()
    return stream.getvalue()


def dock_ligands(receptor_dir, ligands_dir, ligand_ids, cfg, cpu, output_dir):
    """Dock every prepared state of the given ligands (all when ligand_ids is empty)."""
    receptor_dir, ligands_dir, output_dir = Path(receptor_dir), Path(ligands_dir), Path(output_dir)
    box = read_box(receptor_dir)
    outcomes = {row["ligand_id"]: row for row in read_table(ligands_dir / "ligand_outcomes.tsv", "ligand_outcome")}
    unknown = sorted(set(ligand_ids) - outcomes.keys())
    if unknown:
        raise ChemistryError(f"ligands not in the preparation outcomes: {unknown}")
    wanted = set(ligand_ids) or set(outcomes)
    states = [row for row in read_table(ligands_dir / "ligand_states.tsv", "ligand_state") if row["ligand_id"] in wanted]
    config_sha256 = docking_config_sha256(cfg, box)
    engine = engine_version()
    tasks, pose_rows = [], []
    (output_dir / "poses").mkdir(parents=True, exist_ok=True)
    for state in states:
        start = time.monotonic()
        task = {"state_id": state["state_id"], "ligand_id": state["ligand_id"], "status": "failed", "reason": None,
                "n_poses": 0, "best_score": None, "engine": engine, "scoring": cfg["scoring"], "seed": cfg["seed"],
                "exhaustiveness": cfg["exhaustiveness"], "cpu": cpu, "config_sha256": config_sha256,
                "poses_sdf": None, "poses_pdbqt": None}
        try:
            ligand_pdbqt = (ligands_dir / state["pdbqt_file"]).read_text(encoding="utf-8")
            poses, energies = dock_state(receptor_dir / "receptor.pdbqt", box, ligand_pdbqt, cfg, cpu)
            mol = export_poses(poses)
            if mol.GetNumConformers() != len(energies):
                raise ChemistryError(f"{mol.GetNumConformers()} exported poses for {len(energies)} Vina energies")
            rows = [{"state_id": state["state_id"], "ligand_id": state["ligand_id"], "pose_rank": rank,
                     "score": float(energy[0]), "inter": float(energy[1]), "intra": float(energy[2]),
                     "torsional": float(energy[3]), "intra_best_pose": float(energy[4])}
                    for rank, energy in enumerate(energies, start=1)]
            key = safe_key(state["state_id"])
            task["poses_pdbqt"], task["poses_sdf"] = f"poses/{key}.pdbqt", f"poses/{key}.sdf"
            write_text_atomic(output_dir / task["poses_pdbqt"], poses)
            write_text_atomic(output_dir / task["poses_sdf"], poses_sdf(mol, [
                {"state_id": row["state_id"], "ligand_id": row["ligand_id"], "pose_rank": row["pose_rank"],
                 "score": row["score"]} for row in rows]))
            pose_rows += rows
            task.update(status="docked", n_poses=len(rows), best_score=rows[0]["score"] if rows else None)
            if not rows:
                task.update(status="failed", reason="Vina returned no poses")
        except (RuntimeError, ValueError, TypeError, OSError) as error:
            task["reason"] = f"{type(error).__name__}: {error}"[:500]
        task["seconds"] = round(time.monotonic() - start, 2)
        tasks.append(task)
    write_table(output_dir / "docking_tasks.tsv", tasks, "docking_task")
    write_table(output_dir / "docking_poses.tsv", pose_rows, "docking_pose")
    summary = {
        "report_type": "docking",
        "package_version": __version__,
        "engine": engine,
        "config": cfg,
        "config_sha256": config_sha256,
        "box": box,
        "cpu": cpu,
        "receptor_pdbqt": str(receptor_dir / "receptor.pdbqt"),
        "ligands": sorted(wanted),
        "counts": {"states": len(states), "docked": sum(1 for task in tasks if task["status"] == "docked"),
                   "failed": sum(1 for task in tasks if task["status"] == "failed"), "poses": len(pose_rows)},
        "outputs": list(OUTPUT_FILES) + ["poses/"],
    }
    write_json(output_dir / "docking.json", summary)
    return summary
