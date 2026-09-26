"""Identity-only ligand input tables for preparation, docking and prediction.

The tables carry no label column, so the chemistry stages cannot consume labels.
"""

from .config import ConfigError


def _input(row):
    return {"ligand_id": row["ligand_id"], "name": row.get("name"), "smiles": row["smiles"],
            "inchikey": row.get("inchikey")}


def selected_candidates(candidate_rows, max_per_class):
    """Curated selected candidates with selection_rank <= max_per_class, ordered by ligand ID."""
    rows = [row for row in candidate_rows if row["selected"] and row["selection_rank"] <= max_per_class]
    if not rows:
        raise ConfigError("no selected candidates")
    return [_input(row) for row in sorted(rows, key=lambda row: row["ligand_id"])]


def ligands_by_id(ligand_rows, ligand_ids, candidate_rows):
    """Curated ligands to predict; they must not be labelled candidates."""
    by_id = {row["ligand_id"]: row for row in ligand_rows}
    missing = sorted(set(ligand_ids) - by_id.keys())
    if missing:
        raise ConfigError(f"prediction ligands not in the curated ligand table: {missing}")
    labelled = sorted(set(ligand_ids) & {row["ligand_id"] for row in candidate_rows})
    if labelled:
        raise ConfigError(f"prediction ligands are labelled candidates and could leak into evaluation: {labelled}")
    unusable = sorted(ligand_id for ligand_id in ligand_ids if not by_id[ligand_id]["smiles"])
    if unusable:
        raise ConfigError(f"prediction ligands without a structure: {unusable}")
    return [_input(by_id[ligand_id]) for ligand_id in ligand_ids]
