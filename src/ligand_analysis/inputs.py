"""Identity-only ligand input tables for preparation and docking.

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

