"""Feature matrices: PLEC interaction fingerprints of selected poses and RDKit Morgan fingerprints.

Both are written as a compressed SciPy CSR matrix (features.npz), the ordered row identities
(rows.tsv) and a feature schema (feature_schema.json). PLEC keeps the legacy configuration
with ODDT's inherited defaults made explicit. The sparse PLEC output (indices repeated once
per count) is converted to CSR with exact counts, and every row is compared with ODDT's
dense output. ODDT's dense vectors are uint8, so counts above 255 would wrap; such rows are
reported instead of hidden.
"""

import io
import json
from pathlib import Path

import numpy as np

from .. import __version__
from ..fileio import write_bytes_atomic, write_json
from ..tables import read_table, write_table
from . import ChemistryError

OUTPUT_FILES = ("features.npz", "rows.tsv", "feature_schema.json")


def sparse_counts_to_row(indices, size, count_bits):
    """(columns, values) of one CSR row from ODDT's sparse PLEC indices."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= size):
        raise ChemistryError("PLEC index outside the fingerprint size")
    columns, counts = np.unique(indices, return_counts=True)
    values = counts if count_bits else np.ones_like(columns)
    return columns, values.astype(np.int32)


def csr_from_rows(rows, size):
    from scipy import sparse

    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    for number, (columns, _values) in enumerate(rows, start=1):
        indptr[number] = indptr[number - 1] + len(columns)
    indices = np.concatenate([columns for columns, _ in rows]) if rows else np.zeros(0, dtype=np.int64)
    data = np.concatenate([values for _, values in rows]) if rows else np.zeros(0, dtype=np.int32)
    return sparse.csr_matrix((data.astype(np.int32), indices.astype(np.int32), indptr), shape=(len(rows), size))


def write_features(output_dir, matrix, rows, schema):
    from scipy import sparse

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if matrix.shape[0] != len(rows):
        raise ChemistryError("feature rows do not match the matrix")
    stream = io.BytesIO()
    sparse.save_npz(stream, matrix.tocsr(), compressed=True)
    write_bytes_atomic(output_dir / "features.npz", stream.getvalue())
    write_table(output_dir / "rows.tsv", rows, "feature_row")
    write_json(output_dir / "feature_schema.json", schema)


def _sdf_blocks(text):
    blocks, current = [], []
    for line in text.splitlines(keepends=True):
        current.append(line)
        if line.startswith("$$$$"):
            blocks.append("".join(current))
            current = []
    if "".join(current).strip():
        raise ChemistryError("SDF text does not end with $$$$")
    return blocks


def _properties(block):
    properties, lines = {}, block.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(">") and "<" in line:
            name = line[line.index("<") + 1:line.rindex(">")]
            properties[name] = lines[index + 1] if index + 1 < len(lines) else ""
    return properties


def featurize_plec(poses_dir, receptor_dir, cfg, output_dir):
    """PLEC for every selected pose against the prepared receptor PDBQT (as in the legacy pipeline)."""
    import hashlib

    import oddt
    from oddt.fingerprints import PLEC
    from oddt.toolkits import ob as toolkit

    receptor_pdbqt = Path(receptor_dir) / "receptor.pdbqt"
    structure = json.loads((Path(receptor_dir) / "receptor_structure.json").read_text(encoding="utf-8"))
    protein = next(toolkit.readfile("pdbqt", str(receptor_pdbqt)))
    protein.protein = True
    blocks = _sdf_blocks((Path(poses_dir) / "selected_poses.sdf").read_text(encoding="utf-8"))
    blocks.sort(key=lambda block: _properties(block)["ligand_id"])
    options = {"depth_ligand": cfg["depth_ligand"], "depth_protein": cfg["depth_protein"],
               "distance_cutoff": cfg["distance_cutoff"], "size": cfg["size"], "count_bits": cfg["count_bits"],
               "ignore_hoh": cfg["ignore_hoh"]}
    matrix_rows, rows, mismatched, overflow, max_count = [], [], [], [], 0
    for number, block in enumerate(blocks):
        properties = _properties(block)
        ligand = toolkit.readstring("sdf", block)
        sparse_fp = PLEC(ligand, protein, sparse=True, **options)
        dense_fp = PLEC(ligand, protein, sparse=False, **options)
        columns, values = sparse_counts_to_row(sparse_fp, cfg["size"], cfg["count_bits"])
        expanded = np.zeros(cfg["size"], dtype=np.int64)
        expanded[columns] = values
        if values.size:
            max_count = max(max_count, int(values.max()))
        if not np.array_equal(expanded, np.asarray(dense_fp, dtype=np.int64)):
            (overflow if values.size and values.max() > 255 else mismatched).append(properties["sample_id"])
        matrix_rows.append((columns, values))
        rows.append({"row": number, "sample_id": properties["sample_id"], "ligand_id": properties["ligand_id"],
                     "receptor_id": properties["receptor_id"], "structure_id": properties["structure_id"],
                     "state_id": properties["state_id"], "pose_rank": int(properties["pose_rank"]),
                     "group_id": properties["group_id"]})
    if mismatched:
        raise ChemistryError(f"sparse and dense PLEC differ for {mismatched}")
    schema = {
        "representation": "plec",
        "schema_version": 1,
        "package_version": __version__,
        "parameters": cfg,
        "toolkit": {"oddt": oddt.__version__, "backend": "openbabel", "openbabel": toolkit.__version__,
                    "hash": "Python hash() of integer tuples (deterministic; tuple hashing changed in Python 3.8)"},
        "inputs": {"protein": "receptor.pdbqt (Meeko; polar hydrogens) read with Open Babel, protein=True",
                   "ligand": "selected_poses.sdf (Meeko export with hydrogens)",
                   "structure_id": structure["structure_id"],
                   "receptor_pdbqt_sha256": hashlib.sha256(receptor_pdbqt.read_bytes()).hexdigest()},
        "matrix": {"file": "features.npz", "format": "scipy.sparse CSR, int32 counts" if cfg["count_bits"]
                   else "scipy.sparse CSR, int32 bits", "n_rows": len(rows), "n_features": cfg["size"]},
        "dense_sparse_check": {"rows_checked": len(rows), "equal": len(rows) - len(overflow),
                               "uint8_overflow_rows": overflow, "max_count": max_count},
        "outputs": list(OUTPUT_FILES),
    }
    write_features(output_dir, csr_from_rows(matrix_rows, cfg["size"]), rows, schema)
    return schema


def featurize_morgan(ligands_dir, cfg, receptor_id, output_dir):
    """Morgan fingerprints of each prepared parent ligand (ligand-only baseline; no pose or receptor)."""
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=cfg["radius"], fpSize=cfg["size"])
    outcomes = sorted((row for row in read_table(Path(ligands_dir) / "ligand_outcomes.tsv", "ligand_outcome")
                       if row["status"] == "prepared"), key=lambda row: row["ligand_id"])
    matrix_rows, rows = [], []
    for number, outcome in enumerate(outcomes):
        mol = Chem.MolFromSmiles(outcome["parent_smiles"])
        vector = (generator.GetCountFingerprintAsNumPy(mol) if cfg["counts"]
                  else generator.GetFingerprintAsNumPy(mol)).astype(np.int64)
        columns = np.flatnonzero(vector)
        matrix_rows.append((columns, vector[columns].astype(np.int32)))
        rows.append({"row": number, "sample_id": f"{outcome['ligand_id']}@{receptor_id}",
                     "ligand_id": outcome["ligand_id"], "receptor_id": receptor_id, "structure_id": None,
                     "state_id": None, "pose_rank": None,
                     "group_id": (outcome["parent_inchikey"] or outcome["ligand_id"])[:14]})
    schema = {
        "representation": "morgan",
        "schema_version": 1,
        "package_version": __version__,
        "parameters": cfg,
        "toolkit": {"rdkit": rdkit.__version__, "generator": "rdFingerprintGenerator.GetMorganGenerator"},
        "inputs": {"ligand": "standardized neutral parent SMILES from ligand_outcomes.tsv"},
        "matrix": {"file": "features.npz", "format": "scipy.sparse CSR, int32 " + ("counts" if cfg["counts"] else "bits"),
                   "n_rows": len(rows), "n_features": cfg["size"]},
        "outputs": list(OUTPUT_FILES),
    }
    write_features(output_dir, csr_from_rows(matrix_rows, cfg["size"]), rows, schema)
    return schema
