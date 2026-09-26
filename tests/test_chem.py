"""Chemistry-chain tests on a small real fixture (trimmed PDB 2RH1, CC0): receptor preparation,
ligand preparation, docking, pose selection, redocking and PLEC/Morgan features.

They need the ligand-chem image and are skipped elsewhere. Settings are reduced (small box,
exhaustiveness 1) to stay fast, so the poses are not a docking benchmark.
"""

import collections
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ligand_analysis.chem.features import sparse_counts_to_row
from ligand_analysis.pdbfile import THREE_TO_ONE, parse_atoms
from ligand_analysis.tables import read_table

HAVE_CHEM = all(importlib.util.find_spec(name) for name in ("rdkit", "meeko", "vina", "molscrub", "pdbfixer", "oddt"))
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "2rh1_pocket.pdb"
CARAZOLOL = {"resname": "CAU", "name": "carazolol", "smiles": "CC(C)NC[C@H](O)COc1cccc2[nH]c3ccccc3c12",
             "inchikey": "BQXQGZPYHWWCEB-ZDUSSCGKSA-N"}
RECEPTOR_CFG = {"structure": "2rh1", "chain": "A", "activation_state": "inactive", "reference_ligand": CARAZOLOL,
                "ph": 7.4, "residue_variants": {}, "pocket_radius": 5.0, "box": {"size": [20.0, 20.0, 20.0]},
                "seed": 1}
LIGAND_CFG = {"ph": 7.4, "max_stereoisomers": 1, "max_states": 4, "conformer_seed": 42, "forcefield": "mmff94s",
              "ring_fix": False}
DOCKING_CFG = {"engine": "vina", "scoring": "vina", "exhaustiveness": 1, "num_modes": 3, "energy_range": 10.0,
               "min_rmsd": 1.0, "seed": 42}
POSE_CFG = {"rule": "best_valid_score", "clash_distance": 2.2, "box_tolerance": 0.5}
PLEC_CFG = {"depth_ligand": 2, "depth_protein": 4, "size": 65536, "distance_cutoff": 4.5, "count_bits": True,
            "ignore_hoh": True, "backend": "ob"}
LIGANDS = [
    {"ligand_id": "test:s-propranolol", "name": "(S)-propranolol", "smiles": "CC(C)NC[C@H](O)COc1cccc2ccccc12",
     "inchikey": None},
    {"ligand_id": "test:racemic-propranolol", "name": "propranolol", "smiles": "CC(C)NCC(O)COc1cccc2ccccc12",
     "inchikey": None},
    {"ligand_id": "test:broken", "name": "broken", "smiles": "C1CC(", "inchikey": None},
]


def manifest():
    return {"receptor": {"uniprot_accession": "P07550", "gene_symbol": "ADRB2", "species": "Human", "taxon_id": 9606},
            "structures": {"2rh1": {"pdb_id": "2RH1", "format": "pdb", "url": "https://files.rcsb.org/download/2RH1.pdb",
                                    "sha256": "0" * 64, "release": "test", "license": "CC0 1.0"}}}


def uniprot_for_fixture(text):
    """A test sequence consistent with the fixture numbering; position 187 is N (engineered E187)."""
    sequence = ["A"] * 413
    for atom in parse_atoms(text):
        if atom.record == "ATOM" and atom.resnum < 1000:
            sequence[atom.resnum - 1] = THREE_TO_ONE[atom.resname]
    sequence[186] = "N"
    return {"primaryAccession": "P07550", "uniProtkbId": "ADRB2_HUMAN",
            "sequence": {"value": "".join(sequence), "length": 413}}


class SparseRowTests(unittest.TestCase):
    def test_repeated_indices_become_counts(self):
        columns, values = sparse_counts_to_row([5, 1, 5, 5], 8, count_bits=True)
        self.assertEqual((columns.tolist(), values.tolist()), ([1, 5], [1, 3]))
        columns, values = sparse_counts_to_row([5, 1, 5], 8, count_bits=False)
        self.assertEqual(values.tolist(), [1, 1])


@unittest.skipUnless(HAVE_CHEM, "needs the ligand-chem image")
class ChemistryChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ligand_analysis.chem.docking import dock_ligands
        from ligand_analysis.chem.features import featurize_morgan, featurize_plec
        from ligand_analysis.chem.ligands import prepare_ligands
        from ligand_analysis.chem.poses import select_poses
        from ligand_analysis.chem.receptor import prepare_receptor

        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls.tmp.name)
        text = FIXTURE.read_text(encoding="utf-8")
        cls.structure = prepare_receptor(text.encode(), uniprot_for_fixture(text), manifest(), RECEPTOR_CFG,
                                         root / "receptor")
        cls.ligand_summary = prepare_ligands(LIGANDS, LIGAND_CFG, root / "ligands")
        cls.docking = dock_ligands(root / "receptor", root / "ligands", [], DOCKING_CFG, 2, root / "docking")
        cls.selection = select_poses(LIGANDS, root / "ligands", [root / "docking"], root / "receptor", POSE_CFG,
                                     root / "poses")
        cls.plec = featurize_plec(root / "poses", root / "receptor", PLEC_CFG, root / "plec")
        cls.morgan = featurize_morgan(root / "ligands", {"radius": 2, "size": 2048, "counts": False}, "P07550",
                                      root / "morgan")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_receptor_keeps_mapped_residues_and_records_every_change(self):
        structure = self.structure
        self.assertEqual(structure["structure_id"], "pdb:2RH1")
        self.assertEqual(len(structure["retained"]["segments"]), 4)
        self.assertEqual([item["missing_residues"] for item in structure["chain_breaks"]], [63, 71, 8])
        removed = {item.get("resname") or item["source"]: item["count"] for item in structure["removed_components"]}
        self.assertEqual(removed, {"UNP P00720": 3, "HOH": 2, "SO4": 1})
        self.assertEqual([(item["residue"], item["uniprot"], item["structure"]) for item in structure["mutations"]],
                         [("A:187", "N187", "E")])
        self.assertIn("A:113", structure["qc"]["pocket_residues"])
        self.assertTrue(structure["qc"]["passed"])
        self.assertTrue(all(item["near_pocket"] is False for item in structure["chain_breaks"]))
        residues = read_table(self.root / "receptor" / "residues.tsv", "receptor_residue")
        self.assertTrue(all(row["template"] for row in residues))
        self.assertIn("ATOM", (self.root / "receptor" / "receptor.pdbqt").read_text(encoding="utf-8"))

    def test_reference_ligand_keeps_crystal_identity_and_defines_the_box(self):
        from rdkit import Chem

        mol = Chem.MolFromMolFile(str(self.root / "receptor" / "reference_ligand.sdf"), removeHs=False)
        self.assertEqual(Chem.MolToInchiKey(mol), CARAZOLOL["inchikey"])
        self.assertGreater(sum(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()), 0)
        box = json.loads((self.root / "receptor" / "box.json").read_text(encoding="utf-8"))
        self.assertEqual(box["center"], [-30.386, 9.489, 6.751])

    def test_receptor_preparation_is_reproducible(self):
        from ligand_analysis.chem.receptor import prepare_receptor

        text = FIXTURE.read_text(encoding="utf-8")
        prepare_receptor(text.encode(), uniprot_for_fixture(text), manifest(), RECEPTOR_CFG, self.root / "again")
        for name in ("receptor.pdb", "receptor.pdbqt", "residues.tsv", "reference_ligand.sdf"):
            self.assertEqual((self.root / "again" / name).read_bytes(), (self.root / "receptor" / name).read_bytes(),
                             name)

    def test_every_ligand_has_a_preparation_outcome(self):
        outcomes = {row["ligand_id"]: row for row in read_table(self.root / "ligands" / "ligand_outcomes.tsv",
                                                                 "ligand_outcome")}
        self.assertEqual({key: row["reason"] for key, row in outcomes.items()},
                         {"test:s-propranolol": None, "test:racemic-propranolol": "stereoisomer_limit",
                          "test:broken": "invalid_smiles"})
        states = read_table(self.root / "ligands" / "ligand_states.tsv", "ligand_state")
        self.assertEqual({row["ligand_id"] for row in states}, {"test:s-propranolol"})
        self.assertEqual(states[0]["formal_charge"], 1)
        self.assertIn("[NH2+]", states[0]["smiles"])
        self.assertTrue((self.root / "ligands" / states[0]["pdbqt_file"]).is_file())

    def test_docking_keeps_all_poses_and_selection_accounts_for_every_input(self):
        self.assertEqual(self.docking["counts"]["docked"], 1)
        poses = read_table(self.root / "docking" / "docking_poses.tsv", "docking_pose")
        self.assertTrue(1 <= len(poses) <= 3)
        self.assertEqual([row["pose_rank"] for row in poses], list(range(1, len(poses) + 1)))
        self.assertEqual([row["score"] for row in poses], sorted(row["score"] for row in poses))
        selection = {row["ligand_id"]: row for row in read_table(self.root / "poses" / "pose_selection.tsv",
                                                                  "pose_selection")}
        self.assertEqual(collections.Counter(row["status"] for row in selection.values()),
                         collections.Counter({"selected": 1, "rejected": 2}))
        self.assertEqual(selection["test:broken"]["reason"], "not_prepared")
        checks = read_table(self.root / "poses" / "pose_checks.tsv", "pose_check")
        self.assertTrue(all(row["identity_ok"] for row in checks))

    def test_selected_pose_preserves_identity_and_3d_coordinates(self):
        from rdkit import Chem

        from ligand_analysis.chem.poses import smiles_from_3d

        [pose] = Chem.SDMolSupplier(str(self.root / "poses" / "selected_poses.sdf"), removeHs=False)
        state = read_table(self.root / "ligands" / "ligand_states.tsv", "ligand_state")[0]
        self.assertEqual(smiles_from_3d(pose), state["smiles"])
        self.assertEqual(pose.GetConformer().Is3D(), True)
        self.assertEqual(len(pose.GetProp("state_atom_map").split(",")), pose.GetNumAtoms())

    def test_plec_sparse_counts_equal_the_dense_fingerprint(self):
        from scipy import sparse

        check = self.plec["dense_sparse_check"]
        self.assertEqual((check["rows_checked"], check["equal"], check["uint8_overflow_rows"]), (1, 1, []))
        matrix = sparse.load_npz(self.root / "plec" / "features.npz")
        self.assertEqual(matrix.shape, (1, 65536))
        self.assertGreater(matrix.sum(), 0)
        rows = read_table(self.root / "plec" / "rows.tsv", "feature_row")
        self.assertEqual(rows[0]["sample_id"], "test:s-propranolol@P07550")

    def test_morgan_rows_follow_prepared_ligands(self):
        rows = read_table(self.root / "morgan" / "rows.tsv", "feature_row")
        self.assertEqual([row["ligand_id"] for row in rows], ["test:s-propranolol"])

    def test_redocking_reports_rmsd_against_the_crystal_pose(self):
        from ligand_analysis.chem.poses import redock_reference

        report = redock_reference(self.root / "receptor", {**LIGAND_CFG}, DOCKING_CFG, {"rmsd_threshold": 2.0},
                                  POSE_CFG, 2, self.root / "redocking")
        self.assertTrue(1 <= len(report["poses"]) <= 3)
        self.assertTrue(np.isfinite(report["top_pose"]["rmsd"]))
        self.assertEqual(report["passed"], report["top_pose"]["rmsd"] <= 2.0)


if __name__ == "__main__":
    unittest.main()
