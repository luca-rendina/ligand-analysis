"""Small software-regression checks for M4 output comparison."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAVE_CHEM = importlib.util.find_spec("rdkit") is not None


@unittest.skipUnless(HAVE_CHEM, "comparison requires ligand-chem")
class KubernetesComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import runpy

        cls.compare = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "k8s_compare.py"))

    def test_pose_coordinates_cannot_be_aligned_away(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Geometry import Point3D

        with tempfile.TemporaryDirectory() as temporary:
            left, right = (Path(temporary) / name for name in ("podman", "k8s"))
            for root, displacement in ((left, 0.0), (right, 0.11)):
                directory = root / "labelled" / "poses"
                directory.mkdir(parents=True)
                molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
                AllChem.EmbedMolecule(molecule, randomSeed=42)
                conformer = molecule.GetConformer()
                for atom in molecule.GetAtoms():
                    p = conformer.GetAtomPosition(atom.GetIdx())
                    conformer.SetAtomPosition(atom.GetIdx(), Point3D(p.x + displacement, p.y, p.z))
                molecule.SetProp("sample_id", "ethanol@receptor")
                writer = Chem.SDWriter(str(directory / "selected_poses.sdf"))
                writer.write(molecule)
                writer.close()
            with self.assertRaisesRegex(AssertionError, "in-place symmetry-aware RMSD"):
                self.compare["compare_poses"](left, right, "labelled")

    def test_outcome_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = (Path(temporary) / name for name in ("podman", "k8s"))
            for root, status in ((left, "prepared"), (right, "failed")):
                root.mkdir()
                (root / "outcomes.tsv").write_text(
                    "ligand_id\tstatus\nL1\t" + status + "\n", encoding="utf-8"
                )
            with self.assertRaisesRegex(AssertionError, "differing"):
                self.compare["selected_fields"](
                    left, right, "outcomes.tsv", ("status",), ("ligand_id",)
                )

    def test_failed_report_is_not_compared_as_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = (Path(temporary) / name for name in ("podman", "k8s"))
            for root, status in ((left, "success"), (right, "failed")):
                report = root / "report" / "report.json"
                report.parent.mkdir(parents=True)
                report.write_text(json.dumps({"status": status}), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "report status is failed"):
                self.compare["compare"](left, right)


if __name__ == "__main__":
    unittest.main()
