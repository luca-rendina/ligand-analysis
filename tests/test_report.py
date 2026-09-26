"""Run report tests on a minimal synthetic run directory (software regression only)."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from test_ml import MODELS_CFG, RECEPTOR, SPLIT_CFG, candidates, features

from ligand_analysis import cli
from ligand_analysis.chem.features import write_features
from ligand_analysis.fileio import write_json
from ligand_analysis.inputs import selected_candidates
from ligand_analysis.ml import evaluate, make_split, train
from ligand_analysis.report import collect
from ligand_analysis.tables import write_table

FAILED = "iuphar.ligand:4"


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name)
        rows = candidates()
        write_table(root / "curated" / "candidates.tsv", rows, "candidate")
        write_json(root / "curated" / "curation_report.json", {
            "gtopdb_release": {"version": "9999.1", "published": "2000-01-01"},
            "manifest": {"name": "fixture", "sha256": "0" * 64},
            "counts": {"annotated_ligands": 15, "excluded_ligands": 0, "eligible_by_class": {}, "selected_by_class": {}}})
        inputs = selected_candidates(rows, 10)
        write_table(root / "inputs.tsv", inputs, "ligand_input")
        write_json(root / "receptor" / "receptor_structure.json", {
            "structure_id": "pdb:0TST", "receptor_id": RECEPTOR, "gene_symbol": "TST1", "pdb_id": "0TST",
            "chain": "A", "activation_state": "inactive", "resolution_angstrom": 2.0, "source": {"url": "https://x"},
            "retained": {"residues": 10, "segments": [["A:1", "A:10"]]}, "chain_breaks": [], "mutations": [],
            "removed_components": [], "repairs": {"added_heavy_atoms": {}}, "protonation": {"histidines": {}},
            "box": {"center": [0, 0, 0], "size": [20, 20, 20]},
            "qc": {"passed": True, "failures": [], "pocket_residues": ["A:5"]}, "tools": {}})
        outcomes, states, tasks, selections = [], [], [], []
        for row in inputs:
            prepared = row["ligand_id"] != FAILED
            outcomes.append({"ligand_id": row["ligand_id"], "name": row["name"],
                             "status": "prepared" if prepared else "failed",
                             "reason": None if prepared else "invalid_smiles", "input_smiles": row["smiles"],
                             "n_stereoisomers": int(prepared), "n_states": int(prepared)})
            selections.append({"ligand_id": row["ligand_id"], "receptor_id": RECEPTOR, "structure_id": "pdb:0TST",
                               "status": "selected" if prepared else "rejected", "n_states": int(prepared),
                               "n_poses": int(prepared), "n_valid_poses": int(prepared),
                               "score": -7.0 if prepared else None, "reason": None if prepared else "not_prepared"})
            if prepared:
                state_id = f"{row['ligand_id']}#s1p1c1"
                states.append({"state_id": state_id, "ligand_id": row["ligand_id"], "stereo_index": 1,
                               "protomer_index": 1, "conformer_index": 1, "stereo_status": "no_stereo",
                               "smiles": row["smiles"], "inchikey": row["inchikey"], "formal_charge": 0,
                               "heavy_atoms": 3, "sdf_file": "states/x.sdf", "pdbqt_file": "states/x.pdbqt"})
                tasks.append({"state_id": state_id, "ligand_id": row["ligand_id"], "status": "docked", "n_poses": 1,
                              "engine": "test", "scoring": "vina", "seed": 1, "exhaustiveness": 1, "cpu": 1,
                              "config_sha256": "0" * 64, "seconds": 0.1})
        write_table(root / "ligands" / "ligand_outcomes.tsv", outcomes, "ligand_outcome")
        write_table(root / "ligands" / "ligand_states.tsv", states, "ligand_state")
        write_table(root / "docking" / "docking_tasks.tsv", tasks, "docking_task")
        write_table(root / "poses" / "pose_selection.tsv", selections, "pose_selection")
        write_json(root / "redocking" / "redocking.json", {
            "passed": True, "rmsd_threshold": 2.0, "top_pose": {"rmsd": 1.0, "pose_rank": 1, "score": -9.0},
            "best_rmsd_pose": {"rmsd": 1.0, "pose_rank": 1}, "method": "test", "engine": "test"})
        kept = [row["ligand_id"] for row in inputs if row["ligand_id"] != FAILED]
        labels = [next(c["label"] for c in rows if c["ligand_id"] == ligand_id) for ligand_id in kept]
        write_features(root / "features" / "fake", *features(kept, 0, labels))
        write_table(root / "split.tsv", make_split(rows, inputs, SPLIT_CFG), "split")
        model_dir = root / "models" / "fake_dummy"
        train(root / "features" / "fake", root / "split.tsv", root / "curated" / "candidates.tsv", "dummy",
              MODELS_CFG, [], model_dir)
        evaluate(model_dir, root / "features" / "fake", root / "split.tsv", root / "curated" / "candidates.tsv", [],
                 model_dir / "evaluation")

    def tearDown(self):
        self.tmp.cleanup()

    def arguments(self):
        root = self.root
        return ["--curated", root / "curated", "--ligand-inputs", root / "inputs.tsv", "--receptor-dir",
                root / "receptor", "--ligands-dir", root / "ligands", "--docking-dir", root / "docking", "--poses-dir",
                root / "poses", "--redocking-dir", root / "redocking", "--features", root / "features" / "fake",
                "--split", root / "split.tsv", "--model-dir", root / "models" / "fake_dummy", "--output-dir",
                root / "report"]

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["report", *[str(arg) for arg in args]])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_successful_run_accounts_for_every_input_ligand(self):
        root = self.root
        report = collect(root / "curated", root / "inputs.tsv", root / "receptor", root / "ligands",
                         [root / "docking"], root / "poses", root / "redocking", [root / "features" / "fake"],
                         root / "split.tsv", [root / "models" / "fake_dummy"])
        self.assertEqual(report["status"], "success", report["problems"])
        self.assertEqual(report["counts"]["inputs"], 14)
        failed = next(item for item in report["ligands"] if item["ligand_id"] == FAILED)
        self.assertEqual((failed["preparation"], failed["pose"], failed["in_cohort"]), ("failed", "rejected", False))
        self.assertEqual(report["counts"]["cohort"], 13)

    def test_cli_writes_html_and_json(self):
        code, _, err = self.run_cli(*self.arguments())
        self.assertEqual(code, 0, err)
        html = (self.root / "report" / "report.html").read_text(encoding="utf-8")
        self.assertIn("[success]", html)
        self.assertIn(FAILED, html)
        self.assertEqual(json.loads((self.root / "report" / "report.json").read_text(encoding="utf-8"))["status"],
                         "success")

    def test_failed_redocking_qc_gives_a_failed_report_and_exit_1(self):
        path = self.root / "redocking" / "redocking.json"
        content = json.loads(path.read_text(encoding="utf-8"))
        content.update(passed=False, top_pose={"rmsd": 4.2, "pose_rank": 1, "score": -9.0})
        write_json(path, content)
        code, _, err = self.run_cli(*self.arguments())
        self.assertEqual(code, 1)
        self.assertIn("redocking RMSD 4.2 A exceeds 2.0 A", err)
        self.assertIn("[failed]", (self.root / "report" / "report.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
