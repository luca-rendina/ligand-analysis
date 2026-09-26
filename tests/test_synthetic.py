import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ligand_analysis import cli, synthetic
from ligand_analysis.legacy.utility_functions import measure

SMALL = {"n_per_class": 12, "n_features": 32, "n_informative": 6}


class SyntheticExampleTests(unittest.TestCase):
    def run_example(self, folder, seed=3, **kwargs):
        return synthetic.run(folder, seed=seed, **{**SMALL, **kwargs})

    def test_same_seed_gives_identical_outputs(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            reports = [self.run_example(first), self.run_example(second)]
            for report in reports:
                report.pop("created_utc")
            self.assertEqual(reports[0], reports[1])
            for name in ("report.md", "predictions.tsv"):
                self.assertEqual((Path(first) / name).read_bytes(), (Path(second) / name).read_bytes())
            written = json.loads((Path(first) / "report.json").read_text(encoding="utf-8"))
            written.pop("created_utc")
            self.assertEqual(written, reports[0])

    def test_report_is_explicit_about_data_and_conventions(self):
        with tempfile.TemporaryDirectory() as folder:
            report = self.run_example(folder)
            predictions = (Path(folder) / "predictions.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(report["data_kind"], "synthetic")
        self.assertIn("SYNTHETIC DATA", report["disclaimer"])
        self.assertEqual(report["class_mapping"], {"0": "agonist", "1": "antagonist"})
        self.assertEqual(report["conventions"]["legacy_scalar_metrics_positive_class"], 0)
        self.assertEqual(report["conventions"]["roc_auc_positive_class"], 1)

        counts = np.array(report["ensemble"]["test_confusion_counts"])
        test_sizes = [report["split"]["test"][label] for label in ("0", "1")]
        np.testing.assert_array_equal(counts.sum(axis=1), test_sizes)
        self.assertEqual(len(predictions) - 1, counts.sum())
        self.assertEqual(report["ensemble"]["legacy_metrics_class0_positive"],
                         {name: float(value) for name, value in measure("all", counts).items()})
        for label, name in ((0, "agonist"), (1, "antagonist")):
            self.assertEqual(report["ensemble"]["per_class"][name]["support"], counts[label].sum())
        self.assertIsNone(report["ensemble"]["roc_auc_class1"])
        for classifier in report["classifiers"]:
            self.assertEqual(np.sum(classifier["loo_confusion_counts"]),
                             sum(report["split"]["train"].values()))

    def test_global_numpy_random_state_is_restored(self):
        np.random.seed(123)
        expected = np.random.random()
        np.random.seed(123)
        with tempfile.TemporaryDirectory() as folder:
            self.run_example(folder)
        self.assertEqual(np.random.random(), expected)

    def test_existing_outputs_are_not_overwritten_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_example(folder)
            with self.assertRaises(FileExistsError):
                self.run_example(folder)
            self.run_example(folder, overwrite=True)

    def test_class1_scores_follow_class_order(self):
        scores = np.array([0.2, 0.7])
        for classes in ([0, 1], [1, 0]):
            with self.subTest(classes=classes):
                proba = np.column_stack([scores if label == 1 else 1 - scores for label in classes])
                model = SimpleNamespace(classes_=np.array(classes), predict_proba=Mock(return_value=proba))
                np.testing.assert_allclose(synthetic.class1_scores(model, None), scores)
                model = SimpleNamespace(classes_=np.array(classes),
                                        decision_function=Mock(return_value=scores if classes[1] == 1 else -scores))
                np.testing.assert_allclose(synthetic.class1_scores(model, None), scores)


class CliTests(unittest.TestCase):
    def test_synthetic_example_command_uses_data_dir(self):
        with tempfile.TemporaryDirectory() as folder, patch("sys.stdout"):
            self.assertEqual(cli.main(["synthetic-example", "--data-dir", folder, "--seed", "1"]), 0)
            self.assertTrue((Path(folder) / "runs" / "synthetic-seed1" / "report.json").is_file())

    def test_missing_data_dir_is_reported(self):
        with patch.dict("os.environ", clear=True), patch("sys.stderr") as stderr:
            self.assertEqual(cli.main(["synthetic-example"]), 1)
        self.assertIn("LIGAND_ANALYSIS_DATA_DIR", "".join(call.args[0] for call in stderr.write.call_args_list))
