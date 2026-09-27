"""Split, training, evaluation, prediction and identity-only input tests on synthetic features.

Feature values are random bits from a seed, not molecules or docking results; this is a
software regression check only.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import yaml

from ligand_analysis import cli
from ligand_analysis.chem.features import csr_from_rows, write_features
from ligand_analysis.config import ConfigError
from ligand_analysis.inputs import ligands_by_id, selected_candidates
from ligand_analysis.legacy.ensemble_functions import ensemble_model
from ligand_analysis.ml import (
    FeatureProvenanceWarning,
    LegacyEnsemble,
    ModelError,
    binary_metrics,
    evaluate,
    load_bundle,
    make_split,
    predict,
    train,
)
from ligand_analysis.tables import read_table, write_table

ROOT = Path(__file__).resolve().parents[1]
RECEPTOR = "P0TST1"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MODELS_CFG = {"seed": 0, "classifiers": ["dummy", "logistic_regression", "random_forest", "legacy_ensemble"],
              "legacy_loo_threshold": 0.6}
SPLIT_CFG = {"method": "stratified_connectivity_group", "test_fraction": 0.3, "seed": 0}


def block(number):
    return "".join(LETTERS[(number // 26 ** power) % 26] for power in range(14))


def candidate(number, label, rank, group=None):
    key = group or block(number)
    return {"ligand_id": f"iuphar.ligand:{number}", "receptor_id": RECEPTOR, "label": label,
            "class_name": ("agonist", "antagonist")[label], "selected": rank is not None, "selection_rank": rank,
            "selection_note": None, "name": f"fake-{number}", "approved": False, "smiles": "C" * (number % 5 + 1),
            "inchikey": f"{key}-UHFFFAOYSA-N", "connectivity_key": key, "pubchem_cid": number,
            "chembl_id": None, "original_terms": "Agonist / Agonist" if label == 0 else "Antagonist / Antagonist",
            "pubmed_ids": None, "n_annotations": 1}


def candidates():
    rows = [candidate(number, 0, number) for number in range(1, 8)]
    rows += [candidate(number, 1, number - 10) for number in range(11, 18)]
    rows[1] = candidate(2, 0, 2, group=block(1))  # ligands 1 and 2 share a connectivity block
    rows.append(candidate(30, 1, None))  # eligible but not selected
    return rows


def features(sample_ligands, seed, labels, representation="fake", size=64):
    rng = np.random.default_rng(seed)
    rows, matrix_rows = [], []
    for index, (ligand_id, label) in enumerate(zip(sample_ligands, labels, strict=True)):
        probability = np.full(size, 0.15)
        probability[(0 if label == 0 else 8):(8 if label == 0 else 16)] = 0.85
        vector = (rng.random(size) < probability).astype(np.int32)
        columns = np.flatnonzero(vector)
        matrix_rows.append((columns, vector[columns]))
        rows.append({"row": index, "sample_id": f"{ligand_id}@{RECEPTOR}", "ligand_id": ligand_id,
                     "receptor_id": RECEPTOR, "structure_id": None, "state_id": None, "pose_rank": None,
                     "group_id": "GROUP"})
    schema = {"representation": representation, "schema_version": 1, "parameters": {"size": size},
              "toolkit": {}, "matrix": {"file": "features.npz", "n_rows": len(rows), "n_features": size}}
    return csr_from_rows(matrix_rows, size), rows, schema


class Workspace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.candidates = candidates()
        self.labels_path = self.root / "candidates.tsv"
        write_table(self.labels_path, self.candidates, "candidate")
        self.inputs = selected_candidates(self.candidates, 10)
        self.split_rows = make_split(self.candidates, self.inputs, SPLIT_CFG)
        self.split_path = self.root / "split.tsv"
        write_table(self.split_path, self.split_rows, "split")
        ids = [row["ligand_id"] for row in self.inputs]
        labels = [next(c["label"] for c in self.candidates if c["ligand_id"] == ligand_id) for ligand_id in ids]
        self.features_dir = self.root / "features" / "fake"
        write_features(self.features_dir, *features(ids, 0, labels))
        self.ids, self.labels = ids, labels

    def tearDown(self):
        self.tmp.cleanup()


class InputTests(Workspace):
    def test_selected_candidates_are_identities_only(self):
        self.assertEqual(len(self.inputs), 14)
        self.assertEqual(set(self.inputs[0]), {"ligand_id", "name", "smiles", "inchikey"})
        self.assertNotIn("iuphar.ligand:30", {row["ligand_id"] for row in self.inputs})
        self.assertEqual(len(selected_candidates(self.candidates, 2)), 4)

    def test_prediction_inputs_must_not_be_labelled_candidates(self):
        ligand_rows = [{"ligand_id": "iuphar.ligand:99", "name": "x", "smiles": "CCN", "inchikey": None},
                       {"ligand_id": "iuphar.ligand:1", "name": "y", "smiles": "C", "inchikey": None}]
        self.assertEqual([row["ligand_id"] for row in ligands_by_id(ligand_rows, ["iuphar.ligand:99"],
                                                                    self.candidates)], ["iuphar.ligand:99"])
        with self.assertRaisesRegex(ConfigError, "could leak"):
            ligands_by_id(ligand_rows, ["iuphar.ligand:1"], self.candidates)
        with self.assertRaisesRegex(ConfigError, "not in the curated"):
            ligands_by_id(ligand_rows, ["iuphar.ligand:5"], self.candidates)


class SplitTests(Workspace):
    def test_split_is_stratified_grouped_and_deterministic(self):
        partitions = {row["ligand_id"]: row["partition"] for row in self.split_rows}
        self.assertEqual(partitions["iuphar.ligand:1"], partitions["iuphar.ligand:2"])
        for partition in ("train", "test"):
            classes = {row["class_name"] for row in self.split_rows if row["partition"] == partition}
            self.assertEqual(classes, {"agonist", "antagonist"})
        self.assertEqual(make_split(self.candidates, self.inputs, SPLIT_CFG), self.split_rows)

    def test_unlabelled_inputs_and_single_group_classes_are_rejected(self):
        with self.assertRaisesRegex(ModelError, "without a curated label"):
            make_split(self.candidates, [*self.inputs, {"ligand_id": "iuphar.ligand:99", "smiles": "C"}], SPLIT_CFG)
        few = [row for row in self.inputs if row["ligand_id"] in ("iuphar.ligand:1", "iuphar.ligand:11",
                                                                  "iuphar.ligand:12")]
        with self.assertRaisesRegex(ModelError, "agonist: 1 groups"):
            make_split(self.candidates, few, SPLIT_CFG)


class ModelTests(Workspace):
    def test_every_classifier_trains_evaluates_and_reloads_for_prediction(self):
        unlabeled = self.root / "unlabeled"
        write_features(unlabeled, *features(["new:1", "new:2"], 7, [0, 1]))
        for name in MODELS_CFG["classifiers"]:
            with self.subTest(model=name):
                model_dir = self.root / "models" / name
                metadata = train(self.features_dir, self.split_path, self.labels_path, name, MODELS_CFG, [],
                                 model_dir)
                self.assertEqual(sum(metadata["training"]["by_class"].values()),
                                 sum(1 for row in self.split_rows if row["partition"] == "train"))
                report = evaluate(model_dir, self.features_dir, self.split_path, self.labels_path, [],
                                  model_dir / "evaluation")
                test = report["metrics"]["test"]
                self.assertEqual(np.asarray(test["confusion_counts"]).sum(), test["n"])
                self.assertEqual(report["conventions"]["roc_auc_positive_class"], 1)
                rows = read_table(model_dir / "evaluation" / "predictions.tsv", "prediction")
                self.assertEqual({row["partition"] for row in rows}, {"train", "test"})
                predicted = predict(model_dir, unlabeled, self.root / f"{name}.tsv")
                self.assertEqual([row["partition"] for row in predicted], ["unlabeled"] * 2)
                self.assertTrue(all(row["true_label"] is None for row in predicted))

    def test_scores_are_oriented_to_class_1(self):
        model_dir = self.root / "lr"
        train(self.features_dir, self.split_path, self.labels_path, "logistic_regression", MODELS_CFG, [], model_dir)
        evaluate(model_dir, self.features_dir, self.split_path, self.labels_path, [], model_dir / "evaluation")
        rows = read_table(model_dir / "evaluation" / "predictions.tsv", "prediction")
        by_class = {label: np.mean([row["score_class1"] for row in rows if row["true_label"] == label])
                    for label in (0, 1)}
        self.assertGreater(by_class[1], by_class[0])

    def test_cohort_keeps_only_samples_every_representation_has(self):
        other = self.root / "features" / "other"
        kept = [ligand_id for ligand_id in self.ids if ligand_id != "iuphar.ligand:3"]
        write_features(other, *features(kept, 1, [0] * len(kept), representation="other"))
        metadata = train(self.features_dir, self.split_path, self.labels_path, "dummy", MODELS_CFG, [other],
                         self.root / "dummy")
        self.assertNotIn(f"iuphar.ligand:3@{RECEPTOR}", metadata["training"]["samples"])
        self.assertEqual(metadata["training"]["cohort_size"], len(kept))

    def test_legacy_ensemble_votes_match_legacy_predict_ensemble(self):
        matrix, _, _ = features(self.ids, 0, self.labels)
        dense, labels = matrix.toarray().astype(float), np.array(self.labels)
        ensemble = LegacyEnsemble(threshold=0.6).fit(dense, labels)
        counts = np.zeros((2, 2), dtype=int)
        for truth, predicted in zip(labels, ensemble.predict(dense), strict=True):
            counts[truth][predicted] += 1
        legacy = ensemble_model().predict_ensemble(ensemble.members_, dense, labels)
        self.assertEqual(counts.tolist(), np.asarray(legacy).tolist())
        scores = ensemble.score_class1(dense)
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))

    def test_bundle_checksum_and_feature_schema_are_verified(self):
        model_dir = self.root / "rf"
        train(self.features_dir, self.split_path, self.labels_path, "random_forest", MODELS_CFG, [], model_dir)
        other = self.root / "wider"
        write_features(other, *features(self.ids, 0, self.labels, size=32))
        with self.assertRaisesRegex(ModelError, "do not match the model's feature schema"):
            predict(model_dir, other, self.root / "out.tsv")
        with (model_dir / "model.joblib").open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(ModelError, "checksum"):
            load_bundle(model_dir)

    def test_evaluation_rejects_a_split_other_than_the_training_split(self):
        model_dir = self.root / "dummy"
        train(self.features_dir, self.split_path, self.labels_path, "dummy", MODELS_CFG, [], model_dir)
        moved = [dict(row, partition="test" if row["partition"] == "train" else "train") for row in self.split_rows]
        other_split = self.root / "other_split.tsv"
        write_table(other_split, moved, "split")
        with self.assertRaisesRegex(ModelError, "not the split the model was trained on"):
            evaluate(model_dir, self.features_dir, other_split, self.labels_path, [], model_dir / "evaluation")

    def test_feature_provenance_drift_warns_but_still_predicts(self):
        model_dir = self.root / "lr"
        train(self.features_dir, self.split_path, self.labels_path, "logistic_regression", MODELS_CFG, [], model_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FeatureProvenanceWarning)
            predict(model_dir, self.features_dir, self.root / "same.tsv")
        matrix, rows, schema = features(self.ids, 0, self.labels)
        schema.update(schema_version=2, toolkit={"rdkit": "2099.01.1"})
        drifted = self.root / "drifted"
        write_features(drifted, matrix, rows, schema)
        with self.assertWarnsRegex(FeatureProvenanceWarning, r"schema_version 1 -> 2; rdkit None -> 2099\.01\.1"):
            predicted = predict(model_dir, drifted, self.root / "drifted.tsv")
        self.assertEqual(len(predicted), len(self.ids))

    def test_ranking_metrics_use_unrounded_scores(self):
        model_dir = self.root / "lr"
        train(self.features_dir, self.split_path, self.labels_path, "logistic_regression", MODELS_CFG, [], model_dir)
        bundle, _ = load_bundle(model_dir)

        class NearTies:
            classes_ = np.array([0, 1])

            def predict(self, features):
                return np.zeros(features.shape[0], dtype=int)

            def predict_proba(self, features):
                # Class-1 samples score 1e-8 higher: identical after rounding to 6 decimals.
                high = np.asarray(features[:, :8].sum(axis=1) < features[:, 8:16].sum(axis=1), dtype=float)
                class1 = 0.5 + 1e-8 * high
                return np.column_stack([1 - class1, class1])

        bundle["estimator"] = NearTies()
        with patch("ligand_analysis.ml.load_bundle", return_value=(bundle, json.loads(
                (model_dir / "model.json").read_text(encoding="utf-8")))):
            report = evaluate(model_dir, self.features_dir, self.split_path, self.labels_path, [],
                              model_dir / "evaluation")
        rows = read_table(model_dir / "evaluation" / "predictions.tsv", "prediction")
        self.assertEqual({row["score_class1"] for row in rows}, {0.5})
        self.assertGreater(report["metrics"]["train"]["roc_auc_class1"], 0.5)

    def test_undefined_metrics_stay_unset(self):
        metrics = binary_metrics([0, 0, 0], [0, 1, 0], [0.1, 0.9, 0.2])
        self.assertEqual(metrics["confusion_counts"], [[2, 1], [0, 0]])
        self.assertIsNone(metrics["roc_auc_class1"])
        self.assertIsNone(metrics["balanced_accuracy"])
        self.assertEqual(metrics["per_class"]["agonist"]["support"], 3)


class ModelCliTests(Workspace):
    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main([str(arg) for arg in args])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_inputs_split_train_and_predict_commands(self):
        config = self.root / "config.yaml"
        config.write_text(yaml.safe_dump({"ligand_inputs": {"max_per_class": 10}, "split": SPLIT_CFG,
                                          "models": MODELS_CFG}), encoding="utf-8")
        code, out, err = self.run_cli("ligand-inputs", self.labels_path, "--config", config, "--output",
                                      self.root / "inputs.tsv")
        self.assertEqual(code, 0, err)
        self.assertEqual(len(read_table(self.root / "inputs.tsv", "ligand_input")), 14)
        code, out, err = self.run_cli("split", self.labels_path, self.root / "inputs.tsv", "--config", config,
                                      "--output", self.root / "cli_split.tsv")
        self.assertEqual(code, 0, err)
        self.assertEqual(read_table(self.root / "cli_split.tsv", "split"), self.split_rows)
        code, out, err = self.run_cli("train", "--features", self.features_dir, "--split", self.root / "cli_split.tsv",
                                      "--labels", self.labels_path, "--model", "logistic_regression", "--config",
                                      config, "--evaluate", "--output-dir", self.root / "model")
        self.assertEqual(code, 0, err)
        self.assertIn("Test: n=", out)
        metrics = json.loads((self.root / "model" / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["model"], "logistic_regression")
        code, out, err = self.run_cli("predict", "--model-dir", self.root / "model", "--features", self.features_dir,
                                      "--output", self.root / "predictions.tsv")
        self.assertEqual(code, 0, err)
        self.assertIn("14 predictions written", out)

    def test_training_without_both_classes_fails_cleanly(self):
        only_agonists = [row for row in self.split_rows if row["class_name"] == "agonist"]
        write_table(self.split_path, only_agonists, "split")
        config = self.root / "models.yaml"
        config.write_text(yaml.safe_dump({"models": MODELS_CFG}), encoding="utf-8")
        code, _, err = self.run_cli("train", "--features", self.features_dir, "--split", self.split_path, "--labels",
                                    self.labels_path, "--model", "dummy", "--config", config, "--output-dir",
                                    self.root / "m")
        self.assertEqual(code, 1)
        self.assertIn("at least two samples per class", err)


if __name__ == "__main__":
    unittest.main()
