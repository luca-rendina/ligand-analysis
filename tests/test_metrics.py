import unittest

import numpy as np
from sklearn import metrics

from ligand_analysis.legacy import utility_functions as utility


class MetricTests(unittest.TestCase):
    def test_metrics_match_sklearn_with_class_zero_positive(self):
        for counts in ([[80, 20], [5, 5]], [[0, 3], [4, 0]],
                       [[5, 0], [0, 7]], [[0, 5], [0, 8]], [[6, 0], [8, 0]]):
            with self.subTest(counts=counts):
                cm = np.array(counts)
                truth = np.repeat([0, 0, 1, 1], cm.ravel())
                predicted = np.repeat([0, 1, 0, 1], cm.ravel())
                recall = metrics.recall_score(truth, predicted, labels=[0, 1],
                                              average=None, zero_division=0)
                expected = {
                    "accuracy": metrics.accuracy_score(truth, predicted),
                    "precision": metrics.precision_score(truth, predicted, pos_label=0, zero_division=0),
                    "sensitivity": recall[0], "fn_rate": 1 - recall[0],
                    "tn_rate": recall[1], "fp_rate": 1 - recall[1],
                    "f1_score": metrics.f1_score(truth, predicted, pos_label=0, zero_division=0),
                    "balanced_accuracy": metrics.balanced_accuracy_score(truth, predicted),
                    "matthews_correlation": metrics.matthews_corrcoef(truth, predicted),
                    "error_rate": 1 - metrics.accuracy_score(truth, predicted),
                }
                actual = utility.measure("all", cm)
                for name, value in expected.items():
                    self.assertAlmostEqual(actual[name], value, msg=name)
                np.testing.assert_allclose(utility.sensitivity(cm), recall)

    def test_empty_confusion_matrix_has_finite_zero_metrics(self):
        self.assertTrue(all(value == 0 for value in
                            utility.measure("all", np.zeros((2, 2))).values()))

    def test_normalization_handles_missing_class_without_mutating_counts(self):
        counts = np.array([[0, 0], [3, 1]])
        with np.errstate(divide="raise", invalid="raise"):
            normalized = utility.normalize_cm(counts)
        np.testing.assert_allclose(normalized, [[0, 0], [0.75, 0.25]])
        np.testing.assert_array_equal(counts, [[0, 0], [3, 1]])
