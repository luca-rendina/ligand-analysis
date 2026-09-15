import unittest

import numpy as np

from ensemble_functions import ensemble_model
from utility_functions import measure


class FeatureClassifier:
    """Predict the supplied feature so validation counts are known exactly."""

    def fit(self, features, labels):
        self.training_size = len(labels)
        return self

    def predict(self, features):
        return features[:, 0].astype(int)


class EnsembleTests(unittest.TestCase):
    def test_leave_one_out_preserves_counts_and_refits_all_training_samples(self):
        model = ensemble_model()
        model.ClassifierList = [(FeatureClassifier, {})]
        features = np.array([[0]] * 8 + [[1], [0], [1], [1]])
        labels = np.array([0] * 9 + [1] * 3)
        for kwargs in ({}, {"norm": False}):
            with self.subTest(kwargs=kwargs):
                result = model.model_fit_leaveOneOut(features, labels, **kwargs)
                classifier, counts = result[0]
                np.testing.assert_array_equal(counts, [[8, 1], [1, 2]])
                self.assertEqual(classifier.training_size, 12)
                self.assertAlmostEqual(measure("accuracy", counts), 10 / 12)
                self.assertEqual(len(model.filter_ensemble(result, treshold=0.8)), 1)
                self.assertEqual(len(model.filter_ensemble(result, treshold=0.9)), 0)

    def test_vote_conditions_on_predicted_class(self):
        for prediction, counts in [(1, [[99, 1], [70, 30]]),
                                   (0, [[30, 70], [1, 99]])]:
            with self.subTest(prediction=prediction):
                result = ensemble_model().predict_ensemble(
                    [(FeatureClassifier(), np.array(counts))],
                    np.array([[prediction]]), np.array([prediction]))
                self.assertEqual(result[prediction][prediction], 1)

    def test_votes_are_normalized_per_classifier(self):
        class PredictOne(FeatureClassifier):
            def predict(self, features):
                return np.ones(len(features), dtype=int)

        models = [(FeatureClassifier(), np.array([[9, 1], [1, 9]])),
                  (PredictOne(), np.array([[500, 500], [400, 600]]))]
        result = ensemble_model().predict_ensemble(models, np.array([[0]]), np.array([0]))
        self.assertEqual(result, [[1, 0], [0, 0]])

    def test_unseen_prediction_uses_hard_vote(self):
        result = ensemble_model().predict_ensemble(
            [(FeatureClassifier(), np.array([[7, 0], [3, 0]]))],
            np.array([[1]]), np.array([1]))
        self.assertEqual(result, [[0, 0], [0, 1]])

    def test_empty_ensemble_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty ensemble"):
            ensemble_model().predict_ensemble([], np.array([[0]]), np.array([0]))
