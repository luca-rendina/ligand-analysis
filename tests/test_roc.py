from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import pipeline_functions as pipeline


class RocTests(unittest.TestCase):
    def test_roc_uses_continuous_scores_and_respects_class_order(self):
        truth = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.45])
        for classes in ([0, 1], [1, 0]):
            for method in ("predict_proba", "decision_function"):
                with self.subTest(classes=classes, method=method):
                    model = SimpleNamespace(classes_=np.array(classes),
                                            predict=Mock(side_effect=AssertionError("hard labels used")))
                    if method == "predict_proba":
                        values = np.column_stack([scores if label == 1 else 1 - scores
                                                  for label in classes])
                    else:
                        values = scores if classes[1] == 1 else -scores
                    setattr(model, method, Mock(return_value=values))
                    with tempfile.TemporaryDirectory() as folder:
                        with patch.object(pipeline, "output_folder", folder + "/"):
                            plot = pipeline.save_roc([(model, None)], np.zeros((4, 1)),
                                                     truth, "receptor", "structure")
                        line = plot.gca().lines[0]
                        fpr, tpr, _ = roc_curve(truth, scores, pos_label=1)
                        np.testing.assert_allclose(line.get_xdata(), fpr)
                        np.testing.assert_allclose(line.get_ydata(), tpr)
                        self.assertIn(f"auc {roc_auc_score(truth, scores):.2f}", line.get_label())
                        self.assertTrue((Path(folder) / "receptor_structure_ROC.png").is_file())
                        plot.close("all")
                    model.predict.assert_not_called()

    def test_model_without_continuous_scores_is_rejected(self):
        model = SimpleNamespace(predict=Mock(return_value=[0, 0, 1, 1]))
        with self.assertRaisesRegex(ValueError, "predict_proba or decision_function"):
            pipeline.save_roc([(model, None)], np.zeros((4, 1)),
                              np.array([0, 0, 1, 1]), "receptor", "structure")
