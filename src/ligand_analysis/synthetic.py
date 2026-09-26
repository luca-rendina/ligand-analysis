"""Seeded synthetic example that runs the legacy ensemble end to end.

The features are random bits, not molecules, docking poses or measurements. The
example is a software regression check only, never biological evidence.
"""

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import platform

import numpy as np
import pandas
import sklearn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from . import __version__
from .fileio import write_json, write_text_atomic
from .labels import AGONIST, ANTAGONIST, CLASS_NAMES
from .legacy.ensemble_functions import ensemble_model
from .legacy.utility_functions import measure

DISCLAIMER = ("SYNTHETIC DATA: random binary features generated from a seed. They are not "
              "molecules, docking results or pharmacology measurements; the metrics are "
              "software regression values, not biological performance.")
OUTPUT_FILES = ("report.json", "report.md", "predictions.tsv")
CLASSES = (AGONIST, ANTAGONIST)


def make_dataset(seed, n_per_class=30, n_features=64, n_informative=8):
    """Return features, labels and sample IDs with class-dependent bit frequencies."""
    if n_features < 2 * n_informative:
        raise ValueError("n_features must be at least twice n_informative.")
    rng = np.random.default_rng(seed)
    labels = np.repeat(CLASSES, n_per_class)
    probability = np.full((labels.size, n_features), 0.2)
    probability[labels == AGONIST, :n_informative] = 0.7
    probability[labels == ANTAGONIST, n_informative:2 * n_informative] = 0.7
    features = (rng.random(probability.shape) < probability).astype(float)
    sample_ids = np.array([f"synthetic-{index:04d}" for index in range(labels.size)])
    return features, labels, sample_ids


@contextmanager
def _numpy_global_seed(seed):
    # The legacy split shuffles with NumPy's global RNG; seed it only for that call.
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def class1_scores(model, features):
    """Continuous class-1 (antagonist) scores, oriented as in the legacy save_roc."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, list(model.classes_).index(ANTAGONIST)]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return scores if model.classes_[1] == ANTAGONIST else -scores
    raise ValueError("ROC requires predict_proba or decision_function.")


def _metrics(counts):
    return {name: float(value) for name, value in measure("all", np.asarray(counts)).items()}


def run(output_dir, seed=0, n_per_class=30, n_features=64, n_informative=8,
        test_ratio=0.3, threshold=0.6, overwrite=False):
    """Train, evaluate and write report.json, report.md and predictions.tsv."""
    output_dir = Path(output_dir)
    existing = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{output_dir} already contains {', '.join(existing)}; "
                              "choose another --output-dir or pass --overwrite.")

    features, labels, sample_ids = make_dataset(seed, n_per_class, n_features, n_informative)
    model = ensemble_model()
    # Carry the row index through the legacy split so predictions keep their sample IDs.
    indexed = np.column_stack([features, np.arange(labels.size)])
    with _numpy_global_seed(seed):
        train, train_labels, test, test_labels = model.split_data_train_test(
            indexed, labels, ratio=test_ratio)
    test_ids = sample_ids[test[:, -1].astype(int)]
    train, test = train[:, :-1], test[:, :-1]

    loo = model.model_fit_leaveOneOut(train, train_labels)
    selected = model.filter_ensemble(loo, treshold=threshold)
    if len(selected) == 0:
        raise RuntimeError(f"No classifier reached leave-one-out accuracy {threshold}.")
    members = model.fit_ensemble(selected, train, train_labels)

    counts = np.zeros((2, 2), dtype=int)
    predictions = []
    for row, truth in zip(test, test_labels, strict=True):
        votes = np.asarray(model.predict_ensemble(members, row[None, :], np.array([truth])))
        predictions.append(int(np.argmax(votes[truth])))
        counts += votes
    predictions = np.array(predictions)

    selected_ids = {id(classifier) for classifier, _ in selected}
    classifiers = [{
        "name": type(classifier).__name__,
        "loo_confusion_counts": np.asarray(loo_counts).tolist(),
        "loo_metrics_class0_positive": _metrics(loo_counts),
        "selected_for_ensemble": id(classifier) in selected_ids,
        "test_roc_auc_class1": float(roc_auc_score(test_labels, class1_scores(classifier, test))),
    } for classifier, loo_counts in loo]

    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, predictions, labels=list(CLASSES), zero_division=0)
    report = {
        "report_type": "synthetic_ensemble_example",
        "data_kind": "synthetic",
        "disclaimer": DISCLAIMER,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "package_version": __version__,
        "environment": {
            "revision": os.environ.get("LIGAND_ANALYSIS_REVISION"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__,
            "machine": platform.machine(),
        },
        "parameters": {
            "seed": seed, "n_per_class": n_per_class, "n_features": n_features,
            "n_informative_per_class": n_informative, "test_ratio": test_ratio,
            "loo_metric": "accuracy", "loo_threshold": threshold,
        },
        "class_mapping": {str(label): name for label, name in CLASS_NAMES.items()},
        "conventions": {
            "confusion_counts": "raw counts; rows are true classes, columns are predictions",
            "legacy_scalar_metrics_positive_class": AGONIST,
            "roc_auc_positive_class": ANTAGONIST,
            "roc_auc_scores": "continuous predict_proba or decision_function outputs",
        },
        "split": {
            part: {str(label): int(np.sum(values == label)) for label in CLASSES}
            for part, values in (("train", train_labels), ("test", test_labels))
        },
        "classifiers": classifiers,
        "ensemble": {
            "members": [type(classifier).__name__ for classifier, _ in members],
            "test_confusion_counts": counts.tolist(),
            "legacy_metrics_class0_positive": _metrics(counts),
            "per_class": {
                CLASS_NAMES[label]: {"label": label, "precision": float(precision[index]),
                                     "recall": float(recall[index]), "f1": float(f1[index]),
                                     "support": int(support[index])}
                for index, label in enumerate(CLASSES)
            },
            "roc_auc_class1": None,
            "roc_auc_note": "Legacy ensemble voting returns hard classes, not a continuous score.",
        },
        "outputs": list(OUTPUT_FILES),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "report.json", report)
    write_text_atomic(output_dir / "report.md", _markdown(report))
    write_text_atomic(output_dir / "predictions.tsv", _predictions(test_ids, test_labels, predictions))
    return report


def _predictions(sample_ids, truth, predicted):
    stream = io.StringIO()
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(["sample_id", "true_label", "true_class", "predicted_label", "predicted_class"])
    for sample_id, true_label, predicted_label in sorted(zip(sample_ids, truth, predicted, strict=True)):
        writer.writerow([sample_id, int(true_label), CLASS_NAMES[int(true_label)],
                         int(predicted_label), CLASS_NAMES[int(predicted_label)]])
    return stream.getvalue()


def _markdown(report):
    names = [f"{label} {name}" for label, name in report["class_mapping"].items()]
    counts = report["ensemble"]["test_confusion_counts"]
    lines = [
        f"# Synthetic ensemble example (seed {report['parameters']['seed']})", "",
        f"> {report['disclaimer']}", "",
        ("Classes: 0 = agonist, 1 = antagonist. Confusion counts are raw; rows are true classes "
         "and columns are predictions. Legacy scalar metrics treat class 0 as positive; ROC AUC "
         "uses continuous class-1 scores."), "",
        "## Classifiers (leave-one-out on the training split)", "",
        "| Classifier | LOO accuracy | Selected | Test ROC AUC (class 1) |",
        "|---|---|---|---|",
    ]
    for item in report["classifiers"]:
        lines.append(f"| {item['name']} | {item['loo_metrics_class0_positive']['accuracy']:.3f} | "
                     f"{'yes' if item['selected_for_ensemble'] else 'no'} | "
                     f"{item['test_roc_auc_class1']:.3f} |")
    lines += ["", "## Ensemble on the test split", "",
              f"| true \\ predicted | {names[0]} | {names[1]} |", "|---|---|---|",
              f"| {names[0]} | {counts[0][0]} | {counts[0][1]} |",
              f"| {names[1]} | {counts[1][0]} | {counts[1][1]} |", "",
              "| Metric (class 0 positive) | Value |", "|---|---|"]
    lines += [f"| {name} | {value:.3f} |"
              for name, value in report["ensemble"]["legacy_metrics_class0_positive"].items()]
    lines += ["", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    lines += [f"| {item['label']} {name} | {item['precision']:.3f} | {item['recall']:.3f} | "
              f"{item['f1']:.3f} | {item['support']} |"
              for name, item in report["ensemble"]["per_class"].items()]
    lines += ["", f"Ensemble ROC AUC: not available. {report['ensemble']['roc_auc_note']}", ""]
    return "\n".join(lines)
