"""Persisted splits, baseline classifiers, the legacy ensemble, evaluation and prediction.

Labels are joined to feature rows only here, through (ligand_id, receptor_id). Every
representation uses the same persisted split restricted to the common cohort of samples that
all compared representations could featurize, so comparisons stay paired.
"""

import collections
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np

from . import __version__
from .fileio import write_json
from .labels import AGONIST, ANTAGONIST, CLASS_NAMES, join_labels
from .tables import read_table, write_table

CLASSES = (AGONIST, ANTAGONIST)
MODEL_FILES = ("model.joblib", "model.json")
SCORE_TYPES = {
    "dummy": "class prior probability of class 1 (DummyClassifier, strategy=prior)",
    "logistic_regression": "predict_proba for class 1",
    "random_forest": "predict_proba for class 1 (fraction of trees)",
    "legacy_ensemble": "legacy ensemble vote share for class 1 (sum of P(true class | member prediction) "
                       "from raw leave-one-out counts); not a calibrated probability",
}


class ModelError(RuntimeError):
    """Training, evaluation or prediction inputs are inconsistent."""


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_features(directory):
    from scipy import sparse

    directory = Path(directory)
    matrix = sparse.load_npz(directory / "features.npz").tocsr()
    rows = read_table(directory / "rows.tsv", "feature_row")
    schema = json.loads((directory / "feature_schema.json").read_text(encoding="utf-8"))
    if [row["row"] for row in rows] != list(range(matrix.shape[0])):
        raise ModelError(f"{directory}: rows.tsv does not list matrix rows 0..{matrix.shape[0] - 1} in order")
    if matrix.shape[1] != schema["matrix"]["n_features"]:
        raise ModelError(f"{directory}: matrix width differs from the feature schema")
    return matrix, rows, schema


def make_split(candidate_rows, ligand_rows, cfg):
    """Stratified train/test split of the input ligands; ligands of one connectivity block stay together."""
    candidates = {row["ligand_id"]: row for row in candidate_rows}
    missing = sorted(row["ligand_id"] for row in ligand_rows if row["ligand_id"] not in candidates)
    if missing:
        raise ModelError(f"input ligands without a curated label: {missing}")
    rows = [candidates[row["ligand_id"]] for row in ligand_rows]
    labels = join_labels(rows, candidate_rows)
    groups = collections.defaultdict(set)
    for row, label in zip(rows, labels, strict=True):
        groups[row["connectivity_key"]].add(label)
    mixed = sorted(group for group, values in groups.items() if len(values) > 1)
    if mixed:
        raise ModelError(f"connectivity groups with both labels cannot be split: {mixed}")
    rng = np.random.default_rng(cfg["seed"])
    test_groups = set()
    for label in CLASSES:
        names = sorted(group for group, values in groups.items() if label in values)
        n_test = round(len(names) * cfg["test_fraction"])
        if len(names) < 2 or not 1 <= n_test < len(names):
            raise ModelError(f"{CLASS_NAMES[label]}: {len(names)} groups cannot give both a train and a test group")
        test_groups.update(np.asarray(names)[rng.permutation(len(names))[:n_test]].tolist())
    return [{"ligand_id": row["ligand_id"], "receptor_id": row["receptor_id"], "group_id": row["connectivity_key"],
             "label": label, "class_name": CLASS_NAMES[label],
             "partition": "test" if row["connectivity_key"] in test_groups else "train"}
            for row, label in sorted(zip(rows, labels, strict=True), key=lambda item: item[0]["ligand_id"])]


class LegacyEnsemble:
    """The thesis ensemble (ligand_analysis.legacy): leave-one-out selection on the training data,
    then voting with P(true class | member prediction) from the raw out-of-fold counts."""

    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def fit(self, features, labels):
        from .legacy.ensemble_functions import ensemble_model

        model = ensemble_model()
        loo = model.model_fit_leaveOneOut(features, labels)
        selected = model.filter_ensemble(loo, treshold=self.threshold)
        if len(selected) == 0:
            raise ModelError(f"no legacy classifier reached leave-one-out accuracy {self.threshold}")
        self.members_ = model.fit_ensemble(selected, features, labels)
        selected_ids = {id(classifier) for classifier, _ in selected}
        self.leave_one_out_ = [{"classifier": type(classifier).__name__, "confusion_counts": np.asarray(counts).tolist(),
                                "selected": id(classifier) in selected_ids} for classifier, counts in loo]
        self.classes_ = np.array(CLASSES)
        return self

    def votes(self, features):
        """Summed votes per class, exactly as legacy predict_ensemble weighs members."""
        totals = np.zeros((features.shape[0], len(CLASSES)))
        for classifier, counts in self.members_:
            predicted = classifier.predict(features).astype(int)
            counts = np.asarray(counts, dtype=float)
            for row, index in enumerate(predicted):
                column = counts[:, index]
                totals[row] += column / column.sum() if column.sum() else np.eye(len(CLASSES))[index]
        return totals

    def predict(self, features):
        return np.argmax(self.votes(features), axis=1)

    def score_class1(self, features):
        votes = self.votes(features)
        return votes[:, ANTAGONIST] / votes.sum(axis=1)


def build_classifier(name, cfg):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    seed = cfg["seed"]
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logistic_regression":
        return LogisticRegression(C=1.0, solver="liblinear", max_iter=10000, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=1)
    if name == "legacy_ensemble":
        return LegacyEnsemble(threshold=cfg["legacy_loo_threshold"])
    raise ModelError(f"unknown classifier {name}")


def class1_scores(model, features):
    if isinstance(model, LegacyEnsemble):
        return model.score_class1(features)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, list(model.classes_).index(ANTAGONIST)]
    scores = model.decision_function(features)
    return scores if model.classes_[1] == ANTAGONIST else -scores


def cohort_ids(feature_dirs):
    """Sample IDs featurized by every given representation."""
    sets = [{row["sample_id"] for row in read_table(Path(directory) / "rows.tsv", "feature_row")}
            for directory in feature_dirs]
    return set.intersection(*sets) if sets else None


def _partition_rows(rows, split_rows, partition, cohort):
    split = {(row["ligand_id"], row["receptor_id"]): row for row in split_rows}
    return [row for row in rows if split.get((row["ligand_id"], row["receptor_id"]), {}).get("partition") == partition
            and (cohort is None or row["sample_id"] in cohort)]


def _environment():
    import sklearn

    return {"revision": os.environ.get("LIGAND_ANALYSIS_REVISION"), "python": platform.python_version(),
            "numpy": np.__version__, "scikit-learn": sklearn.__version__, "machine": platform.machine()}


def train(features_dir, split_path, labels_path, model_name, cfg, cohort_dirs, output_dir):
    import joblib

    matrix, rows, schema = load_features(features_dir)
    split_rows = read_table(split_path, "split")
    cohort = cohort_ids([features_dir, *cohort_dirs])
    train_rows = _partition_rows(rows, split_rows, "train", cohort)
    labels = np.array(join_labels(train_rows, read_table(labels_path, "candidate")))
    counts = {CLASS_NAMES[label]: int((labels == label).sum()) for label in CLASSES}
    if min(counts.values()) < 2:
        raise ModelError(f"training needs at least two samples per class, got {counts}")
    features = matrix[[row["row"] for row in train_rows]].toarray().astype(float)
    estimator = build_classifier(model_name, cfg).fit(features, labels)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {"estimator": estimator, "model": model_name, "representation": schema["representation"],
              "feature_parameters": schema["parameters"], "n_features": schema["matrix"]["n_features"]}
    joblib.dump(bundle, output_dir / "model.joblib")
    details = ({"leave_one_out": estimator.leave_one_out_,
                "members": [type(classifier).__name__ for classifier, _ in estimator.members_]}
               if isinstance(estimator, LegacyEnsemble) else {"params": _jsonable(estimator.get_params())})
    metadata = {
        "record_type": "model",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "package_version": __version__,
        "environment": _environment(),
        "model": model_name,
        "representation": schema["representation"],
        "feature_schema": {key: schema[key] for key in ("representation", "schema_version", "parameters", "toolkit")},
        "class_mapping": {str(label): name for label, name in CLASS_NAMES.items()},
        "score_type": SCORE_TYPES[model_name],
        "config": cfg,
        "training": {"samples": [row["sample_id"] for row in train_rows], "by_class": counts,
                     "split_sha256": file_sha256(split_path), "cohort_size": len(cohort) if cohort else None},
        "estimator": details,
        "model_sha256": file_sha256(output_dir / "model.joblib"),
        "outputs": list(MODEL_FILES),
        "security": "model.joblib is a pickle: load it only from runs you trust",
    }
    write_json(output_dir / "model.json", metadata)
    return metadata


def _jsonable(params):
    return {key: value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
            for key, value in sorted(params.items())}


def load_bundle(model_dir, schema=None):
    """Load a model bundle after checking its checksum and, if given, its feature schema."""
    import joblib

    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    if file_sha256(model_dir / "model.joblib") != metadata["model_sha256"]:
        raise ModelError(f"{model_dir / 'model.joblib'} does not match the checksum in model.json")
    bundle = joblib.load(model_dir / "model.joblib")
    if schema is not None and (schema["representation"] != bundle["representation"]
                               or schema["parameters"] != bundle["feature_parameters"]
                               or schema["matrix"]["n_features"] != bundle["n_features"]):
        raise ModelError(f"features ({schema['representation']}) do not match the model's feature schema "
                         f"({bundle['representation']}); prepare them with the same configuration")
    return bundle, metadata


def _prediction_rows(bundle, metadata, rows, features, partition, labels=None):
    estimator = bundle["estimator"]
    predicted = np.asarray(estimator.predict(features)).astype(int)
    scores = class1_scores(estimator, features)
    return [{"sample_id": row["sample_id"], "ligand_id": row["ligand_id"], "receptor_id": row["receptor_id"],
             "representation": bundle["representation"], "model": bundle["model"], "partition": partition,
             "true_label": None if labels is None else int(labels[index]), "predicted_label": int(predicted[index]),
             "predicted_class": CLASS_NAMES[int(predicted[index])], "score_class1": round(float(scores[index]), 6),
             "score_type": metadata["score_type"]}
            for index, row in enumerate(rows)]


def binary_metrics(truth, predicted, scores):
    """Metrics with raw counts (rows true, columns predicted); undefined values stay None."""
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    from .legacy.utility_functions import measure

    truth, predicted, scores = np.asarray(truth), np.asarray(predicted), np.asarray(scores, dtype=float)
    counts = confusion_matrix(truth, predicted, labels=list(CLASSES))
    both = len(set(truth.tolist())) == 2
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=list(CLASSES),
                                                                     zero_division=0)
    return {
        "n": int(truth.size),
        "confusion_counts": counts.tolist(),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)) if both else None,
        "mcc": float(matthews_corrcoef(truth, predicted)) if both else None,
        "per_class": {CLASS_NAMES[label]: {"label": label, "precision": float(precision[index]),
                                           "recall": float(recall[index]), "f1": float(f1[index]),
                                           "support": int(support[index])}
                      for index, label in enumerate(CLASSES)},
        "roc_auc_class1": float(roc_auc_score(truth, scores)) if both else None,
        "pr_auc_class1": float(average_precision_score(truth, scores)) if both else None,
        "pr_auc_class0": float(average_precision_score(1 - truth, -scores)) if both else None,
        "legacy_metrics_class0_positive": {name: float(value) for name, value in measure("all", counts).items()},
    }


def evaluate(model_dir, features_dir, split_path, labels_path, cohort_dirs, output_dir):
    matrix, rows, schema = load_features(features_dir)
    bundle, metadata = load_bundle(model_dir, schema)
    split_rows = read_table(split_path, "split")
    label_rows = read_table(labels_path, "candidate")
    cohort = cohort_ids([features_dir, *cohort_dirs])
    predictions, results = [], {}
    for partition in ("train", "test"):
        part_rows = _partition_rows(rows, split_rows, partition, cohort)
        if not part_rows:
            raise ModelError(f"no {partition} samples in the cohort")
        labels = np.array(join_labels(part_rows, label_rows))
        features = matrix[[row["row"] for row in part_rows]].toarray().astype(float)
        part_predictions = _prediction_rows(bundle, metadata, part_rows, features, partition, labels)
        predictions += part_predictions
        results[partition] = binary_metrics(labels, [row["predicted_label"] for row in part_predictions],
                                            [row["score_class1"] for row in part_predictions])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(output_dir / "predictions.tsv", predictions, "prediction")
    report = {
        "report_type": "evaluation",
        "package_version": __version__,
        "environment": _environment(),
        "model": bundle["model"],
        "representation": bundle["representation"],
        "model_sha256": metadata["model_sha256"],
        "split_sha256": file_sha256(split_path),
        "cohort_size": len(cohort) if cohort else None,
        "class_mapping": {str(label): name for label, name in CLASS_NAMES.items()},
        "conventions": {
            "confusion_counts": "raw counts; rows are true classes, columns are predictions",
            "legacy_scalar_metrics_positive_class": AGONIST,
            "roc_auc_positive_class": ANTAGONIST,
            "score": metadata["score_type"],
        },
        "metrics": results,
        "outputs": ["predictions.tsv", "metrics.json"],
    }
    write_json(output_dir / "metrics.json", report)
    return report


def predict(model_dir, features_dir, output_path):
    """Predict every feature row without labels, from a saved model bundle."""
    matrix, rows, schema = load_features(features_dir)
    bundle, metadata = load_bundle(model_dir, schema)
    if not rows:
        raise ModelError("no feature rows to predict")
    predictions = _prediction_rows(bundle, metadata, rows, matrix.toarray().astype(float), "unlabeled")
    write_table(output_path, predictions, "prediction")
    return predictions
