"""Static HTML/JSON run report that accounts for every input ligand.

The report joins labels only for display and evaluation summaries. Its status is "success"
only when the receptor and redocking QC pass, every input ligand has an outcome, both classes
remain in the training and test cohorts, and every requested model was evaluated.
"""

import collections
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path

from . import __version__
from .fileio import write_json, write_text_atomic
from .labels import CLASS_NAMES
from .tables import read_table

DISCLAIMER = ("Execution demonstration on a small public example (one receptor structure, about 20 ligands). "
              "The metrics show that the workflow runs end to end; they are not a scientific benchmark. "
              "Docking scores and poses are model inputs, never labels.")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect(curated_dir, ligand_inputs, receptor_dir, ligands_dir, docking_dirs, poses_dir, redocking_dir,
            feature_dirs, split_path, model_dirs, prediction_files=(), run_metadata=None):
    """Assemble the report dictionary from published stage outputs."""
    curation = _json(Path(curated_dir) / "curation_report.json")
    candidates = {row["ligand_id"]: row for row in read_table(Path(curated_dir) / "candidates.tsv", "candidate")}
    inputs = read_table(ligand_inputs, "ligand_input")
    structure = _json(Path(receptor_dir) / "receptor_structure.json")
    preparation = {row["ligand_id"]: row for row in read_table(Path(ligands_dir) / "ligand_outcomes.tsv",
                                                               "ligand_outcome")}
    states = collections.Counter(row["ligand_id"] for row in read_table(Path(ligands_dir) / "ligand_states.tsv",
                                                                        "ligand_state"))
    tasks = collections.defaultdict(list)
    for directory in docking_dirs:
        for task in read_table(Path(directory) / "docking_tasks.tsv", "docking_task"):
            tasks[task["ligand_id"]].append(task)
    selection = {row["ligand_id"]: row for row in read_table(Path(poses_dir) / "pose_selection.tsv", "pose_selection")}
    redocking = _json(Path(redocking_dir) / "redocking.json")
    features = {}
    for directory in feature_dirs:
        schema = _json(Path(directory) / "feature_schema.json")
        rows = read_table(Path(directory) / "rows.tsv", "feature_row")
        features[schema["representation"]] = {"schema": schema, "samples": {row["sample_id"] for row in rows}}
    split = {row["ligand_id"]: row for row in read_table(split_path, "split")}
    models = []
    for directory in model_dirs:
        metadata = _json(Path(directory) / "model.json")
        evaluation = _json(Path(directory) / "evaluation" / "metrics.json")
        predictions = read_table(Path(directory) / "evaluation" / "predictions.tsv", "prediction")
        models.append({"metadata": metadata, "evaluation": evaluation, "predictions": predictions})
    unlabeled = [row for path in prediction_files for row in read_table(path, "prediction")]

    receptor_id = structure["receptor_id"]
    cohort = set.intersection(*(item["samples"] for item in features.values())) if features else set()
    test_predictions = collections.defaultdict(dict)
    for model in models:
        for row in model["predictions"]:
            if row["partition"] == "test":
                key = f"{row['representation']}/{row['model']}"
                test_predictions[row["ligand_id"]][key] = row["predicted_class"]
    ligands = []
    for row in inputs:
        ligand_id = row["ligand_id"]
        prepared, chosen = preparation.get(ligand_id), selection.get(ligand_id)
        docked = [task for task in tasks.get(ligand_id, []) if task["status"] == "docked"]
        sample_id = f"{ligand_id}@{receptor_id}"
        label = candidates[ligand_id]["label"] if ligand_id in candidates else None
        ligands.append({
            "ligand_id": ligand_id, "name": row.get("name"),
            "class_name": CLASS_NAMES[label] if label is not None else None,
            "preparation": prepared["status"] if prepared else "missing",
            "preparation_reason": prepared["reason"] if prepared else None,
            "states": states.get(ligand_id, 0),
            "docked_states": len(docked),
            "failed_states": len(tasks.get(ligand_id, [])) - len(docked),
            "pose": chosen["status"] if chosen else "missing",
            "pose_reason": chosen["reason"] if chosen else None,
            "best_score": chosen["score"] if chosen else None,
            "selected_state": chosen["state_id"] if chosen else None,
            "features": sorted(name for name, item in features.items() if sample_id in item["samples"]),
            "in_cohort": sample_id in cohort,
            "partition": split[ligand_id]["partition"] if ligand_id in split else None,
            "test_predictions": test_predictions.get(ligand_id, {}),
        })

    problems = []
    if not structure["qc"]["passed"]:
        problems.append("receptor QC failed: " + "; ".join(structure["qc"]["failures"]))
    if not redocking["passed"]:
        problems.append(f"redocking RMSD {redocking['top_pose']['rmsd']} A exceeds {redocking['rmsd_threshold']} A")
    unaccounted = [item["ligand_id"] for item in ligands if item["preparation"] == "missing" or item["pose"] == "missing"]
    if unaccounted:
        problems.append(f"ligands without an outcome: {unaccounted}")
    by_partition = collections.Counter((item["partition"], item["class_name"]) for item in ligands if item["in_cohort"])
    for partition in ("train", "test"):
        for name in CLASS_NAMES.values():
            if by_partition[(partition, name)] < (2 if partition == "train" else 1):
                problems.append(f"too few {name} samples in the {partition} cohort")
    if not models:
        problems.append("no models were evaluated")
    comparison = [{
        "representation": model["metadata"]["representation"], "model": model["metadata"]["model"],
        "train_by_class": model["metadata"]["training"]["by_class"],
        "test": model["evaluation"]["metrics"]["test"], "train": model["evaluation"]["metrics"]["train"],
        "score_type": model["metadata"]["score_type"],
    } for model in sorted(models, key=lambda item: (item["metadata"]["representation"], item["metadata"]["model"]))]

    return {
        "report_type": "pipeline_run",
        "status": "failed" if problems else "success",
        "problems": problems,
        "disclaimer": DISCLAIMER,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "package_version": __version__,
        "revision": os.environ.get("LIGAND_ANALYSIS_REVISION"),
        "run": run_metadata or {},
        "class_mapping": {str(label): name for label, name in CLASS_NAMES.items()},
        "conventions": {"confusion_counts": "raw counts; rows are true classes, columns are predictions",
                        "roc_pr": "class 1 (antagonist) scores; PR AUC also reported for class 0",
                        "legacy_scalar_metrics_positive_class": 0},
        "curation": {"gtopdb_release": curation["gtopdb_release"], "manifest": curation["manifest"],
                     "counts": {key: curation["counts"][key] for key in ("annotated_ligands", "excluded_ligands",
                                                                          "eligible_by_class", "selected_by_class")}},
        "receptor": {key: structure[key] for key in ("structure_id", "receptor_id", "gene_symbol", "pdb_id", "chain",
                                                     "activation_state", "resolution_angstrom", "source", "retained",
                                                     "chain_breaks", "mutations", "removed_components", "repairs",
                                                     "protonation", "box", "qc", "tools")},
        "redocking": {key: redocking[key] for key in ("passed", "rmsd_threshold", "top_pose", "best_rmsd_pose",
                                                      "method", "engine")},
        "ligands": ligands,
        "counts": {
            "inputs": len(ligands),
            "prepared": sum(1 for item in ligands if item["preparation"] == "prepared"),
            "pose_selected": sum(1 for item in ligands if item["pose"] == "selected"),
            "cohort": sum(1 for item in ligands if item["in_cohort"]),
            "cohort_by_partition": {f"{partition}/{name}": count
                                    for (partition, name), count in sorted(by_partition.items(), key=str)},
        },
        "features": {name: {key: item["schema"][key] for key in ("parameters", "matrix", "toolkit")
                            if key in item["schema"]} | ({"dense_sparse_check": item["schema"]["dense_sparse_check"]}
                                                         if "dense_sparse_check" in item["schema"] else {})
                     for name, item in sorted(features.items())},
        "models": comparison,
        "unlabeled_predictions": unlabeled,
    }


def _cell(value):
    if value is None:
        return "&ndash;"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return html.escape(", ".join(map(str, value)))
    if isinstance(value, dict):
        return html.escape(", ".join(f"{key}: {item}" for key, item in value.items()))
    return html.escape(str(value))


def _table(headers, rows):
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{_cell(value)}</td>" for value in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


STYLE = ("<style>body{font-family:system-ui,sans-serif;margin:2em;max-width:1200px}table{border-collapse:collapse;"
         "margin:0.5em 0 1.5em}th,td{border:1px solid #ccc;padding:3px 8px;text-align:left;font-size:0.9em}"
         "th{background:#f0f0f0}.ok{color:#176117}.fail{color:#a00}.note{background:#fff8dc;padding:0.6em;"
         "border-left:4px solid #e0b000}</style>")


def render_html(report):
    receptor, redocking = report["receptor"], report["redocking"]
    status_class = "ok" if report["status"] == "success" else "fail"
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>ligand-analysis report: {html.escape(report['run'].get('name', 'run'))}</title>",
        STYLE + "</head><body>",
        f"<h1>ligand-analysis run report <span class='{status_class}'>[{html.escape(report['status'])}]</span></h1>",
        f"<p class='note'>{html.escape(report['disclaimer'])}</p>",
    ]
    if report["problems"]:
        parts.append("<h2 class='fail'>Problems</h2><ul>" +
                     "".join(f"<li>{html.escape(problem)}</li>" for problem in report["problems"]) + "</ul>")
    run = report["run"]
    containers = "; ".join(
        f"{key}: {value.get('image')} ({value.get('id') or 'id unknown'})" if isinstance(value, dict) else f"{key}: {value}"
        for key, value in (run.get("containers") or {}).items()) or None
    parts.append("<h2>Provenance</h2>" + _table(["Item", "Value"], [
        ["Created (UTC)", report["created_utc"]], ["Package version", report["package_version"]],
        ["Image revision", report["revision"]], ["Git revision", run.get("git_revision")],
        ["Nextflow", run.get("nextflow_version")], ["Run name / session", f"{run.get('run_name')} / {run.get('session_id')}"],
        ["Containers", containers], ["Configuration", run.get("config_file")],
        ["Class mapping", report["class_mapping"]], ["Conventions", report["conventions"]],
    ]))
    curation = report["curation"]
    parts.append("<h2>Data</h2>" + _table(["Item", "Value"], [
        ["GtoPdb release", curation["gtopdb_release"]], ["Manifest", curation["manifest"]],
        *[[key, value] for key, value in curation["counts"].items()], *[[key, value] for key, value in report["counts"].items()],
    ]))
    parts.append("<h2>Receptor</h2>" + _table(["Item", "Value"], [
        ["Structure", (f"{receptor['structure_id']} chain {receptor['chain']} ({receptor['activation_state']}, "
                       f"{receptor['resolution_angstrom']} A)")],
        ["Receptor", f"{receptor['gene_symbol']} {receptor['receptor_id']}"],
        ["Source", receptor["source"]], ["Retained segments", receptor["retained"]["segments"]],
        ["Chain breaks", [f"{item['after']}/{item['before']} ({item['missing_residues']} missing)"
                          for item in receptor["chain_breaks"]]],
        ["Mutations", [f"{item['uniprot']}{item['structure']} ({item['seqadv']})" for item in receptor["mutations"]]],
        ["Removed", [f"{item.get('resname') or item.get('source')} x{item['count']}"
                     for item in receptor["removed_components"]]],
        ["Added heavy atoms", receptor["repairs"]["added_heavy_atoms"]],
        ["Histidines", receptor["protonation"]["histidines"]],
        ["Box centre / size (A)", f"{receptor['box']['center']} / {receptor['box']['size']}"],
        ["Pocket residues", receptor["qc"]["pocket_residues"]],
        ["QC", "passed" if receptor["qc"]["passed"] else "; ".join(receptor["qc"]["failures"])],
        ["Tools", receptor["tools"]],
    ]))
    top, best = redocking["top_pose"], redocking["best_rmsd_pose"]
    parts.append("<h2>Redocking diagnostic</h2>" +
                 f"<p class='{'ok' if redocking['passed'] else 'fail'}'>Top-ranked pose RMSD {top['rmsd']:.2f} A "
                 f"(threshold {redocking['rmsd_threshold']} A): {'passed' if redocking['passed'] else 'failed'}. "
                 f"Best pose RMSD {best['rmsd']:.2f} A (rank {best['pose_rank']}).</p>"
                 f"<p>{html.escape(redocking['method'])}; {html.escape(redocking['engine'])}.</p>")
    parts.append("<h2>Every input ligand</h2>" + _table(
        ["Ligand", "Name", "Class", "Prepared", "States", "Docked", "Pose", "Score", "Features", "Cohort", "Partition",
         "Test predictions"],
        [[item["ligand_id"], item["name"], item["class_name"],
          item["preparation"] + (f" ({item['preparation_reason']})" if item["preparation_reason"] else ""),
          item["states"], f"{item['docked_states']} ok / {item['failed_states']} failed",
          item["pose"] + (f" ({item['pose_reason']})" if item["pose_reason"] else ""), item["best_score"],
          item["features"], item["in_cohort"], item["partition"],
          dict(collections.Counter(item["test_predictions"].values())) or None]
         for item in report["ligands"]]))
    parts.append("<h2>Features</h2>" + _table(["Representation", "Parameters", "Matrix", "Dense/sparse check"], [
        [name, item["parameters"], item["matrix"], item.get("dense_sparse_check")]
        for name, item in report["features"].items()]))
    parts.append("<h2>Models (test partition of the common cohort)</h2>" + _table(
        ["Representation", "Model", "Train by class", "Test n", "Confusion counts", "Balanced accuracy", "MCC",
         "ROC AUC (class 1)", "PR AUC (class 1)", "Recall agonist", "Recall antagonist"],
        [[item["representation"], item["model"], item["train_by_class"], item["test"]["n"],
          item["test"]["confusion_counts"], item["test"]["balanced_accuracy"], item["test"]["mcc"],
          item["test"]["roc_auc_class1"], item["test"]["pr_auc_class1"],
          item["test"]["per_class"]["agonist"]["recall"], item["test"]["per_class"]["antagonist"]["recall"]]
         for item in report["models"]]))
    columns = [f"{item['representation']}/{item['model']}" for item in report["models"]]
    tested = [item for item in report["ligands"] if item["test_predictions"]]
    if tested:
        parts.append("<h2>Test predictions per model</h2>" + _table(
            ["Ligand", "Name", "True class", *columns],
            [[item["ligand_id"], item["name"], item["class_name"],
              *[item["test_predictions"].get(column) for column in columns]] for item in tested]))
    score_types = sorted({(item["model"], item["score_type"]) for item in report["models"]})
    parts.append("<h2>Continuous scores</h2>" + _table(["Model", "Class-1 score"], score_types))
    if report["unlabeled_predictions"]:
        parts.append("<h2>Unlabeled predictions (not evaluated)</h2>" + _table(
            ["Ligand", "Representation", "Model", "Predicted class", "Class-1 score"],
            [[row["ligand_id"], row["representation"], row["model"], row["predicted_class"], row["score_class1"]]
             for row in report["unlabeled_predictions"]]))
    parts.append("</body></html>\n")
    return "\n".join(parts)


def write_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "report.json", report)
    write_text_atomic(output_dir / "report.html", render_html(report))
