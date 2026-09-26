"""Command-line interface: ``ligand-analysis <command>``."""

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .config import ConfigError, data_dir
from .labels import LabelJoinError


def _synthetic_example(args):
    from . import synthetic

    output_dir = args.output_dir or data_dir(args.data_dir) / "runs" / f"synthetic-seed{args.seed}"
    report = synthetic.run(output_dir, seed=args.seed, overwrite=args.overwrite)
    ensemble = report["ensemble"]
    print(f"Synthetic example (software regression only) written to {output_dir}")
    print(f"Ensemble members: {', '.join(ensemble['members'])}")
    print(f"Test confusion counts (rows true, columns predicted): {ensemble['test_confusion_counts']}")
    print(f"Accuracy: {ensemble['legacy_metrics_class0_positive']['accuracy']:.3f}")


def _snapshot_dir(args, manifest):
    return args.snapshot_dir or data_dir(args.data_dir) / "sources" / manifest["name"]


def _fetch(args):
    from . import curation
    from .config import load_manifest
    from .sources import Snapshot

    manifest = load_manifest(args.manifest)
    snapshot = Snapshot(_snapshot_dir(args, manifest), offline=args.offline, refresh=args.refresh)
    events = curation.fetch(manifest, snapshot)
    for source_id, url, digest, status in events:
        print(f"{status:10} {source_id:20} {digest[:12]} {url}")
    downloaded = sum(1 for event in events if event[3] == "downloaded")
    print(f"{downloaded} downloaded, {len(events) - downloaded} reused from {snapshot.root}")


def _curate(args):
    from . import curation
    from .config import load_manifest, manifest_sha256
    from .sources import Snapshot

    manifest = load_manifest(args.manifest)
    output_dir = args.output_dir or data_dir(args.data_dir) / "curated" / manifest["name"]
    snapshot = Snapshot(_snapshot_dir(args, manifest), offline=True)
    report = curation.curate(manifest, snapshot, output_dir, manifest_sha256=manifest_sha256(args.manifest))
    counts = report["counts"]
    print(f"Curated {report['receptor']['gene_symbol']} ({report['receptor']['receptor_id']}), "
          f"GtoPdb {report['gtopdb_release']['version']}, written to {output_dir}")
    print(f"Annotations: {counts['annotations']} for {counts['annotated_ligands']} ligands; "
          f"excluded ligands: {counts['excluded_ligands']}")
    print(f"Eligible: {counts['eligible_by_class']}; selected: {counts['selected_by_class']}")
    print(f"Issues: {counts['issues_by_type']}")


def _section(args, name):
    from .config import load_config

    return load_config(args.config, [name])[name]


def _ligand_inputs(args):
    from .inputs import selected_candidates
    from .tables import read_table, write_table

    rows = selected_candidates(read_table(args.candidates, "candidate"), _section(args, "ligand_inputs")["max_per_class"])
    write_table(args.output, rows, "ligand_input")
    print(f"{len(rows)} ligand inputs (identities only, no labels) written to {args.output}")


def _prediction_inputs(args):
    from .inputs import ligands_by_id
    from .tables import read_table, write_table

    ids = _section(args, "prediction")["ligand_ids"]
    rows = ligands_by_id(read_table(args.ligands, "ligand"), ids, read_table(args.candidates, "candidate"))
    write_table(args.output, rows, "ligand_input")
    print(f"{len(rows)} unlabeled prediction inputs written to {args.output}")


def _prepare_receptor(args):
    from .chem.receptor import prepare_receptor
    from .config import load_manifest
    from .curation import structure_source_id
    from .sources import Snapshot

    cfg = _section(args, "receptor")
    manifest = load_manifest(args.manifest)
    structure = manifest.get("structures", {}).get(cfg["structure"])
    if structure is None:
        raise ConfigError(f"structure {cfg['structure']!r} is not in {args.manifest}")
    snapshot = Snapshot(_snapshot_dir(args, manifest), offline=True)
    pdb = snapshot.get(structure_source_id(cfg["structure"]), structure["url"], structure["sha256"])
    uniprot_source = manifest["sources"]["uniprot_entry"]
    uniprot = json.loads(snapshot.get("uniprot_entry", uniprot_source["url"], uniprot_source.get("sha256")))
    metadata = prepare_receptor(pdb, uniprot, manifest, cfg, args.output_dir)
    qc = metadata["qc"]
    print(f"Prepared {metadata['structure_id']} chain {metadata['chain']}: {metadata['retained']['residues']} residues "
          f"in {len(metadata['retained']['segments'])} segments; box centre {metadata['box']['center']}")
    print(f"Pocket: {len(qc['pocket_residues'])} residues; QC {'passed' if qc['passed'] else 'FAILED'}; "
          f"written to {args.output_dir}")


def _prepare_ligands(args):
    from .chem.ligands import prepare_ligands
    from .tables import read_table

    summary = prepare_ligands(read_table(args.ligands, "ligand_input"), _section(args, "ligand_preparation"),
                              args.output_dir)
    counts = summary["counts"]
    print(f"Prepared {counts['prepared']} of {counts['inputs']} ligands ({counts['states']} states); "
          f"failed: {counts['failures_by_reason'] or 0}; written to {args.output_dir}")


def _dock(args):
    from .chem.docking import dock_ligands

    summary = dock_ligands(args.receptor_dir, args.ligands_dir, args.ligand_id or [], _section(args, "docking"),
                           args.cpu, args.output_dir)
    counts = summary["counts"]
    print(f"Docked {counts['docked']} of {counts['states']} states ({counts['poses']} poses, "
          f"{counts['failed']} failed) with {summary['engine']}; written to {args.output_dir}")


def _select_poses(args):
    from .chem.poses import select_poses
    from .tables import read_table

    summary = select_poses(read_table(args.ligands, "ligand_input"), args.ligands_dir, args.docking_dir,
                           args.receptor_dir, _section(args, "pose_selection"), args.output_dir)
    counts = summary["counts"]
    print(f"Selected poses for {counts['selected']} of {counts['ligands']} ligands; rejected: "
          f"{counts['rejected_by_reason'] or 0}; written to {args.output_dir}")


def _redock_reference(args):
    from .chem.poses import redock_reference
    from .config import load_config

    cfg = load_config(args.config, ["ligand_preparation", "docking", "redocking", "pose_selection"])
    report = redock_reference(args.receptor_dir, cfg["ligand_preparation"], cfg["docking"], cfg["redocking"],
                              cfg["pose_selection"], args.cpu, args.output_dir)
    top = report["top_pose"]
    print(f"Redocked {report['reference_ligand']['name']}: top pose RMSD {top['rmsd']:.2f} A (score {top['score']}), "
          f"threshold {report['rmsd_threshold']} A: {'passed' if report['passed'] else 'FAILED'}")


def _featurize(args):
    from .chem import features

    cfg = _section(args, "featurization")[args.representation]
    if args.representation == "plec":
        if not (args.poses_dir and args.receptor_dir):
            raise ConfigError("plec needs --poses-dir and --receptor-dir")
        schema = features.featurize_plec(args.poses_dir, args.receptor_dir, cfg, args.output_dir)
        check = schema["dense_sparse_check"]
        print(f"PLEC: {schema['matrix']['n_rows']} rows x {schema['matrix']['n_features']} features; dense/sparse "
              f"equal for {check['equal']} of {check['rows_checked']} rows (max count {check['max_count']})")
    else:
        receptor_id = args.receptor_id
        if args.receptor_table:
            from .tables import read_table

            rows = read_table(args.receptor_table, "receptor")
            if len(rows) != 1:
                raise ConfigError(f"{args.receptor_table} must hold exactly one receptor")
            receptor_id = rows[0]["receptor_id"]
        if not (args.ligands_dir and receptor_id):
            raise ConfigError("morgan needs --ligands-dir and --receptor-id or --receptor-table")
        schema = features.featurize_morgan(args.ligands_dir, cfg, receptor_id, args.output_dir)
        print(f"Morgan: {schema['matrix']['n_rows']} rows x {schema['matrix']['n_features']} features")


def _split(args):
    from .ml import make_split
    from .tables import read_table, write_table

    rows = make_split(read_table(args.candidates, "candidate"), read_table(args.ligands, "ligand_input"),
                      _section(args, "split"))
    write_table(args.output, rows, "split")
    counts = {partition: {name: sum(1 for row in rows if row["partition"] == partition and row["class_name"] == name)
                          for name in ("agonist", "antagonist")} for partition in ("train", "test")}
    print(f"Split written to {args.output}: {counts}")


def _train(args):
    from .ml import evaluate, train

    metadata = train(args.features, args.split, args.labels, args.model, _section(args, "models"), args.cohort or [],
                     args.output_dir)
    print(f"Trained {metadata['model']} on {metadata['representation']} ({metadata['training']['by_class']})")
    if args.evaluate:
        report = evaluate(args.output_dir, args.features, args.split, args.labels, args.cohort or [],
                          Path(args.output_dir) / "evaluation")
        test = report["metrics"]["test"]
        print(f"Test: n={test['n']}, confusion counts {test['confusion_counts']}, balanced accuracy "
              f"{test['balanced_accuracy']}, ROC AUC (class 1) {test['roc_auc_class1']}")


def _evaluate(args):
    from .ml import evaluate

    report = evaluate(args.model_dir, args.features, args.split, args.labels, args.cohort or [], args.output_dir)
    test = report["metrics"]["test"]
    print(f"Test: n={test['n']}, confusion counts {test['confusion_counts']}, balanced accuracy "
          f"{test['balanced_accuracy']}, ROC AUC (class 1) {test['roc_auc_class1']}")


def _predict(args):
    from .ml import predict

    rows = predict(args.model_dir, args.features, args.output)
    for row in rows:
        print(f"{row['ligand_id']:24} {row['predicted_class']:10} score_class1={row['score_class1']:.3f}")
    print(f"{len(rows)} predictions written to {args.output}")


def _report(args):
    from . import report

    run = json.loads(Path(args.run_metadata).read_text(encoding="utf-8")) if args.run_metadata else None
    content = report.collect(args.curated, args.ligand_inputs, args.receptor_dir, args.ligands_dir, args.docking_dir,
                             args.poses_dir, args.redocking_dir, args.features, args.split, args.model_dir,
                             args.predictions or [], run)
    report.write_report(content, args.output_dir)
    print(f"Report [{content['status']}] written to {Path(args.output_dir) / 'report.html'}")
    for problem in content["problems"]:
        print(f"problem: {problem}", file=sys.stderr)
    if content["status"] != "success" and not args.record_failure:
        raise RuntimeError("the run did not meet the success criteria; see the report")


def build_parser():
    parser = argparse.ArgumentParser(prog="ligand-analysis", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    synthetic = commands.add_parser(
        "synthetic-example",
        help="train and evaluate the legacy ensemble on seeded synthetic features")
    synthetic.add_argument("--output-dir", help="default: DATA_DIR/runs/synthetic-seed<SEED>")
    synthetic.add_argument("--data-dir", help="overrides LIGAND_ANALYSIS_DATA_DIR")
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.add_argument("--overwrite", action="store_true", help="replace existing report files")
    synthetic.set_defaults(handler=_synthetic_example)

    fetch = commands.add_parser("fetch", help="download manifest sources into a checksummed snapshot")
    fetch.add_argument("manifest")
    fetch.add_argument("--snapshot-dir", help="default: DATA_DIR/sources/<manifest name>")
    fetch.add_argument("--data-dir", help="overrides LIGAND_ANALYSIS_DATA_DIR")
    fetch.add_argument("--offline", action="store_true", help="fail instead of downloading")
    fetch.add_argument("--refresh", action="store_true", help="download again even if cached")
    fetch.set_defaults(handler=_fetch)

    curate = commands.add_parser("curate", help="build label, issue and candidate tables offline")
    curate.add_argument("manifest")
    curate.add_argument("--snapshot-dir", help="default: DATA_DIR/sources/<manifest name>")
    curate.add_argument("--output-dir", help="default: DATA_DIR/curated/<manifest name>")
    curate.add_argument("--data-dir", help="overrides LIGAND_ANALYSIS_DATA_DIR")
    curate.set_defaults(handler=_curate)

    def stage(name, handler, help_text, config=True):
        sub = commands.add_parser(name, help=help_text)
        if config:
            sub.add_argument("--config", required=True, help="pipeline configuration (YAML/JSON), e.g. configs/demo.yaml")
        sub.set_defaults(handler=handler)
        return sub

    sub = stage("ligand-inputs", _ligand_inputs, "write identity-only inputs from curated selected candidates")
    sub.add_argument("candidates", help="curated candidates.tsv")
    sub.add_argument("--output", required=True)

    sub = stage("prediction-inputs", _prediction_inputs, "write unlabeled ligands to predict (config 'prediction')")
    sub.add_argument("ligands", help="curated ligands.tsv")
    sub.add_argument("--candidates", required=True, help="curated candidates.tsv; prediction ligands must not be in it")
    sub.add_argument("--output", required=True)

    sub = stage("prepare-receptor", _prepare_receptor, "select, repair, protonate and parameterize the receptor (chem)")
    sub.add_argument("manifest")
    sub.add_argument("--snapshot-dir", help="default: DATA_DIR/sources/<manifest name>")
    sub.add_argument("--data-dir", help="overrides LIGAND_ANALYSIS_DATA_DIR")
    sub.add_argument("--output-dir", required=True)

    sub = stage("prepare-ligands", _prepare_ligands, "standardize ligands and write docking states (chem)")
    sub.add_argument("ligands", help="ligand input table (ligand_id, name, smiles, inchikey)")
    sub.add_argument("--output-dir", required=True)

    sub = stage("dock", _dock, "dock prepared ligand states with AutoDock Vina (chem)")
    sub.add_argument("--receptor-dir", required=True)
    sub.add_argument("--ligands-dir", required=True)
    sub.add_argument("--ligand-id", action="append", help="dock only these ligands (repeatable); default all")
    sub.add_argument("--cpu", type=int, default=1, help="Vina threads (default 1)")
    sub.add_argument("--output-dir", required=True)

    sub = stage("select-poses", _select_poses, "check poses and select one per ligand without labels (chem)")
    sub.add_argument("ligands", help="ligand input table; every row gets an outcome")
    sub.add_argument("--ligands-dir", required=True)
    sub.add_argument("--docking-dir", required=True, nargs="+")
    sub.add_argument("--receptor-dir", required=True)
    sub.add_argument("--output-dir", required=True)

    sub = stage("redock-reference", _redock_reference, "redock the reference ligand and report its RMSD (chem)")
    sub.add_argument("--receptor-dir", required=True)
    sub.add_argument("--cpu", type=int, default=1)
    sub.add_argument("--output-dir", required=True)

    sub = stage("featurize", _featurize, "PLEC (poses) or Morgan (ligands) feature matrices (chem)")
    sub.add_argument("representation", choices=["plec", "morgan"])
    sub.add_argument("--poses-dir", help="plec: select-poses output")
    sub.add_argument("--receptor-dir", help="plec: prepare-receptor output")
    sub.add_argument("--ligands-dir", help="morgan: prepare-ligands output")
    sub.add_argument("--receptor-id", help="morgan: receptor of the samples (UniProt accession)")
    sub.add_argument("--receptor-table", help="morgan: curated receptor.tsv giving the receptor ID")
    sub.add_argument("--output-dir", required=True)

    sub = stage("split", _split, "persist a stratified connectivity-group train/test split")
    sub.add_argument("candidates", help="curated candidates.tsv (labels)")
    sub.add_argument("ligands", help="ligand input table")
    sub.add_argument("--output", required=True)

    def model_inputs(sub):
        sub.add_argument("--features", required=True, help="featurize output directory")
        sub.add_argument("--split", required=True)
        sub.add_argument("--labels", required=True, help="curated candidates.tsv")
        sub.add_argument("--cohort", nargs="*", help="other feature directories; use only samples all of them have")

    sub = stage("train", _train, "fit one classifier on the training partition and save a model bundle")
    model_inputs(sub)
    sub.add_argument("--model", required=True, choices=["dummy", "logistic_regression", "random_forest",
                                                         "legacy_ensemble"])
    sub.add_argument("--evaluate", action="store_true", help="also evaluate into OUTPUT_DIR/evaluation")
    sub.add_argument("--output-dir", required=True)

    sub = stage("evaluate", _evaluate, "evaluate a saved model on the persisted split", config=False)
    model_inputs(sub)
    sub.add_argument("--model-dir", required=True)
    sub.add_argument("--output-dir", required=True)

    sub = stage("predict", _predict, "predict unlabeled samples with a saved model bundle", config=False)
    sub.add_argument("--model-dir", required=True)
    sub.add_argument("--features", required=True)
    sub.add_argument("--output", required=True)

    sub = stage("report", _report, "write the HTML/JSON run report; exit 1 unless the run succeeded", config=False)
    sub.add_argument("--curated", required=True)
    sub.add_argument("--ligand-inputs", required=True)
    sub.add_argument("--receptor-dir", required=True)
    sub.add_argument("--ligands-dir", required=True)
    sub.add_argument("--docking-dir", required=True, nargs="+")
    sub.add_argument("--poses-dir", required=True)
    sub.add_argument("--redocking-dir", required=True)
    sub.add_argument("--features", required=True, nargs="+")
    sub.add_argument("--split", required=True)
    sub.add_argument("--model-dir", required=True, nargs="+", help="train outputs with an evaluation/ folder")
    sub.add_argument("--predictions", nargs="*", help="unlabeled prediction tables")
    sub.add_argument("--run-metadata", help="JSON with workflow provenance")
    sub.add_argument("--record-failure", action="store_true",
                     help="exit 0 even when the run fails its success criteria (the workflow publishes the report, "
                          "then fails the run itself)")
    sub.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except (ConfigError, LabelJoinError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
