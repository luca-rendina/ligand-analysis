"""Command-line interface: ``ligand-analysis <command>``."""

import argparse
import sys

from . import __version__
from .config import ConfigError, data_dir


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
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except (ConfigError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
