Updating my master thesis work: agonist/antagonist classification of receptor ligands.

Everything runs in Podman containers; no host Python is needed. Plan: [docs/LOCAL_PIPELINE_PLAN.md](docs/LOCAL_PIPELINE_PLAN.md); status: [docs/progress.md](docs/progress.md).

## Setup (Windows, PowerShell)

Install Podman, then once:

```powershell
podman machine init --now      # later sessions: podman machine start
$rev = git describe --always --dirty
podman build -f containers/ml/Containerfile -t ligand-ml:dev --build-arg REVISION=$rev .
podman build -f containers/tools/Containerfile -t ligand-tools:dev .
```

Images install exactly the packages in `containers/*/linux-64.lock`. After editing an `environment.yml`, regenerate its lock (example for `ml`):

```powershell
podman run --rm -v "${PWD}\containers\ml:/spec" docker.io/mambaorg/micromamba:2.9.0-debian13-slim@sha256:e0a99b0f17a759e14c2f967dc0ca2d3a3c1ca3c62955f4d20bba770eaaf0184d bash -c "micromamba create -y -q -n lock -f /spec/environment.yml && micromamba env export -n lock --explicit > /spec/linux-64.lock"
```

## Tests

```powershell
# against the package installed in the image
podman run --rm -v "${PWD}:/work:ro" -w /work -e PYTHONPATH=code/ml_protocol ligand-ml:dev python -m unittest discover -s tests -v
# against the working tree without rebuilding
podman run --rm -v "${PWD}:/work:ro" -w /work -e PYTHONPATH=src:code/ml_protocol ligand-ml:dev python -m unittest discover -s tests -v
# lint (configuration in pyproject.toml)
podman run --rm -v "${PWD}:/io:ro" -w /io ghcr.io/astral-sh/ruff:0.16.9 check --no-cache src tests
```

Tests use small synthetic fixtures and never touch the network. When the ADRB2 snapshot is present (after `dvc pull`), they also check that `curate` reproduces `data/curated/adrb2` byte for byte.

## Synthetic example (software regression only)

```powershell
podman run --rm -v "${PWD}\data\local:/data" ligand-ml:dev ligand-analysis synthetic-example --seed 0
```

Writes `report.json`, `report.md` and `predictions.tsv` to `data/local/runs/synthetic-seed0/` (git-ignored). Synthetic data is not biological ground truth.

## Curated ligands (ADRB2)

`data/manifests/adrb2.yaml` pins the GtoPdb release files by SHA-256 and states the label rules. `fetch` downloads into a checksummed snapshot (reused afterwards); `curate` works offline from it.

```powershell
podman run --rm -v "${PWD}:/work" -w /work ligand-ml:dev ligand-analysis fetch data/manifests/adrb2.yaml --snapshot-dir data/sources/adrb2
podman run --rm --network=none -v "${PWD}:/work" -w /work ligand-ml:dev ligand-analysis curate data/manifests/adrb2.yaml --snapshot-dir data/sources/adrb2 --output-dir data/curated/adrb2
```

Outputs in `data/curated/adrb2/`: `candidates.tsv` (eligible ligands, `selected` marks the demo subset), `exclusions.tsv`, `issues.tsv` (duplicates, missing records, conflicts), `annotations.tsv`, `ligands.tsv`, `receptor.tsv`, `curation_report.json`. The GtoPdb CSV downloads are public; no account or API key is used.

## Data versioning (DVC)

The snapshot `data/sources/adrb2` is tracked by DVC. Configure a local remote outside the repository once per machine, then push or pull:

```powershell
New-Item -ItemType Directory -Force C:\workspace\ligand-analysis-dvc-remote
function dvc { podman run --rm -v "${PWD}:/work" -v "C:\workspace\ligand-analysis-dvc-remote:/dvc-remote" ligand-tools:dev dvc @args }
dvc remote add -f -d --local localremote /dvc-remote
dvc push
dvc pull
```

DVC may log harmless `chmod` debug messages on Windows-mounted folders.

## Conventions

Label mapping `0 = agonist`, `1 = antagonist`. Evaluation uses raw confusion counts (true classes in rows, predictions in columns). Scalar precision/recall/F1 use class 0 (agonist); ROC curves use class 1 (antagonist). `model_fit_leaveOneOut(..., norm=True)` is for displaying matrices only. Previously saved ensembles must be retrained to retain the raw validation counts.

## Layout

- `src/ligand_analysis/`: the package and its CLI; `schemas/` holds the JSON Schemas for manifests and tables; `legacy/` holds the metric and ensemble functions moved from `code/ml_protocol` (only the imports and one redundant default argument changed).
- `containers/`: one Containerfile, environment and lockfile per image.
- `data/manifests/` (source manifests) and `data/curated/` (small curated tables) are committed; `data/sources/` is tracked by DVC; `data/local/` is ignored scratch space.
- `tests/`: unit tests named after the module they cover (`test_sources.py`, `test_tables.py`, ...), plus the legacy regression tests (`test_ensemble.py`, `test_metrics.py`, `test_roc.py`).

## Legacy code

`code/ml_protocol` keeps the thesis notebooks and `pipeline_functions.py` (filename-labelled feature preprocessing and result saving) until later milestones replace them. `pipeline_functions.py` imports the metric and ensemble functions from `ligand_analysis.legacy`; `code_test/Test Tensorflow.ipynb` still imports the removed `utility_functions` and `ensemble_functions` modules. The notebooks need packages that are not in `ligand-ml` (pygtop, biopandas, pubchempy, unidecode, tensorflow, ODDT/Open Babel). The original Dockerfile, docker-compose file and readme.txt (proprietary OpenEye, unpinned packages) were removed; they remain in the Git history.
