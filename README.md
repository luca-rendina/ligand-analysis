Updating my master thesis work: agonist/antagonist classification of receptor ligands.

Everything runs in Podman containers; no host Python is needed. Plan: [docs/LOCAL_PIPELINE_PLAN.md](docs/LOCAL_PIPELINE_PLAN.md); status: [docs/progress.md](docs/progress.md); molecular protocol: [docs/protocol.md](docs/protocol.md).

## Setup (Windows, PowerShell)

Install Podman, then once:

```powershell
podman machine init --now      # later sessions: podman machine start
$rev = git describe --always --dirty
podman build -f containers/ml/Containerfile -t ligand-ml:dev --build-arg REVISION=$rev .
podman build -f containers/chem/Containerfile -t ligand-chem:dev --build-arg REVISION=$rev .
podman build -f containers/runner/Containerfile -t ligand-runner:dev --build-arg REVISION=$rev .
podman build -f containers/tools/Containerfile -t ligand-tools:dev .
```

| Image | Contents |
| --- | --- |
| `ligand-ml` | the package with NumPy, SciPy, pandas, scikit-learn: curation, models, evaluation, report |
| `ligand-chem` | the package with RDKit, molscrub, Meeko, AutoDock Vina, PDBFixer/OpenMM, ODDT/Open Babel (numpy<2, required by ODDT 0.7) |
| `ligand-runner` | Nextflow 26.04, OpenJDK 21 and a Podman client (workflow controller) |
| `ligand-tools` | DVC and Git |

Images install exactly the packages in `containers/*/linux-64.lock`. After editing an `environment.yml`, regenerate its lock (example for `chem`; use the same command for `ml`, `runner` or `tools`):

```powershell
$image = 'chem'
podman run --rm -v "${PWD}\containers\${image}:/spec" docker.io/mambaorg/micromamba:2.9.0-debian13-slim@sha256:e0a99b0f17a759e14c2f967dc0ca2d3a3c1ca3c62955f4d20bba770eaaf0184d bash -c "micromamba create -y -q -n lock -f /spec/environment.yml && micromamba env export -n lock --explicit > /spec/linux-64.lock"
```

## Tests

```powershell
# against the package installed in the images (chemistry tests run only in ligand-chem)
podman run --rm --network=none -v "${PWD}:/work:ro" -w /work -e PYTHONPATH=code/ml_protocol ligand-ml:dev python -m unittest discover -s tests -v
podman run --rm --network=none -v "${PWD}:/work:ro" -w /work -e PYTHONPATH=code/ml_protocol ligand-chem:dev python -m unittest discover -s tests -v
# against the working tree without rebuilding
podman run --rm --network=none -v "${PWD}:/work:ro" -w /work -e PYTHONPATH=src:code/ml_protocol ligand-chem:dev python -m unittest discover -s tests -v
# lint (configuration in pyproject.toml) and the Nextflow strict-syntax check
podman run --rm -v "${PWD}:/io:ro" -w /io ghcr.io/astral-sh/ruff:0.16.9 check --no-cache src tests
podman run --rm -v "${PWD}:/mnt/w:ro" -w /mnt/w ligand-runner:dev nextflow lint main.nf workflow nextflow.config conf
```

Tests use small synthetic fixtures and a trimmed 2RH1 pocket (`tests/fixtures/2rh1_pocket.pdb`, CC0) and never touch the network. When the ADRB2 snapshot is present (after `dvc pull`), they also check that `curate` reproduces `data/curated/adrb2` byte for byte.

## CI/CD

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs on GitHub-hosted Ubuntu 26.04 runners, which ship Podman 5.7:

- On every pull request, push to `main` or `v*` tag, and manual run (Actions → CI/CD → Run workflow): Ruff, the four image builds, the offline test suite against the installed package in `ligand-ml` and in `ligand-chem` (which adds the chemistry tests on the 2RH1 pocket fixture), `nextflow lint`, and CLI smoke tests. The ADRB2 reproduction test is skipped because the DVC remote is local, and the full molecular demo is not run in CI (it downloads sources and docks for several minutes). Other branches are tested through a pull request or a manual run.
- After a push to `main` or a `v*` tag passes, the tested images are pushed to `ghcr.io/luca-rendina/ligand-ml`, `ligand-chem`, `ligand-runner` and `ligand-tools`, tagged with the commit SHA or the version tag (there is no `latest`). The run summary lists the digests; pin them for published runs.

New GHCR packages are private; their visibility can be changed in the package settings (making a package public cannot be undone). Dependabot proposes monthly updates for the SHA-pinned actions.

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

Outputs in `data/curated/adrb2/`: `candidates.tsv` (eligible ligands, `selected` marks the demo subset), `exclusions.tsv`, `issues.tsv` (duplicates, missing records, conflicts), `annotations.tsv`, `ligands.tsv`, `receptor.tsv`, `curation_report.json`. The GtoPdb CSV downloads are public; no account or API key is used. The manifest also pins the receptor structure (PDB 2RH1), which `fetch` stores in the same snapshot.

## Molecular workflow (public ADRB2 demo)

One command runs acquisition, curation, receptor and ligand preparation, docking, pose selection, PLEC and Morgan features, the persisted split, four models per representation, evaluation, unlabeled prediction and the HTML report. Nextflow does not run on Windows, so [scripts/nextflow.ps1](scripts/nextflow.ps1) starts it in `ligand-runner`, which launches each task as a sibling `ligand-ml`/`ligand-chem` container through the Podman machine's API socket:

```powershell
.\scripts\nextflow.ps1 -profile podman -params-file configs/demo.yaml
```

On Linux, enable the user Podman socket once (`systemctl --user enable --now podman.socket`) and run `scripts/nextflow.sh` with the same arguments. The underlying command is `nextflow run main.nf -profile podman -params-file configs/demo.yaml`.

- Results go to `data/local/results/adrb2-demo/` (`--outdir` to change): `report/report.html` and `report.json`, `curated/`, `inputs/`, `receptor/`, `redocking/`, `labelled/` and `prediction/` (prepared ligands, docking, poses, features), `split/`, `models/` (bundles and evaluations), `prediction/predictions/` and `pipeline_info/` (Nextflow trace, timeline and execution report). Task work directories are in `data/local/work/`.
- `FETCH` stores the snapshot in `data/sources/<manifest name>` (`--sources_dir` to change) and is skipped when it exists. `curate` verifies every checksum when it reads the snapshot.
- Add `,offline` to the profile (`-profile podman,offline`) to give every task container `--network=none`; the controller runs with `NXF_OFFLINE=true`.
- `-resume` reuses completed tasks. Each stage receives only its own configuration sections, so editing, say, `docking` in a copy of `configs/demo.yaml` reruns docking and what depends on it, while curation, preparation and Morgan features stay cached.
- The report is written even when the run fails its success criteria (receptor or redocking QC, missing outcomes, too few samples per class); the `REPORT` task then exits 1.

The stages are also individual commands (`ligand-analysis <command> --help`): `ligand-inputs`, `prediction-inputs`, `prepare-receptor`, `prepare-ligands`, `dock`, `select-poses`, `redock-reference`, `featurize plec|morgan` (in `ligand-chem`), and `split`, `train`, `evaluate`, `predict`, `report` (in `ligand-ml`). For example, to predict new ligands with a saved model after preparing, docking and featurizing them with the same configuration:

```powershell
podman run --rm -v "${PWD}:/work" -w /work ligand-ml:dev ligand-analysis predict --model-dir data/local/results/adrb2-demo/models/plec_logistic_regression --features data/local/results/adrb2-demo/prediction/features/plec --output data/local/predictions.tsv
```

`model.joblib` is a pickle: load bundles only from runs you trust. `predict` checks the bundle checksum and that the features use the model's representation and parameters.

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

- `src/ligand_analysis/`: the package and its CLI; `schemas/` holds the JSON Schemas for manifests, the pipeline configuration and every table; `chem/` holds the chemistry stages (imported only in `ligand-chem`); `legacy/` holds the metric and ensemble functions moved from `code/ml_protocol` (only the imports and one redundant default argument changed).
- `main.nf`, `workflow/` (modules and the ligand-features subworkflow), `nextflow.config` and `conf/` (execution profiles): the Nextflow workflow. `configs/demo.yaml`: scientific parameters of the demo. `scripts/`: Nextflow launchers for Windows and Linux.
- `containers/`: one Containerfile, environment and lockfile per image.
- `.github/`: the CI/CD workflow, the Dependabot configuration and the Copilot instructions.
- `data/manifests/` (source manifests) and `data/curated/` (small curated tables) are committed; `data/sources/` is tracked by DVC; `data/local/` is ignored scratch space (results and Nextflow work directories).
- `tests/`: unit tests named after the module they cover (`test_sources.py`, `test_tables.py`, `test_chem.py`, ...), the legacy regression tests (`test_ensemble.py`, `test_metrics.py`, `test_roc.py`) and `fixtures/`.

## Legacy code

`code/ml_protocol` keeps the thesis notebooks and `pipeline_functions.py` (filename-labelled feature preprocessing and result saving) for reference. The workflow above replaces that preprocessing: labels are joined by ligand and receptor identity, and PLEC comes from `ligand-analysis featurize plec` with the same depths and size. `pipeline_functions.py` and `code_test/Test Tensorflow.ipynb` import the metric and ensemble functions from `ligand_analysis.legacy`. The notebooks need packages that are not in `ligand-ml` (pygtop, biopandas, pubchempy, unidecode, tensorflow, ODDT/Open Babel). The original Dockerfile, docker-compose file and readme.txt (proprietary OpenEye, unpinned packages) were removed; they remain in the Git history.
