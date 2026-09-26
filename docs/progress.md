# Progress

Status of [LOCAL_PIPELINE_PLAN.md](LOCAL_PIPELINE_PLAN.md). The migration is developed on the `migration/local-pipeline` branch.

## M0: reproducible baseline (done)

| Acceptance criterion | Evidence |
| --- | --- |
| Existing 10 tests pass in the image | The full suite (31 tests: the 10 legacy tests plus 21 new ones) passes against the package installed in `ligand-ml:dev` |
| No proprietary dependency | `containers/ml/linux-64.lock` holds conda-forge packages only. OpenEye, xgboost, pydotplus, IPython and openbabel imports were removed from the packaged/legacy modules |
| Seeded synthetic example writes an explicit report | `ligand-analysis synthetic-example --seed 0` gives confusion counts `[[8,1],[0,9]]` and accuracy 0.944, and writes report.json, report.md and predictions.tsv |
| CLI works outside the repository | Run from `/tmp` in the image with no repository mounted |

Changes:
- Legacy code moved to `src/ligand_analysis/legacy/`. The `code/ml_protocol/*_functions.py` shims were later removed (see the review follow-up below).
- The image is pinned by digest and built from a lockfile.
- One behaviour-neutral edit: `criterion='friedman_mse'` was removed from the GradientBoosting parameters. scikit-learn 1.9 ignores the parameter and warns about it.

## M1: data acquisition and curation for ADRB2 (P07550) (done)

| Acceptance criterion | Evidence |
| --- | --- |
| Labels no longer depend on filenames | Labels come only from the `label_mapping` (GtoPdb Type/Action) rules in `data/manifests/adrb2.yaml`. They are joined by `join_labels` on the GtoPdb ligand ID and the UniProt accession. A test checks that an "agonist" filename does not override an antagonist annotation |
| Duplicate, missing and conflicting identities reported | `issues.tsv` and `exclusions.tsv`. Tests cover duplicate rows and structures, missing records and structures, PubChem mismatches, conflicting and ambiguous labels |
| Pinned download reused from cache | `fetch --offline` with `--network=none`: 0 downloaded, 4 reused |
| DVC restores a tracked artifact from the local remote | Ran `dvc add data/sources/adrb2`, `dvc push`, deleted the folder, ran `dvc pull`; all 4 SHA-256 checks OK |
| Curated candidate list for one receptor | `data/curated/adrb2/candidates.tsv` |

Results (GtoPdb 2026.3, 130 annotations, 68 ligands):
- 36 ligands are eligible (14 agonists, 22 antagonists). 29 are selected (14 agonists, 15 antagonists).
- 32 ligands are excluded:
  - 28 unmapped actions, mainly partial agonists and allosteric modulators;
  - 3 ambiguous;
  - labelled/radioactive, missing structure and ligand type.
- Issues:
  - 1 missing structure;
  - 3 ambiguous labels;
  - 1 shared connectivity.
- No PubChem mismatches were found.

Notes:
- Mirabegron, vibegron and solabegron are β3-selective agonists that GtoPdb also annotates at β2. They were kept as published; review whether they belong in the demo.
- The GtoPdb bulk CSVs are public. No account, password or API key was used or stored.
- No chemistry stages (preparation, docking, PLEC) were run; they belong to later milestones.

Commands (PowerShell, from the repository root):

```powershell
podman run --rm -v "${PWD}:/work" -w /work ligand-ml:dev ligand-analysis fetch data/manifests/adrb2.yaml --snapshot-dir data/sources/adrb2
podman run --rm --network=none -v "${PWD}:/work" -w /work ligand-ml:dev ligand-analysis fetch data/manifests/adrb2.yaml --snapshot-dir data/sources/adrb2 --offline
podman run --rm --network=none -v "${PWD}:/work" -w /work ligand-ml:dev ligand-analysis curate data/manifests/adrb2.yaml --snapshot-dir data/sources/adrb2 --output-dir data/curated/adrb2
podman run --rm -v "${PWD}:/work:ro" -w /work -e PYTHONPATH=code/ml_protocol ligand-ml:dev python -m unittest discover -s tests -v
```

## Review follow-up (after M1)

Removed (still in the Git history):
- `Dockerfile`, `docker-compose.yml` and `readme.txt`: the original thesis environment with OpenEye, replaced by `containers/`.
- The `code/ml_protocol/utility_functions.py` and `ensemble_functions.py` shims. `pipeline_functions.py`, `code_test/Test Tensorflow.ipynb` and the tests now import `ligand_analysis.legacy` directly.

Kept: `code/ml_protocol/pipeline_functions.py` and the notebooks. Their preprocessing still labels ligands by filename and is replaced in M2/M3.

Tests:
- New modules `test_sources.py` (snapshot cache and pin checks; HTTPS retries, redirects and size limit), `test_tables.py`, `test_labels.py` and `test_config.py`.
- `test_curation.py` now covers every issue type and exclusion reason, the class limit, the `fetch` and `curate` commands and their error exits.
- A golden test re-runs `curate` on the DVC snapshot and compares the result with `data/curated/adrb2` byte for byte. It is skipped when the snapshot is absent.
- Ruff settings are in `pyproject.toml`. The legacy code keeps its original style and is not linted.
- Result: 58 tests pass offline (`--network=none`), both against the package installed in `ligand-ml:dev` and against the working tree, with none skipped. Ruff reports no issues.

Fixes found while writing the tests:
- The manifest digest in `curation_report.json` changed when Git converted line endings on Windows. `manifest_sha256` now hashes with LF endings, and `.gitattributes` keeps `data/curated/` LF.
- Malformed YAML manifests and invalid UniProt or PubChem JSON now give a clear error and exit code 1 instead of a traceback.
- The receptor name is chosen deterministically when GtoPdb rows give different names for the same target.

`data/curated/adrb2` was regenerated offline with the same counts; the report now records the LF-normalised manifest digest.

Commands:

```powershell
podman run --rm -v "${PWD}:/io:ro" -w /io ghcr.io/astral-sh/ruff:0.16.9 check --no-cache src tests
podman run --rm --network=none -v "${PWD}:/work:ro" -w /work -e PYTHONPATH=code/ml_protocol ligand-ml:dev python -m unittest discover -s tests -v
```

## CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on `ubuntu-26.04` runners, which ship Podman 5.7 (`ubuntu-24.04` has Podman 4.9):
- `lint`: Ruff on `src` and `tests`, with GitHub annotations.
- `test`: builds both images with revision and source labels, runs the offline suite (`--network=none`) against the installed package, and smoke-tests `ligand-analysis synthetic-example` and `dvc version` outside the repository.
- `publish` (pushes to `main` and `v*` tags only, after `lint` and `test`): loads the tested images and pushes them to GHCR as `ghcr.io/luca-rendina/ligand-ml` and `ligand-tools`, tagged with the commit SHA or the version tag. There are no moving tags; the run summary lists the digests.

Actions are pinned by commit SHA and updated monthly by Dependabot (`.github/dependabot.yml`). The workflow token is read-only except for `packages: write` in `publish`, and checkout does not keep credentials.

Checked locally:
- actionlint 1.7.12 with shellcheck: no findings, apart from `ubuntu-26.04` missing from its runner-label list, which predates that runner.
- Both images build with the CI flags and carry the expected labels.
- In a fresh clone without the DVC snapshot, the CI test command ran 58 tests with 1 skipped (the ADRB2 golden test), and the smoke tests passed.
- The `podman save`/`podman load` hand-off between jobs keeps the tags and labels.

Not verified yet: a run on GitHub and the GHCR push. The golden test stays skipped in CI until CI can reach a DVC remote (M6).

## M0 and M1 re-check (before M2)

On `migration/local-pipeline` at `c4bee14`: 58 tests passed in `ligand-ml:dev`. The ADRB2 golden test also passed once the DVC snapshot was present. `synthetic-example --seed 0` from `/tmp` again gave `[[8,1],[0,9]]` and accuracy 0.944. One M1 defect was fixed: the GtoPdb published date kept the closing quote of the CSV header (`"2026-09-16\""`). It is now parsed without it, and a test covers it.

## M2: open source molecular example (done)

| Acceptance criterion | Evidence |
| --- | --- |
| The complete chemistry dependency chain works in the image | `ligand-chem` (RDKit 2026.03.1, molscrub 0.3.0, Meeko 0.8.0, Vina 1.2.7, PDBFixer 1.12/OpenMM 8.6.1, ODDT 0.7 with Open Babel 3.2.1, numpy 1.26.4). The chain ran in it from preparation to PLEC for 2RH1, 20 labelled and 3 unlabeled ligands. Chemistry tests on a trimmed 2RH1 pocket fixture run the same stages offline |
| Outputs preserve molecular identity and 3D coordinates | The carazolol coordinates reproduce the configured InChIKey. All 20 parent InChIKeys match the curated ones. Every 3D state must reproduce its stereochemistry, which is how molscrub's ring-fix inversion of nadolol was found. 351 of 351 labelled poses and 54 of 54 unlabeled poses pass the identity check after the Meeko export, and selected poses carry an atom map onto the prepared state |
| Structural QC and rejection records exist | Receptor QC: 20 pocket residues, none needing added atoms, no chain break or mutation in the pocket (N187E lies outside). Redocking carazolol from SMILES gave a top-ranked pose with 0.996 Å in-place RMSD (threshold 2.0 Å, fixed beforehand; score −10.12 kcal/mol). `pose_checks.tsv` (identity, box, clashes) and rejection reasons in `ligand_outcomes.tsv` and `pose_selection.tsv` |
| Dense/sparse feature equivalence is checked | Every PLEC row is compared: 20 of 20 labelled and 3 of 3 unlabeled rows are equal. The largest bit count is 18, so no uint8 overflow |
| Every input has an outcome | 20 of 20 ligands prepared (39 states, 32 of them from enumerated stereocentres), 39 of 39 states docked, 20 of 20 poses selected. Tests cover failures (invalid SMILES, stereoisomer limit) |

Choices (details in [protocol.md](protocol.md)):
- Receptor 2RH1 chain A: DBREF-mapped residues 29–230 and 263–342. T4L, waters, ions, lipids and additives are removed and listed. The T4L gap is split with charged termini. Only the Asp29 side chain and terminal OXT atoms are added. Protonation follows OpenMM at pH 7.4 with a seeded start. Meeko templates matched every residue. The box is 26 Å, centred on carazolol.
- Ligands: 10 agonists and 10 antagonists (`selection_rank <= 10`). Unassigned stereocentres are enumerated (at most 4 stereoisomers), molscrub states are built at pH 7.4 (at most 4), with one seeded ETKDGv3/MMFF94s conformer per state. molscrub ring fixing is off because it inverted a nadolol stereocentre for some seeds.
- Docking: Vina, exhaustiveness 8, 9 modes, seed 42. The selected pose is the best valid score over all states; labels are never read.
- Features: PLEC with ligand depth 2, protein depth 4, 65,536 bits, 4.5 Å cutoff, counts, waters ignored, Open Babel backend; the protein is the Meeko PDBQT. Morgan: radius 2, 2,048 bits.

Determinism: a manual run (Vina with 16 threads, all ligands in one process) and the Nextflow run (2 threads per ligand task) gave identical scores for all 351 poses, the same 20 selections and a byte-identical PLEC matrix. Receptor hydrogens were first not reproducible, because OpenMM places them from Python's unseeded random generator. That generator is now seeded, and a test checks that the prepared receptor is byte-identical between runs.

The snapshot `data/sources/adrb2` now also holds 2RH1 (6 files, 14.8 MB). It was re-added to DVC and pushed. A restore into a fresh checkout passed all 5 SHA-256 checks. Commit `data/sources/adrb2.dvc` together with the push.

## M3: complete Podman workflow (done)

| Acceptance criterion | Evidence |
| --- | --- |
| One documented command runs the public demo from acquisition through report | `.\scripts\nextflow.ps1 -profile podman -params-file configs/demo.yaml --sources_dir data/local/sources` started from an empty snapshot directory. FETCH downloaded the 5 sources, with checksums identical to the DVC snapshot. 55 tasks completed in 4 min 11 s, and the report status is `success` |
| A cached rerun works without public database access | A fresh run (no `-resume`) with `-profile podman,offline` skipped FETCH (stored snapshot). All 54 tasks were launched with `--network=none`, checked in each `.command.run`, and it succeeded in 4 min 11 s. All 264 data outputs are byte-identical to the online run, as are the metrics of the 8 models |
| Resume reuses valid tasks, and changing a docking parameter reruns affected downstream work | `-resume` without changes: 53 tasks cached, and only REPORT reran (14 s; its run record changes each run). `-resume` with `docking.exhaustiveness: 16`: CURATE, both inputs tasks, PREPARE_RECEPTOR, both PREPARE_LIGANDS, both FEATURIZE_MORGAN and SPLIT were cached. All 23 DOCK tasks, REDOCK_REFERENCE, both SELECT_POSES, both FEATURIZE_PLEC, 8 TRAIN, 8 PREDICT and REPORT reran (7 min 1 s). 9 of 20 selected poses changed |
| Models can be reloaded and used for prediction | The PREDICT tasks reload each checksummed bundle and predict the unlabeled ligands. `ligand-analysis predict` run by hand on a published bundle (`plec_logistic_regression`) reproduced the workflow output byte for byte. Mismatched feature schemas and tampered bundles are rejected (tests) |

Workflow: `main.nf` with modules in `workflow/` and the ligand-features subworkflow, used for the labelled and the unlabeled ligands. Stage configurations come from `configs/demo.yaml`, and each stage gets only its sections. `nextflow.config`, `conf/podman.config` and an `offline` profile hold the execution settings. Nextflow 26.04.6 runs in `ligand-runner` (OpenJDK 21, Podman 5.8 client) against the Podman machine's API socket. The project is mounted at the machine's path (`/mnt/c/...`), so work directories resolve identically for the controller and the tasks. `nextflow lint` reports no errors under the strict syntax parser, which is the default in Nextflow 26.

Demo results (test partition of the common cohort, 3 agonists and 3 antagonists: isoprenaline, vilanterol, mirabegron / bupranolol, nadolol, practolol):

| Representation | Model | Confusion counts | Balanced accuracy | ROC AUC (class 1) |
| --- | --- | --- | --- | --- |
| Morgan | dummy | [[3,0],[3,0]] | 0.50 | 0.50 |
| Morgan | logistic regression / random forest / legacy ensemble | [[3,0],[0,3]] | 1.00 | 1.00 |
| PLEC | dummy | [[3,0],[3,0]] | 0.50 | 0.50 |
| PLEC | logistic regression, random forest | [[2,1],[0,3]] | 0.83 | 1.00 |
| PLEC | legacy ensemble | [[3,0],[0,3]] | 1.00 | 1.00 |

Mirabegron, the β3-selective agonist flagged in M1, is the one ligand the PLEC logistic regression and random forest call antagonist. With exhaustiveness 16, two PLEC models still misclassify one test ligand: now the logistic regression and the legacy ensemble (the random forest is correct). Six test ligands cannot support any performance claim. The two classes are largely separable by chemotype, so ligand-only models score perfectly here without any help from the structure. Unlabeled predictions (all non-dummy models agree): salbutamol and terbutaline are called agonist and alprenolol antagonist. They are not evaluated.

Tests and checks: 97 tests. In `ligand-ml`, 88 pass and 9 are skipped (the chemistry chain); in `ligand-chem`, all 97 pass. Both runs are offline against the installed packages, with the ADRB2 golden test included. Ruff and `nextflow lint` report no issues, and actionlint reports only the known `ubuntu-26.04` label. CI now builds `ligand-chem` and `ligand-runner`, runs the tests in both package images, lints the workflow and publishes all four images from `main`.

Fixes made while running the workflow:
- The strict parser evaluates `publishDir`/`storeDir` when the process is defined, so input variables cannot appear there. Outputs are written under an input-named folder instead.
- PowerShell split an unquoted `-profile podman,offline` into two arguments, which silently dropped the offline profile. The launcher joins them again.
- Build caching kept a stale revision label in `ligand-runner`; rebuild it with `--no-cache` when its revision must be exact.

Fixes from a code review of the branch:
- A failing `report` task was never published, so an earlier success report could stay in the output folder. `REPORT` now writes and publishes the report even when the run fails its criteria (`--record-failure`), and a separate `CHECK_REPORT` task then fails the run. Checked with `-resume` and `redocking.rmsd_threshold: 0.5`: the run exited 1 and the published report showed `failed` ("redocking RMSD 0.996 A exceeds 0.5 A"). A rerun with the demo settings published `success` again; only REPORT and CHECK_REPORT ran.
- Classifier names and the manifest name become shell arguments and file names. They are now checked in `main.nf` before any task runs, and quoted in the task scripts.

Commands (PowerShell, repository root):

```powershell
.\scripts\nextflow.ps1 -profile podman -params-file configs/demo.yaml --sources_dir data/local/sources
.\scripts\nextflow.ps1 -profile podman,offline -params-file configs/demo.yaml --sources_dir data/local/sources --outdir data/local/results/adrb2-demo-offline
.\scripts\nextflow.ps1 -profile podman,offline -params-file configs/demo.yaml --sources_dir data/local/sources --outdir data/local/results/adrb2-demo-offline -resume
# a copy of configs/demo.yaml with docking.exhaustiveness: 16
.\scripts\nextflow.ps1 -profile podman,offline -params-file data/local/demo-exhaustiveness16.yaml --sources_dir data/local/sources --outdir data/local/results/adrb2-demo-offline -resume
podman run --rm --network=none -v "${PWD}:/work" -w /work ligand-ml:dev ligand-analysis predict --model-dir data/local/results/adrb2-demo/models/plec_logistic_regression --features data/local/results/adrb2-demo/prediction/features/plec --output data/local/predict-check.tsv
```

Without `--sources_dir`, FETCH uses the DVC snapshot `data/sources/adrb2` (after `dvc pull`) and the run needs no download.

The Linux wrapper `scripts/nextflow.sh` was run inside the Podman machine (Fedora 44, Podman 6.0.2) on the same project path with `-profile podman,offline ... -resume`: 53 tasks were cached, REPORT reran, and the status is `success`. Git cannot resolve this Windows worktree from Linux, so the recorded revision is `unknown`; in a normal Linux clone the wrapper records `git describe`. A native Linux host has not been tested.

## Next task

M4: local Kubernetes (kind with Podman), running the same workflow and images. `ligand-runner` already contains the controller. The remaining work is the cluster, image loading, shared storage, service account and runner pod, plus comparing the outputs with the Podman run above.
