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
- The `code/ml_protocol/utility_functions.py` and `ensemble_functions.py` shims. `pipeline_functions.py` and the tests now import `ligand_analysis.legacy` directly. `code_test/Test Tensorflow.ipynb` still imports the removed modules; update its first cell before running it.

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

## Next task

M2: receptor structure selection (e.g. 2RH1) and ligand preparation for the selected candidates.
