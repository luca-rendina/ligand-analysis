# Project Guidelines

See [the local pipeline plan](../docs/LOCAL_PIPELINE_PLAN.md) for milestone scope, architecture, and scientific protocol. Keep this file concise and update it when project conventions change.

## Scientific Invariants

- Preserve the label mapping `0 = agonist` and `1 = antagonist`. Keep original pharmacology terms and assay context; do not silently map ambiguous or missing labels.
- Confusion matrices use true classes in rows and predictions in columns. Preserve raw counts for model selection and ensemble voting; normalize only for display.
- Legacy scalar precision, recall, and F1 use class 0 as positive. ROC/AUC use class 1 and continuous classifier scores, not hard predictions.
- New and migrated workflows must not infer pharmacology labels from filenames, docking scores, generated conformers, or docking poses. The legacy preprocessing currently uses filenames; do not extend that pattern. Join annotations to features through validated molecular and receptor identities, and keep preparation and docking independent of labels (chemistry stages take `ligand_input` tables, whose schema has no label column).
- Synthetic data is for software regression only, never biological ground truth.
- Every requested input needs a recorded outcome (prepared/failed, docked/failed, selected/rejected with a reason). Limits on stereoisomers, states and conformers exclude with a reason; they never truncate silently.
- Keep the split persisted and shared by all representations, restricted to the common cohort; fit and select models on training data only.

## Dependencies and Data

- Use open-source runtime dependencies. Do not add proprietary toolkits such as OpenEye; audit actual imports and supported workflows before adding packages.
- Keep downloaded datasets, caches, models, and run outputs outside Git and container image layers. Commit only source, configuration, lockfiles, small manifests, and permitted small fixtures.
- Preserve the existing PLEC feature configuration unless a deliberate, documented scientific change is requested: protein depth 4, ligand depth 2, and 65,536 features (explicit in `configs/demo.yaml`, ODDT with the Open Babel backend).
- Scientific parameters belong in `configs/*.yaml` (validated by `schemas/pipeline_config.schema.json`); execution settings belong in `nextflow.config` and `conf/`. Record method choices in `docs/protocol.md`.
- Nextflow 26 runs the strict syntax parser: no imports, no top-level statements, explicit closure parameters, and no input variables inside `publishDir`/`storeDir` (write outputs under an input-named folder instead).

## Verification

Never use host Python. Run the offline regression suite from the repository root in the Podman images (see README for building them). Chemistry tests run only in `ligand-chem`:

```powershell
podman run --rm --network=none -v "${PWD}:/work:ro" -w /work -e PYTHONPATH=src:code/ml_protocol ligand-ml:dev python -m unittest discover -s tests -v
podman run --rm --network=none -v "${PWD}:/work:ro" -w /work -e PYTHONPATH=src:code/ml_protocol ligand-chem:dev python -m unittest discover -s tests -v
podman run --rm -v "${PWD}:/io:ro" -w /io ghcr.io/astral-sh/ruff:0.16.9 check --no-cache src tests
podman run --rm -v "${PWD}:/mnt/w:ro" -w /mnt/w ligand-runner:dev nextflow lint main.nf workflow nextflow.config conf
```

Add or update focused tests for changed behavior, then run the regression suite and the lint checks. Workflow tasks use the package installed in the images, so rebuild `ligand-ml`/`ligand-chem` before running Nextflow after code changes. Do not claim molecular workflow validation unless the chemistry stages were actually executed.

CI (`.github/workflows/ci.yml`) runs the same checks on Linux against the installed package; keep the commands in sync.