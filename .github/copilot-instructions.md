# Project Guidelines

See [the local pipeline plan](../docs/LOCAL_PIPELINE_PLAN.md) for milestone scope, architecture, and scientific protocol. Keep this file concise and update it when project conventions change.

## Scientific Invariants

- Preserve the label mapping `0 = agonist` and `1 = antagonist`. Keep original pharmacology terms and assay context; do not silently map ambiguous or missing labels.
- Confusion matrices use true classes in rows and predictions in columns. Preserve raw counts for model selection and ensemble voting; normalize only for display.
- Legacy scalar precision, recall, and F1 use class 0 as positive. ROC/AUC use class 1 and continuous classifier scores, not hard predictions.
- New and migrated workflows must not infer pharmacology labels from filenames, docking scores, generated conformers, or docking poses. The legacy preprocessing currently uses filenames; do not extend that pattern. Join annotations to features through validated molecular and receptor identities, and keep preparation and docking independent of labels.
- Synthetic data is for software regression only, never biological ground truth.

## Dependencies and Data

- Use open-source runtime dependencies. Do not add proprietary toolkits such as OpenEye; audit actual imports and supported workflows before adding packages.
- Keep downloaded datasets, caches, models, and run outputs outside Git and container image layers. Commit only source, configuration, lockfiles, small manifests, and permitted small fixtures.
- Preserve the existing PLEC feature configuration unless a deliberate, documented scientific change is requested: protein depth 4, ligand depth 2, and 65,536 features.

## Verification

Never use host Python. Run the offline regression suite from the repository root in the Podman image (see README for building `ligand-ml:dev`):

```powershell
podman run --rm -v "${PWD}:/work:ro" -w /work -e PYTHONPATH=src:code/ml_protocol ligand-ml:dev python -m unittest discover -s tests -v
podman run --rm -v "${PWD}:/io:ro" -w /io ghcr.io/astral-sh/ruff:0.16.9 check --no-cache src tests
```

Add or update focused tests for changed behavior, then run the regression suite and the lint check. Do not claim molecular workflow validation unless the chemistry stages were actually executed.