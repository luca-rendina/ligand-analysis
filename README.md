Updating my master thesis work.

Run the synthetic regression tests from the repository root in the project's Python environment:

```sh
PYTHONPATH=code/ml_protocol MPLBACKEND=Agg python -m unittest discover -s tests -v
```

Evaluation uses raw confusion counts (true classes in rows, predictions in columns).
Scalar precision/recall/F1 use class 0 (agonist); ROC curves retain class 1 (antagonist).
`model_fit_leaveOneOut(..., norm=True)` is for displaying matrices only.
Previously saved ensembles must be retrained to retain the raw validation counts.
