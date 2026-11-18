# Person 2 – Machine Learning Pipeline

This contribution adds the candidate-classification utilities, baseline model
comparison, held-out-trip evaluation, final model training, and Optuna tuning.

## Expected candidate data

Place one or more CSV files in `results/` with filenames ending in
`_candidates.csv`. Each file must contain:

- `trip_id`: identifier used to keep entire trips together during validation
- `gps_index`: GPS observation index
- `candidate_arc_id`: candidate road-arc identifier
- `distance`, `heading_diff`, `speed`, `time_gap`: model features
- `label`: binary candidate label (0 or 1)

The current repository sample contains only trip `00000001`. GroupKFold
requires at least two distinct trips, so comparison, evaluation, and Optuna
scripts intentionally stop until more trips are added. Do not replace this
check with random row-level splitting: that would leak observations from the
same trip across train and validation sets.

## Install and run

From the repository root:

```bash
pip install -r requirements.txt
python -m src.compare_models
python -m src.evaluation
python -m src.bayesian_optimization
python -m src.train
python -m src.plot_results
python -m pytest -q
```

Outputs are written under `results/` and the fitted model plus metadata are
written under `models/`. The plotting script creates report-ready PNGs when
comparison and Optuna result files exist. The unit tests run locally and also
run automatically through GitHub Actions on pushes and pull requests.

## Important label and evaluation caveat

The current candidate generator assigns labels by proximity to the supplied
reference route. These are reference-route-derived labels, not independently
verified ground truth. Model scores should be described as agreement with
those provisional labels unless the team validates the labels independently.
The requested 10-trip student-testing evaluation also requires 10 separate
testing trips; it cannot be claimed from the one sample trip currently present.
