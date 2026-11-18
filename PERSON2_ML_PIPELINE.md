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

The uploaded dataset contains 80 distinct trips. The pipeline reserves 10 whole trips for final evaluation using a deterministic `GroupShuffleSplit` (`random_state=42`), leaving 70 trips for model selection and tuning. Five-fold `GroupKFold` runs only inside those 70 training trips. Do not replace this with random row-level splitting: that would leak observations from the same trip across partitions. The fixed 10-trip holdout is `1, 13, 20, 22, 29, 33, 39, 41, 42, 88` for this dataset and split seed.

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

`compare_models` selects among the five baselines using grouped CV on the 70 training trips. `evaluation` fits a baseline Random Forest on those 70 trips and reports the untouched 10-trip holdout metrics. Optuna also tunes only on the 70 training trips. Finally, `train` fits the deployment model on all 80 trips after holdout evaluation has been recorded. Outputs are written under `results/` and the fitted model plus metadata are written under `models/`. The plotting script creates report-ready PNGs when
comparison and Optuna result files exist. The unit tests run locally and also
run automatically through GitHub Actions on pushes and pull requests.

## Important label and evaluation caveat

The current candidate generator assigns labels by proximity to the supplied
reference route. These are reference-route-derived labels, not independently
verified ground truth. Model scores should be described as agreement with
those provisional labels unless the team validates the labels independently.
The requested 10-trip student-testing evaluation also requires 10 separate
testing trips; it cannot be claimed from the one sample trip currently present.
