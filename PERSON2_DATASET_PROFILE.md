# Person 2 – Uploaded Training Dataset Profile

## Source and handling

The dataset supplied for the first evaluation is `all_train_candidates.csv`, inside
`all_train_candidates(1).zip`. The CSV is approximately 135.4 MB and contains
1,667,336 rows. Keep the raw CSV out of Git history; extract it locally into
`results/all_train_candidates.csv` (the existing pipeline already looks for
that path). The archive itself should not be committed to the repository.

## Schema compatibility

The uploaded CSV uses these columns:

- `trip_id`
- `gps_point_id`
- `candidate_road_id`
- `distance_to_road`
- `heading_difference`
- `speed`
- `time_difference`
- `label`

`src/ml_utils.py` already normalises these aliases to the pipeline schema:
`gps_index`, `candidate_arc_id`, `distance`, `heading_diff`, and `time_gap`.

Use only `distance`, `heading_diff`, `speed`, and `time_gap` as the initial
model features. Keep `trip_id` as the GroupKFold grouping variable and `label`
as the target. The three ID columns are identifiers, not model features.

## Dataset profile

| Check | Result |
|---|---:|
| Candidate rows | 1,667,336 |
| Distinct trips | 80 |
| Distinct GPS points (trip + point ID) | 208,417 |
| Candidate roads per GPS point | 8 |
| Missing values | 0 |
| Positive labels (1) | 187,942 |
| Negative labels (0) | 1,479,394 |
| GPS points with at least one positive candidate | 187,942 |
| GPS points with no positive candidate | 20,475 |

Each GPS point has exactly eight candidate rows. Labels are imbalanced:
approximately 11.27% positive and 88.73% negative.

## Feature-distribution checks

| Feature | Median | 95th percentile | Maximum |
|---|---:|---:|---:|
| distance_to_road | 28.512 | 231.040 | 1,571.853 |
| heading_difference | 87.112 | 175.149 | 180.000 |
| speed | 11.069 | 33.242 | 180.138 |
| time_difference | 1.000 | 1.000 | 153,064.000 |

`time_difference` has a particularly long upper tail: its 99th percentile is
3, while its maximum is 153,064. Inspect this extreme value and the feature's
units/semantics before deciding whether to cap, transform, or exclude it. Do
not silently remove it.

## Validation and interpretation

The 80 distinct trip groups make trip-grouped cross-validation possible.
Continue to split by `trip_id`; do not use random row-level splits.

The 20,475 GPS points without a positive candidate must be accounted for when
interpreting candidate-ranking / top-1 metrics. A top-1 prediction cannot be
correct against a positive candidate label for those points.

The repository's existing documentation notes that candidate labels are
derived from proximity to the supplied reference route. Until the team
independently verifies those labels, describe model metrics as agreement with
the provisional labels, not as independently verified map-matching accuracy.

## Local setup

1. Extract the ZIP.
2. Copy `all_train_candidates.csv` to `results/all_train_candidates.csv`.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run the grouped model comparison and evaluation scripts from the repository root.

Do not commit the 135.4 MB CSV or ZIP to normal Git history.