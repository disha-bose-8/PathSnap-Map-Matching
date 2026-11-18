# PathSnap Map Matching

PathSnap maps a noisy GPS trip to directed road arcs. The inference path
generates nearby candidates, scores them with the fitted candidate classifier,
and uses road-network transitions with log-space Viterbi decoding.

## Install and run

From the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

The default model artifact is `models/candidate_classifier.joblib`. The default
trip partition is `data/Student folder/student_test`. Configure a different
location or partition before starting the server when the evaluator provides
separate trip files:

```powershell
$env:PATHSNAP_DATA_ROOT = "C:\path\to\Student folder"
$env:PATHSNAP_EVALUATION_PARTITION = "student_test"
$env:PATHSNAP_MODEL_PATH = "C:\path\to\candidate_classifier.joblib"
```

The API provides `GET /health`, `GET /trips`, and `POST /predict` (also
available as `POST /match`). The prediction request contains only a trip ID:

```json
{"trip_id": "00000000"}
```

The prediction response includes the arc ID for each GPS observation, the
classifier-only baseline, route geometry, continuity, and server-side
reference comparison when the trip folder contains a `.route` file. Reference
arc IDs stay on the server and are not returned to the UI. The comparison
currently uses the nearest reference arc for each GPS point; it is a spatial
proxy and must be replaced if the evaluator specifies a different route metric
or request schema.

Open `http://127.0.0.1:8000/docs` to inspect the API. For the second laptop,
replace `127.0.0.1` with the server laptop's Wi-Fi IPv4 address and allow the
server port through the private-network firewall when prompted.

To compare classifier-only choices with HMM output on the permitted
development partition, run:

```powershell
python -m src.route_evaluation
```

This writes per-trip results to `results/hmm_comparison.csv`. Review the
evaluator's official route-scoring rule before presenting the spatial proxy
accuracy as a final score.

To tune the selected HistGradientBoosting model and train the API artifact,
run these commands in order:

```powershell
python -m src.bayesian_optimization
python -m src.train
python -m src.evaluation
```

Optuna uses five-fold `GroupKFold` on the 70 training trips and reserves the
10-trip holdout. Its default search runs 20 trials on a reproducible sample of
up to 250,000 rows. `src.train` reads the best parameters from
`results/bayesian_optimization.json` when that file exists. The saved model is
trained on the 70 training trips; `src.evaluation` reports the untouched
holdout result. Set `PATHSNAP_OPTUNA_TRIALS` and
`PATHSNAP_OPTUNA_MAX_ROWS` to adjust the search budget.

The recorded tuning, holdout, and student-test route comparison results are in
[PERSON3_RESULTS.md](PERSON3_RESULTS.md).


## Person 2 — Candidate classifier model comparison

The following table records the offline comparison performed for model selection
using the supplied `all_train_candidates.csv` dataset. All five models were
scored on the same held-out set of **186,368 candidate rows from 10 trips**.

| Rank | Model | Accuracy | Precision | Recall | Macro F1 | Top-1 Accuracy | Log Loss |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **HistGradientBoosting** | **93.14%** | **66.99%** | **62.51%** | **80.44%** | **66.17%** | **0.1786** |
| 2 | Random Forest | 92.30% | 62.79% | 57.43% | 77.87% | 59.92% | 0.2854 |
| 3 | KNN | 91.08% | 55.33% | 58.38% | 75.92% | 56.54% | 0.7105 |
| 4 | Decision Tree | 88.14% | 42.84% | 53.88% | 70.52% | 53.67% | 4.2755 |
| 5 | Logistic Regression | 70.08% | 24.16% | 92.39% | 59.28% | 61.68% | 0.5418 |

### Evaluation setup

| Item | Details |
|---|---|
| Dataset size | 1,667,336 candidate rows across 80 trips |
| Split | 70 training trips; 10 held-out trips |
| Held-out rows | 186,368 |
| Features | `distance_to_road`, `heading_difference`, `speed`, `time_difference` |
| Target | `label` |
| Grouping key | `trip_id` |
| Split method | Trip-grouped holdout (`GroupShuffleSplit`), random state 42 |
| Fit sample sizes | 250,000 rows for Logistic Regression, Decision Tree, Random Forest and HistGradientBoosting; 50,000 rows for KNN |

**Selected model: HistGradientBoosting.** It achieved the highest macro F1
(80.44%), accuracy (93.14%), precision (66.99%) and top-1 accuracy (66.17%),
and the lowest log loss (0.1786) in this comparison.

**Interpretation and limitations:** This is a single trip-grouped holdout
comparison, not completed five-fold cross-validation. The labels are derived
from the supplied reference-route data, so these metrics measure agreement
with those labels rather than independently verified real-world map-matching
accuracy. The dataset is imbalanced (88.73% negative and 11.27% positive
labels), so accuracy should be read alongside precision, recall and macro F1.
The saved HistGradientBoosting pipeline was subsequently refitted on all
available rows from the 70 training trips; the table above reports the
comparison-fit metrics, not a separate evaluation of that final refit.

## Implementation notes

- `.arcs` row positions are retained as arc IDs; `.nodes` row positions are
  retained as node IDs.
- HMM states are candidate arcs for one GPS observation.
- Emissions use the trained model's positive-class probability.
- Transitions compare GPS displacement with directed shortest-path distance.
- Missing candidate layers recover with the nearest road arc and are listed in
  `fallback_observation_indices`.
- The route and model APIs do not expose reference arc sequences to the UI.
- Do not report experiment scores until the team has run the experiments and
  recorded their outputs.
