# Person 3 — Tuning and route reconstruction results

## HistGradientBoosting hyperparameter search

- Search: 20 Optuna TPE trials, random seed 42.
- Objective: mean Macro F1 across five `GroupKFold` splits.
- Tuning data: a reproducible proportional sample of 250,000 candidate rows
  drawn from the 70 training trips. The 10-trip holdout was excluded.
- Best grouped-CV Macro F1: **0.75276**.
- Artifacts: `results/bayesian_optimization.json` and
  `results/optuna_trials.csv`.

Best parameters:

| Parameter | Value |
|---|---:|
| `learning_rate` | 0.07205896 |
| `max_iter` | 100 |
| `max_leaf_nodes` | 47 |
| `max_depth` | 3 |
| `min_samples_leaf` | 70 |
| `l2_regularization` | 0.00000108 |
| `max_bins` | 64 |

The tuned model was refit on all 1,480,968 rows from the 70 training trips.
On the untouched 10-trip holdout (186,368 rows), it reached:

| Metric | Result |
|---|---:|
| Accuracy | 0.93093 |
| Macro F1 | 0.80325 |
| Precision | 0.66739 |
| Recall | 0.62361 |
| Top-1 accuracy | 0.66406 |
| Log loss | 0.17952 |

These are agreement scores against provisional reference-route-derived labels,
not independently verified real-world accuracy. The saved local model is
`models/candidate_classifier.joblib`; model artifacts are ignored by Git.

## Classifier-only versus HMM on the 10 student-test trips

The route accuracy below is a nearest-reference-arc spatial proxy. It should
not be presented as the official score unless the evaluator confirms that
metric. No student-test trips were used to tune the classifier.

| Trip | Classifier proxy accuracy | HMM proxy accuracy | Classifier continuity | HMM continuity |
|---|---:|---:|---:|---:|
| 00000000 | 0.86896 | 0.88454 | 0.95484 | 0.99321 |
| 00000007 | 0.91654 | 0.84638 | 0.93718 | 0.99852 |
| 00000009 | 0.76072 | 0.79504 | 0.96756 | 0.99618 |
| 00000040 | 0.83639 | 0.81736 | 0.92051 | 0.98530 |
| 00000052 | 0.88168 | 0.91603 | 0.95785 | 0.99617 |
| 00000056 | 0.73345 | 0.70970 | 0.89942 | 0.99292 |
| 00000065 | 0.69531 | 0.59245 | 0.86701 | 0.99739 |
| 00000076 | 0.74361 | 0.76312 | 0.93603 | 0.99865 |
| 00000077 | 0.60880 | 0.54539 | 0.90639 | 0.99335 |
| 00000091 | 0.75629 | 0.74371 | 0.92211 | 0.99542 |

Across 19,160 GPS observations, weighted classifier-only proxy accuracy was
**0.75856** and HMM proxy accuracy was **0.73398**. HMM route continuity rose
from **0.92115** to **0.99321**; it improved continuity on all 10 trips and
improved proxy accuracy on 4 of 10. Candidate miss rate was **0.03596** and
there were **0** fallback observations.

Interpretation: the current HMM strongly improves directed-road continuity,
while this pointwise proxy accuracy decreases overall. Confirm the evaluator's
official route metric before deciding whether to tune transition/emission
weights; do not tune those weights on these student-test trips.
