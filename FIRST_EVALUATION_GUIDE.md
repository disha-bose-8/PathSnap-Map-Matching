# PathSnap – First Evaluation Guide

## Review scope

This review covers the shared preprocessing/feature-engineering pipeline and Person 2's classifier selection/training work. Use only results produced by the repository; do not quote placeholder or invented scores.

## 1. Data preprocessing and feature engineering

The source data is organised by trip. Each trip contains GPS observations and a road network (nodes, arcs, and a reference route). Keeping a trip identifier is essential because observations from one trip must not be randomly split across training and validation.

The candidate generator:
1. Loads the GPS track, road nodes, road arcs, and the reference route for a trip.
2. Projects latitude/longitude into a local metre-based coordinate system for efficient spatial search.
3. Builds a KD-tree over road-arc midpoints and retrieves nearby road candidates for each GPS observation.
4. Calculates the point-to-road-segment distance for each candidate.
5. Calculates GPS movement speed from haversine distance divided by elapsed time.
6. Calculates GPS travel heading and the angular difference between GPS heading and road-segment heading.
7. Records elapsed time since the previous GPS observation.
8. Adds a provisional binary label indicating whether a candidate matches the nearest arc from the supplied reference route.

### Features used by the classifier

| Feature | Meaning | Why it matters |
|---|---|---|
| distance | Perpendicular distance from GPS point to candidate road segment, in metres | Smaller distance is generally more plausible |
| heading_diff | Smallest angular difference between GPS direction and road direction, in degrees | Smaller difference indicates better directional alignment |
| speed | Approximate GPS movement speed, in metres/second | Helps distinguish plausible movement patterns |
| time_gap | Time elapsed from the previous GPS point, in seconds | Helps interpret movement between observations |

The candidate generator currently selects up to 8 candidates from the 20 nearest road-arc midpoints. These are configurable parameters, not model hyperparameters.

### Preprocessing caveats to explain honestly

- Invalid/missing feature values are converted to numeric values and imputed with the median inside each model pipeline.
- Scaling is applied for models sensitive to feature magnitude, such as Logistic Regression and KNN.
- The candidate labels are derived from the provided reference route and are therefore provisional. They are not independent ground truth.
- The uploaded candidate dataset contains 80 trip groups. For this review, reserve 10 complete trips as a final holdout (fixed seed 42) and use the other 70 for model selection/tuning. The holdout trip IDs are 1, 13, 20, 22, 29, 33, 39, 41, 42, and 88. These are dataset trip IDs; do not describe them as student identities unless the team confirms that mapping.

## 2. Model selection and training (Person 2)

Five classifiers are compared:
1. Logistic Regression – simple linear baseline and interpretable coefficients.
2. K-Nearest Neighbours – predicts from similar feature vectors.
3. Decision Tree – learns rule-based feature splits.
4. Random Forest – combines many decision trees to reduce variance.
5. HistGradientBoosting – builds boosted trees sequentially to correct earlier errors.

### Validation design

The 10-trip holdout is separated first using GroupShuffleSplit by trip_id. The five-fold GroupKFold model-selection and Optuna tuning runs only inside the remaining 70 training trips. All candidate rows from a trip stay in one partition, preventing journey-level leakage.

### Metrics

- Macro F1: balances precision and recall across both classes.
- Precision: fraction of predicted positive candidates that are positive.
- Recall: fraction of positive candidates identified.
- Top-1 accuracy: for each GPS point, select the candidate with the highest positive-class score and check its label.
- Log loss: evaluates the quality of predicted class probabilities; lower is better.

The comparison script saves results/model_comparison.csv. The plot script creates results/model_comparison.png.

### Final training and optimisation

The holdout evaluation fits the selected HistGradientBoosting model on the 70 training trips and reports results only on the untouched 10-trip holdout. Optuna tunes its hyperparameters with five-fold GroupKFold on those same 70 trips. The final training script uses the saved Optuna parameters when available and refits the model on the 70 training trips, keeping the 10-trip holdout untouched. It saves:
- models/candidate_classifier.joblib
- models/metadata.json

Optuna uses Bayesian optimisation to search HistGradientBoosting hyperparameters against grouped-CV Macro F1. By default, the search uses a reproducible proportional sample of up to 250,000 rows from the 70 training trips and runs 20 trials. Its output is `results/bayesian_optimization.json` and `results/optuna_trials.csv`; the deployment training script reads the saved best parameters.

## 3. Commands for a live demonstration

Run from the repository root:

```bash
pip install -r requirements.txt
python -m pytest -q
python -m src.compare_models
python -m src.bayesian_optimization
python -m src.train
python -m src.evaluation
python -m src.plot_results
```

The supplied route-derived labels are provisional, so report these metrics as
agreement with the supplied labels rather than independent ground-truth
accuracy.

## 4. Short presentation script

"We first preprocess each trip by loading GPS observations and the road network. For every GPS point, we generate nearby road-segment candidates using a spatial KD-tree. We then engineer four features: distance to the road, heading difference, speed, and time gap. These features describe spatial closeness, direction, and movement behaviour.

For model selection, we compare Logistic Regression, KNN, Decision Tree, Random Forest, and HistGradientBoosting. We use Macro F1, precision, recall, Top-1 accuracy, and log loss. We keep trips together in validation and use Optuna with five-fold GroupKFold to tune the selected HistGradientBoosting model without using the 10-trip holdout. We then fit the final model on the 70 training trips and evaluate it on the reserved holdout.

At this stage, the candidate labels come from the supplied reference route, so we describe them as provisional labels. We need more independently verified trips before claiming final generalisation or the 10-student testing result."

## 5. Current evidence and limitations

- The full candidate dataset contains 1,667,336 rows across 80 trips. The full CSV is in `results/all_train_candidates.zip`; the checked-in loose CSV is only a two-trip sample. The loader detects this and resolves the full archive.
- The selected HistGradientBoosting model was tuned with 20 Optuna trials and five-fold trip-grouped validation on a proportional 250,000-row sample from the 70 training trips.
- The tuned model was evaluated on the untouched 10-trip holdout. Tuning, holdout, and HMM comparison results are recorded in [PERSON3_RESULTS.md](PERSON3_RESULTS.md).
- The test workflow exists, but its status for the current code revision has not been confirmed in this guide.
- The 10 held-out trip evaluation is a trip-level holdout; it is not proof that the trips correspond to ten different students. Confirm student ownership before calling it a 10-student test.
