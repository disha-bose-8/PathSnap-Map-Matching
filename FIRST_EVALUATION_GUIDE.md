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
- The current generated candidate summary covers two trips. More trip groups are needed for a reliable generalisation estimate and the separate 10-trip student evaluation.

## 2. Model selection and training (Person 2)

Five classifiers are compared:
1. Logistic Regression – simple linear baseline and interpretable coefficients.
2. K-Nearest Neighbours – predicts from similar feature vectors.
3. Decision Tree – learns rule-based feature splits.
4. Random Forest – combines many decision trees to reduce variance.
5. HistGradientBoosting – builds boosted trees sequentially to correct earlier errors.

### Validation design

GroupKFold uses trip_id as the group. All candidate rows from one trip stay together in either training or validation. This avoids leakage from nearly identical GPS observations within the same journey. The number of folds is capped by the number of distinct trips.

### Metrics

- Macro F1: balances precision and recall across both classes.
- Precision: fraction of predicted positive candidates that are positive.
- Recall: fraction of positive candidates identified.
- Top-1 accuracy: for each GPS point, select the candidate with the highest positive-class score and check its label.
- Log loss: evaluates the quality of predicted class probabilities; lower is better.

The comparison script saves results/model_comparison.csv. The plot script creates results/model_comparison.png.

### Final training and optimisation

The training script fits a Random Forest on all available labelled training candidates and saves:
- models/candidate_classifier.joblib
- models/metadata.json

Optuna uses Bayesian optimisation to search Random Forest hyperparameters against grouped-CV Macro F1. Run it only after enough independent trip groups and trustworthy labels are available; its output is results/bayesian_optimization.json and results/optuna_trials.csv.

## 3. Commands for a live demonstration

Run from the repository root:

```bash
pip install -r requirements.txt
python -m pytest -q
python -m src.compare_models
python -m src.train
python -m src.plot_results
```

For Bayesian optimisation after the team has added enough independently labelled trips:

```bash
python -m src.bayesian_optimization
```

## 4. Short presentation script

"We first preprocess each trip by loading GPS observations and the road network. For every GPS point, we generate nearby road-segment candidates using a spatial KD-tree. We then engineer four features: distance to the road, heading difference, speed, and time gap. These features describe spatial closeness, direction, and movement behaviour.

For model selection, we compare Logistic Regression, KNN, Decision Tree, Random Forest, and HistGradientBoosting. We use Macro F1, precision, recall, Top-1 accuracy, and log loss. We use GroupKFold by trip ID rather than randomly splitting rows, so points from the same trip cannot appear in both training and validation. We select a model based on validation performance and then train the final classifier on the available training data. We also prepared Optuna Bayesian optimisation for Random Forest hyperparameters.

At this stage, the candidate labels come from the supplied reference route, so we describe them as provisional labels. We need more independently verified trips before claiming final generalisation or the 10-student testing result."

## 5. Current evidence and limitations

- The repository has a candidate-generation summary for two trips.
- The automated unit-test workflow has passed for the latest test commit.
- A separate GitHub Actions workflow runs baseline model comparison, final baseline training, and creates a downloadable artifact.
- Do not report model scores until the ML review workflow completes and its artifact has been inspected.
- Do not claim 10-student testing is complete unless ten separate test trips have actually been evaluated.
