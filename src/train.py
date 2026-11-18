
"""Train the selected HistGradientBoosting model on the 70-trip training split."""
from pathlib import Path
import json

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.ml_utils import (
    FEATURES,
    load_candidate_data,
    make_holdout_split,
    resolve_candidate_files,
)

ROOT = Path(__file__).resolve().parents[1]
TUNING_RESULTS = ROOT / "results" / "bayesian_optimization.json"
TUNABLE_PARAMETERS = {
    "learning_rate",
    "max_iter",
    "max_leaf_nodes",
    "max_depth",
    "min_samples_leaf",
    "l2_regularization",
    "max_bins",
}


def load_tuned_parameters() -> dict:
    """Load Optuna parameters for the selected HistGradientBoosting model."""
    if not TUNING_RESULTS.is_file():
        print("No Optuna results found; using HistGradientBoosting defaults.")
        return {}

    tuning = json.loads(TUNING_RESULTS.read_text())
    if tuning.get("model") != "HistGradientBoostingClassifier":
        raise ValueError(
            "results/bayesian_optimization.json must contain tuning results "
            "for HistGradientBoostingClassifier."
        )
    parameters = tuning.get("best_params")
    if not isinstance(parameters, dict):
        raise ValueError("Optuna results are missing a best_params object.")
    unexpected = set(parameters) - TUNABLE_PARAMETERS
    if unexpected:
        raise ValueError(
            f"Optuna results contain unsupported parameters: {sorted(unexpected)}"
        )
    print(
        "Using Optuna parameters with grouped-CV Macro F1 "
        f"{tuning.get('best_macro_f1', float('nan')):.4f}."
    )
    return parameters


def main():
    data = load_candidate_data(resolve_candidate_files(ROOT))

    # Match Person 2's comparison: 70 training trips, 10 held-out trips.
    train_idx, test_idx = make_holdout_split(
        data, test_groups=10, random_state=42
    )

    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    train_trips = train_data["trip_id"].nunique()
    test_trips = test_data["trip_id"].nunique()

    print(f"Training trips: {train_trips}")
    print(f"Held-out test trips: {test_trips}")
    print(f"Training rows: {len(train_data)}")
    print(f"Test rows (not used for fitting): {len(test_data)}")

    tuned_parameters = load_tuned_parameters()
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", HistGradientBoostingClassifier(
            **tuned_parameters,
            random_state=42,
            early_stopping=False,
        )),
    ])

    # Fit ONLY on the 70 training trips.
    model.fit(train_data[FEATURES], train_data["label"])

    out = ROOT / "models"
    out.mkdir(exist_ok=True)

    joblib.dump(model, out / "candidate_classifier.joblib")

    metadata = {
        "model": "HistGradientBoostingClassifier",
        "features": FEATURES,
        "training_rows": int(len(train_data)),
        "training_trips": int(train_trips),
        "held_out_rows": int(len(test_data)),
        "held_out_trips": int(test_trips),
        "random_state": 42,
        "tuned_parameters": tuned_parameters,
        "tuning_results": (
            str(TUNING_RESULTS.relative_to(ROOT))
            if TUNING_RESULTS.is_file() else None
        ),
        "label_note": (
            "Candidate labels are derived heuristically from reference-route "
            "data and are not independently verified ground truth."
        ),
    }

    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("HistGradientBoosting training complete.")
    print(f"Saved model to {out / 'candidate_classifier.joblib'}")


if __name__ == "__main__":
    main()
