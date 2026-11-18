"""Fit the selected candidate classifier and save it with metadata."""
from pathlib import Path
import json
import warnings

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.ml_utils import FEATURES, load_candidate_data, resolve_candidate_files

ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = resolve_candidate_files(ROOT)


def main():
    data = load_candidate_data(DATA_FILES)
    trips = data["trip_id"].nunique()
    if trips < 2:
        warnings.warn(
            "Only one trip is present. The model can be fitted for a pipeline "
            "smoke test, but its metrics are not evidence of generalisation. "
            "Add independently labelled trips before evaluation.",
            RuntimeWarning,
        )
    X, y = data[FEATURES], data["label"]
    tuning_file = ROOT / "results" / "bayesian_optimization.json"
    tuned_params = {}
    if tuning_file.exists():
        tuning = json.loads(tuning_file.read_text())
        tuned_params = tuning.get("best_params", {})
        print("Using Optuna-selected hyperparameters.")
    else:
        print("No Optuna results found; using baseline Random Forest settings.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            **tuned_params, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        )),
    ])
    model.fit(X, y)
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "candidate_classifier.joblib")
    metadata = {
        "model": "RandomForestClassifier",
        "features": FEATURES,
        "training_rows": int(len(data)),
        "training_trips": int(trips),
        "label_note": (
            "Current candidate labels are generated from the reference route "
            "heuristically; verify their provenance before treating metrics "
            "as independent ground truth."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved model using {len(data)} rows from {trips} trip(s).")
    print("Saved to models/candidate_classifier.joblib")


if __name__ == "__main__":
    main()
