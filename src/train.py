
"""Train the selected HistGradientBoosting model on the 70-trip training split."""
from pathlib import Path
import json
import warnings

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

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", HistGradientBoostingClassifier(random_state=42)),
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
