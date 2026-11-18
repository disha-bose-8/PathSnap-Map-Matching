"""Evaluate a fitted classifier on held-out trips without row-level leakage."""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, log_loss)

from src.ml_utils import FEATURES, load_candidate_data, resolve_candidate_files, make_holdout_split, top1_accuracy

ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = resolve_candidate_files(ROOT)


def evaluate_holdout(data: pd.DataFrame, test_indices, model):
    test = data.iloc[test_indices]
    X, y = test[FEATURES], test["label"]
    pred = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        metrics["log_loss"] = log_loss(y, probabilities, labels=model.classes_)
        metrics["top1_accuracy"] = top1_accuracy(test, probabilities[:, list(model.classes_).index(1)])
    return metrics


def main():
    data = load_candidate_data(DATA_FILES)
    train_idx, test_idx = make_holdout_split(data, test_groups=10, random_state=42)
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        )),
    ])
    model.fit(train[FEATURES], train["label"])
    metrics = evaluate_holdout(data, test_idx, model)
    summary = {
        "protocol": "GroupShuffleSplit by trip_id; 10 held-out trips; random_state=42",
        "training_trip_count": int(train["trip_id"].nunique()),
        "heldout_trip_count": int(test["trip_id"].nunique()),
        "heldout_trip_ids": sorted(test["trip_id"].astype(str).unique().tolist()),
        "training_rows": int(len(train)),
        "heldout_rows": int(len(test)),
        "metrics": metrics,
        "label_caveat": "Labels are provisional reference-route-derived labels; metrics measure agreement with these labels.",
    }
    output = ROOT / "results" / "evaluation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
