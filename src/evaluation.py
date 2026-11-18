"""Evaluate a fitted classifier on held-out trips without row-level leakage."""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, log_loss)

from src.ml_utils import FEATURES, load_candidate_data, make_group_splits

ROOT = Path(__file__).resolve().parents[1]
COMBINED_DATA = ROOT / "results" / "all_train_candidates.csv"
DATA_FILES = ([COMBINED_DATA] if COMBINED_DATA.exists()
              else sorted((ROOT / "results").glob("*_candidates.csv")))


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
        metrics["log_loss"] = log_loss(y, model.predict_proba(X),
                                       labels=model.classes_)
    return metrics


def main():
    data = load_candidate_data(DATA_FILES)
    splits = make_group_splits(data, n_splits=5)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
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
        train = data.iloc[train_idx]
        model.fit(train[FEATURES], train["label"])
        rows.append({"fold": fold, **evaluate_holdout(data, test_idx, model)})
    output = ROOT / "results" / "evaluation.json"
    output.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    print(f"Saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
