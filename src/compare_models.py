"""Compare the five baseline classifiers with trip-grouped cross-validation."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, log_loss)

from src.ml_utils import FEATURES, load_candidate_data, make_group_splits, make_models

ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = sorted((ROOT / "results").glob("*_candidates.csv"))
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def main():
    data = load_candidate_data(DATA_FILES)
    splits = make_group_splits(data, n_splits=5)
    X, y = data[FEATURES], data["label"]
    rows = []
    for name, model in make_models().items():
        fold_scores = []
        for train_idx, test_idx in splits:
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            row = {
                "macro_f1": f1_score(y.iloc[test_idx], pred, average="macro",
                                     zero_division=0),
                "top1_accuracy": np.nan,
                "precision": precision_score(y.iloc[test_idx], pred,
                                              zero_division=0),
                "recall": recall_score(y.iloc[test_idx], pred,
                                       zero_division=0),
                "log_loss": np.nan,
            }
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X.iloc[test_idx])
                row["log_loss"] = log_loss(y.iloc[test_idx], probs,
                                           labels=model.classes_)
                positive = probs[:, list(model.classes_).index(1)]
                part = data.iloc[test_idx][["trip_id", "gps_index", "label"]].copy()
                part["score"] = positive
                chosen = part.loc[part.groupby(["trip_id", "gps_index"])["score"].idxmax()]
                row["top1_accuracy"] = accuracy_score(chosen["label"],
                                                       np.ones(len(chosen), dtype=int))
            fold_scores.append(row)
        rows.append({"model": name, **{
            metric: float(np.nanmean([fold[metric] for fold in fold_scores]))
            for metric in fold_scores[0]
        }})
    result = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    result.to_csv(OUT / "model_comparison.csv", index=False)
    print(result.to_string(index=False))
    print("\nSaved results/model_comparison.csv")


if __name__ == "__main__":
    main()
