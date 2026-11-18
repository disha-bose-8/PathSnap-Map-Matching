"""Tune a Random Forest with Optuna using trip-grouped cross-validation."""
from pathlib import Path
import json

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from src.ml_utils import FEATURES, load_candidate_data, make_group_splits

ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = sorted((ROOT / "results").glob("*_candidates.csv"))


def main():
    data = load_candidate_data(DATA_FILES)
    splits = make_group_splits(data, n_splits=5)
    X, y = data[FEATURES], data["label"]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
        scores = []
        for train_idx, test_idx in splits:
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(
                    **params, class_weight="balanced_subsample",
                    random_state=42, n_jobs=-1,
                )),
            ])
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            scores.append(f1_score(y.iloc[test_idx], pred,
                                   average="macro", zero_division=0))
        return sum(scores) / len(scores)

    study = optuna.create_study(direction="maximize",
                                study_name="pathsnap_random_forest")
    study.optimize(objective, n_trials=50)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    summary = {
        "best_macro_f1": study.best_value,
        "best_params": study.best_params,
        "trials": len(study.trials),
        "note": "Scores are valid only with multiple independent trip groups "
                "and trustworthy labels.",
    }
    (out / "bayesian_optimization.json").write_text(json.dumps(summary, indent=2))
    study.trials_dataframe().to_csv(out / "optuna_trials.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
