"""Tune HistGradientBoosting with Optuna and trip-grouped cross-validation."""
from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.ml_utils import (
    FEATURES,
    load_candidate_data,
    make_group_splits,
    make_holdout_split,
    resolve_candidate_files,
)


ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
DEFAULT_TRIALS = 20
DEFAULT_MAX_ROWS = 250_000
CV_SPLITS = 5


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def sample_by_trip(
    data: pd.DataFrame, max_rows: int, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Take a reproducible proportional sample while retaining every trip."""
    if max_rows <= 0 or len(data) <= max_rows:
        return data.reset_index(drop=True)

    groups = list(data.groupby("trip_id", sort=True))
    if max_rows < len(groups):
        raise ValueError(
            "The tuning sample must have at least one row per training trip."
        )

    exact_sizes = np.asarray(
        [len(group) * max_rows / len(data) for _, group in groups],
        dtype=float,
    )
    sample_sizes = np.floor(exact_sizes).astype(int)
    sample_sizes = np.maximum(sample_sizes, 1)
    remaining = max_rows - int(sample_sizes.sum())

    if remaining > 0:
        order = np.argsort(-(exact_sizes - np.floor(exact_sizes)), kind="stable")
        for index in order[:remaining]:
            sample_sizes[index] += 1
    elif remaining < 0:
        order = np.argsort(exact_sizes - np.floor(exact_sizes), kind="stable")
        for index in order:
            if remaining == 0:
                break
            if sample_sizes[index] > 1:
                sample_sizes[index] -= 1
                remaining += 1

    sampled = [
        group.sample(n=int(sample_sizes[index]), random_state=random_state + index)
        for index, (_, group) in enumerate(groups)
    ]
    return pd.concat(sampled, ignore_index=True)


def main() -> None:
    trial_count = _env_int("PATHSNAP_OPTUNA_TRIALS", DEFAULT_TRIALS)
    max_rows = _env_int("PATHSNAP_OPTUNA_MAX_ROWS", DEFAULT_MAX_ROWS)

    data = load_candidate_data(resolve_candidate_files(ROOT))
    training_indices, _ = make_holdout_split(
        data, test_groups=10, random_state=RANDOM_STATE
    )
    training_data = data.iloc[training_indices].reset_index(drop=True)
    tuning_data = sample_by_trip(training_data, max_rows)
    splits = make_group_splits(tuning_data, n_splits=CV_SPLITS)
    features = tuning_data[FEATURES]
    labels = tuning_data["label"]

    print(
        "Optuna tuning HistGradientBoosting on "
        f"{len(tuning_data):,} sampled rows from "
        f"{training_data['trip_id'].nunique()} training trips; "
        f"the 10-trip holdout is reserved."
    )

    def objective(trial: optuna.Trial) -> float:
        parameters = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.2, log=True
            ),
            "max_iter": trial.suggest_int("max_iter", 100, 400, step=50),
            "max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes", 15, 63, step=8
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 10, 100, step=10
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-6, 10.0, log=True
            ),
            "max_bins": trial.suggest_categorical("max_bins", [64, 128, 255]),
        }

        fold_scores = []
        for train_indices, validation_indices in splits:
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", HistGradientBoostingClassifier(
                    **parameters,
                    random_state=RANDOM_STATE,
                    # GroupKFold is the only validation split used for tuning.
                    early_stopping=False,
                )),
            ])
            model.fit(features.iloc[train_indices], labels.iloc[train_indices])
            prediction = model.predict(features.iloc[validation_indices])
            fold_scores.append(
                f1_score(
                    labels.iloc[validation_indices],
                    prediction,
                    average="macro",
                    zero_division=0,
                )
            )

        trial.set_user_attr("fold_macro_f1", [float(score) for score in fold_scores])
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        study_name="pathsnap_hist_gradient_boosting",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=trial_count, show_progress_bar=True)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    summary = {
        "model": "HistGradientBoostingClassifier",
        "objective": "mean 5-fold GroupKFold Macro F1",
        "best_macro_f1": float(study.best_value),
        "best_params": study.best_params,
        "completed_trials": len(study.trials),
        "training_rows_available": int(len(training_data)),
        "tuning_rows_sampled": int(len(tuning_data)),
        "training_trip_count": int(training_data["trip_id"].nunique()),
        "holdout_trips_reserved": 10,
        "cv_splits": CV_SPLITS,
        "random_state": RANDOM_STATE,
        "label_note": (
            "Candidate labels are derived from the supplied reference-route "
            "data; metrics measure agreement with those provisional labels."
        ),
    }
    (results_dir / "bayesian_optimization.json").write_text(
        json.dumps(summary, indent=2)
    )
    study.trials_dataframe().to_csv(
        results_dir / "optuna_trials.csv", index=False
    )
    print(json.dumps(summary, indent=2))
    print("Saved results/bayesian_optimization.json and results/optuna_trials.csv")


if __name__ == "__main__":
    main()
