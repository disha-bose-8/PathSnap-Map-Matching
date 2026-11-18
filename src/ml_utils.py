"""Shared utilities for the PathSnap candidate-classification pipeline.

Expected input: one row per GPS-point/road-arc candidate, with columns
trip_id, gps_index, distance, heading_diff, speed, time_gap, label.
Labels must be independently verified ground truth before reporting metrics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import shutil
import zipfile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

FEATURES = ["distance", "heading_diff", "speed", "time_gap"]
REQUIRED_COLUMNS = ["trip_id", "gps_index", *FEATURES, "label"]


def resolve_candidate_files(root: str | Path) -> list[Path]:
    """Find the complete candidate dataset, preferring the full ZIP if needed."""
    results_dir = Path(root) / "results"
    combined = results_dir / "all_train_candidates.csv"
    archive = results_dir / "all_train_candidates.zip"

    # A stale or partial combined CSV must not hide the full archived dataset.
    if combined.exists():
        try:
            trip_count = pd.read_csv(combined, usecols=["trip_id"])["trip_id"].nunique()
        except (ValueError, KeyError, pd.errors.ParserError):
            trip_count = 0
        if trip_count > 10:
            print(f"Using {combined.name} ({trip_count} distinct trips).")
            return [combined]
        print(
            f"Found only {trip_count} distinct trips in {combined.name}; "
            "checking the full ZIP archive."
        )

    if archive.exists():
        with zipfile.ZipFile(archive) as zipped:
            members = [
                name for name in zipped.namelist()
                if Path(name).name == "all_train_candidates.csv"
            ]
            if len(members) != 1:
                raise ValueError(
                    "Expected exactly one all_train_candidates.csv inside "
                    f"{archive.name}; found {len(members)}."
                )
            with zipped.open(members[0]) as source:
                header = pd.read_csv(source, nrows=0).columns.tolist()
            if "trip_id" not in header:
                raise ValueError("The CSV inside the ZIP has no trip_id column.")
            with zipped.open(members[0]) as source:
                trip_count = pd.read_csv(source, usecols=["trip_id"])["trip_id"].nunique()
            if trip_count <= 10:
                raise ValueError(
                    f"The ZIP archive contains only {trip_count} distinct trips; "
                    "at least 11 are required for a 10-trip holdout."
                )
            with zipped.open(members[0]) as source, combined.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        print(f"Extracted full dataset ({trip_count} trips) to {combined}")
        return [combined]

    per_trip_files = sorted(results_dir.glob("*_candidates.csv"))
    if per_trip_files:
        raise ValueError(
            "Full dataset unavailable: found only per-trip candidate CSV files. "
            "Restore results/all_train_candidates.zip before running model evaluation."
        )
    raise FileNotFoundError(
        "Candidate data not found. Place results/all_train_candidates.csv "
        "or results/all_train_candidates.zip in the repository."
    )


def load_candidate_data(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and validate one or more candidate CSV files."""
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("No candidate CSV files were supplied.")
    data = pd.concat(frames, ignore_index=True)
    # Accept Person 1's candidate-generator column names and normalise them
    # to the stable schema consumed by the ML pipeline.
    aliases = {
        "gps_point_id": "gps_index",
        "candidate_road_id": "candidate_arc_id",
        "distance_to_road": "distance",
        "heading_difference": "heading_diff",
        "time_difference": "time_gap",
    }
    rename = {old: new for old, new in aliases.items()
              if old in data.columns and new not in data.columns}
    data = data.rename(columns=rename)
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Candidate data is missing columns: {missing}")
    data = data.dropna(subset=["trip_id", "gps_index", "label"]).copy()
    data["label"] = pd.to_numeric(data["label"], errors="raise").astype(int)
    if not set(data["label"].unique()).issubset({0, 1}):
        raise ValueError("label must contain only binary values 0 and 1.")
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def make_group_splits(data: pd.DataFrame, n_splits: int = 5):
    """Return trip-grouped splits; refuses leakage-prone single-trip CV."""
    groups = data["trip_id"].astype(str)
    n_groups = groups.nunique()
    if n_groups < 2:
        raise ValueError(
            "At least two distinct trips are required for GroupKFold; "
            f"only {n_groups} trip(s) are available."
        )
    splitter = GroupKFold(n_splits=min(n_splits, n_groups))
    return list(splitter.split(data[FEATURES], data["label"], groups))


def make_models(random_state: int = 42):
    """Build five baseline classifiers with preprocessing pipelines."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    prep = ColumnTransformer([("numeric", numeric, FEATURES)])
    return {
        "Logistic Regression": Pipeline([
            ("prep", prep),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                         random_state=random_state)),
        ]),
        "KNN": Pipeline([
            ("prep", prep),
            ("model", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ]),
        "Decision Tree": Pipeline([
            ("prep", prep),
            ("model", DecisionTreeClassifier(class_weight="balanced",
                                             random_state=random_state)),
        ]),
        "Random Forest": Pipeline([
            ("prep", prep),
            ("model", RandomForestClassifier(n_estimators=300,
                class_weight="balanced_subsample", random_state=random_state,
                n_jobs=-1)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", prep),
            ("model", HistGradientBoostingClassifier(random_state=random_state)),
        ]),
    }


def top1_accuracy(data: pd.DataFrame, probabilities: np.ndarray) -> float:
    """Accuracy after selecting the highest-scoring candidate per GPS point."""
    scored = data[["trip_id", "gps_index", "label"]].copy()
    scored["score"] = np.asarray(probabilities)
    chosen = scored.loc[scored.groupby(
        ["trip_id", "gps_index"]
    )["score"].idxmax()]
    return float(chosen["label"].mean()) if len(chosen) else float("nan")



def make_holdout_split(data: pd.DataFrame, test_groups: int = 10,
                       random_state: int = 42):
    """Reserve whole trips for final testing; return positional train/test indices."""
    groups = data["trip_id"].astype(str)
    n_groups = groups.nunique()
    if n_groups <= test_groups:
        raise ValueError(
            f"Need more than {test_groups} distinct trips; found {n_groups}."
        )
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_groups, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(data[FEATURES], data["label"], groups))
    train_groups = set(groups.iloc[train_idx])
    test_groups_found = set(groups.iloc[test_idx])
    if train_groups & test_groups_found:
        raise RuntimeError("Trip leakage detected between train and holdout sets.")
    if data.iloc[train_idx]["label"].nunique() < 2 or data.iloc[test_idx]["label"].nunique() < 2:
        raise ValueError("Both train and holdout partitions must contain both label classes.")
    return train_idx, test_idx
