"""Unit tests for the Person 2 candidate-classification utilities."""
import numpy as np
import pandas as pd
import pytest

from src.ml_utils import (
    FEATURES,
    load_candidate_data,
    make_group_splits,
    make_models,
    top1_accuracy,
)


def sample_frame(n_trips=3):
    rows = []
    for trip in range(n_trips):
        for gps_index in range(4):
            # Positive candidate has smaller distance and heading difference.
            rows.append({
                "trip_id": f"trip-{trip}",
                "gps_index": gps_index,
                "candidate_arc_id": trip * 100 + gps_index * 2,
                "distance": 2.0 + gps_index,
                "heading_diff": 5.0,
                "speed": 30.0 + gps_index,
                "time_gap": 1.0,
                "label": 1,
            })
            rows.append({
                "trip_id": f"trip-{trip}",
                "gps_index": gps_index,
                "candidate_arc_id": trip * 100 + gps_index * 2 + 1,
                "distance": 30.0 + gps_index,
                "heading_diff": 150.0,
                "speed": 30.0 + gps_index,
                "time_gap": 1.0,
                "label": 0,
            })
    return pd.DataFrame(rows)


def test_load_candidate_data_validates_and_converts_features(tmp_path):
    path = tmp_path / "candidates.csv"
    frame = sample_frame(n_trips=2)
    frame["speed"] = frame["speed"].astype(object)
    frame.loc[0, "speed"] = "not-a-number"
    frame.to_csv(path, index=False)

    loaded = load_candidate_data([path])

    assert set(FEATURES).issubset(loaded.columns)
    assert pd.isna(loaded.loc[0, "speed"])
    assert set(loaded["label"].unique()) == {0, 1}


def test_load_candidate_data_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"trip_id": ["a"], "label": [1]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_candidate_data([path])


def test_load_candidate_data_rejects_non_binary_labels(tmp_path):
    path = tmp_path / "bad_labels.csv"
    frame = sample_frame(n_trips=2)
    frame.loc[0, "label"] = 2
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="binary"):
        load_candidate_data([path])


def test_group_splits_keep_trips_separate():
    frame = sample_frame(n_trips=3)

    splits = make_group_splits(frame, n_splits=3)

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        train_trips = set(frame.iloc[train_idx]["trip_id"])
        test_trips = set(frame.iloc[test_idx]["trip_id"])
        assert train_trips.isdisjoint(test_trips)


def test_group_splits_refuse_single_trip():
    frame = sample_frame(n_trips=1)

    with pytest.raises(ValueError, match="At least two distinct trips"):
        make_group_splits(frame)


def test_all_five_models_fit_and_predict():
    frame = sample_frame(n_trips=3)
    X, y = frame[FEATURES], frame["label"]

    models = make_models()
    assert len(models) == 5

    for model in models.values():
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert set(np.unique(predictions)).issubset({0, 1})


def test_top1_accuracy_selects_highest_scoring_candidate():
    frame = sample_frame(n_trips=1)
    scores = np.where(frame["label"].to_numpy() == 1, 0.9, 0.1)

    assert top1_accuracy(frame, scores) == pytest.approx(1.0)


def test_loader_normalises_person1_candidate_column_names(tmp_path):
    path = tmp_path / "person1_candidates.csv"
    frame = sample_frame(n_trips=2).rename(columns={
        "gps_index": "gps_point_id",
        "candidate_arc_id": "candidate_road_id",
        "distance": "distance_to_road",
        "heading_diff": "heading_difference",
        "time_gap": "time_difference",
    })
    frame.to_csv(path, index=False)
    loaded = load_candidate_data([path])
    assert set(FEATURES).issubset(loaded.columns)
    assert {"gps_index", "candidate_arc_id"}.issubset(loaded.columns)
