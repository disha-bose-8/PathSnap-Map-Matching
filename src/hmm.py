"""Candidate-emission and road-transition scoring for PathSnap."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.road_graph import RoadGraph, haversine_m
from src.viterbi import ViterbiResult, viterbi_decode


FEATURE_COLUMNS = ["distance", "heading_diff", "speed", "time_gap"]
CANDIDATE_COLUMNS = ["trip_id", "gps_index", "candidate_arc_id", *FEATURE_COLUMNS]


@dataclass(frozen=True)
class HMMConfig:
    """Small set of interpretable route-transition settings."""

    probability_floor: float = 1e-9
    base_transition_scale_m: float = 50.0
    transition_scale_m_per_second: float = 5.0
    disconnected_log_score: float = -18.0
    implausible_speed_mps: float = 55.0
    implausible_speed_slack_m: float = 100.0
    implausible_speed_penalty: float = 6.0


@dataclass(frozen=True)
class HMMResult:
    """One predicted arc per GPS observation plus a classifier-only baseline."""

    arc_ids: tuple[int, ...]
    classifier_only_arc_ids: tuple[int, ...]
    fallback_observation_indices: tuple[int, ...]
    log_score: float


def predict_positive_probabilities(
    model: object, candidates: pd.DataFrame
) -> np.ndarray:
    """Get class-1 probabilities from the trained candidate classifier."""
    missing_features = sorted(set(FEATURE_COLUMNS) - set(candidates.columns))
    if missing_features:
        raise ValueError(f"Candidate rows are missing model features: {missing_features}")
    if not hasattr(model, "predict_proba"):
        raise TypeError("The candidate model must implement predict_proba().")

    features = candidates[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(getattr(model, "classes_", []))
    positive_columns = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(candidates):
        raise ValueError("predict_proba() returned an unexpected array shape.")
    if len(positive_columns) != 1:
        raise ValueError("The fitted model must have a positive class labelled 1.")
    scores = probabilities[:, int(positive_columns[0])]
    if not np.isfinite(scores).all():
        raise ValueError("The model returned a non-finite candidate probability.")
    return np.clip(scores, 0.0, 1.0)


def _normalise_candidate_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "gps_point_id": "gps_index",
        "candidate_road_id": "candidate_arc_id",
        "distance_to_road": "distance",
        "heading_difference": "heading_diff",
        "time_difference": "time_gap",
    }
    rename = {
        old: new for old, new in aliases.items()
        if old in candidates.columns and new not in candidates.columns
    }
    frame = candidates.rename(columns=rename).copy()
    required = {"gps_index", "candidate_arc_id"}
    missing = sorted(required - set(frame.columns))
    if frame.empty and missing:
        return pd.DataFrame(columns=["gps_index", "candidate_arc_id", *FEATURE_COLUMNS])
    if missing:
        raise ValueError(f"Candidate rows are missing columns: {missing}")
    frame["gps_index"] = pd.to_numeric(frame["gps_index"], errors="coerce")
    frame["candidate_arc_id"] = pd.to_numeric(
        frame["candidate_arc_id"], errors="coerce"
    )
    frame = frame.dropna(subset=["gps_index", "candidate_arc_id"])
    finite_ids = np.isfinite(frame[["gps_index", "candidate_arc_id"]]).all(axis=1)
    frame = frame[
        finite_ids
        & np.equal(frame["gps_index"], np.floor(frame["gps_index"]))
        & np.equal(
            frame["candidate_arc_id"], np.floor(frame["candidate_arc_id"])
        )
    ].copy()
    frame["gps_index"] = frame["gps_index"].astype(int)
    frame["candidate_arc_id"] = frame["candidate_arc_id"].astype(int)
    return frame


def decode_candidates(
    track: pd.DataFrame,
    candidates: pd.DataFrame,
    graph: RoadGraph,
    model: object | None = None,
    probabilities: Sequence[float] | None = None,
    config: HMMConfig | None = None,
) -> HMMResult:
    """Decode candidate arcs using classifier probabilities and road distance.

    ``track`` uses the loader's longitude, latitude, timestamp columns.
    Candidate rows may use either Person 1's current column names or Person
    2's normalised aliases. If a GPS observation has no valid candidate, the
    nearest network arc is inserted as a recovery state and reported in the
    result.
    """
    if track.empty:
        raise ValueError("Cannot decode an empty GPS track.")
    for column in ("longitude", "latitude", "timestamp"):
        if column not in track.columns:
            raise ValueError(f"GPS track is missing the {column!r} column.")
    config = config or HMMConfig()
    if config.probability_floor <= 0.0 or config.probability_floor >= 1.0:
        raise ValueError("probability_floor must be between 0 and 1.")

    gps = track[["longitude", "latitude", "timestamp"]].apply(
        pd.to_numeric, errors="raise"
    ).reset_index(drop=True)
    if not np.isfinite(gps.to_numpy(dtype=float)).all():
        raise ValueError("GPS coordinates and timestamps must be finite.")

    frame = _normalise_candidate_columns(candidates)
    if frame.empty:
        raw_probabilities = np.empty(0, dtype=float)
        if probabilities is not None and len(probabilities) != 0:
            raise ValueError("probabilities must have one value per candidate row.")
    elif probabilities is None:
        if model is None:
            raise ValueError("Pass either a fitted model or candidate probabilities.")
        frame = frame.reset_index(drop=True)
        raw_probabilities = predict_positive_probabilities(model, frame)
    else:
        raw_probabilities = np.asarray(probabilities, dtype=float)
        if raw_probabilities.shape != (len(frame),):
            raise ValueError("probabilities must have one value per candidate row.")
    if not np.isfinite(raw_probabilities).all():
        raise ValueError("Candidate probabilities must be finite.")
    if ((raw_probabilities < 0.0) | (raw_probabilities > 1.0)).any():
        raise ValueError("Candidate probabilities must be between 0 and 1.")

    frame = frame.reset_index(drop=True)
    frame["_probability"] = raw_probabilities
    frame = frame[
        frame["gps_index"].between(0, len(gps) - 1)
        & frame["candidate_arc_id"].between(0, graph.number_of_arcs - 1)
    ].copy()

    candidate_layers: list[list[tuple[int, float, float]]] = []
    emission_layers: list[np.ndarray] = []
    classifier_only_ids: list[int] = []
    fallback_indices: list[int] = []

    for gps_index, point in gps.iterrows():
        group = frame.loc[frame["gps_index"] == gps_index]
        group = group.sort_values("_probability", ascending=False).drop_duplicates(
            "candidate_arc_id", keep="first"
        )
        layer: list[tuple[int, float, float]] = []
        scores: list[float] = []

        if group.empty:
            arc_id, _, fraction = graph.nearest_arc(
                float(point["longitude"]), float(point["latitude"])
            )
            layer.append((arc_id, fraction, config.probability_floor))
            scores.append(config.probability_floor)
            fallback_indices.append(int(gps_index))
        else:
            for arc_value, probability_value in group[
                ["candidate_arc_id", "_probability"]
            ].itertuples(index=False, name=None):
                arc_id = int(arc_value)
                fraction, _ = graph.project_point_to_arc(
                    float(point["longitude"]), float(point["latitude"]), arc_id
                )
                probability = float(probability_value)
                layer.append((arc_id, fraction, probability))
                scores.append(probability)

        score_array = np.clip(
            np.asarray(scores, dtype=float), config.probability_floor, 1.0
        )
        emission = score_array / score_array.sum()
        emission_layers.append(np.log(emission))
        candidate_layers.append(layer)
        classifier_only_ids.append(layer[int(np.argmax(score_array))][0])

    transition_layers: list[np.ndarray] = []
    timestamps = gps["timestamp"].to_numpy(dtype=float)
    longitudes = gps["longitude"].to_numpy(dtype=float)
    latitudes = gps["latitude"].to_numpy(dtype=float)
    for index in range(len(candidate_layers) - 1):
        elapsed = max(0.0, float(timestamps[index + 1] - timestamps[index]))
        gps_distance = haversine_m(
            latitudes[index], longitudes[index],
            latitudes[index + 1], longitudes[index + 1],
        )
        scale = max(
            config.base_transition_scale_m,
            config.transition_scale_m_per_second * elapsed,
        )
        previous_layer = candidate_layers[index]
        next_layer = candidate_layers[index + 1]
        transition = np.empty((len(previous_layer), len(next_layer)), dtype=float)

        for previous_index, (previous_arc, previous_fraction, _) in enumerate(
            previous_layer
        ):
            for next_index, (next_arc, next_fraction, _) in enumerate(next_layer):
                allowed_distance = (
                    config.implausible_speed_mps * max(elapsed, 1.0)
                    + config.implausible_speed_slack_m
                )
                network_distance = graph.transition_distance_m(
                    previous_arc,
                    previous_fraction,
                    next_arc,
                    next_fraction,
                    max_distance_m=allowed_distance,
                )
                if network_distance is None:
                    score = config.disconnected_log_score
                else:
                    score = -abs(network_distance - gps_distance) / scale
                transition[previous_index, next_index] = score
        transition_layers.append(transition)

    initial = np.full(
        len(candidate_layers[0]), -np.log(len(candidate_layers[0])), dtype=float
    )
    decoded: ViterbiResult = viterbi_decode(
        emission_layers,
        transition_layers,
        initial_log_scores=initial,
        disconnected_fallback_log_score=config.disconnected_log_score,
    )
    path_arc_ids = tuple(
        candidate_layers[index][state_index][0]
        for index, state_index in enumerate(decoded.state_indices)
    )
    return HMMResult(
        arc_ids=path_arc_ids,
        classifier_only_arc_ids=tuple(classifier_only_ids),
        fallback_observation_indices=tuple(fallback_indices),
        log_score=decoded.log_score,
    )
