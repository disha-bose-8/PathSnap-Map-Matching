"""Compare independent classifier choices with classifier + HMM decoding.

The route accuracy written by this script uses nearest reference arcs as
pointwise labels. It is an explicit spatial proxy, not an official route score
unless the evaluator confirms that alignment rule.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm

from src.hmm import decode_candidates
from src.inference import (
    generate_inference_candidates,
    load_reference_route,
    load_trip_inputs,
    nearest_reference_arc_labels,
)
from src.road_graph import RoadGraph, route_continuity


ROOT = Path(__file__).resolve().parents[1]


def evaluate_partition(
    data_root: str | Path,
    partition: str,
    model_path: str | Path,
) -> pd.DataFrame:
    """Return per-trip classifier/HMM metrics for one permitted partition."""
    data_root = Path(data_root).resolve()
    model_path = Path(model_path).resolve()
    partition_dir = data_root / partition
    if not partition_dir.is_dir():
        raise FileNotFoundError(f"Trip partition was not found: {partition_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact was not found: {model_path}")

    model = joblib.load(model_path)
    rows: list[dict[str, object]] = []
    trip_ids = sorted(
        folder.name
        for folder in partition_dir.iterdir()
        if folder.is_dir() and (folder / f"{folder.name}.track").is_file()
    )
    for trip_id in tqdm(trip_ids, desc="Evaluating trips"):
        track, nodes, arcs, trip_dir = load_trip_inputs(
            data_root, partition, trip_id
        )
        graph = RoadGraph(nodes, arcs)
        candidates = generate_inference_candidates(trip_id, track, nodes, arcs)
        decoded = decode_candidates(track, candidates, graph, model=model)
        reference_ids = load_reference_route(trip_dir, trip_id)
        reference_labels = nearest_reference_arc_labels(
            track, reference_ids, graph
        )
        candidate_sets = (
            candidates.groupby("gps_index")["candidate_arc_id"]
            .apply(lambda values: set(values.astype(int)))
            .to_dict()
        )
        candidate_miss_rate = float(np.mean([
            int(label) not in candidate_sets.get(index, set())
            for index, label in enumerate(reference_labels)
        ]))
        rows.append({
            "trip_id": trip_id,
            "gps_observations": len(track),
            "candidate_miss_rate": candidate_miss_rate,
            "classifier_only_nearest_reference_accuracy": float(
                np.mean(
                    np.asarray(decoded.classifier_only_arc_ids)
                    == reference_labels
                )
            ),
            "hmm_nearest_reference_accuracy": float(
                np.mean(np.asarray(decoded.arc_ids) == reference_labels)
            ),
            "classifier_only_route_continuity": route_continuity(
                graph, track, decoded.classifier_only_arc_ids
            ),
            "hmm_route_continuity": route_continuity(
                graph, track, decoded.arc_ids
            ),
            "fallback_observations": len(decoded.fallback_observation_indices),
        })
    return pd.DataFrame(rows)


def main() -> None:
    data_root = Path(os.getenv(
        "PATHSNAP_DATA_ROOT", ROOT / "data" / "Student folder"
    ))
    partition = os.getenv("PATHSNAP_EVALUATION_PARTITION", "student_test")
    model_path = Path(os.getenv(
        "PATHSNAP_MODEL_PATH", ROOT / "models" / "candidate_classifier.joblib"
    ))
    results = evaluate_partition(data_root, partition, model_path)
    output = ROOT / "results" / "hmm_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(results.to_string(index=False))
    print(f"Saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
