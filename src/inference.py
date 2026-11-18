"""Unlabelled trip loading and candidate generation for API inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.candidate_generation import (
    K,
    SEARCH_K,
    bearing,
    haversine,
    point_to_segment_distance,
    project_coordinates,
)


def load_trip_inputs(
    data_root: str | Path, partition: str, trip_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """Load only GPS and network inputs; reference labels are not read here."""
    if not partition or Path(partition).name != partition:
        raise ValueError("partition must be a single directory name.")
    if not trip_id or Path(trip_id).name != trip_id or trip_id in {".", ".."}:
        raise ValueError("trip_id must be a single directory name.")

    root = Path(data_root).resolve()
    partition_dir = (root / partition).resolve()
    trip_dir = (partition_dir / trip_id).resolve()
    if not trip_dir.is_relative_to(partition_dir):
        raise ValueError("trip_id resolves outside the configured data directory.")
    prefix = trip_dir / trip_id

    track = pd.read_csv(
        str(prefix) + ".track",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"],
    )
    nodes = pd.read_csv(
        str(prefix) + ".nodes",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"],
    )
    arcs = pd.read_csv(
        str(prefix) + ".arcs",
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
    )
    if track.empty or nodes.empty or arcs.empty:
        raise ValueError(f"Trip {trip_id} has an empty track or road network.")
    return track, nodes, arcs, trip_dir


def generate_inference_candidates(
    trip_id: str,
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    k: int = K,
    search_k: int = SEARCH_K,
) -> pd.DataFrame:
    """Generate the same four candidate features without reading route labels."""
    if k < 1 or search_k < 1:
        raise ValueError("k and search_k must be positive integers.")
    for column in ("longitude", "latitude", "timestamp"):
        if column not in track.columns:
            raise ValueError(f"Track is missing the {column!r} column.")

    from_nodes = pd.to_numeric(arcs["from_node"], errors="raise").to_numpy(int)
    to_nodes = pd.to_numeric(arcs["to_node"], errors="raise").to_numpy(int)
    if (
        from_nodes.min() < 0 or to_nodes.min() < 0
        or from_nodes.max() >= len(nodes) or to_nodes.max() >= len(nodes)
    ):
        raise ValueError("An arc refers to a node outside the nodes table.")

    node_lats = pd.to_numeric(nodes["latitude"], errors="raise").to_numpy(float)
    node_lons = pd.to_numeric(nodes["longitude"], errors="raise").to_numpy(float)
    lat_a, lon_a = node_lats[from_nodes], node_lons[from_nodes]
    lat_b, lon_b = node_lats[to_nodes], node_lons[to_nodes]
    gps_lats = pd.to_numeric(track["latitude"], errors="raise").to_numpy(float)
    gps_lons = pd.to_numeric(track["longitude"], errors="raise").to_numpy(float)
    timestamps = pd.to_numeric(track["timestamp"], errors="raise").to_numpy(float)
    if not all(
        np.isfinite(array).all()
        for array in (node_lats, node_lons, gps_lats, gps_lons, timestamps)
    ):
        raise ValueError("Track and road coordinates/timestamps must be finite.")

    reference_latitude = float(np.mean(gps_lats))
    ax, ay = project_coordinates(lat_a, lon_a, reference_latitude)
    bx, by = project_coordinates(lat_b, lon_b, reference_latitude)
    midpoints = np.column_stack([(ax + bx) / 2.0, (ay + by) / 2.0])
    tree = cKDTree(midpoints)

    gps_x, gps_y = project_coordinates(
        gps_lats, gps_lons, reference_latitude
    )
    time_gaps = np.diff(timestamps, prepend=timestamps[0])
    speeds = np.zeros(len(track), dtype=float)
    headings = np.zeros(len(track), dtype=float)
    for index in range(len(track)):
        if index < len(track) - 1:
            start, end = index, index + 1
        elif index > 0:
            start, end = index - 1, index
        else:
            continue
        elapsed = abs(timestamps[end] - timestamps[start])
        if elapsed > 0.0:
            speeds[index] = haversine(
                gps_lats[start], gps_lons[start],
                gps_lats[end], gps_lons[end],
            ) / elapsed
        headings[index] = bearing(
            gps_lats[start], gps_lons[start],
            gps_lats[end], gps_lons[end],
        )

    nearby_count = min(search_k, len(arcs))
    candidate_count = min(k, len(arcs))
    records: list[dict[str, object]] = []
    for gps_index in range(len(track)):
        _, nearby_ids = tree.query(
            [gps_x[gps_index], gps_y[gps_index]], k=nearby_count
        )
        nearby_ids = np.atleast_1d(nearby_ids).astype(int)
        distances = np.asarray(
            [
                point_to_segment_distance(
                    gps_x[gps_index], gps_y[gps_index],
                    ax[arc_id], ay[arc_id], bx[arc_id], by[arc_id],
                )
                for arc_id in nearby_ids
            ],
            dtype=float,
        )
        for position in np.argsort(distances)[:candidate_count]:
            arc_id = int(nearby_ids[position])
            road_heading = bearing(
                lat_a[arc_id], lon_a[arc_id], lat_b[arc_id], lon_b[arc_id]
            )
            heading_difference = abs(headings[gps_index] - road_heading)
            heading_difference = min(heading_difference, 360.0 - heading_difference)
            records.append({
                "trip_id": trip_id,
                "gps_index": gps_index,
                "candidate_arc_id": arc_id,
                "distance": float(distances[position]),
                "heading_diff": float(heading_difference),
                "speed": float(speeds[gps_index]),
                "time_gap": float(time_gaps[gps_index]),
            })

    return pd.DataFrame.from_records(records)


def load_reference_route(trip_dir: str | Path, trip_id: str) -> np.ndarray:
    """Read reference arc IDs for server-side scoring after prediction."""
    route_path = Path(trip_dir) / f"{trip_id}.route"
    route = pd.read_csv(
        route_path, sep=r"\s+", header=None, names=["arc_id"]
    )
    return pd.to_numeric(route["arc_id"], errors="raise").to_numpy(dtype=int)


def nearest_reference_arc_labels(
    track: pd.DataFrame,
    reference_arc_ids: np.ndarray,
    graph,
) -> np.ndarray:
    """Create the documented spatial proxy labels for development reporting."""
    route_arcs = np.unique(reference_arc_ids)
    route_arcs = route_arcs[
        (route_arcs >= 0) & (route_arcs < graph.number_of_arcs)
    ]
    if route_arcs.size == 0:
        raise ValueError("The reference route contains no valid arc IDs.")

    labels: list[int] = []
    for point in track.itertuples(index=False):
        longitude = float(getattr(point, "longitude"))
        latitude = float(getattr(point, "latitude"))
        best_arc = int(route_arcs[0])
        best_distance = float("inf")
        for arc_id in route_arcs:
            _, distance = graph.project_point_to_arc(
                longitude, latitude, int(arc_id)
            )
            if distance < best_distance:
                best_arc = int(arc_id)
                best_distance = distance
        labels.append(best_arc)
    return np.asarray(labels, dtype=int)
