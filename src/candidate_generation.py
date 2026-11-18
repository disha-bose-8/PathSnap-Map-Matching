
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_trip


# CONFIGURATION
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Student folder"
TRAIN_DIR = DATA / "train"
OUTPUT_DIR = ROOT / "results"

K = 8
SEARCH_K = 20

# First test with 2 trips.
# Change to None to process all training trips.
MAX_TRIPS = 2

EARTH_RADIUS = 6371000


# GEOGRAPHICAL FUNCTIONS
def project_coordinates(lat, lon, ref_lat):
    x = lon * 111320 * np.cos(np.radians(ref_lat))
    y = lat * 110540
    return x, y


def bearing(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)

    y = np.sin(dlon) * np.cos(lat2)
    x = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))


def point_to_segment_distance(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay

    denominator = dx * dx + dy * dy

    if denominator == 0:
        return np.hypot(px - ax, py - ay)

    t = ((px - ax) * dx + (py - ay) * dy) / denominator
    t = np.clip(t, 0, 1)

    nearest_x = ax + t * dx
    nearest_y = ay + t * dy

    return np.hypot(px - nearest_x, py - nearest_y)


# PROCESS ONE TRIP
def generate_candidates(trip_id):

    track, nodes, arcs, route = load_trip("train", trip_id)

    if track.empty or nodes.empty or arcs.empty or route.empty:
        raise ValueError(f"Empty data found in trip {trip_id}")

    edge_start = arcs["from_node"].to_numpy(dtype=int)
    edge_end = arcs["to_node"].to_numpy(dtype=int)

    if (
        edge_start.min() < 0 or edge_end.min() < 0
        or edge_start.max() >= len(nodes)
        or edge_end.max() >= len(nodes)
    ):
        raise ValueError(f"Invalid node IDs in trip {trip_id}")

    lat_a = nodes.iloc[edge_start]["latitude"].to_numpy()
    lon_a = nodes.iloc[edge_start]["longitude"].to_numpy()
    lat_b = nodes.iloc[edge_end]["latitude"].to_numpy()
    lon_b = nodes.iloc[edge_end]["longitude"].to_numpy()

    ref_lat = track["latitude"].mean()

    ax, ay = project_coordinates(lat_a, lon_a, ref_lat)
    bx, by = project_coordinates(lat_b, lon_b, ref_lat)

    midpoints = np.column_stack([
        (ax + bx) / 2,
        (ay + by) / 2
    ])

    tree = cKDTree(midpoints)

    gps_lat = track["latitude"].to_numpy()
    gps_lon = track["longitude"].to_numpy()

    gps_x, gps_y = project_coordinates(
        gps_lat, gps_lon, ref_lat
    )

    timestamps = track["timestamp"].to_numpy()
    time_gaps = np.diff(timestamps, prepend=timestamps[0])

    speeds = np.zeros(len(track))
    headings = np.zeros(len(track))

    for i in range(len(track)):

        if i < len(track) - 1:
            j = i + 1
            start, end = i, j
        elif i > 0:
            j = i - 1
            start, end = j, i
        else:
            continue

        dt = abs(timestamps[j] - timestamps[i])

        if dt > 0:
            distance = haversine(
                gps_lat[start], gps_lon[start],
                gps_lat[end], gps_lon[end]
            )
            speeds[i] = distance / dt

        headings[i] = bearing(
            gps_lat[start], gps_lon[start],
            gps_lat[end], gps_lon[end]
        )

    # PROVISIONAL REFERENCE LABELS
    reference_ids = np.unique(route["arc_id"].to_numpy(dtype=int))

    reference_ids = reference_ids[
        (reference_ids >= 0) & (reference_ids < len(arcs))
    ]

    if len(reference_ids) == 0:
        raise ValueError(f"No valid reference arcs in trip {trip_id}")

    ref_labels = []

    for i in range(len(track)):
        distances = np.array([
            point_to_segment_distance(
                gps_x[i], gps_y[i],
                ax[r], ay[r], bx[r], by[r]
            )
            for r in reference_ids
        ])

        ref_labels.append(reference_ids[np.argmin(distances)])

    # CANDIDATE GENERATION
    records = []
    search_k = min(SEARCH_K, len(arcs))
    candidate_k = min(K, len(arcs))

    for i in range(len(track)):

        _, nearby_ids = tree.query(
            [gps_x[i], gps_y[i]], k=search_k
        )

        nearby_ids = np.atleast_1d(nearby_ids).astype(int)

        distances = np.array([
            point_to_segment_distance(
                gps_x[i], gps_y[i],
                ax[r], ay[r], bx[r], by[r]
            )
            for r in nearby_ids
        ])

        order = np.argsort(distances)[:candidate_k]

        for position in order:
            arc_id = nearby_ids[position]
            distance = distances[position]

            road_heading = bearing(
                lat_a[arc_id], lon_a[arc_id],
                lat_b[arc_id], lon_b[arc_id]
            )

            heading_diff = abs(headings[i] - road_heading)
            heading_diff = min(heading_diff, 360 - heading_diff)

            records.append({
                "trip_id": trip_id,
                "gps_point_id": i,
                "candidate_road_id": int(arc_id),
                "distance_to_road": float(distance),
                "heading_difference": float(heading_diff),
                "speed": float(speeds[i]),
                "time_difference": float(time_gaps[i]),
                "label": int(arc_id == ref_labels[i])
            })

    candidates = pd.DataFrame(records)

    # Provisional candidate recall
    candidate_sets = (
        candidates.groupby("gps_point_id")["candidate_road_id"]
        .apply(set)
    )

    recall = np.mean([
        ref_labels[i] in candidate_sets.get(i, set())
        for i in range(len(track))
    ])

    print(
        f"{trip_id}: GPS={len(track)}, "
        f"candidates={len(candidates)}, "
        f"provisional recall={recall:.4f}"
    )

    return candidates, recall


# PROCESS TRAINING TRIPS
def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trip_folders = sorted([
        folder for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    ])

    if MAX_TRIPS is not None:
        trip_folders = trip_folders[:MAX_TRIPS]

    all_candidates = []
    summaries = []

    for folder in trip_folders:

        trip_id = folder.name

        try:
            candidates, recall = generate_candidates(trip_id)

            all_candidates.append(candidates)

            summaries.append({
                "trip_id": trip_id,
                "gps_observations": candidates["gps_point_id"].nunique(),
                "candidate_rows": len(candidates),
                "provisional_recall": recall
            })

        except Exception as error:
            print(f"ERROR processing {trip_id}: {error}")

    if not all_candidates:
        print("No candidate datasets were generated.")
        return

    combined = pd.concat(all_candidates, ignore_index=True)

    combined_file = OUTPUT_DIR / "all_train_candidates.csv"
    summary_file = OUTPUT_DIR / "candidate_generation_summary.csv"

    combined.to_csv(combined_file, index=False)
    pd.DataFrame(summaries).to_csv(summary_file, index=False)

    print("\nCANDIDATE GENERATION FINISHED")
    print("Trips processed:", len(all_candidates))
    print("Total candidate rows:", len(combined))
    print("Combined dataset:", combined_file)
    print("Summary:", summary_file)


if __name__ == "__main__":
    main()
