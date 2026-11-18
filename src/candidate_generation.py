import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)

from data_loader import load_trip


# --------------------------------
# CONFIGURATION
# --------------------------------

TRIP_ID = "00000001"

K = 8
SEARCH_K = 20

EARTH_RADIUS = 6371000


# --------------------------------
# GEOGRAPHICAL FUNCTIONS
# --------------------------------

def project_coordinates(lat, lon, ref_lat):

    x = lon * 111320 * np.cos(
        np.radians(ref_lat)
    )

    y = lat * 110540

    return x, y


def bearing(lat1, lon1, lat2, lon2):

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlon = np.radians(lon2 - lon1)

    y = np.sin(dlon) * np.cos(lat2)

    x = (
        np.cos(lat1) * np.sin(lat2)
        -
        np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    return (
        np.degrees(np.arctan2(y, x)) + 360
    ) % 360


def haversine(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS * np.arcsin(
        np.sqrt(a)
    )


# --------------------------------
# POINT TO ROAD DISTANCE
# --------------------------------

def point_to_segment_distance(px, py, ax, ay, bx, by):

    dx = bx - ax
    dy = by - ay

    denominator = dx * dx + dy * dy

    if denominator == 0:
        return np.hypot(px - ax, py - ay)

    t = (
        (px - ax) * dx
        +
        (py - ay) * dy
    ) / denominator

    t = np.clip(t, 0, 1)

    nearest_x = ax + t * dx
    nearest_y = ay + t * dy

    return np.hypot(
        px - nearest_x,
        py - nearest_y
    )


# --------------------------------
# LOAD DATA
# --------------------------------

track, nodes, arcs, route = load_trip(
    "train", TRIP_ID
)

# Node IDs correspond to row indices.
edge_start = arcs["from_node"].to_numpy(dtype=int)
edge_end = arcs["to_node"].to_numpy(dtype=int)

lat_a = nodes.iloc[edge_start]["latitude"].to_numpy()
lon_a = nodes.iloc[edge_start]["longitude"].to_numpy()

lat_b = nodes.iloc[edge_end]["latitude"].to_numpy()
lon_b = nodes.iloc[edge_end]["longitude"].to_numpy()

ref_lat = track["latitude"].mean()


# Project road endpoints

ax, ay = project_coordinates(
    lat_a, lon_a, ref_lat
)

bx, by = project_coordinates(
    lat_b, lon_b, ref_lat
)

mid_x = (ax + bx) / 2
mid_y = (ay + by) / 2


# Build spatial index

midpoints = np.column_stack(
    [mid_x, mid_y]
)

tree = cKDTree(midpoints)

number_of_arcs = len(arcs)

search_k = min(SEARCH_K, number_of_arcs)
candidate_k = min(K, number_of_arcs)


# --------------------------------
# GPS FEATURES
# --------------------------------

gps_x, gps_y = project_coordinates(
    track["latitude"].to_numpy(),
    track["longitude"].to_numpy(),
    ref_lat
)

timestamps = track["timestamp"].to_numpy()

gps_lat = track["latitude"].to_numpy()
gps_lon = track["longitude"].to_numpy()

time_gaps = np.diff(timestamps, prepend=timestamps[0])

speeds = np.zeros(len(track))

headings = np.zeros(len(track))

for i in range(len(track)):

    if i < len(track) - 1:
        j = i + 1
    else:
        j = max(0, i - 1)

    dt = abs(timestamps[j] - timestamps[i])

    if dt > 0:

        distance = haversine(
            gps_lat[i], gps_lon[i],
            gps_lat[j], gps_lon[j]
        )

        speeds[i] = distance / dt

    headings[i] = bearing(
        gps_lat[i], gps_lon[i],
        gps_lat[j], gps_lon[j]
    )


# --------------------------------
# REFERENCE LABELS
# --------------------------------

# These labels use the nearest reference-route
# arc spatially. They are provisional heuristic
# labels, not guaranteed temporal ground truth.

reference_ids = np.unique(
    route["arc_id"].to_numpy(dtype=int)
)

reference_ids = reference_ids[
    (reference_ids >= 0)
    &
    (reference_ids < number_of_arcs)
]

ref_labels = []

for i in range(len(track)):

    distances = np.array([
        point_to_segment_distance(
            gps_x[i], gps_y[i],
            ax[r], ay[r],
            bx[r], by[r]
        )
        for r in reference_ids
    ])

    nearest_reference_arc = reference_ids[
        np.argmin(distances)
    ]

    ref_labels.append(nearest_reference_arc)


# --------------------------------
# CANDIDATE GENERATION
# --------------------------------

records = []

for i in range(len(track)):

    # Find nearby road-segment midpoints
    _, nearby_ids = tree.query(
        [gps_x[i], gps_y[i]],
        k=search_k
    )

    nearby_ids = np.atleast_1d(
        nearby_ids
    ).astype(int)

    # Calculate exact point-to-segment distances
    distances = []

    for arc_id in nearby_ids:

        d = point_to_segment_distance(
            gps_x[i], gps_y[i],
            ax[arc_id], ay[arc_id],
            bx[arc_id], by[arc_id]
        )

        distances.append(d)

    distances = np.array(distances)

    # Select K closest candidates
    order = np.argsort(distances)[:candidate_k]

    selected_ids = nearby_ids[order]
    selected_distances = distances[order]

    for arc_id, distance in zip(
        selected_ids,
        selected_distances
    ):

        road_heading = bearing(
            lat_a[arc_id],
            lon_a[arc_id],
            lat_b[arc_id],
            lon_b[arc_id]
        )

        heading_difference = abs(
            headings[i] - road_heading
        )

        heading_difference = min(
            heading_difference,
            360 - heading_difference
        )

        records.append({

            "trip_id": TRIP_ID,

            "gps_index": i,

            "candidate_arc_id": int(arc_id),

            "distance": float(distance),

            "heading_diff": float(
                heading_difference
            ),

            "speed": float(speeds[i]),

            "time_gap": float(time_gaps[i]),

            "label": int(
                arc_id == ref_labels[i]
            )

        })


# --------------------------------
# SAVE OUTPUT
# --------------------------------

candidates = pd.DataFrame(records)

output_dir = (
    Path(__file__).resolve().parents[1]
    / "results"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    output_dir / f"{TRIP_ID}_candidates.csv"
)

candidates.to_csv(
    output_file,
    index=False
)


# --------------------------------
# SUMMARY
# --------------------------------

candidate_recall = np.mean([
    ref_labels[i] in set(
        candidates.loc[
            candidates["gps_index"] == i,
            "candidate_arc_id"
        ]
    )
    for i in range(len(track))
])

print("\nCANDIDATE GENERATION COMPLETE")

print("GPS observations:", len(track))

print("Total candidates:", len(candidates))

print("Features:")
print(candidates.head())

print(
    "Provisional candidate recall:",
    round(candidate_recall, 4)
)

print("Saved to:", output_file)