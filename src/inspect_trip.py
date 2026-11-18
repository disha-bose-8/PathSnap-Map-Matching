import sys
from pathlib import Path

import pandas as pd
import folium

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)

from data_loader import load_trip


TRIP_ID = "00000001"

# Load data
track, nodes, arcs, route = load_trip(
    "train", TRIP_ID
)


# --------------------------------
# 1. BASIC DATASET INSPECTION
# --------------------------------

datasets = {
    "GPS TRACK": track,
    "ROAD NODES": nodes,
    "ROAD ARCS": arcs,
    "REFERENCE ROUTE": route
}

for name, df in datasets.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(df.head())

    print("\nShape:", df.shape)

    print(
        "Missing values:",
        df.isna().sum().sum()
    )


# --------------------------------
# 2. TIMESTAMP ANALYSIS
# --------------------------------

time_gaps = track["timestamp"].diff().dropna()

print("\nTIMESTAMP GAP DISTRIBUTION")

print(time_gaps.value_counts().head(10))

print("\nGPS COORDINATE STATISTICS")

print(
    track[
        ["longitude", "latitude"]
    ].describe()
)


# --------------------------------
# 3. VALIDATE ROAD NETWORK
# --------------------------------

valid_nodes = (
    arcs["from_node"].between(
        0, len(nodes) - 1
    )
    &
    arcs["to_node"].between(
        0, len(nodes) - 1
    )
)

valid_route = route["arc_id"].between(
    0, len(arcs) - 1
)

print("\nNETWORK VALIDATION")

print(
    "Invalid arc endpoints:",
    (~valid_nodes).sum()
)

print(
    "Invalid route arc IDs:",
    (~valid_route).sum()
)


if not valid_nodes.all():
    raise ValueError("Invalid node references!")

if not valid_route.all():
    raise ValueError("Invalid route arc IDs!")


# --------------------------------
# 4. CREATE FOLIUM MAP
# --------------------------------

center = [
    track["latitude"].mean(),
    track["longitude"].mean()
]

m = folium.Map(
    location=center,
    zoom_start=13
)


# GPS trajectory

gps_coords = track[
    ["latitude", "longitude"]
].values.tolist()

folium.PolyLine(
    gps_coords,
    color="blue",
    weight=3,
    tooltip="GPS trajectory"
).add_to(m)


# GPS observation markers

gps_layer = folium.FeatureGroup(
    name="GPS observations"
)

for i in range(0, len(gps_coords), 50):

    folium.CircleMarker(
        gps_coords[i],
        radius=3,
        color="blue",
        fill=True,
        popup=f"GPS point {i}"
    ).add_to(gps_layer)

gps_layer.add_to(m)


# Reference route

route_layer = folium.FeatureGroup(
    name="Reference route"
)

for arc_id in route["arc_id"].astype(int):

    edge = arcs.iloc[arc_id]

    start = nodes.iloc[
        int(edge["from_node"])
    ]

    end = nodes.iloc[
        int(edge["to_node"])
    ]

    folium.PolyLine(
        [
            [
                start["latitude"],
                start["longitude"]
            ],
            [
                end["latitude"],
                end["longitude"]
            ]
        ],
        color="green",
        weight=4
    ).add_to(route_layer)


route_layer.add_to(m)

folium.LayerControl().add_to(m)


# Save map

output = (
    Path(__file__).resolve().parents[1]
    / "results"
    / f"{TRIP_ID}_map.html"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True
)

m.save(str(output))

print("\nMAP SAVED SUCCESSFULLY:")
print(output)

print("\nDATA INSPECTION COMPLETED!")