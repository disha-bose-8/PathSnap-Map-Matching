from pathlib import Path
import sys
import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)

from data_loader import load_trip

ROOT = Path(__file__).resolve().parents[1]

train_dir = ROOT / "data" / "Student folder" / "train"

results = []

for folder in sorted(train_dir.iterdir()):

    if not folder.is_dir():
        continue

    trip_id = folder.name

    try:
        track, nodes, arcs, route = load_trip(
            "train", trip_id
        )

        results.append({

            "trip_id": trip_id,

            "gps_points": len(track),

            "road_nodes": len(nodes),

            "road_arcs": len(arcs),

            "route_length": len(route),

            "missing_gps": track.isna().sum().sum(),

            "missing_nodes": nodes.isna().sum().sum(),

            "missing_arcs": arcs.isna().sum().sum(),

            "missing_route": route.isna().sum().sum(),

            "avg_time_gap": (
                track["timestamp"].diff().dropna().mean()
            )

        })

    except Exception as e:

        print("Error in trip:", trip_id, e)


summary = pd.DataFrame(results)

print("\nDATASET SUMMARY")
print(summary.to_string(index=False))

print("\nOVERALL STATISTICS")
print(summary.describe())

output = ROOT / "results" / "dataset_summary.csv"

output.parent.mkdir(exist_ok=True)

summary.to_csv(output, index=False)

print("\nSaved:", output)