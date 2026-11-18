from pathlib import Path
import pandas as pd

# Project directory
ROOT = Path(__file__).resolve().parents[1]

# Dataset directory
DATA = ROOT / "data" / "Student folder"


def load_trip(partition, trip_id):

    folder = DATA / partition / trip_id

    prefix = folder / trip_id

    # GPS observations
    track = pd.read_csv(
        str(prefix) + ".track",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    # Road network nodes
    nodes = pd.read_csv(
        str(prefix) + ".nodes",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    # Road network arcs
    arcs = pd.read_csv(
        str(prefix) + ".arcs",
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"]
    )

    # Reference route
    route = pd.read_csv(
        str(prefix) + ".route",
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )

    return track, nodes, arcs, route