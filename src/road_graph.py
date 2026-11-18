"""Directed road-network helpers used by the sequence decoder.

Arc IDs are the zero-based row positions in the ``.arcs`` file. Node IDs are
the zero-based row positions in the ``.nodes`` file; preserving those IDs is
important because ``.route`` files refer to arc IDs directly.
"""
from __future__ import annotations

from heapq import heappop, heappush
from typing import Optional, Sequence

import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in metres between two coordinates."""
    lat1_r, lon1_r, lat2_r, lon2_r = np.radians(
        [lat1, lon1, lat2, lon2]
    )
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    value = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    return float(2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(value)))


class RoadGraph:
    """A directed graph built from one trip's ``nodes`` and ``arcs`` tables."""

    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame) -> None:
        required_nodes = {"longitude", "latitude"}
        required_arcs = {"from_node", "to_node"}
        if not required_nodes.issubset(nodes.columns):
            raise ValueError(
                f"nodes must contain columns {sorted(required_nodes)}"
            )
        if not required_arcs.issubset(arcs.columns):
            raise ValueError(
                f"arcs must contain columns {sorted(required_arcs)}"
            )
        if nodes.empty or arcs.empty:
            raise ValueError("A road graph requires at least one node and arc.")

        self._longitudes = pd.to_numeric(
            nodes["longitude"], errors="raise"
        ).to_numpy(dtype=float)
        self._latitudes = pd.to_numeric(
            nodes["latitude"], errors="raise"
        ).to_numpy(dtype=float)
        if not np.isfinite(self._longitudes).all() or not np.isfinite(
            self._latitudes
        ).all():
            raise ValueError("Road-node coordinates must be finite numbers.")

        endpoints = arcs[["from_node", "to_node"]].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=float)
        if not np.equal(endpoints, np.floor(endpoints)).all():
            raise ValueError("Arc endpoints must be integer node IDs.")
        self._from_nodes = endpoints[:, 0].astype(int)
        self._to_nodes = endpoints[:, 1].astype(int)
        if (
            self._from_nodes.min() < 0
            or self._to_nodes.min() < 0
            or self._from_nodes.max() >= len(nodes)
            or self._to_nodes.max() >= len(nodes)
        ):
            raise ValueError("An arc refers to a node outside the nodes table.")

        self.arc_ids = tuple(range(len(arcs)))
        self._reference_latitude = float(np.mean(self._latitudes))
        self._node_x, self._node_y = self._project(
            self._latitudes, self._longitudes
        )
        self._arc_lengths_m = np.asarray(
            [
                haversine_m(
                    self._latitudes[start], self._longitudes[start],
                    self._latitudes[end], self._longitudes[end],
                )
                for start, end in zip(self._from_nodes, self._to_nodes)
            ],
            dtype=float,
        )

        self._adjacency: dict[int, list[tuple[int, float]]] = {
            node_id: [] for node_id in range(len(nodes))
        }
        for start, end, length_m in zip(
            self._from_nodes, self._to_nodes, self._arc_lengths_m
        ):
            self._adjacency[int(start)].append((int(end), float(length_m)))

        self._node_distances: dict[int, dict[int, float]] = {}
        self._bounded_node_distances: dict[
            int, tuple[float, dict[int, float]]
        ] = {}

    @property
    def number_of_arcs(self) -> int:
        return len(self.arc_ids)

    def arc_length_m(self, arc_id: int) -> float:
        self._validate_arc_id(arc_id)
        return float(self._arc_lengths_m[arc_id])

    def arc_endpoint_coordinates(
        self, arc_id: int
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the arc's (start, end) coordinates as (latitude, longitude)."""
        self._validate_arc_id(arc_id)
        start = int(self._from_nodes[arc_id])
        end = int(self._to_nodes[arc_id])
        return (
            (float(self._latitudes[start]), float(self._longitudes[start])),
            (float(self._latitudes[end]), float(self._longitudes[end])),
        )

    def project_point_to_arc(
        self, longitude: float, latitude: float, arc_id: int
    ) -> tuple[float, float]:
        """Return (fraction along arc, perpendicular distance in metres)."""
        self._validate_arc_id(arc_id)
        point_x, point_y = self._project(
            np.asarray([latitude], dtype=float),
            np.asarray([longitude], dtype=float),
        )
        start = int(self._from_nodes[arc_id])
        end = int(self._to_nodes[arc_id])
        ax, ay = float(self._node_x[start]), float(self._node_y[start])
        bx, by = float(self._node_x[end]), float(self._node_y[end])
        dx, dy = bx - ax, by - ay
        denominator = dx * dx + dy * dy
        if denominator <= 0.0:
            fraction = 0.0
        else:
            fraction = float(
                np.clip(
                    ((float(point_x[0]) - ax) * dx
                     + (float(point_y[0]) - ay) * dy)
                    / denominator,
                    0.0,
                    1.0,
                )
            )
        nearest_x = ax + fraction * dx
        nearest_y = ay + fraction * dy
        distance_m = float(
            np.hypot(float(point_x[0]) - nearest_x,
                     float(point_y[0]) - nearest_y)
        )
        return fraction, distance_m

    def nearest_arc(
        self, longitude: float, latitude: float
    ) -> tuple[int, float, float]:
        """Return nearest (arc ID, distance in metres, fraction along arc)."""
        best_arc = -1
        best_distance = float("inf")
        best_fraction = 0.0
        for arc_id in self.arc_ids:
            fraction, distance = self.project_point_to_arc(
                longitude, latitude, arc_id
            )
            if distance < best_distance:
                best_arc = arc_id
                best_distance = distance
                best_fraction = fraction
        return best_arc, best_distance, best_fraction

    def shortest_node_distance_m(
        self,
        source_node: int,
        target_node: int,
        max_distance_m: Optional[float] = None,
    ) -> Optional[float]:
        """Return directed shortest-path distance, or None if disconnected."""
        if source_node not in self._adjacency or target_node not in self._adjacency:
            raise ValueError("Unknown node ID in shortest-path query.")
        if source_node == target_node:
            return 0.0
        if max_distance_m is None:
            if source_node not in self._node_distances:
                self._node_distances[source_node] = self._dijkstra(source_node)
            return self._node_distances[source_node].get(target_node)
        if not np.isfinite(max_distance_m) or max_distance_m < 0.0:
            raise ValueError("max_distance_m must be a finite non-negative value.")
        if source_node in self._node_distances:
            distance = self._node_distances[source_node].get(target_node)
            return distance if distance is not None and distance <= max_distance_m else None

        cached = self._bounded_node_distances.get(source_node)
        if cached is not None and cached[0] >= max_distance_m:
            distance = cached[1].get(target_node)
            return distance if distance is not None and distance <= max_distance_m else None

        distances = self._dijkstra(source_node, max_distance_m)
        if cached is None or max_distance_m > cached[0]:
            self._bounded_node_distances[source_node] = (max_distance_m, distances)
        distance = distances.get(target_node)
        return distance if distance is not None and distance <= max_distance_m else None

    def transition_distance_m(
        self,
        previous_arc_id: int,
        previous_fraction: float,
        next_arc_id: int,
        next_fraction: float,
        max_distance_m: Optional[float] = None,
    ) -> Optional[float]:
        """Estimate directed network distance between two projected points."""
        self._validate_arc_id(previous_arc_id)
        self._validate_arc_id(next_arc_id)
        previous_fraction = float(np.clip(previous_fraction, 0.0, 1.0))
        next_fraction = float(np.clip(next_fraction, 0.0, 1.0))

        previous_length = self.arc_length_m(previous_arc_id)
        if previous_arc_id == next_arc_id and next_fraction >= previous_fraction:
            direct_distance = (next_fraction - previous_fraction) * previous_length
            if max_distance_m is not None and direct_distance > max_distance_m:
                return None
            return direct_distance

        previous_end = int(self._to_nodes[previous_arc_id])
        next_start = int(self._from_nodes[next_arc_id])
        remaining_previous = (1.0 - previous_fraction) * previous_length
        partial_next = next_fraction * self.arc_length_m(next_arc_id)
        middle_limit = None
        if max_distance_m is not None:
            middle_limit = max_distance_m - remaining_previous - partial_next
            if middle_limit < 0.0:
                return None
        middle_distance = self.shortest_node_distance_m(
            previous_end, next_start, max_distance_m=middle_limit
        )
        if middle_distance is None:
            return None

        return remaining_previous + middle_distance + partial_next

    def _dijkstra(
        self, source_node: int, max_distance_m: Optional[float] = None
    ) -> dict[int, float]:
        distances = {source_node: 0.0}
        queue: list[tuple[float, int]] = [(0.0, source_node)]
        while queue:
            distance, node_id = heappop(queue)
            if distance != distances.get(node_id):
                continue
            if max_distance_m is not None and distance > max_distance_m:
                break
            for next_node, edge_length in self._adjacency[node_id]:
                new_distance = distance + edge_length
                if max_distance_m is not None and new_distance > max_distance_m:
                    continue
                if new_distance < distances.get(next_node, float("inf")):
                    distances[next_node] = new_distance
                    heappush(queue, (new_distance, next_node))
        return distances

    def _project(
        self, latitudes: np.ndarray, longitudes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x = longitudes * 111_320.0 * np.cos(
            np.radians(self._reference_latitude)
        )
        y = latitudes * 110_540.0
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def _validate_arc_id(self, arc_id: int) -> None:
        if not isinstance(arc_id, (int, np.integer)):
            raise ValueError(f"Arc ID must be an integer, got {arc_id!r}.")
        if arc_id < 0 or arc_id >= self.number_of_arcs:
            raise ValueError(f"Arc ID {arc_id} is outside this trip's network.")


def route_continuity(
    graph: RoadGraph, track: pd.DataFrame, arc_ids: Sequence[int]
) -> float:
    """Fraction of consecutive projected arcs connected in the directed graph.

    Project each observation once and use NumPy arrays in the transition loop.
    This avoids duplicate projections and repeated pandas row lookups.
    """
    arc_array = np.asarray(arc_ids, dtype=int)
    if arc_array.ndim != 1:
        raise ValueError("arc_ids must be a one-dimensional sequence.")
    if len(arc_array) <= 1:
        return 1.0
    if len(track) != len(arc_array):
        raise ValueError(
            "track and arc_ids must contain the same number of observations."
        )
    if (arc_array < 0).any() or (arc_array >= graph.number_of_arcs).any():
        raise ValueError("arc_ids contains an ID outside this road network.")

    longitudes = pd.to_numeric(track["longitude"], errors="raise").to_numpy(dtype=float)
    latitudes = pd.to_numeric(track["latitude"], errors="raise").to_numpy(dtype=float)
    timestamps = pd.to_numeric(track["timestamp"], errors="raise").to_numpy(dtype=float)
    if not all(np.isfinite(values).all() for values in (longitudes, latitudes, timestamps)):
        raise ValueError("Track coordinates and timestamps must be finite.")

    fractions = np.empty(len(arc_array), dtype=float)
    for index, arc_id in enumerate(arc_array):
        fractions[index], _ = graph.project_point_to_arc(
            float(longitudes[index]), float(latitudes[index]), int(arc_id)
        )

    connected = 0
    for index in range(len(arc_array) - 1):
        elapsed = abs(float(timestamps[index + 1] - timestamps[index]))
        max_distance = 55.0 * max(elapsed, 1.0) + 100.0
        if graph.transition_distance_m(
            int(arc_array[index]), float(fractions[index]),
            int(arc_array[index + 1]), float(fractions[index + 1]),
            max_distance_m=max_distance,
        ) is not None:
            connected += 1
    return connected / (len(arc_array) - 1)
