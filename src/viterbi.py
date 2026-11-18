"""Log-space Viterbi decoding for variable-size candidate layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ViterbiResult:
    """Decoded candidate-state indexes and their total log score."""

    state_indices: tuple[int, ...]
    log_score: float


def viterbi_decode(
    emission_log_scores: Sequence[Sequence[float]],
    transition_log_scores: Sequence[Sequence[Sequence[float]]],
    initial_log_scores: Sequence[float] | None = None,
    disconnected_fallback_log_score: float = -30.0,
) -> ViterbiResult:
    """Return the highest-scoring state sequence.

    ``emission_log_scores[t]`` has one value per candidate at observation t.
    ``transition_log_scores[t]`` has shape (states at t, states at t + 1).
    Unreachable transitions may use negative infinity; if an entire next layer
    is unreachable, the decoder applies the configured fallback penalty so a
    single disconnected segment does not discard the rest of the trip.
    """
    if not emission_log_scores:
        raise ValueError("At least one emission layer is required.")

    emissions = [np.asarray(layer, dtype=float) for layer in emission_log_scores]
    if any(layer.ndim != 1 or layer.size == 0 for layer in emissions):
        raise ValueError("Every observation must have at least one candidate.")
    if any(np.isnan(layer).any() or np.isposinf(layer).any() for layer in emissions):
        raise ValueError("Emission scores may be finite or -inf, not NaN/+inf.")
    if len(transition_log_scores) != len(emissions) - 1:
        raise ValueError("There must be one transition matrix between each pair of layers.")

    if initial_log_scores is None:
        scores = np.full(emissions[0].size, -np.log(emissions[0].size))
    else:
        scores = np.asarray(initial_log_scores, dtype=float)
        if scores.shape != emissions[0].shape:
            raise ValueError("Initial scores must match the first candidate layer.")
        if np.isnan(scores).any() or np.isposinf(scores).any():
            raise ValueError("Initial scores may be finite or -inf, not NaN/+inf.")
    scores = scores + emissions[0]

    backpointers: list[np.ndarray] = []
    for time_index, raw_transition in enumerate(transition_log_scores):
        transition = np.asarray(raw_transition, dtype=float)
        expected_shape = (emissions[time_index].size, emissions[time_index + 1].size)
        if transition.shape != expected_shape:
            raise ValueError(
                f"Transition matrix {time_index} has shape {transition.shape}; "
                f"expected {expected_shape}."
            )
        if np.isnan(transition).any() or np.isposinf(transition).any():
            raise ValueError("Transition scores may be finite or -inf, not NaN/+inf.")

        accumulated = scores[:, None] + transition
        best_previous = np.argmax(accumulated, axis=0)
        best_scores = accumulated[
            best_previous, np.arange(emissions[time_index + 1].size)
        ]

        unreachable = ~np.isfinite(best_scores)
        if unreachable.any():
            fallback_previous = int(np.argmax(scores))
            best_previous[unreachable] = fallback_previous
            best_scores[unreachable] = (
                scores[fallback_previous] + disconnected_fallback_log_score
            )

        backpointers.append(best_previous)
        scores = best_scores + emissions[time_index + 1]

    final_state = int(np.argmax(scores))
    final_score = float(scores[final_state])
    if not np.isfinite(final_score):
        raise ValueError("No finite Viterbi path could be constructed.")

    path = [final_state]
    for pointer in reversed(backpointers):
        path.append(int(pointer[path[-1]]))
    path.reverse()
    return ViterbiResult(tuple(path), final_score)
