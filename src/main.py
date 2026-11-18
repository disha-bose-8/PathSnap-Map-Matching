"""FastAPI entry point for the two-laptop PathSnap demonstration."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.hmm import decode_candidates
from src.inference import (
    generate_inference_candidates,
    load_reference_route,
    load_trip_inputs,
    nearest_reference_arc_labels,
)
from src.road_graph import RoadGraph, route_continuity


ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("pathsnap.api")
TRIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")


def _configured_path(environment_name: str, default: Path) -> Path:
    configured = os.getenv(environment_name)
    if not configured:
        return default.resolve()
    path = Path(configured).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    data_root: Path
    partition: str
    model_path: Path

    @classmethod
    def from_environment(cls) -> "Settings":
        partition = os.getenv("PATHSNAP_EVALUATION_PARTITION", "student_test")
        if not partition or Path(partition).name != partition:
            raise ValueError("PATHSNAP_EVALUATION_PARTITION must be one folder name.")
        return cls(
            data_root=_configured_path(
                "PATHSNAP_DATA_ROOT", ROOT / "data" / "Student folder"
            ),
            partition=partition,
            model_path=_configured_path(
                "PATHSNAP_MODEL_PATH",
                ROOT / "models" / "candidate_classifier.joblib",
            ),
        )


class PredictionRequest(BaseModel):
    """The evaluator interface selects a trip by ID; raw data stays server-side."""

    trip_id: str = Field(min_length=1, max_length=32)


def _evaluate_reference(
    trip_dir: Path,
    trip_id: str,
    track: pd.DataFrame,
    graph: RoadGraph,
    predicted_arc_ids: tuple[int, ...],
) -> dict[str, Any] | None:
    """Score against server-side reference data after generating predictions."""
    route_path = trip_dir / f"{trip_id}.route"
    if not route_path.is_file():
        return None
    reference_ids = load_reference_route(trip_dir, trip_id)
    reference_labels = nearest_reference_arc_labels(track, reference_ids, graph)
    accuracy = float(np.mean(np.asarray(predicted_arc_ids) == reference_labels))
    return {
        "metric": "nearest_reference_arc_accuracy",
        "accuracy": accuracy,
        "observations": int(len(reference_labels)),
        "note": (
            "Spatial nearest-reference-arc proxy; use the evaluator's official "
            "route metric if it specifies a different alignment."
        ),
    }


def _make_response(
    trip_id: str,
    track: pd.DataFrame,
    graph: RoadGraph,
    decoded,
    evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    route_geometry = []
    for gps_index, arc_id in enumerate(decoded.arc_ids):
        start, end = graph.arc_endpoint_coordinates(int(arc_id))
        route_geometry.append({
            "gps_index": gps_index,
            "arc_id": int(arc_id),
            "coordinates": [list(start), list(end)],
        })
    return {
        "trip_id": trip_id,
        "gps_points": [
            [float(row.latitude), float(row.longitude)]
            for row in track.itertuples(index=False)
        ],
        "predicted_arc_ids": list(decoded.arc_ids),
        "classifier_only_arc_ids": list(decoded.classifier_only_arc_ids),
        "predicted_route": route_geometry,
        "route_continuity": route_continuity(graph, track, decoded.arc_ids),
        "fallback_observation_indices": list(
            decoded.fallback_observation_indices
        ),
        "evaluation": evaluation,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.settings = settings
        application.state.model = None
        application.state.model_error = None
        if settings.model_path.is_file():
            try:
                application.state.model = joblib.load(settings.model_path)
                LOGGER.info("Loaded candidate model from %s", settings.model_path)
            except Exception as error:  # Keep /health usable when an artifact is invalid.
                application.state.model_error = str(error)
                LOGGER.exception("Could not load the candidate model")
        else:
            application.state.model_error = "Model artifact not found."
            LOGGER.warning("Candidate model is missing at %s", settings.model_path)
        yield

    application = FastAPI(
        title="PathSnap Map Matching API",
        version="1.0.0",
        description="Trip-ID-based candidate classification and HMM/Viterbi decoding.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": application.state.model is not None,
            "model_error": application.state.model_error,
        }

    @application.get("/trips")
    def trips() -> dict[str, list[str]]:
        partition_dir = settings.data_root / settings.partition
        if not partition_dir.is_dir():
            raise HTTPException(status_code=503, detail="Trip data directory is unavailable.")
        trip_ids = sorted(
            folder.name
            for folder in partition_dir.iterdir()
            if folder.is_dir()
            and (folder / f"{folder.name}.track").is_file()
        )
        return {"trip_ids": trip_ids}

    def predict(request: PredictionRequest) -> dict[str, Any]:
        if not TRIP_ID_PATTERN.fullmatch(request.trip_id):
            raise HTTPException(status_code=422, detail="Invalid trip_id format.")
        if application.state.model is None:
            raise HTTPException(
                status_code=503,
                detail="Candidate model is not loaded; train or configure the model artifact.",
            )
        try:
            track, nodes, arcs, trip_dir = load_trip_inputs(
                settings.data_root, settings.partition, request.trip_id
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Trip was not found.") from None
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        try:
            graph = RoadGraph(nodes, arcs)
            candidates = generate_inference_candidates(
                request.trip_id, track, nodes, arcs
            )
            decoded = decode_candidates(
                track, candidates, graph, model=application.state.model
            )
            evaluation = _evaluate_reference(
                trip_dir, request.trip_id, track, graph, decoded.arc_ids
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Trip reference data is unavailable.") from None
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        LOGGER.info(
            "trip_id=%s observations=%d fallback_points=%d",
            request.trip_id,
            len(track),
            len(decoded.fallback_observation_indices),
        )
        return _make_response(
            request.trip_id, track, graph, decoded, evaluation
        )

    application.post("/predict")(predict)
    application.post("/match", include_in_schema=False)(predict)
    return application


app = create_app()
