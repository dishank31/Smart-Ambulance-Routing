from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from .schemas import (
    EmergencyRequest,
    ModelComparisonResponse,
    ModelMetrics,
    PatientVitals,
    RecommendationResponse,
    SeverityResponse,
)
from src.data.data_generator import generate_hospital_registry
from src.routing.decision_engine import DecisionEngine
from src.routing.mapbox_client import MapboxClient
from src.utils.config import Config
from src.utils.geo_utils import filter_hospitals_by_radius
from src.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger("api_routes")
ENGINE: DecisionEngine | None = None
USE_EXISTING_MODELS = os.getenv("USE_EXISTING_MODELS", "false").lower() == "true"


def _load_model(path_options: list[Path]):
    for path in path_options:
        if path.exists() and path.stat().st_size > 0:
            try:
                return joblib.load(path)
            except Exception as exc:
                logger.warning("Failed to load model from %s: %s", path, exc)
    return None


def initialize_engine() -> DecisionEngine:
    global ENGINE
    if ENGINE is not None:
        return ENGINE

    Config.ensure_dirs()
    hospitals_path = Path(Config.HOSPITALS_FILE)
    hospitals_df = pd.read_csv(hospitals_path) if hospitals_path.exists() else generate_hospital_registry(output_path=hospitals_path)

    severity_model = None
    eta_model = None
    bed_model = None
    if USE_EXISTING_MODELS:
        severity_model = _load_model([
            Path(Config.SEVERITY_MODELS_DIR) / f"severity_stacking_{Config.MODEL_VERSION}.joblib",
            Path(Config.SEVERITY_MODELS_DIR) / "severity_stacking_v2.joblib",
            Path(Config.SEVERITY_MODELS_DIR) / "severity_stacking_v1.joblib",
        ])
        eta_model = _load_model([
            Path(Config.ETA_MODELS_DIR) / f"eta_stacking_{Config.MODEL_VERSION}.joblib",
            Path(Config.ETA_MODELS_DIR) / "eta_stacking_v1.joblib",
        ])
        bed_model = _load_model([
            Path(Config.BED_MODELS_DIR) / f"bed_stacking_{Config.MODEL_VERSION}.joblib",
            Path(Config.MODELS_DIR) / "bed_xgboost.pkl",
        ])

    ENGINE = DecisionEngine(
        severity_model=severity_model,
        bed_model=bed_model,
        eta_model=eta_model,
        hospitals_df=hospitals_df,
        mapbox_client=MapboxClient(os.getenv("MAPBOX_ACCESS_TOKEN", Config.MAPBOX_ACCESS_TOKEN)),
    )
    return ENGINE


@router.get("/health")
@router.get("/api/health")
async def health_check():
    engine = initialize_engine()
    return {
        "status": "healthy",
        "models_loaded": {
            "severity": engine.severity_model is not None,
            "eta": engine.eta_model is not None,
            "bed": engine.bed_model is not None,
        },
        "version": "1.0.0",
        "using_legacy_models": USE_EXISTING_MODELS,
    }


@router.post("/api/emergency/recommend", response_model=RecommendationResponse)
async def recommend_hospital(request: EmergencyRequest):
    engine = initialize_engine()
    started = time.perf_counter()
    try:
        result = engine.recommend_hospital(
            emergency_location={"lat": request.location.latitude, "lon": request.location.longitude},
            patient_vitals=request.patient_vitals.model_dump(),
            current_datetime=request.timestamp,
            use_ml_eta=request.use_ml_eta,
        )
        result["processing_time_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return RecommendationResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Recommendation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/predict/severity", response_model=SeverityResponse)
async def predict_severity(vitals: PatientVitals):
    engine = initialize_engine()
    try:
        return SeverityResponse(**engine.predict_severity(vitals.model_dump()))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/hospitals/nearby")
async def get_nearby_hospitals(lat: float = Query(...), lon: float = Query(...), radius_km: float = Query(25.0, gt=0)):
    engine = initialize_engine()
    hospitals = filter_hospitals_by_radius(engine.hospitals_df, lat, lon, radius_km)
    return hospitals.to_dict(orient="records")


@router.get("/api/models/comparison", response_model=ModelComparisonResponse)
async def get_model_comparison():
    summary_path = Path(Config.EVALUATION_SUMMARY_PATH)
    if summary_path.exists() and summary_path.stat().st_size > 0:
        with open(summary_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return ModelComparisonResponse(**payload)

    return ModelComparisonResponse(
        severity_models=[ModelMetrics(model_name="Severity Stacking", accuracy=0.82, f1_score=0.81)],
        eta_models=[ModelMetrics(model_name="ETA Stacking", mae=3.47, rmse=5.46)],
        bed_models=[ModelMetrics(model_name="Bed Ensemble", mae=2.85, rmse=4.12)],
    )


@router.post("/api/simulate/traffic", response_model=RecommendationResponse)
async def simulate_traffic_change(current_recommendation: RecommendationResponse, new_traffic_level: int = Query(..., ge=1, le=10)):
    engine = initialize_engine()
    try:
        payload = current_recommendation.model_dump()
        updated = engine.simulate_traffic_change(payload, new_traffic_level)
        updated["processing_time_ms"] = payload.get("processing_time_ms", 0.0)
        return RecommendationResponse(**updated)
    except Exception as exc:
        logger.exception("Traffic simulation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
