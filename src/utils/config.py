"""Central configuration for the Smart Ambulance project."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DATA_DIR = DATASETS_DIR / "raw"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
SYNTHETIC_DATA_DIR = DATASETS_DIR / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"
SEVERITY_MODELS_DIR = MODELS_DIR / "severity"
ETA_MODELS_DIR = MODELS_DIR / "eta"
BED_MODELS_DIR = MODELS_DIR / "bed_availability"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
LOGS_DIR = PROJECT_ROOT / "logs"

SEVERITY_DATA_FILE = SYNTHETIC_DATA_DIR / "severity_data.csv"
BED_DATA_FILE = SYNTHETIC_DATA_DIR / "bed_availability.csv"
ETA_DATA_FILE = SYNTHETIC_DATA_DIR / "eta_data.csv"
HOSPITALS_FILE = SYNTHETIC_DATA_DIR / "hospitals.csv"
MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"
EVALUATION_SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.json"

SEVERITY_TARGET = "severity"
ETA_TARGET = "trip_duration_minutes"
BED_TARGET = "available_beds"

SEVERITY_FEATURES = ["heart_rate", "bp_systolic", "bp_diastolic", "spo2", "respiratory_rate", "temperature", "gcs_score", "pain_scale", "age", "gender", "has_chronic_condition", "chief_complaint"]
ETA_FEATURES = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude", "distance_km", "hour", "day_of_week", "month", "is_weekend", "is_rush_hour", "traffic_level", "temperature"]
BED_FEATURES = ["hospital_id", "department", "total_beds", "hour", "day_of_week", "month", "is_weekend", "is_holiday", "occupancy_rate", "admissions_last_1h", "discharges_last_1h", "admissions_rolling_6h", "discharges_rolling_6h", "occupancy_rolling_avg_24h"]

SEVERITY_LABELS = {1: "Critical", 2: "Emergent", 3: "Urgent", 4: "Less Urgent", 5: "Non-urgent"}
SEVERITY_DEPARTMENT_MAP = {1: "ICU", 2: "ICU", 3: "Emergency", 4: "General", 5: "General"}
RF_CLASSIFIER_PARAMS = {"n_estimators": 300, "max_depth": 14, "min_samples_split": 4, "min_samples_leaf": 2, "class_weight": "balanced_subsample", "random_state": 42, "n_jobs": -1}
XGB_CLASSIFIER_PARAMS = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "eval_metric": "mlogloss", "random_state": 42, "n_jobs": -1}
LGB_CLASSIFIER_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.85, "colsample_bytree": 0.85, "class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1}
ADABOOST_PARAMS = {"n_estimators": 150, "learning_rate": 0.05, "random_state": 42}
RF_REGRESSOR_PARAMS = {"n_estimators": 300, "max_depth": 16, "min_samples_split": 4, "random_state": 42, "n_jobs": -1}
XGB_REGRESSOR_PARAMS = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "random_state": 42, "n_jobs": -1}
LGB_REGRESSOR_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.85, "colsample_bytree": 0.85, "random_state": 42, "n_jobs": -1, "verbose": -1}
GBR_PARAMS = {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 4, "random_state": 42}
DECISION_WEIGHTS = {1: {"time": 0.8, "beds": 0.2}, 2: {"time": 0.8, "beds": 0.2}, 3: {"time": 0.6, "beds": 0.4}, 4: {"time": 0.4, "beds": 0.6}, 5: {"time": 0.4, "beds": 0.6}}
DEFAULT_DATABASE_URL = "sqlite:///./smart_ambulance.db"
DEFAULT_MAPBOX_PROFILE = "mapbox/driving"
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
CV_FOLDS = 5
HOSPITAL_SEARCH_RADIUS_KM = 25.0
MODEL_VERSION = _env("MODEL_VERSION", "v1")
DEBUG = _env("DEBUG", "false").lower() == "true"
DATABASE_URL = _env("DATABASE_URL", DEFAULT_DATABASE_URL)
MAPBOX_ACCESS_TOKEN = _env("MAPBOX_ACCESS_TOKEN", "")
OPENROUTESERVICE_API_KEY = _env("OPENROUTESERVICE_API_KEY", "")
LOCAL_OSRM_URL = _env("LOCAL_OSRM_URL", "http://127.0.0.1:5000/route/v1")


class Config:
    PROJECT_ROOT = PROJECT_ROOT
    DATASETS_DIR = DATASETS_DIR
    RAW_DATA_DIR = RAW_DATA_DIR
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
    SYNTHETIC_DATA_DIR = SYNTHETIC_DATA_DIR
    MODELS_DIR = MODELS_DIR
    SEVERITY_MODELS_DIR = SEVERITY_MODELS_DIR
    ETA_MODELS_DIR = ETA_MODELS_DIR
    BED_MODELS_DIR = BED_MODELS_DIR
    RESULTS_DIR = RESULTS_DIR
    DOCS_DIR = DOCS_DIR
    LOGS_DIR = LOGS_DIR
    SEVERITY_DATA_FILE = SEVERITY_DATA_FILE
    BED_DATA_FILE = BED_DATA_FILE
    ETA_DATA_FILE = ETA_DATA_FILE
    HOSPITALS_FILE = HOSPITALS_FILE
    MODEL_REGISTRY_PATH = MODEL_REGISTRY_PATH
    EVALUATION_SUMMARY_PATH = EVALUATION_SUMMARY_PATH
    DATABASE_URL = DATABASE_URL
    MAPBOX_ACCESS_TOKEN = MAPBOX_ACCESS_TOKEN
    OPENROUTESERVICE_API_KEY = OPENROUTESERVICE_API_KEY
    LOCAL_OSRM_URL = LOCAL_OSRM_URL
    MODEL_VERSION = MODEL_VERSION
    DEBUG = DEBUG
    DEFAULT_MAPBOX_PROFILE = DEFAULT_MAPBOX_PROFILE
    SEVERITY_TARGET = SEVERITY_TARGET
    ETA_TARGET = ETA_TARGET
    BED_TARGET = BED_TARGET
    SEVERITY_FEATURES = SEVERITY_FEATURES
    ETA_FEATURES = ETA_FEATURES
    BED_FEATURES = BED_FEATURES
    SEVERITY_LABELS = SEVERITY_LABELS
    SEVERITY_DEPARTMENT_MAP = SEVERITY_DEPARTMENT_MAP
    DECISION_WEIGHTS = DECISION_WEIGHTS
    RF_CLASSIFIER_PARAMS = RF_CLASSIFIER_PARAMS
    XGB_CLASSIFIER_PARAMS = XGB_CLASSIFIER_PARAMS
    LGB_CLASSIFIER_PARAMS = LGB_CLASSIFIER_PARAMS
    ADABOOST_PARAMS = ADABOOST_PARAMS
    RF_REGRESSOR_PARAMS = RF_REGRESSOR_PARAMS
    XGB_REGRESSOR_PARAMS = XGB_REGRESSOR_PARAMS
    LGB_REGRESSOR_PARAMS = LGB_REGRESSOR_PARAMS
    GBR_PARAMS = GBR_PARAMS
    TEST_SIZE = TEST_SIZE
    VAL_SIZE = VAL_SIZE
    RANDOM_STATE = RANDOM_STATE
    CV_FOLDS = CV_FOLDS
    HOSPITAL_SEARCH_RADIUS_KM = HOSPITAL_SEARCH_RADIUS_KM

    @classmethod
    def ensure_dirs(cls) -> None:
        for path in (cls.DATASETS_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.SYNTHETIC_DATA_DIR, cls.MODELS_DIR, cls.SEVERITY_MODELS_DIR, cls.ETA_MODELS_DIR, cls.BED_MODELS_DIR, cls.RESULTS_DIR, cls.DOCS_DIR, cls.LOGS_DIR):
            Path(path).mkdir(parents=True, exist_ok=True)
