"""
Configuration constants for the Smart Ambulance ML system.
Centralizes all paths, feature definitions, and hyperparameters.
"""

import os

# =============================================================================
# BASE PATHS
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dataset paths
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
RAW_DATA_DIR = os.path.join(DATASETS_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATASETS_DIR, "processed")
SYNTHETIC_DATA_DIR = os.path.join(DATASETS_DIR, "synthetic")
BED_DATA_DIR = os.path.join(DATASETS_DIR, "bed_availability")

# Model paths
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SEVERITY_MODELS_DIR = os.path.join(MODELS_DIR, "severity")
ETA_MODELS_DIR = os.path.join(MODELS_DIR, "eta")
BED_MODELS_DIR = os.path.join(MODELS_DIR, "bed_availability")

# Results path
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# =============================================================================
# DATASET FILES
# =============================================================================
SEVERITY_CLEANED_CSV = os.path.join(PROCESSED_DATA_DIR, "severity_cleaned.csv")
ETA_PROCESSED_CSV = os.path.join(PROCESSED_DATA_DIR, "eta_processed.csv")
BED_PROCESSED_CSV = os.path.join(PROCESSED_DATA_DIR, "bed_availability_processed.csv")
HOSPITAL_BEDS_CSV = os.path.join(BED_DATA_DIR, "hospital_beds.csv")
SYNTHETIC_TRIAGE_CSV = os.path.join(SYNTHETIC_DATA_DIR, "patient_triage_50k.csv")

# =============================================================================
# FEATURE DEFINITIONS (matched to actual dataset columns)
# =============================================================================

# Severity model — real dataset uses ESI target with 971 one-hot-encoded columns
# We use all non-target columns as features
SEVERITY_TARGET = "esi"

# Key severity features (from the real dataset)
SEVERITY_KEY_FEATURES = [
    "age", "gender", "ethnicity", "race", "lang",
    "religion", "maritalstatus", "employstatus", "insurance_status",
    "arrivalmode", "arrivalhour_bin",
    "triage_vital_hr", "triage_vital_sbp", "triage_vital_dbp",
    "triage_vital_rr", "triage_vital_o2sat", "triage_vital_temp",
    "n_edvisits", "n_admissions", "n_surgeries",
    "previousdispo_bin", "n_medications",
]

# ETA model features (from NYC taxi processed data)
ETA_FEATURES = [
    "passenger_count",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "hour",
    "day_of_week",
    "month",
    "distance_km",
]
ETA_TARGET = "duration_min"

# Bed availability features from bed_preprocessing (small dataset)
BED_SIMPLE_FEATURES = [
    "age", "gender", "arrivalhour_bin",
    "triage_vital_hr", "triage_vital_sbp", "triage_vital_temp",
]
BED_SIMPLE_TARGET = "available_beds"

# Bed availability features from Hospital_data.py (rich synthetic dataset)
BED_RICH_FEATURES = [
    "hospital_id", "total_beds", "occupied_beds",
    "occupancy_rate", "admissions_last_1h", "discharges_last_1h",
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "admissions_rolling_6h", "discharges_rolling_6h",
    "occupancy_rolling_avg_24h",
]
BED_RICH_TARGET = "available_beds"

# =============================================================================
# SEVERITY LABEL MAPPING
# =============================================================================
SEVERITY_LABELS = {
    1: "Critical",
    2: "Emergent",
    3: "Urgent",
    4: "Less Urgent",
    5: "Non-urgent",
}

SEVERITY_EMOJI = {
    1: "🔴 CRITICAL",
    2: "🟠 EMERGENT",
    3: "🟡 URGENT",
    4: "🟢 LESS URGENT",
    5: "⚪ NON-URGENT",
}

SEVERITY_DEPARTMENT_MAP = {
    1: "ICU",
    2: "Emergency",
    3: "Emergency",
    4: "General",
    5: "General",
}

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Random Forest (Classification)
RF_CLASSIFIER_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

# XGBoost (Classification)
XGB_CLASSIFIER_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}

# LightGBM (Classification)
LGB_CLASSIFIER_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# AdaBoost (Classification)
ADABOOST_PARAMS = {
    "n_estimators": 50,
    "learning_rate": 0.1,
    "random_state": 42,
}
ADABOOST_BASE_ESTIMATOR_DEPTH = 3

# Random Forest (Regression)
RF_REGRESSOR_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": 42,
    "n_jobs": -1,
}

# XGBoost (Regression)
XGB_REGRESSOR_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

# LightGBM (Regression)
LGB_REGRESSOR_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "subsample": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# Gradient Boosting (Regression)
GBR_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
}

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
CV_FOLDS = 2

# =============================================================================
# DECISION ENGINE WEIGHTS
# =============================================================================
DECISION_WEIGHTS = {
    "eta_weight": 0.35,        # Lower ETA = better
    "bed_weight": 0.40,        # More beds = better
    "severity_match_weight": 0.25,  # Department match = better
}

# Hospital search radius in km
HOSPITAL_SEARCH_RADIUS_KM = 20.0


class Config:
    """Configuration class for easy access."""
    
    # All attributes from module level
    PROJECT_ROOT = PROJECT_ROOT
    DATASETS_DIR = DATASETS_DIR
    RAW_DATA_DIR = RAW_DATA_DIR
    PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
    SYNTHETIC_DATA_DIR = SYNTHETIC_DATA_DIR
    BED_DATA_DIR = BED_DATA_DIR
    MODELS_DIR = MODELS_DIR
    SEVERITY_MODELS_DIR = SEVERITY_MODELS_DIR
    ETA_MODELS_DIR = ETA_MODELS_DIR
    BED_MODELS_DIR = BED_MODELS_DIR
    RESULTS_DIR = RESULTS_DIR
    LOGS_DIR = LOGS_DIR
    
    # Dataset Files
    SEVERITY_CLEANED_CSV = SEVERITY_CLEANED_CSV
    ETA_PROCESSED_CSV = ETA_PROCESSED_CSV
    BED_PROCESSED_CSV = BED_PROCESSED_CSV
    HOSPITAL_BEDS_CSV = HOSPITAL_BEDS_CSV
    SYNTHETIC_TRIAGE_CSV = SYNTHETIC_TRIAGE_CSV

    # Features and Targets
    SEVERITY_TARGET = SEVERITY_TARGET
    SEVERITY_KEY_FEATURES = SEVERITY_KEY_FEATURES
    ETA_FEATURES = ETA_FEATURES
    ETA_TARGET = ETA_TARGET
    BED_SIMPLE_FEATURES = BED_SIMPLE_FEATURES
    BED_SIMPLE_TARGET = BED_SIMPLE_TARGET
    BED_RICH_FEATURES = BED_RICH_FEATURES
    BED_RICH_TARGET = BED_RICH_TARGET

    # Mappings
    SEVERITY_LABELS = SEVERITY_LABELS
    SEVERITY_EMOJI = SEVERITY_EMOJI
    SEVERITY_DEPARTMENT_MAP = SEVERITY_DEPARTMENT_MAP

    # Hyperparameters
    RF_CLASSIFIER_PARAMS = RF_CLASSIFIER_PARAMS
    XGB_CLASSIFIER_PARAMS = XGB_CLASSIFIER_PARAMS
    LGB_CLASSIFIER_PARAMS = LGB_CLASSIFIER_PARAMS
    ADABOOST_PARAMS = ADABOOST_PARAMS
    ADABOOST_BASE_ESTIMATOR_DEPTH = ADABOOST_BASE_ESTIMATOR_DEPTH
    RF_REGRESSOR_PARAMS = RF_REGRESSOR_PARAMS
    XGB_REGRESSOR_PARAMS = XGB_REGRESSOR_PARAMS
    LGB_REGRESSOR_PARAMS = LGB_REGRESSOR_PARAMS
    GBR_PARAMS = GBR_PARAMS

    # Decision Weights
    DECISION_WEIGHTS = DECISION_WEIGHTS
    HOSPITAL_SEARCH_RADIUS_KM = HOSPITAL_SEARCH_RADIUS_KM

    RANDOM_STATE = RANDOM_STATE
    CV_FOLDS = CV_FOLDS
    TEST_SIZE = TEST_SIZE
    VAL_SIZE = VAL_SIZE
    
    @classmethod
    def ensure_dirs(cls):
        """Create all required directories."""
        for d in [cls.SEVERITY_MODELS_DIR, cls.ETA_MODELS_DIR, cls.BED_MODELS_DIR,
                  cls.RESULTS_DIR, cls.LOGS_DIR, cls.PROCESSED_DATA_DIR,
                  cls.SYNTHETIC_DATA_DIR]:
            os.makedirs(d, exist_ok=True)
