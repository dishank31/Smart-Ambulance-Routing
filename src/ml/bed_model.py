"""Bed availability prediction model for hospital capacity forecasting."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

from ..utils.config import Config
from ..utils.logger import setup_logger


logger = setup_logger("bed_model")


class BedPredictor:
    """Ensemble regressor for predicting hospital bed availability."""

    def __init__(self):
        self.models: dict[str, any] = {}
        self.stacking_model = None
        self.is_trained = False
        self.feature_names: list[str] = []
        self.training_time: dict[str, float] = {}
        self.metrics: dict[str, dict] = {}

    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Preprocess features for bed availability prediction."""
        df = df.copy()

        # Ensure numeric types
        numeric_cols = ["hospital_id", "total_beds", "hour", "day_of_week", "month",
                       "is_weekend", "is_holiday", "admissions_last_1h", "discharges_last_1h"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Encode department
        if "department" in df.columns:
            dept_map = {"ICU": 0, "Emergency": 1, "General": 2}
            df["department"] = df["department"].map(dept_map).fillna(2)

        # Encode season if present
        if "season" in df.columns:
            season_map = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}
            df["season"] = df["season"].map(season_map).fillna(1)

        # Cyclical encoding for hour
        if "hour" in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Cyclical encoding for day_of_week
        if "day_of_week" in df.columns:
            df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Time features
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["is_night"] = (df["hour"] >= 20) | (df["hour"] <= 5).astype(int)
            df["is_peak"] = df["hour"].isin([10, 11, 12, 13, 18, 19, 20, 21]).astype(int)

        # Handle missing occupancy_rate
        if "occupancy_rate" in df.columns:
            df["occupancy_rate"] = df["occupancy_rate"].fillna(0.75)

        # Rolling features should already be present
        self.feature_names = [c for c in df.columns
                               if c not in ["available_beds", "occupied_beds", "timestamp",
                                          "hospital_name", "is_peak", "is_night"]]
        return df

    def add_base_models(self):
        """Add base regression models."""
        self.models["Random Forest"] = RandomForestRegressor(**Config.RF_REGRESSOR_PARAMS)

        if XGBRegressor:
            self.models["XGBoost"] = XGBRegressor(**Config.XGB_REGRESSOR_PARAMS)
        else:
            logger.warning("XGBRegressor not available")

        if LGBMRegressor:
            self.models["LightGBM"] = LGBMRegressor(**Config.LGB_REGRESSOR_PARAMS)
        else:
            logger.warning("LGBMRegressor not available")

        self.models["Gradient Boosting"] = GradientBoostingRegressor(**Config.GBR_PARAMS)

        logger.info(f"Added {len(self.models)} base models")

    def train(self, X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray,
              X_val: pd.DataFrame | np.ndarray = None, y_val: pd.Series | np.ndarray = None) -> dict:
        """Train all models and stacking ensemble."""
        self.add_base_models()
        results = {}

        # Train individual models
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            start = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - start
            self.training_time[name] = elapsed

            # Validation evaluation
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "time": elapsed}
                logger.info(f"  {name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

        # Build and train stacking ensemble
        logger.info("Building Stacking Regressor...")
        base_estimators = [(name, model) for name, model in self.models.items()]
        meta_learner = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)

        self.stacking_model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )

        start = time.time()
        self.stacking_model.fit(X_train, y_train)
        self.training_time["Stacking"] = time.time() - start

        if X_val is not None and y_val is not None:
            y_pred = self.stacking_model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            results["Stacking"] = {"MAE": mae, "RMSE": rmse, "R2": r2, "time": self.training_time["Stacking"]}
            logger.info(f"  Stacking: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

        self.is_trained = True
        return results

    def predict(self, X: pd.DataFrame | np.ndarray, use_stacking: bool = True) -> np.ndarray:
        """Predict available bed counts."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        model = self.stacking_model if use_stacking else list(self.models.values())[0]
        predictions = model.predict(X)
        # Ensure non-negative integers
        return np.maximum(0, np.round(predictions)).astype(int)

    def save(self, filepath: str | Path, model_name: str = "bed_stacking") -> None:
        """Save trained model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model": self.stacking_model if self.stacking_model else list(self.models.values())[0],
            "feature_names": self.feature_names,
            "training_time": self.training_time,
            "version": Config.MODEL_VERSION,
            "task": "bed_availability_prediction"
        }
        joblib.dump(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")

        # Update registry
        self._update_registry(filepath, model_name)

    def _update_registry(self, filepath: Path, model_name: str) -> None:
        """Update model registry."""
        registry_path = Path(Config.MODEL_REGISTRY_PATH)
        registry = {}
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
            except:
                pass

        registry["bed"] = {
            "model_name": model_name,
            "filepath": str(filepath),
            "version": Config.MODEL_VERSION,
            "features": self.feature_names,
            "training_time": self.training_time
        }

        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

    def load(self, filepath: str | Path) -> None:
        """Load trained model."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        save_dict = joblib.load(filepath)
        loaded_model = save_dict.get("model", save_dict)

        self.stacking_model = loaded_model
        self.feature_names = save_dict.get("feature_names", [])
        self.training_time = save_dict.get("training_time", {})
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

    def get_feature_importance(self) -> dict:
        """Get feature importance from tree-based models."""
        importance_dict = {}

        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = dict(zip(self.feature_names, model.feature_importances_))

        return importance_dict


def train_bed_model(data_path: str | Path = None, save: bool = True) -> BedPredictor:
    """Convenience function to train bed availability model."""
    if data_path is None:
        data_path = Config.BED_DATA_FILE

    logger.info(f"Loading bed availability data from {data_path}")
    df = pd.read_csv(data_path)

    # Preprocess
    predictor = BedPredictor()
    df_processed = predictor.preprocess_features(df, fit=True)

    # Split features and target
    X = df_processed.drop(["available_beds", "occupied_beds"], axis=1, errors='ignore')
    y = df["available_beds"]

    # Temporal split (preserve time ordering)
    if "timestamp" in df.columns:
        df_sorted = df.sort_values("timestamp")
        n = len(df_sorted)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        X = df_processed.drop(["available_beds", "occupied_beds", "timestamp"], axis=1, errors='ignore')
        X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
        y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train
    results = predictor.train(X_train, y_train, X_val, y_val)

    # Test evaluation
    y_pred = predictor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logger.info(f"\nTest Set Performance: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

    # Save
    if save:
        save_path = Path(Config.BED_MODELS_DIR) / f"bed_stacking_{Config.MODEL_VERSION}.joblib"
        predictor.save(save_path, model_name="bed_stacking")

    return predictor


if __name__ == "__main__":
    # Train when run directly
    Config.ensure_dirs()
    model = train_bed_model()
    print("Bed availability model training complete!")
