"""Severity classification model for triage prediction."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

from ..utils.config import Config
from ..utils.logger import setup_logger


logger = setup_logger("severity_model")


class SeverityClassifier:
    """Ensemble classifier for emergency severity prediction (ESI 1-5)."""

    def __init__(self):
        self.models: dict[str, any] = {}
        self.stacking_model = None
        self.is_trained = False
        self.feature_names: list[str] = []
        self.label_encoder = None
        self.training_time: dict[str, float] = {}

    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Preprocess raw features for severity classification."""
        df = df.copy()

        # Encode gender
        if "gender" in df.columns:
            df["gender"] = df["gender"].apply(lambda x: 1 if str(x).lower().startswith("f") else 0)

        # Encode chronic condition
        if "has_chronic_condition" in df.columns:
            df["has_chronic_condition"] = df["has_chronic_condition"].astype(int)

        # One-hot encode chief complaint
        if "chief_complaint" in df.columns:
            complaint_dummies = pd.get_dummies(df["chief_complaint"], prefix="complaint")
            df = pd.concat([df.drop("chief_complaint", axis=1), complaint_dummies], axis=1)

        # Ensure all expected features exist
        expected_features = ["heart_rate", "bp_systolic", "bp_diastolic", "spo2",
                           "respiratory_rate", "temperature", "gcs_score", "pain_scale",
                           "age", "gender", "has_chronic_condition"]

        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = 0

        self.feature_names = [c for c in df.columns if c not in ["severity", "recommended_department"]]
        return df

    def add_base_models(self):
        """Add base models to the ensemble."""
        self.models["Random Forest"] = RandomForestClassifier(**Config.RF_CLASSIFIER_PARAMS)

        if XGBClassifier:
            self.models["XGBoost"] = XGBClassifier(**Config.XGB_CLASSIFIER_PARAMS)
        else:
            logger.warning("XGBClassifier not available")

        if LGBMClassifier:
            self.models["LightGBM"] = LGBMClassifier(**Config.LGB_CLASSIFIER_PARAMS)
        else:
            logger.warning("LGBMClassifier not available")

        self.models["AdaBoost"] = AdaBoostClassifier(**Config.ADABOOST_PARAMS)

        logger.info(f"Added {len(self.models)} base models")

    def train(self, X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray,
              X_val: pd.DataFrame | np.ndarray = None, y_val: pd.Series | np.ndarray = None) -> dict:
        """Train all base models and stacking ensemble."""
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
                acc = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                results[name] = {"accuracy": acc, "f1": f1, "time": elapsed}
                logger.info(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}, Time={elapsed:.2f}s")

        # Build and train stacking ensemble
        logger.info("Building Stacking Ensemble...")
        base_estimators = [(name, model) for name, model in self.models.items()]
        meta_learner = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

        self.stacking_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )

        start = time.time()
        self.stacking_model.fit(X_train, y_train)
        self.training_time["Stacking"] = time.time() - start

        if X_val is not None and y_val is not None:
            y_pred = self.stacking_model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            results["Stacking"] = {"accuracy": acc, "f1": f1, "time": self.training_time["Stacking"]}
            logger.info(f"  Stacking: Acc={acc:.4f}, F1={f1:.4f}")

        self.is_trained = True
        return results

    def predict(self, X: pd.DataFrame | np.ndarray, use_stacking: bool = True) -> np.ndarray:
        """Predict severity levels."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        model = self.stacking_model if use_stacking else list(self.models.values())[0]
        return model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray, use_stacking: bool = True) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        model = self.stacking_model if use_stacking else list(self.models.values())[0]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        return None

    def save(self, filepath: str | Path, model_name: str = "severity_stacking") -> None:
        """Save trained model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "model": self.stacking_model if self.stacking_model else list(self.models.values())[0],
            "feature_names": self.feature_names,
            "training_time": self.training_time,
            "version": Config.MODEL_VERSION,
            "task": "severity_classification"
        }
        joblib.dump(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")

        # Update model registry
        self._update_registry(filepath, model_name)

    def _update_registry(self, filepath: Path, model_name: str) -> None:
        """Update model registry with new model metadata."""
        registry_path = Path(Config.MODEL_REGISTRY_PATH)
        registry = {}
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
            except:
                pass

        registry["severity"] = {
            "model_name": model_name,
            "filepath": str(filepath),
            "version": Config.MODEL_VERSION,
            "features": self.feature_names,
            "training_time": self.training_time,
            "classes": [1, 2, 3, 4, 5]
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


def train_severity_model(data_path: str | Path = None, save: bool = True) -> SeverityClassifier:
    """Convenience function to train severity model from data file."""
    if data_path is None:
        data_path = Config.SEVERITY_DATA_FILE

    logger.info(f"Loading severity data from {data_path}")
    df = pd.read_csv(data_path)

    # Preprocess
    classifier = SeverityClassifier()
    df_processed = classifier.preprocess_features(df, fit=True)

    # Split features and target
    X = df_processed.drop(["severity", "recommended_department"], axis=1, errors='ignore')
    y = df["severity"]

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train
    results = classifier.train(X_train, y_train, X_val, y_val)

    # Test evaluation
    y_pred = classifier.predict(X_test)
    logger.info("\nTest Set Performance:")
    logger.info(classification_report(y_test, y_pred))

    # Save
    if save:
        save_path = Path(Config.SEVERITY_MODELS_DIR) / f"severity_stacking_{Config.MODEL_VERSION}.joblib"
        classifier.save(save_path, model_name="severity_stacking")

    return classifier


if __name__ == "__main__":
    # Train when run directly
    Config.ensure_dirs()
    model = train_severity_model()
    print("Severity model training complete!")
