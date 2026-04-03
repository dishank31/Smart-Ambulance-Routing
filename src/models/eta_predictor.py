import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgb

from ..utils.config import Config
from ..data.preprocessing import DataPreprocessor
from ..data.feature_engineering import create_eta_features
from ..data.data_loader import load_eta_data
from ..evaluation.evaluator import ModelEvaluator
from ..utils.logger import setup_logger
from ..utils.model_registry import ModelRegistry

logger = setup_logger('eta_trainer')
registry = ModelRegistry()

def train_eta_models():
    logger.info("=" * 60)
    logger.info("TRAINING ETA PREDICTION MODELS")
    logger.info("=" * 60)
    
    df = load_eta_data()
    df = create_eta_features(df)
    
    feature_cols = ['distance_km', 'bearing', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                    'is_rush_hour', 'is_weekend', 'month', 'manhattan_dist',
                    'pickup_latitude', 'pickup_longitude']
    
    # Filter to only the columns that exist
    actual_features = [col for col in feature_cols if col in df.columns]
    
    X = df[actual_features].values
    y = df['duration_minutes'].values if 'duration_minutes' in df.columns else df[Config.ETA_TARGET].values
    
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y, temporal=True)
    
    X_train = preprocessor.scaler.fit_transform(X_train)
    X_val = preprocessor.scaler.transform(X_val)
    X_test = preprocessor.scaler.transform(X_test)
    
    os.makedirs(Config.ETA_MODELS_DIR, exist_ok=True)
    preprocessor.save_scaler(os.path.join(Config.ETA_MODELS_DIR, 'scaler_eta.joblib'))
    
    models = {
        'Random Forest': RandomForestRegressor(**Config.RF_REGRESSOR_PARAMS),
        'XGBoost': xgb.XGBRegressor(**Config.XGB_REGRESSOR_PARAMS),
        'LightGBM': lgb.LGBMRegressor(**Config.LGB_REGRESSOR_PARAMS),
        'Gradient Boosting': GradientBoostingRegressor(**Config.GBR_PARAMS),
    }
    
    evaluator = ModelEvaluator(task='regression')
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluator.evaluate_regression(model, X_test, y_test, name)
        registry.register_model(model, f"eta_{name.lower().replace(' ', '_')}", metrics)
    
    logger.info("Training Stacking Ensemble...")
    stacking_eta = StackingRegressor(
        estimators=[(k.lower().replace(' ', '_'), v) for k, v in models.items()],
        final_estimator=ElasticNet(alpha=0.1, l1_ratio=0.5),
        cv=3, n_jobs=1
    )
    
    stacking_eta.fit(X_train, y_train)
    metrics = evaluator.evaluate_regression(stacking_eta, X_test, y_test, 'Stacking Ensemble')
    registry.register_model(stacking_eta, "eta_stacking", metrics)
    
    evaluator.plot_model_comparison_bar()
    
    return stacking_eta
