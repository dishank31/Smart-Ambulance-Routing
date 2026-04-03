import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb

from ..utils.config import Config
from ..data.preprocessing import DataPreprocessor
from ..data.data_loader import load_bed_data
from ..evaluation.evaluator import ModelEvaluator
from ..utils.logger import setup_logger
from ..utils.model_registry import ModelRegistry

logger = setup_logger('bed_trainer')
registry = ModelRegistry()

def train_bed_models():
    logger.info("=" * 60)
    logger.info("TRAINING BED AVAILABILITY MODELS (YOUR REAL DATA)")
    logger.info("=" * 60)
    
    # Load from your specific `bed_availability_processed.csv` !
    df = load_bed_data(use_rich=False)
    
    preprocessor = DataPreprocessor()
    df_encoded = preprocessor.encode_categorical(df, columns=['gender'], method='label')
    
    y_ser = df_encoded[Config.BED_SIMPLE_TARGET].fillna(df_encoded[Config.BED_SIMPLE_TARGET].mode()[0])
    y = y_ser.values.astype(int)
    df_features = df_encoded.drop(columns=[Config.BED_SIMPLE_TARGET, 'gender']) # drop raw string gender as it is encoded now
    
    actual_features = list(df_features.columns)
    X = df_features.values
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    X_train = preprocessor.scaler.fit_transform(X_train)
    X_val = preprocessor.scaler.transform(X_val)
    X_test = preprocessor.scaler.transform(X_test)
    
    os.makedirs(Config.BED_MODELS_DIR, exist_ok=True)
    joblib.dump(actual_features, os.path.join(Config.BED_MODELS_DIR, 'bed_features.joblib'))
    preprocessor.save_scaler(os.path.join(Config.BED_MODELS_DIR, 'scaler_bed.joblib'))
    
    models = {
        'Random Forest': RandomForestRegressor(**Config.RF_REGRESSOR_PARAMS),
        'XGBoost': xgb.XGBRegressor(**Config.XGB_REGRESSOR_PARAMS),
        'LightGBM': lgb.LGBMRegressor(**Config.LGB_REGRESSOR_PARAMS),
    }
    
    evaluator = ModelEvaluator(task='regression')
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluator.evaluate_regression(model, X_test, y_test, name)
        registry.register_model(model, f"bed_{name.lower().replace(' ', '_')}", metrics)
    
    logger.info("Training Stacking Ensemble...")
    stacking_bed = StackingRegressor(
        estimators=[(k.lower().replace(' ', '_'), v) for k, v in models.items()],
        final_estimator=Ridge(alpha=1.0),
        cv=3, n_jobs=1
    )
    
    stacking_bed.fit(X_train, y_train)
    metrics = evaluator.evaluate_regression(stacking_bed, X_test, y_test, 'Stacking Ensemble')
    registry.register_model(stacking_bed, "bed_stacking", metrics)
    
    evaluator.plot_model_comparison_bar()
    
    return stacking_bed
