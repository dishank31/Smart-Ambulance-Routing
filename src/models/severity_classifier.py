import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

from ..utils.config import Config
from ..data.preprocessing import DataPreprocessor
from ..data.data_loader import load_severity_data
from ..evaluation.evaluator import ModelEvaluator
from ..utils.logger import setup_logger
from ..utils.model_registry import ModelRegistry

logger = setup_logger('severity_trainer')
registry = ModelRegistry()

def train_severity_models():
    logger.info("=" * 60)
    logger.info("TRAINING SEVERITY CLASSIFICATION MODELS (YOUR REAL DATA)")
    logger.info("=" * 60)
    
    df = load_severity_data()
    preprocessor = DataPreprocessor()
    
    y_ser = df['esi'].fillna(df['esi'].mode()[0])
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_ser.values)
    df_features = df.drop(columns=['esi'])
    
    actual_features = list(df_features.columns)
    
    # If the real dataset has any remaining strings, encode them, though I assume it's mostly cleaned
    df_clean = df_features.select_dtypes(exclude=['object'])
    actual_features = list(df_clean.columns)
    X = df_clean.values
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    X_train = preprocessor.scaler.fit_transform(X_train)
    X_val = preprocessor.scaler.transform(X_val)
    X_test = preprocessor.scaler.transform(X_test)
    
    os.makedirs(Config.SEVERITY_MODELS_DIR, exist_ok=True)
    joblib.dump(actual_features, os.path.join(Config.SEVERITY_MODELS_DIR, 'severity_features.joblib'))
    preprocessor.save_scaler(os.path.join(Config.SEVERITY_MODELS_DIR, 'scaler_severity.joblib'))
    
    models = {
        'Random Forest': RandomForestClassifier(**Config.RF_CLASSIFIER_PARAMS),
        'XGBoost': xgb.XGBClassifier(**Config.XGB_CLASSIFIER_PARAMS),
        'LightGBM': lgb.LGBMClassifier(**Config.LGB_CLASSIFIER_PARAMS),
    }
    
    evaluator = ModelEvaluator(task='classification')
    
    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluator.evaluate_classification(model, X_test, y_test, name)
        registry.register_model(model, f"severity_{name.lower().replace(' ', '_')}", metrics)
    
    logger.info("\nTraining Stacking Ensemble...")
    stacking = StackingClassifier(
        estimators=[
            ('rf', models['Random Forest']),
            ('xgb', models['XGBoost']),
            ('lgb', models['LightGBM']),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=2, stack_method='predict_proba', n_jobs=-1
    )
    
    stacking.fit(X_train, y_train)
    metrics = evaluator.evaluate_classification(stacking, X_test, y_test, 'Stacking Ensemble')
    registry.register_model(stacking, "severity_stacking", metrics)
    
    evaluator.plot_model_comparison_bar()
    
    return stacking
