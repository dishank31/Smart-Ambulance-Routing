import joblib
import os
from src.utils.config import Config

def load_bed_model():
    """
    STEP 5: LOAD MODEL IN YOUR PROJECT
    Loads the trained XGBoost model for bed availability.
    """
    model_path = os.path.join(Config.MODELS_DIR, 'bed_xgboost.pkl')
    try:
        model = joblib.load(model_path)
        print(f"Successfully loaded bed model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading bed model: {e}")
        return None
