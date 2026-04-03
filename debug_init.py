import sys
import os
import joblib

# Add project root to path
sys.path.append(os.getcwd())

from backend.services.prediction_service import PredictionService
from src.utils.config import Config

try:
    print("Testing PredictionService initialization...")
    # Manually check paths
    print(f"SEVERITY_MODELS_DIR: {Config.SEVERITY_MODELS_DIR}")
    print(f"ETA_MODELS_DIR: {Config.ETA_MODELS_DIR}")
    
    service = PredictionService()
    if service.is_ready():
        print("✅ PredictionService is READY.")
    else:
        print("❌ PredictionService is NOT READY (Setup failed in __init__).")
except Exception as e:
    import traceback
    print(f"💥 Exception during init: {e}")
    traceback.print_exc()
