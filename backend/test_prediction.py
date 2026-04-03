import joblib
import numpy as np
import os

# Ensure we can load the model regardless of where the script is run from
model_path = os.path.join(os.getcwd(), 'models', 'bed_xgboost.pkl')

if not os.path.exists(model_path):
    # Try parent directory if we are inside backend/
    model_path = os.path.join(os.path.dirname(os.getcwd()), 'models', 'bed_xgboost.pkl')

print(f"Loading model from: {model_path}")

# =========================
# STEP 5: LOAD MODEL
# =========================
try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# =========================
# STEP 6: TEST PREDICTION
# =========================
# Input features: [age, gender, arrivalhour_bin, triage_vital_hr, triage_vital_sbp, triage_vital_temp]
# Note: gender is label encoded (0 or 1)
your_input_data = [45, 0, 14, 85, 130, 37.2]

# Perform prediction (XGBoost expects a 2D array-like input)
prediction = model.predict([your_input_data])

print("\n--- Prediction Results ---")
print(f"Input Data: {your_input_data}")
print(f"Predicted Available Beds: {prediction[0]:.2f}")
