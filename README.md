# 🚑 Smart Ambulance Dispatch & Hospital Recommendation System

An end-to-end Machine Learning pipeline, FastAPI Backend, and Streamlit Dashboard designed to optimize ambulance dispatching by predicting patient severity, real-time hospital ETA, and hospital bed availability using ensemble stacking ML models.

## Features
- **Severity Prediction**: Classifies emergency level (ESI 1-5) using Stacking (RF, XGB, LGB, AdaBoost).
- **Dynamic ETA**: Predicts real-time ambulance travel times considering rush hour, seasonality, and Manhattan distance.
- **Bed Availability**: Forecasts hospital bed availability based on historical hourly occupancy.
- **Decision Engine**: Ranks hospitals using a weighted multi-objective function (35% ETA, 40% Beds, 25% Dept Match).

## Subsystems
- **Data Generator**: Synthesizes highly realistic triage and hospital availability data.
- **Model Training**: Automated pipeline to train base learners and meta-learners.
- **Backend API**: High-performance asynchronous FastAPI server handling predictions natively.
- **Dashboard**: Professional Streamlit UI with `folium` interactive maps.

## Tech Stack
- Frontend: Streamlit, Folium
- Backend: FastAPI, Pydantic, Uvicorn
- Base ML: Scikit-learn, XGBoost, LightGBM
- Utilities: Optuna, SHAP

## Installation & Setup

1. **Install Requirements**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. **Generate Synthetic Data**
```bash
python datasets/synthetic/generate_severity_data.py
python datasets/synthetic/generate_bed_data.py
```

3. **Train Models**
```bash
python train_all_models.py
```
*(This will take 2-4 minutes and will create `.joblib` files in `models/` directory)*

4. **Run Backend API \*(IN A SEPARATE TERMINAL)\***
```bash
uvicorn backend.main:app --reload --port 8000
```
API Documentation will be available at `http://localhost:8000/docs`

5. **Run Streamlit Dashboard \*(IN ANOTHER TERMINAL)\***
```bash
streamlit run frontend/app_streamlit.py
```
