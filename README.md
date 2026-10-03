# 🚑 Smart Ambulance Dispatch & Hospital Recommendation System

<div align="center">
  <p>An end-to-end Machine Learning pipeline, asynchronous FastAPI Backend, and interactive Streamlit Dashboard designed to optimize emergency medical services through advanced predictive modeling and dynamic routing.</p>
</div>

---

## 📖 Project Overview

In urban environments, emergency medical services face critical challenges in patient triage, ambulance routing, and hospital bed allocation. Unbalanced hospital loads and static dispatching rules can lead to delayed treatments. 

This project solves these issues by acting as a **Decision Engine for Emergency Dispatchers**. Upon receiving an emergency call, the system leverages a powerful **Stacking Ensemble Machine Learning Pipeline** to dynamically calculate the severity of the patient's condition, the estimated time of arrival (ETA) through urban traffic, and live hospital bed availability to route the ambulance to the most optimal healthcare facility.

---

## 🧠 Core Machine Learning Methodology

The intelligence of the system relies on three predictive components built using advanced **Stacking Ensemble techniques**:

### 1. Patient Severity Prediction (ESI Classification)
- **Objective:** Classifies patients into the 5-level Emergency Severity Index (ESI).
- **Features:** Age, heart rate, blood pressure, oxygen saturation, pain level, and symptomatic flags.
- **Models Used:** A Stacking Classifier combining Random Forest (for non-linear relationships), XGBoost (high-performance gradient boosting), and LightGBM (categorical feature handling). 
- **Result:** Mitigates the bias of individual models and achieves robust classification even on noisy real-world data.

### 2. Dynamic ETA Prediction (Regression)
- **Objective:** Dynamically calculates ambulance travel times.
- **Features:** Spatial coordinates, Manhattan distance, hour of the day, day of the week, and simulated traffic density.
- **Models Used:** A regression-based Stacking Ensemble (Gradient Boosting, LightGBM, and XGBoost). Lowered variance significantly during simulated edge-case traffic scenarios.

### 3. Multi-Objective Decision Engine
- Rather than strictly sending ambulances to the closest geographical point, our routing engine ranks all nearby hospitals using a custom weighted optimization formula:
  > **Score = (0.35 × Normalized ETA) + (0.40 × Normalized Bed Availability) + (0.25 × Specialty Match)**
- This ensures critically ill patients are assigned to hospitals with guaranteed ICU/Emergency Room capabilities.

---

## 🏗️ System Architecture & Tech Stack

- **Frontend Application (Streamlit & Folium):** An interactive data dashboard providing real-time geocoding and map-based visualizations for dispatchers.
- **Backend API (FastAPI & Uvicorn):** High-performance asynchronous REST API handling high volumes of concurrent prediction requests.
- **Data Engine (Pandas & NumPy):** Heavily engineered synthetic data generator mathematically mimicking real-world triage demographics and localized NYC-area hospital metrics.
- **Machine Learning Core:** Scikit-learn, XGBoost, and LightGBM.

---

## 📊 Experimental Results & Evaluation

The system was rigorously evaluated using industry-standard metrics:

### Classification Performance
The Stacking Ensemble model achieved an overall accuracy of **82.0%** on highly noisy real-world data distributions, with near-perfect classification on cleaner subsets. It significantly outperformed isolated base models in terms of both macro F1 scores and weighted precision.

### Regression Metrics (ETA)
The Stacking Regressor minimized absolute error margins critical for life-saving operations:
- **Mean Absolute Error (MAE):** 3.46 minutes
- **Root Mean Square Error (RMSE):** 5.45
- **R² Score:** 0.753

### Interpretable AI (SHAP Analysis)
To guarantee transparency in the medical decision-making process:
- **Severity Prediction:** SHAP plots verify that `Oxygen Saturation` and `Systolic Blood Pressure` are the highest predictive indicators for critical trauma.
- **ETA Prediction:** `Distance` and `Hour_Of_Day` logically dominate the regression trees, capturing rush hour traffic effects effectively.

---

## 📁 Repository Structure

| Path | Purpose |
|------|---------|
| `notebooks/` | Data exploration, preparation, model training, evaluation, and SHAP pipelines |
| `src/` | Library code encompassing data engineering, routing algorithms, and core ML |
| `backend/` | Complete FastAPI server application |
| `frontend/` | Streamlit dispatcher dashboard |
| `datasets/` | Directory for raw, processed, and synthetic generated CSVs |
| `models/` | Trained Machine Learning `.joblib` artifacts |

---

## 🛠️ Installation & Setup

Ensure you are located in the project's root directory (`smart-ambulance-ml/`) before starting.

### 1. Virtual Environment & Dependencies
Set up the Python environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```
*(For working with Jupyter and SHAP visualization notebooks, optionally run: `pip install -r requirements-dev.txt`)*

### 2. Dataset Generation
You can use the provided CSVs in `datasets/` or rebuild them entirely by running the Jupyter Notebooks `01_data_exploration.ipynb` and `02_data_preparation.ipynb`.

### 3. Model Training
To retrain the complete ensemble ML pipeline:
```bash
python train_all_models.py
```
*(Takes ~2-4 minutes and persists `.joblib` files directly into the `models/` directory).*

### 4. Running the Complete System
To easily start both the FastAPI backend and Streamlit dashboard at the same time, you can run the provided launcher script:
```bash
python run_system.py
```
- API endpoints map to: `http://localhost:8000`
- Interactive Swagger Docs: `http://localhost:8000/docs`
- Streamlit UI maps to: `http://localhost:8501`

Alternatively, you can start them separately:
**Backend API:**
```bash
uvicorn backend.main:app --reload --port 8000
```
**Frontend Dashboard:**
```bash
streamlit run frontend/app_streamlit.py
```
