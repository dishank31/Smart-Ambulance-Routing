from __future__ import annotations

import os
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import shap

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets" / "processed"


def save_summary_plot(model, frame: pd.DataFrame, output_path: Path, title: str, max_display: int = 20) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(frame)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, frame, show=False, max_display=max_display)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_severity_plot() -> Path:
    features = joblib.load(MODELS_DIR / "severity" / "severity_features.joblib")
    model = joblib.load(MODELS_DIR / "severity" / "severity_lightgbm_v4.joblib")
    frame = pd.read_csv(DATASETS_DIR / "severity_cleaned.csv", usecols=features).head(300)
    output_path = RESULTS_DIR / "shap_severity.png"
    save_summary_plot(model, frame, output_path, "SHAP - Severity Model", max_display=20)
    return output_path


def generate_eta_plot() -> Path:
    features = joblib.load(MODELS_DIR / "eta" / "eta_features.joblib")
    model = joblib.load(MODELS_DIR / "eta" / "eta_xgboost_v2.joblib")
    frame = pd.read_csv(DATASETS_DIR / "eta_processed.csv", usecols=features).head(400)
    output_path = RESULTS_DIR / "shap_eta.png"
    save_summary_plot(model, frame, output_path, "SHAP - ETA Model", max_display=9)
    return output_path


def generate_bed_plot() -> Path:
    features = ["age", "gender", "arrivalhour_bin", "triage_vital_hr", "triage_vital_sbp", "triage_vital_temp"]
    model = joblib.load(MODELS_DIR / "bed_xgboost.pkl")
    frame = pd.read_csv(DATASETS_DIR / "bed_availability_processed.csv", usecols=features).head(400)
    frame["gender"] = pd.Categorical(frame["gender"]).codes
    frame["arrivalhour_bin"] = pd.Categorical(frame["arrivalhour_bin"]).codes
    output_path = RESULTS_DIR / "shap_bed.png"
    save_summary_plot(model, frame, output_path, "SHAP - Bed Availability Model", max_display=6)
    return output_path


def main() -> None:
    generated = [
        generate_severity_plot(),
        generate_eta_plot(),
        generate_bed_plot(),
    ]
    for path in generated:
        print(f"Generated {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
