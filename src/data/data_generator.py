"""Synthetic dataset generators for the Smart Ambulance project."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import Config
from ..utils.geo_utils import haversine_distance


RNG = np.random.default_rng(42)

HOSPITAL_NAMES = [
    "Mount Sinai Medical Center",
    "NYU Langone Health",
    "NewYork-Presbyterian",
    "Bellevue Hospital Center",
    "Lenox Hill Hospital",
    "Brooklyn Methodist Hospital",
    "Montefiore Medical Center",
    "Jamaica Hospital Medical Center",
    "Richmond University Medical Center",
    "Harlem Hospital Center",
]

HOSPITAL_PROFILES = [
    {"latitude": 40.7901, "longitude": -73.9533, "total_icu_beds": 24, "total_emergency_beds": 52, "total_general_beds": 135, "has_trauma_center": False, "has_cardiac_center": True, "has_stroke_center": True, "has_burn_unit": False},
    {"latitude": 40.7420, "longitude": -73.9749, "total_icu_beds": 20, "total_emergency_beds": 45, "total_general_beds": 120, "has_trauma_center": False, "has_cardiac_center": True, "has_stroke_center": False, "has_burn_unit": False},
    {"latitude": 40.8412, "longitude": -73.9418, "total_icu_beds": 30, "total_emergency_beds": 60, "total_general_beds": 160, "has_trauma_center": True, "has_cardiac_center": True, "has_stroke_center": True, "has_burn_unit": False},
    {"latitude": 40.7391, "longitude": -73.9754, "total_icu_beds": 18, "total_emergency_beds": 58, "total_general_beds": 110, "has_trauma_center": True, "has_cardiac_center": False, "has_stroke_center": False, "has_burn_unit": False},
    {"latitude": 40.7736, "longitude": -73.9602, "total_icu_beds": 16, "total_emergency_beds": 38, "total_general_beds": 100, "has_trauma_center": False, "has_cardiac_center": True, "has_stroke_center": False, "has_burn_unit": False},
    {"latitude": 40.6681, "longitude": -73.9752, "total_icu_beds": 14, "total_emergency_beds": 42, "total_general_beds": 105, "has_trauma_center": True, "has_cardiac_center": False, "has_stroke_center": False, "has_burn_unit": True},
    {"latitude": 40.8840, "longitude": -73.8801, "total_icu_beds": 26, "total_emergency_beds": 54, "total_general_beds": 145, "has_trauma_center": True, "has_cardiac_center": False, "has_stroke_center": True, "has_burn_unit": False},
    {"latitude": 40.7008, "longitude": -73.8067, "total_icu_beds": 12, "total_emergency_beds": 36, "total_general_beds": 92, "has_trauma_center": False, "has_cardiac_center": False, "has_stroke_center": False, "has_burn_unit": False},
    {"latitude": 40.5865, "longitude": -74.0924, "total_icu_beds": 15, "total_emergency_beds": 34, "total_general_beds": 95, "has_trauma_center": False, "has_cardiac_center": False, "has_stroke_center": False, "has_burn_unit": True},
    {"latitude": 40.8146, "longitude": -73.9400, "total_icu_beds": 17, "total_emergency_beds": 44, "total_general_beds": 108, "has_trauma_center": False, "has_cardiac_center": False, "has_stroke_center": True, "has_burn_unit": False},
]


def _ensure_parent(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def generate_hospital_registry(n_hospitals: int = 10, output_path: str | Path = Config.HOSPITALS_FILE) -> pd.DataFrame:
    records = []
    for idx in range(n_hospitals):
        profile = HOSPITAL_PROFILES[idx % len(HOSPITAL_PROFILES)]
        records.append(
            {
                "hospital_id": idx + 1,
                "name": HOSPITAL_NAMES[idx % len(HOSPITAL_NAMES)],
                "latitude": round(profile["latitude"], 6),
                "longitude": round(profile["longitude"], 6),
                "total_icu_beds": profile["total_icu_beds"],
                "total_emergency_beds": profile["total_emergency_beds"],
                "total_general_beds": profile["total_general_beds"],
                "has_trauma_center": profile["has_trauma_center"],
                "has_cardiac_center": profile["has_cardiac_center"],
                "has_stroke_center": profile["has_stroke_center"],
                "has_burn_unit": profile["has_burn_unit"],
            }
        )

    df = pd.DataFrame(records)
    _ensure_parent(output_path)
    df.to_csv(output_path, index=False)
    return df


def generate_severity_data(n_samples: int = 50000, output_path: str | Path = Config.SEVERITY_DATA_FILE) -> pd.DataFrame:
    severity_classes = np.array([1, 2, 3, 4, 5])
    class_probs = np.array([0.05, 0.15, 0.35, 0.30, 0.15])
    complaints = {
        1: ["cardiac_arrest", "respiratory_failure", "massive_bleeding", "stroke"],
        2: ["chest_pain", "severe_trauma", "seizure", "sepsis"],
        3: ["abdominal_pain", "fracture", "asthma_attack", "moderate_burn"],
        4: ["laceration", "headache", "minor_fracture", "vomiting"],
        5: ["cold", "minor_rash", "sore_throat", "medication_refill"],
    }

    rows = []
    for _ in range(n_samples):
        severity = int(RNG.choice(severity_classes, p=class_probs))
        age = int(np.clip(RNG.normal(52 if severity <= 2 else 38, 22), 1, 95))
        gender = RNG.choice(["Male", "Female"])
        chronic = int(RNG.choice([0, 1], p=[0.55, 0.45])) if age >= 40 else int(RNG.choice([0, 1], p=[0.8, 0.2]))

        if severity == 1:
            hr, sbp, dbp, spo2, rr, temp = RNG.normal(132, 22), RNG.normal(78, 12), RNG.normal(48, 10), RNG.normal(83, 6), RNG.normal(30, 7), RNG.normal(38.6, 1.1)
            gcs, pain = int(np.clip(RNG.normal(7, 3), 3, 13)), int(np.clip(RNG.normal(8.5, 1.5), 0, 10))
        elif severity == 2:
            hr, sbp, dbp, spo2, rr, temp = RNG.normal(116, 18), RNG.normal(96, 14), RNG.normal(62, 10), RNG.normal(90, 4), RNG.normal(24, 5), RNG.normal(38.0, 0.9)
            gcs, pain = int(np.clip(RNG.normal(12, 2), 7, 15)), int(np.clip(RNG.normal(7.5, 1.7), 0, 10))
        elif severity == 3:
            hr, sbp, dbp, spo2, rr, temp = RNG.normal(98, 14), RNG.normal(114, 16), RNG.normal(74, 10), RNG.normal(95, 3), RNG.normal(20, 4), RNG.normal(37.6, 0.8)
            gcs, pain = int(np.clip(RNG.normal(14, 1), 10, 15)), int(np.clip(RNG.normal(6, 2), 0, 10))
        elif severity == 4:
            hr, sbp, dbp, spo2, rr, temp = RNG.normal(84, 10), RNG.normal(122, 12), RNG.normal(79, 8), RNG.normal(98, 1.5), RNG.normal(17, 2), RNG.normal(37.0, 0.5)
            gcs, pain = int(np.clip(RNG.normal(15, 0.5), 13, 15)), int(np.clip(RNG.normal(4, 2), 0, 10))
        else:
            hr, sbp, dbp, spo2, rr, temp = RNG.normal(76, 8), RNG.normal(118, 10), RNG.normal(76, 7), RNG.normal(99, 1), RNG.normal(15, 2), RNG.normal(36.8, 0.4)
            gcs, pain = 15, int(np.clip(RNG.normal(2, 1.5), 0, 10))

        if RNG.random() < 0.03:
            hr = max(25, hr + RNG.normal(0, 35))
            spo2 = np.clip(spo2 + RNG.normal(0, 8), 55, 100)
            sbp = max(50, sbp + RNG.normal(0, 18))

        complaint = RNG.choice(complaints[severity])
        rows.append(
            {
                "heart_rate": round(float(np.clip(hr, 25, 220)), 1),
                "bp_systolic": round(float(np.clip(sbp, 50, 230)), 1),
                "bp_diastolic": round(float(np.clip(dbp, 30, 150)), 1),
                "spo2": round(float(np.clip(spo2, 55, 100)), 1),
                "respiratory_rate": round(float(np.clip(rr, 6, 45)), 1),
                "temperature": round(float(np.clip(temp, 34.0, 41.5)), 1),
                "gcs_score": gcs,
                "pain_scale": pain,
                "age": age,
                "gender": gender,
                "has_chronic_condition": chronic,
                "chief_complaint": complaint,
                "severity": severity,
                "recommended_department": Config.SEVERITY_DEPARTMENT_MAP[severity],
            }
        )

    df = pd.DataFrame(rows)
    _ensure_parent(output_path)
    df.to_csv(output_path, index=False)
    return df


def generate_bed_availability_data(n_days: int = 365, n_hospitals: int = 10, output_path: str | Path = Config.BED_DATA_FILE) -> pd.DataFrame:
    hospitals = generate_hospital_registry(n_hospitals=n_hospitals, output_path=Config.HOSPITALS_FILE)
    departments = ["ICU", "Emergency", "General"]
    start = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = [start + timedelta(hours=i) for i in range(n_days * 24)]
    rows = []

    for _, hospital in hospitals.iterrows():
        dept_totals = {
            "ICU": int(hospital["total_icu_beds"]),
            "Emergency": int(hospital["total_emergency_beds"]),
            "General": int(hospital["total_general_beds"]),
        }
        for department in departments:
            total_beds = dept_totals[department]
            admissions_hist: list[int] = []
            discharges_hist: list[int] = []
            occ_hist: list[float] = []
            for ts in timestamps:
                hour = ts.hour
                month = ts.month
                day_of_week = ts.weekday()
                is_weekend = int(day_of_week >= 5)
                is_holiday = int((ts.month, ts.day) in {(1, 1), (7, 4), (12, 25)})
                season = "winter" if month in (12, 1, 2) else "spring" if month in (3, 4, 5) else "summer" if month in (6, 7, 8) else "fall"

                if department == "ICU":
                    base_occ = 0.82 + (0.05 if hour >= 20 or hour <= 5 else 0.0)
                    admissions, discharges = max(0, int(RNG.poisson(2 if is_weekend else 3))), max(0, int(RNG.poisson(1)))
                elif department == "Emergency":
                    peak = 0.12 if hour in range(10, 15) or hour in range(18, 23) else 0.0
                    base_occ = 0.68 + peak
                    admissions, discharges = max(0, int(RNG.poisson(7 if peak else 4))), max(0, int(RNG.poisson(4)))
                else:
                    base_occ = 0.72 + (0.08 if month in (12, 1, 2) else 0.0)
                    admissions, discharges = max(0, int(RNG.poisson(4 if not is_weekend else 3))), max(0, int(RNG.poisson(3)))

                if season == "winter":
                    base_occ += 0.15
                if is_weekend:
                    admissions = max(0, admissions - 1)
                if is_holiday:
                    base_occ += 0.05

                occupancy_rate = float(np.clip(base_occ + RNG.normal(0, 0.05), 0.25, 0.98))
                occupied_beds = int(np.clip(round(total_beds * occupancy_rate), 0, total_beds))
                available_beds = int(max(0, total_beds - occupied_beds))

                admissions_hist.append(admissions)
                discharges_hist.append(discharges)
                occ_hist.append(occupancy_rate)

                rows.append(
                    {
                        "timestamp": ts.isoformat(),
                        "hospital_id": int(hospital["hospital_id"]),
                        "hospital_name": hospital["name"],
                        "department": department,
                        "total_beds": total_beds,
                        "occupied_beds": occupied_beds,
                        "available_beds": available_beds,
                        "occupancy_rate": round(occupancy_rate, 4),
                        "admissions_last_1h": admissions,
                        "discharges_last_1h": discharges,
                        "hour": hour,
                        "day_of_week": day_of_week,
                        "month": month,
                        "is_weekend": is_weekend,
                        "is_holiday": is_holiday,
                        "season": season,
                        "admissions_rolling_6h": round(float(np.mean(admissions_hist[-6:])), 3),
                        "discharges_rolling_6h": round(float(np.mean(discharges_hist[-6:])), 3),
                        "occupancy_rolling_avg_24h": round(float(np.mean(occ_hist[-24:])), 4),
                    }
                )

    df = pd.DataFrame(rows)
    _ensure_parent(output_path)
    df.to_csv(output_path, index=False)
    return df


def generate_eta_data(n_samples: int = 100000, city: str = "NYC", output_path: str | Path = Config.ETA_DATA_FILE) -> pd.DataFrame:
    if city.upper() != "NYC":
        raise ValueError("Only NYC bounds are supported in this generator.")

    weather_options = ["clear", "cloudy", "rain", "snow"]
    weather_probs = [0.58, 0.23, 0.15, 0.04]
    start = datetime(2025, 1, 1)
    rows = []

    for _ in range(n_samples):
        pickup_lat = RNG.uniform(40.5, 41.0)
        pickup_lon = RNG.uniform(-74.3, -73.7)
        dropoff_lat = RNG.uniform(40.5, 41.0)
        dropoff_lon = RNG.uniform(-74.3, -73.7)
        pickup_datetime = start + timedelta(minutes=int(RNG.integers(0, 365 * 24 * 60)))
        hour = pickup_datetime.hour
        day_of_week = pickup_datetime.weekday()
        month = pickup_datetime.month
        is_weekend = int(day_of_week >= 5)
        is_rush_hour = int(hour in [7, 8, 9, 17, 18, 19])
        weather = str(RNG.choice(weather_options, p=weather_probs))
        temperature = float(RNG.normal(4, 6) if month in (12, 1, 2) else RNG.normal(26, 5) if month in (6, 7, 8) else RNG.normal(16, 6))
        traffic_level = int(np.clip(RNG.normal(8 if is_rush_hour else 4 if not is_weekend else 3, 1.5), 1, 10))

        distance_km = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        if distance_km < 0.5:
            distance_km += float(RNG.uniform(0.5, 1.2))

        if is_rush_hour:
            speed = RNG.uniform(15, 25)
        elif 0 <= hour <= 5:
            speed = RNG.uniform(40, 60)
        else:
            speed = RNG.uniform(30, 45)

        if weather == "rain":
            speed *= 0.8
        elif weather == "snow":
            speed *= 0.6

        if RNG.random() < 0.02:
            speed *= 0.45
        elif RNG.random() < 0.03:
            speed *= 1.35

        duration = (distance_km / max(speed, 5)) * 60
        duration += traffic_level * RNG.uniform(0.3, 1.0)
        duration = float(np.clip(duration + RNG.normal(0, 2), 2, 120))

        rows.append(
            {
                "pickup_latitude": round(pickup_lat, 6),
                "pickup_longitude": round(pickup_lon, 6),
                "dropoff_latitude": round(dropoff_lat, 6),
                "dropoff_longitude": round(dropoff_lon, 6),
                "pickup_datetime": pickup_datetime.isoformat(),
                "distance_km": round(distance_km, 3),
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": is_weekend,
                "is_rush_hour": is_rush_hour,
                "weather_condition": weather,
                "temperature": round(temperature, 1),
                "traffic_level": traffic_level,
                "trip_duration_minutes": round(duration, 2),
            }
        )

    df = pd.DataFrame(rows)
    _ensure_parent(output_path)
    df.to_csv(output_path, index=False)
    return df
