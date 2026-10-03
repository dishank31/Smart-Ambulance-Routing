"""Smart routing decision engine for hospital recommendation."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ..data.data_generator import generate_hospital_registry
from ..utils.config import Config
from ..utils.geo_utils import filter_hospitals_by_radius, haversine_distance
from ..utils.logger import setup_logger


class DecisionEngine:
    def __init__(self, severity_model, bed_model, eta_model, hospitals_df: pd.DataFrame | None, mapbox_client):
        self.severity_model = severity_model
        self.bed_model = bed_model
        self.eta_model = eta_model
        self.hospitals_df = hospitals_df.copy() if hospitals_df is not None and not hospitals_df.empty else generate_hospital_registry()
        self.hospital_index = self.hospitals_df.set_index("hospital_id", drop=False)
        self.mapbox_client = mapbox_client
        self.logger = setup_logger("decision_engine")
        self.scoring_weights = Config.DECISION_WEIGHTS

    def infer_case_profile(self, patient_vitals: dict, severity_info: dict) -> dict:
        complaint = str(patient_vitals.get("chief_complaint", "")).lower()
        specialty = None
        if any(term in complaint for term in ("cardiac", "chest", "heart")):
            specialty = "cardiac"
        elif "stroke" in complaint:
            specialty = "stroke"
        elif any(term in complaint for term in ("trauma", "fracture", "bleed", "laceration")):
            specialty = "trauma"
        elif "burn" in complaint:
            specialty = "burn"
        elif any(term in complaint for term in ("respir", "asthma", "seizure")):
            specialty = "emergency"

        severity = int(severity_info["severity"])
        capability_weight = 0.12 if severity <= 2 else 0.08 if severity == 3 else 0.05
        return {
            "complaint": complaint,
            "specialty": specialty,
            "capability_weight": capability_weight,
            "required_department": severity_info["required_department"],
        }

    def _severity_feature_vector(self, patient_vitals: dict) -> pd.DataFrame:
        row = {feature: patient_vitals.get(feature, 0) for feature in Config.SEVERITY_FEATURES}
        row["gender"] = 1 if str(row.get("gender", "")).lower().startswith("f") else 0
        row["has_chronic_condition"] = int(bool(row.get("has_chronic_condition", False)))
        complaint = str(patient_vitals.get("chief_complaint", "other")).lower()
        row.update(
            {
                "complaint_chest_pain": int("chest" in complaint),
                "complaint_stroke": int("stroke" in complaint),
                "complaint_respiratory": int("respir" in complaint or "asthma" in complaint),
                "complaint_trauma": int("trauma" in complaint or "fracture" in complaint or "bleed" in complaint),
                "complaint_minor": int("minor" in complaint or "cold" in complaint or "headache" in complaint),
            }
        )
        return pd.DataFrame([row])

    def predict_severity(self, patient_vitals: dict) -> dict:
        hr = float(patient_vitals["heart_rate"])
        sbp = float(patient_vitals["bp_systolic"])
        spo2 = float(patient_vitals["spo2"])
        rr = float(patient_vitals["respiratory_rate"])
        gcs = int(patient_vitals["gcs_score"])
        complaint = str(patient_vitals["chief_complaint"]).lower()

        severity = None
        confidence = 0.0
        if hasattr(self.severity_model, "predict"):
            try:
                features = self._severity_feature_vector(patient_vitals)
                prediction = int(np.asarray(self.severity_model.predict(features))[0])
                severity = int(np.clip(prediction, 1, 5))
                if hasattr(self.severity_model, "predict_proba"):
                    confidence = float(np.max(np.asarray(self.severity_model.predict_proba(features))[0]))
            except Exception as exc:
                self.logger.warning("Severity model inference failed, using rules: %s", exc)

        if severity is None:
            if gcs <= 8 or spo2 < 85 or sbp < 80 or "arrest" in complaint:
                severity = 1
            elif spo2 < 90 or sbp < 95 or hr > 125 or "stroke" in complaint or "chest" in complaint:
                severity = 2
            elif rr > 24 or hr > 105 or patient_vitals["pain_scale"] >= 7:
                severity = 3
            elif patient_vitals["pain_scale"] >= 4 or float(patient_vitals["temperature"]) > 38:
                severity = 4
            else:
                severity = 5
            confidence = 0.82

        return {
            "severity": severity,
            "severity_label": Config.SEVERITY_LABELS[severity],
            "confidence": round(confidence if confidence else 0.82, 3),
            "required_department": Config.SEVERITY_DEPARTMENT_MAP[severity],
        }

    def predict_bed_availability(self, hospital_id: int, department: str, current_datetime: datetime) -> int:
        hospital = self.hospital_index.loc[hospital_id]
        totals = {
            "ICU": int(hospital.get("total_icu_beds", 10)),
            "Emergency": int(hospital.get("total_emergency_beds", 30)),
            "General": int(hospital.get("total_general_beds", 60)),
        }
        total_beds = totals.get(department, totals["General"])

        if self.bed_model is not None and hasattr(self.bed_model, "predict"):
            features = pd.DataFrame(
                [
                    {
                        "hospital_id": hospital_id,
                        "department": {"ICU": 0, "Emergency": 1, "General": 2}.get(department, 2),
                        "total_beds": total_beds,
                        "hour": current_datetime.hour,
                        "day_of_week": current_datetime.weekday(),
                        "month": current_datetime.month,
                        "is_weekend": int(current_datetime.weekday() >= 5),
                        "is_holiday": 0,
                        "occupancy_rate": 0.78,
                        "admissions_last_1h": 4,
                        "discharges_last_1h": 2,
                        "admissions_rolling_6h": 4.2,
                        "discharges_rolling_6h": 2.1,
                        "occupancy_rolling_avg_24h": 0.77,
                    }
                ]
            )
            try:
                predicted = int(round(float(np.asarray(self.bed_model.predict(features))[0])))
                return max(0, min(predicted, total_beds))
            except Exception as exc:
                self.logger.warning("Bed model inference failed, using heuristic: %s", exc)

        base_occupancy = 0.84 if department == "ICU" else 0.72 if department == "Emergency" else 0.68
        if current_datetime.month in (12, 1, 2):
            base_occupancy += 0.08
        if current_datetime.hour in range(18, 23):
            base_occupancy += 0.05
        if current_datetime.weekday() >= 5:
            base_occupancy -= 0.03
        available = int(round(total_beds * (1 - np.clip(base_occupancy + np.random.normal(0, 0.04), 0.25, 0.98))))
        return max(0, min(available, total_beds))

    def predict_eta(self, emergency_coords: tuple[float, float], hospital_coords: tuple[float, float], current_datetime: datetime, traffic_level: int = 5) -> float:
        distance = haversine_distance(*emergency_coords, *hospital_coords)
        rush = current_datetime.hour in [7, 8, 9, 17, 18, 19]

        if self.eta_model is not None and hasattr(self.eta_model, "predict"):
            features = pd.DataFrame(
                [
                    {
                        "pickup_latitude": emergency_coords[0],
                        "pickup_longitude": emergency_coords[1],
                        "dropoff_latitude": hospital_coords[0],
                        "dropoff_longitude": hospital_coords[1],
                        "distance_km": distance,
                        "hour": current_datetime.hour,
                        "day_of_week": current_datetime.weekday(),
                        "month": current_datetime.month,
                        "is_weekend": int(current_datetime.weekday() >= 5),
                        "is_rush_hour": int(rush),
                        "traffic_level": traffic_level,
                        "temperature": 22.0,
                    }
                ]
            )
            try:
                predicted = float(np.asarray(self.eta_model.predict(features))[0])
                return round(max(predicted, 1.5), 2)
            except Exception as exc:
                self.logger.warning("ETA model inference failed, using heuristic: %s", exc)

        speed = 20 if rush else 35 if 6 <= current_datetime.hour <= 22 else 45
        speed *= max(0.45, 1 - ((traffic_level - 5) * 0.06))
        return round(max((distance / max(speed, 8)) * 60, 2.0), 2)

    def filter_hospitals(self, emergency_coords: tuple[float, float], required_department: str, radius_km: float = 25, specialty: str | None = None):
        candidates = filter_hospitals_by_radius(self.hospitals_df, emergency_coords[0], emergency_coords[1], radius_km)
        if required_department == "ICU":
            candidates = candidates.loc[candidates["total_icu_beds"] > 0]
        elif required_department == "Emergency":
            candidates = candidates.loc[candidates["total_emergency_beds"] > 0]
        else:
            candidates = candidates.loc[candidates["total_general_beds"] > 0]

        specialty_columns = {
            "cardiac": "has_cardiac_center",
            "stroke": "has_stroke_center",
            "trauma": "has_trauma_center",
            "burn": "has_burn_unit",
        }
        specialty_col = specialty_columns.get(specialty)
        if specialty_col and specialty_col in candidates.columns:
            matched = candidates.loc[candidates[specialty_col] == True]
            if not matched.empty:
                candidates = matched

        return candidates.reset_index(drop=True)

    def score_hospital(self, eta_minutes: float, available_beds: int, severity_level: int, min_eta: float, max_eta: float, capability_score: float = 0.0, capability_weight: float = 0.0) -> float:
        time_score = 1.0 if max_eta == min_eta else 1 - ((eta_minutes - min_eta) / (max_eta - min_eta))
        bed_score = min(available_beds, 5) / 5.0
        weights = self.scoring_weights[int(severity_level)]
        weighted_total = weights["time"] * time_score + weights["beds"] * bed_score + capability_weight * capability_score
        return round(max(0.0, min(1.0, weighted_total)), 4)

    def capability_score(self, hospital: dict, case_profile: dict) -> float:
        specialty = case_profile.get("specialty")
        if not specialty:
            return 0.0
        capability_map = {
            "cardiac": "has_cardiac_center",
            "stroke": "has_stroke_center",
            "trauma": "has_trauma_center",
            "burn": "has_burn_unit",
        }
        flag = capability_map.get(specialty)
        if flag and hospital.get(flag):
            return 1.0
        if specialty == "emergency":
            return 0.6
        return 0.0

    def build_recommendation_breakdown(self, optimal: dict, naive: dict, case_profile: dict) -> list[str]:
        breakdown: list[str] = []
        specialty = case_profile.get("specialty")
        specialty_label = {
            "cardiac": "cardiac center",
            "stroke": "stroke center",
            "trauma": "trauma center",
            "burn": "burn unit",
            "emergency": "high-acuity emergency coverage",
        }.get(specialty)

        if specialty_label and self.capability_score(optimal, case_profile) > 0:
            breakdown.append(f"Specialty match: {specialty_label}.")

        breakdown.append(
            f"Resource position: {optimal['predicted_beds_available']} beds available with ETA {optimal['predicted_eta_minutes']:.1f} min."
        )

        if optimal["hospital_id"] != naive["hospital_id"]:
            eta_delta = float(optimal["predicted_eta_minutes"]) - float(naive["predicted_eta_minutes"])
            if eta_delta <= 0:
                breakdown.append(f"Time advantage: {abs(eta_delta):.1f} min faster than the nearest option.")
            else:
                breakdown.append(f"Accepts a {eta_delta:.1f} min ETA penalty for a better overall clinical fit.")

            bed_delta = int(optimal["predicted_beds_available"]) - int(naive["predicted_beds_available"])
            if bed_delta > 0:
                breakdown.append(f"Capacity advantage: {bed_delta} more beds than the nearest option.")
            elif bed_delta < 0:
                breakdown.append(f"Capacity tradeoff: {abs(bed_delta)} fewer beds, offset by stronger routing score.")

        breakdown.append(f"Composite routing score: {optimal['score']:.2f}.")
        return breakdown

    def _attach_route_details(self, hospital: dict, emergency_coords: tuple[float, float], override_eta: bool = False) -> None:
        try:
            directions = self.mapbox_client.get_directions(
                (emergency_coords[1], emergency_coords[0]),
                (hospital["location"]["longitude"], hospital["location"]["latitude"]),
            )
            hospital["route_geometry"] = directions.get("geometry")
            hospital["route_steps"] = directions.get("steps", [])
            if override_eta and directions.get("duration"):
                hospital["predicted_eta_minutes"] = round(float(directions["duration"]) / 60.0, 2)
        except Exception as exc:
            self.logger.warning("Route details unavailable for %s: %s", hospital["name"], exc)
            hospital.setdefault("route_geometry", None)
            hospital.setdefault("route_steps", [])

    def recommend_hospital(self, emergency_location: dict, patient_vitals: dict, current_datetime: datetime, use_ml_eta: bool = True) -> dict:
        emergency_coords = (float(emergency_location["lat"]), float(emergency_location["lon"]))
        severity_info = self.predict_severity(patient_vitals)
        case_profile = self.infer_case_profile(patient_vitals, severity_info)
        required_department = case_profile["required_department"]
        candidates = self.filter_hospitals(
            emergency_coords,
            required_department,
            Config.HOSPITAL_SEARCH_RADIUS_KM,
            specialty=case_profile["specialty"],
        )
        if candidates.empty:
            raise ValueError("No hospitals found within the configured radius.")

        hospital_results = []
        for hospital in candidates.itertuples(index=False):
            hospital_coords = (float(hospital.latitude), float(hospital.longitude))
            available_beds = self.predict_bed_availability(int(hospital.hospital_id), required_department, current_datetime)
            route_geometry = None
            if use_ml_eta:
                eta_minutes = self.predict_eta(emergency_coords, hospital_coords, current_datetime)
            else:
                try:
                    directions = self.mapbox_client.get_directions((emergency_coords[1], emergency_coords[0]), (hospital_coords[1], hospital_coords[0]))
                    eta_minutes = round(float(directions["duration"]) / 60.0, 2)
                    route_geometry = directions["geometry"]
                except Exception:
                    eta_minutes = self.predict_eta(emergency_coords, hospital_coords, current_datetime)

            hospital_results.append(
                {
                    "hospital_id": int(hospital.hospital_id),
                    "name": hospital.name,
                    "location": {"latitude": hospital_coords[0], "longitude": hospital_coords[1]},
                    "department": required_department,
                    "predicted_eta_minutes": eta_minutes,
                    "predicted_beds_available": int(available_beds),
                    "distance_km": round(float(hospital.distance_km), 2),
                    "route_geometry": route_geometry,
                    "route_steps": [],
                    "has_cardiac_center": bool(getattr(hospital, "has_cardiac_center", False)),
                    "has_stroke_center": bool(getattr(hospital, "has_stroke_center", False)),
                    "has_trauma_center": bool(getattr(hospital, "has_trauma_center", False)),
                    "has_burn_unit": bool(getattr(hospital, "has_burn_unit", False)),
                }
            )

        ranked_pool = [h for h in hospital_results if h["predicted_beds_available"] > 0] or hospital_results
        min_eta = min(h["predicted_eta_minutes"] for h in ranked_pool)
        max_eta = max(h["predicted_eta_minutes"] for h in ranked_pool)
        for hospital in ranked_pool:
            specialty_score = self.capability_score(hospital, case_profile)
            hospital["score"] = self.score_hospital(
                hospital["predicted_eta_minutes"],
                hospital["predicted_beds_available"],
                severity_info["severity"],
                min_eta,
                max_eta,
                capability_score=specialty_score,
                capability_weight=case_profile["capability_weight"],
            )
        ranked_pool.sort(key=lambda item: item["score"], reverse=True)
        optimal = ranked_pool[0]
        nearest = min(hospital_results, key=lambda item: item["predicted_eta_minutes"])

        self._attach_route_details(optimal, emergency_coords, override_eta=not use_ml_eta)
        if nearest["hospital_id"] != optimal["hospital_id"]:
            self._attach_route_details(nearest, emergency_coords, override_eta=not use_ml_eta)

        naive_choice = {
            "hospital_id": nearest["hospital_id"],
            "name": nearest["name"],
            "location": nearest["location"],
            "department": nearest["department"],
            "predicted_eta_minutes": nearest["predicted_eta_minutes"],
            "predicted_beds_available": nearest["predicted_beds_available"],
            "score": nearest.get("score", 0.0),
            "route_geometry": nearest.get("route_geometry"),
            "route_steps": nearest.get("route_steps", []),
            "why_not_optimal": self.compare_with_naive(optimal, nearest, case_profile),
        }
        recommendation_breakdown = self.build_recommendation_breakdown(optimal, naive_choice, case_profile)

        return {
            "severity_info": severity_info,
            "optimal_hospital": optimal,
            "alternative_hospitals": ranked_pool[1:4],
            "naive_choice": naive_choice,
            "recommendation_reason": self.compare_with_naive(optimal, nearest, case_profile),
            "recommendation_breakdown": recommendation_breakdown,
            "emergency_location": {"latitude": emergency_coords[0], "longitude": emergency_coords[1]},
        }

    def simulate_traffic_change(self, current_recommendation: dict, new_traffic_level: int) -> dict:
        updated = current_recommendation.copy()
        all_hospitals = [updated["optimal_hospital"], *updated.get("alternative_hospitals", [])]
        origin = updated.get("emergency_location")
        if not origin or not all_hospitals:
            return updated
        origin_lat = float(origin.get("lat", origin.get("latitude")))
        origin_lon = float(origin.get("lon", origin.get("longitude")))

        severity = updated["severity_info"]["severity"]
        tracked_hospitals = {}
        for hospital in [*all_hospitals, updated.get("naive_choice")]:
            if hospital:
                tracked_hospitals[hospital["hospital_id"]] = hospital

        etas = []
        for hospital in tracked_hospitals.values():
            eta = self.predict_eta((origin_lat, origin_lon), (hospital["location"]["latitude"], hospital["location"]["longitude"]), datetime.utcnow(), traffic_level=new_traffic_level)
            hospital["predicted_eta_minutes"] = eta
            if hospital["hospital_id"] != updated.get("naive_choice", {}).get("hospital_id") or hospital in all_hospitals:
                etas.append(eta)

        min_eta, max_eta = min(etas), max(etas)
        case_profile = self.infer_case_profile(
            {"chief_complaint": updated.get("severity_info", {}).get("severity_label", "")},
            updated["severity_info"],
        )
        for hospital in all_hospitals:
            hospital["score"] = self.score_hospital(
                hospital["predicted_eta_minutes"],
                hospital["predicted_beds_available"],
                severity,
                min_eta,
                max_eta,
                capability_score=self.capability_score(hospital, case_profile),
                capability_weight=case_profile["capability_weight"],
            )
        all_hospitals.sort(key=lambda item: item["score"], reverse=True)
        updated["optimal_hospital"] = all_hospitals[0]
        updated["alternative_hospitals"] = all_hospitals[1:4]
        if updated.get("naive_choice"):
            naive = tracked_hospitals[updated["naive_choice"]["hospital_id"]]
            updated["naive_choice"] = naive
        self._attach_route_details(updated["optimal_hospital"], (origin_lat, origin_lon))
        if updated.get("naive_choice"):
            self._attach_route_details(updated["naive_choice"], (origin_lat, origin_lon))
        updated["recommendation_reason"] = f"Recalculated with traffic level {new_traffic_level}. {self.compare_with_naive(all_hospitals[0], updated['naive_choice'], case_profile)}"
        updated["recommendation_breakdown"] = self.build_recommendation_breakdown(updated["optimal_hospital"], updated["naive_choice"], case_profile)
        return updated

    def compare_with_naive(self, optimal: dict, naive: dict, case_profile: dict | None = None) -> str:
        specialty = (case_profile or {}).get("specialty")
        specialty_label = {
            "cardiac": "cardiac center",
            "stroke": "stroke center",
            "trauma": "trauma center",
            "burn": "burn unit",
        }.get(specialty)
        if optimal["hospital_id"] == naive["hospital_id"]:
            return (
                f"The nearest hospital is also the best option because it balances ETA "
                f"({optimal['predicted_eta_minutes']:.1f} min) with available beds "
                f"({optimal['predicted_beds_available']})."
            )
        if specialty_label:
            return (
                f"The nearest hospital is {naive['name']} at {naive['predicted_eta_minutes']:.1f} minutes, "
                f"but {optimal['name']} is preferred because it is better aligned for this case "
                f"({specialty_label}) while still offering {optimal['predicted_beds_available']} beds."
            )
        if naive["predicted_beds_available"] <= 0:
            return (
                f"The nearest hospital is {naive['name']} at {naive['predicted_eta_minutes']:.1f} minutes, "
                f"but it has no available {naive['department']} beds. {optimal['name']} is recommended with "
                f"{optimal['predicted_beds_available']} available beds and a stronger overall score."
            )
        return (
            f"The nearest hospital is {naive['name']} at {naive['predicted_eta_minutes']:.1f} minutes, "
            f"but {optimal['name']} offers a better balance of travel time and bed availability "
            f"({optimal['predicted_beds_available']} beds, score {optimal['score']:.2f})."
        )
