import numpy as np
import pandas as pd
from src.utils.model_registry import ModelRegistry
from src.data.preprocessing import DataPreprocessor
from src.utils.config import Config
import joblib

class PredictionService:
    def __init__(self):
        self.registry = ModelRegistry()
        try:
            self.sev_model = self.registry.get_active_model('severity_stacking')
            self.eta_model = self.registry.get_active_model('eta_stacking')
            
            # LOAD ENSEMBLE MODELS FOR BEDS
            self.bed_xgboost = joblib.load('models/bed_xgboost.pkl')
            self.bed_lightgbm = joblib.load('models/bed_lightgbm.pkl')
            
            self.sev_prep = DataPreprocessor()
            self.sev_prep.load_scaler(f"{Config.SEVERITY_MODELS_DIR}/scaler_severity.joblib")
            self.sev_features = joblib.load(f"{Config.SEVERITY_MODELS_DIR}/severity_features.joblib")
            
            self.eta_prep = DataPreprocessor()
            self.eta_prep.load_scaler(f"{Config.ETA_MODELS_DIR}/scaler_eta.joblib")
            self.eta_features = joblib.load(f"{Config.ETA_MODELS_DIR}/eta_features.joblib")
            
            # Define exact features expected by bed models
            self.bed_features = ['age', 'gender', 'arrivalhour_bin', 'triage_vital_hr', 'triage_vital_sbp', 'triage_vital_temp']
            
            self._is_ready = True
        except Exception as e:
            print(f"Prediction Service Setup Issue: {e}")
            self._is_ready = False
            self.sev_features = []
            self.bed_features = []
            
    def is_ready(self):
        return getattr(self, '_is_ready', False)

    def predict_severity(self, req_dict):
        # 1. Start with an array of zeros corresponding to all 970 features
        row = np.zeros(len(self.sev_features))
        
        # 2. Extract input values
        # Numeric standard columns you collected
        numeric_maps = {
            'age': req_dict['age'],
            'triage_vital_hr': req_dict['heart_rate'],
            'triage_vital_sbp': req_dict['bp_systolic'],
            'triage_vital_dbp': req_dict['bp_diastolic'],
            'triage_vital_rr': req_dict['respiratory_rate'],
            'triage_vital_o2sat': req_dict['spo2'],
            'triage_vital_temp': req_dict['temperature'],
            'n_admissions': req_dict.get('has_chronic_condition', 0) * 2
        }
        
        for k, v in numeric_maps.items():
            if k in self.sev_features:
                idx = self.sev_features.index(k)
                row[idx] = v
                
        # 3. Handle Gender
        g = req_dict.get('gender', 'M')
        if f'gender_{g}' in self.sev_features:
            row[self.sev_features.index(f'gender_{g}')] = 1
            
        # 4. Handle Chief Complaint (Standardize: remove ALL spaces and underscores)
        cc_raw = req_dict.get('chief_complaint', '').lower().replace(' ', '').replace('_', '')
        cc_col = f'cc_{cc_raw}'
        if cc_col in self.sev_features:
            row[self.sev_features.index(cc_col)] = 1
            
        # 5. Handle Chronic Condition Mapping
        if req_dict.get('has_chronic_condition', 0):
            # Map general chronic flag to common model indicators
            chronic_indicators = ['history_htn_on_meds', 'history_diabetes', 'history_asthma', 'history_cad']
            for ind in chronic_indicators:
                if ind in self.sev_features:
                    row[self.sev_features.index(ind)] = 1
        
        X = np.array([row])
        X_scaled = self.sev_prep.scaler.transform(X)
        
        pred_idx = self.sev_model.predict(X_scaled)[0]
        # Your target was `esi` 1-5, which was mapped to 0-4 for xgboost
        severity = int(pred_idx) + 1
        
        probas = self.sev_model.predict_proba(X_scaled)[0]
        confidence = float(np.max(probas))
        
        return {
            'severity_level': severity,
            'severity_label': Config.SEVERITY_LABELS.get(severity, "Unknown"),
            'recommended_department': Config.SEVERITY_DEPARTMENT_MAP.get(severity, 'Emergency'),
            'confidence': confidence,
            'emoji': Config.SEVERITY_EMOJI.get(severity, "ℹ️")
        }
        
    def batch_predict_eta(self, hospital_coords, loc_context):
        """
        Predict ETA using the ML model as a baseline, then scale
        proportionally by actual distance so each hospital gets a
        different, realistic ETA.
        """
        if not hospital_coords: return []
        
        from src.utils.geo_utils import haversine_vectorized

        plat, plon = loc_context['lat'], loc_context['lon']
        hlats = np.array([c[0] for c in hospital_coords])
        hlons = np.array([c[1] for c in hospital_coords])
        
        hour = loc_context.get('hour', 12)
        dow  = loc_context.get('day_of_week', 2)
        month = loc_context.get('month', 4)

        # Actual haversine distances (km) from incident to each hospital
        dists = haversine_vectorized(
            np.full_like(hlats, plat), np.full_like(hlons, plon),
            hlats, hlons
        )

        # Get ML model's base prediction for the median-distance hospital
        try:
            median_dist = float(np.median(dists))
            X_base = np.zeros((1, len(self.eta_features)))
            feature_map = {
                'passenger_count': 1,
                'pickup_latitude': plat,
                'pickup_longitude': plon,
                'dropoff_latitude': float(np.median(hlats)),
                'dropoff_longitude': float(np.median(hlons)),
                'hour': hour,
                'day_of_week': dow,
                'month': month,
                'distance_km': median_dist,
            }
            for k, v in feature_map.items():
                if k in self.eta_features:
                    X_base[0, self.eta_features.index(k)] = v
            X_scaled = self.eta_prep.scaler.transform(X_base)
            base_eta = float(self.eta_model.predict(X_scaled)[0])
        except Exception:
            base_eta = 12.0  # sensible fallback

        base_eta = max(base_eta, 3.0)  # floor at 3 min

        # Scale ETA per hospital proportional to its distance
        # Use: ETA_i = base_eta * (dist_i / median_dist)
        median_dist = float(np.median(dists)) if float(np.median(dists)) > 0.1 else 1.0
        etas = base_eta * (dists / median_dist)

        # Time-of-day modifiers for realism
        rush_factor = 1.35 if ((7 <= hour <= 9) or (17 <= hour <= 19)) else 1.0
        weekend_factor = 0.85 if dow >= 5 else 1.0
        etas = etas * rush_factor * weekend_factor

        # Clamp to reasonable range (2 min – 60 min)
        etas = np.clip(etas, 2.0, 60.0)
        return list(np.round(etas, 1))

    def batch_predict_beds(self, hospital_data_list, loc_context, triage_data, department="General"):
        """
        Predict available beds using XGBoost + LightGBM ensemble.
        Each hospital gets a different bed count based on its own capacity
        for the specific department and the ML model's patient-stress prediction.
        """
        if not self.is_ready():
            return [10] * len(hospital_data_list)

        age   = triage_data.get('age', 45)
        gender = 1 if triage_data.get('gender') == 'F' else 0
        hour  = loc_context.get('hour', 12)
        hr    = triage_data.get('heart_rate', 80)
        sbp   = triage_data.get('bp_systolic', 120)
        temp  = triage_data.get('temperature', 98.6)

        input_data = [age, gender, hour, hr, sbp, temp]
        X = np.array([input_data])

        try:
            pred_xgb = float(self.bed_xgboost.predict(X)[0])
            pred_lgb = float(self.bed_lightgbm.predict(X)[0])
            ml_pred = (pred_xgb + pred_lgb) / 2.0
            occupancy_rate = max(0.0, min(1.0, 1.0 - (ml_pred / 120.0)))
        except Exception as e:
            print(f"Bed Ensemble Prediction Error: {e}")
            occupancy_rate = 0.3

        if 7 <= hour <= 10 or 17 <= hour <= 21:
            occupancy_rate = min(1.0, occupancy_rate + 0.15)

        # Map department name to capacity key
        dept_map = {
            "ICU": "icu_total",
            "Emergency": "emergency_total",
            "General": "general_total"
        }
        cap_key = dept_map.get(department, "general_total")

        results = []
        for h in hospital_data_list:
            total_cap = h.get(cap_key, h.get('general_total', 100))

            np.random.seed(int(h.get('id', 1)) * 137 + hour)
            jitter = np.random.uniform(0.85, 1.15)
            avail = total_cap * (1.0 - occupancy_rate) * jitter

            avail = max(1, int(round(avail)))
            avail = min(avail, int(total_cap))
            results.append(avail)

        return results

