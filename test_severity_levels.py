"""Test the AI for Level 1 and Level 2 triage results."""
import requests

API = "http://localhost:8000/api/v1/dispatch"

def test_scenario(label, hr, sbp, spo2, gcs, cc, chronic=1):
    payload = {
        "triage": {
            "heart_rate": hr, "bp_systolic": sbp, "bp_diastolic": 110,
            "spo2": spo2, "respiratory_rate": 30, "temperature": 101.5,
            "gcs_score": gcs, "pain_scale": 10, "age": 70,
            "has_chronic_condition": chronic, "gender": "M",
            "chief_complaint": cc
        },
        "location": {"lat": 40.7128, "lon": -74.0060, "hour": 10, "day_of_week": 2, "month": 4}
    }
    r = requests.post(API, json=payload)
    t = r.json()['triage_result']
    print(f"--- {label} ---")
    print(f"Input: CC={cc}, HR={hr}, O2={spo2}, GCS={gcs}")
    print(f"Result: Level {t['severity_level']} ({t['severity_label']}) | Conf: {t['confidence']:.1%}")
    print(f"Dept: {t['recommended_department']}\n")

# Level 1 Target: Resuscitative (Immediate action)
test_scenario("LEVEL 1 TEST", hr=0, sbp=0, spo2=0, gcs=3, cc="cardiac_arrest")

# Level 2 Target: Emergent (High risk)
test_scenario("LEVEL 2 TEST", hr=140, sbp=190, spo2=88, gcs=10, cc="chest_pain")
