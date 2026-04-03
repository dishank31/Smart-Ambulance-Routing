"""Verify confidence variance across different vital combinations."""
import requests
import json

API = "http://localhost:8000/api/v1/dispatch"

def test_vitals(hr, sbp, spo2, rr, cc="headache", chronic=0):
    payload = {
        "triage": {
            "heart_rate": hr, "bp_systolic": sbp, "bp_diastolic": 80,
            "spo2": spo2, "respiratory_rate": rr, "temperature": 37.0,
            "gcs_score": 15, "pain_scale": 3, "age": 30,
            "has_chronic_condition": chronic, "gender": "M",
            "chief_complaint": cc
        },
        "location": {"lat": 40.7128, "lon": -74.0060, "hour": 10, "day_of_week": 2, "month": 4}
    }
    r = requests.post(API, json=payload)
    data = r.json()
    t = data['triage_result']
    return f"HR:{hr:<3} SBP:{sbp:<3} O2:{spo2:<3} Chronic:{chronic} | Severity: {t['severity_level']} | Conf: {t['confidence']:.2%}"

scenarios = [
    (70, 120, 98, 16, "headache", 0),
    (85, 130, 97, 18, "headache", 0),
    (100, 140, 95, 20, "chest_pain", 1),
    (130, 160, 90, 26, "chest_pain", 1),
    (160, 190, 85, 30, "cardiac_arrest", 1),
]

for s in scenarios:
    print(test_vitals(*s))
