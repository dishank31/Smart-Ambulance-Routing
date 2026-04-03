"""Find the 'Critical' (Level 1) trigger combination."""
import requests

API = "http://localhost:8000/api/v1/dispatch"

def test(hr, sbp, spo2, rr, cc, chronic=1, gcs=3):
    payload = {
        "triage": {
            "heart_rate": hr, "bp_systolic": sbp, "bp_diastolic": 70,
            "spo2": spo2, "respiratory_rate": rr, "temperature": 39.5,
            "gcs_score": gcs, "pain_scale": 10, "age": 75,
            "has_chronic_condition": chronic, "gender": "M",
            "chief_complaint": cc
        },
        "location": {"lat": 40.7128, "lon": -74.0060, "hour": 10, "day_of_week": 2, "month": 4}
    }
    r = requests.post(API, json=payload)
    data = r.json()
    t = data['triage_result']
    return f"CC:{cc:<15} HR:{hr:<3} O2:{spo2:<3} GCS:{gcs:<2} | Severity: {t['severity_level']} ({t['severity_label']}) | Conf: {t['confidence']:.2%}"

tests = [
    (160, 200, 80, 35, "cardiac_arrest", 1, 3),
    (180, 220, 75, 40, "cardiac_arrest", 1, 3),
    (140, 180, 85, 30, "unresponsive", 1, 5),
    (120, 160, 90, 25, "chest_pain", 1, 10),
]

for t in tests:
    print(test(*t))
