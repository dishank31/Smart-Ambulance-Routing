"""Find the Level 1 trigger."""
import requests

API = "http://localhost:8000/api/v1/dispatch"

def test(hr, sbp, spo2, rr, cc, gcs=3):
    payload = {
        "triage": {
            "heart_rate": hr, "bp_systolic": sbp, "bp_diastolic": 50,
            "spo2": spo2, "respiratory_rate": rr, "temperature": 40.0,
            "gcs_score": gcs, "pain_scale": 10, "age": 85,
            "has_chronic_condition": 1, "gender": "F",
            "chief_complaint": cc
        },
        "location": {"lat": 40.7128, "lon": -74.0060, "hour": 10, "day_of_week": 2, "month": 4}
    }
    r = requests.post(API, json=payload)
    t = r.json()['triage_result']
    return f"CC:{cc:<15} HR:{hr} O2:{spo2} | Lvl:{t['severity_level']} ({t['severity_label']})"

scenarios = [
    (180, 240, 70, 50, "cardiacarrest"),
    (20, 50, 60, 5, "respiratoryarrest"),
    (0, 0, 0, 0, "cardiacarrest"), # Dead?
    (150, 190, 80, 30, "trauma"),
]

for s in scenarios: print(test(*s))
