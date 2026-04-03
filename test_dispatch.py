"""End-to-end test: call the dispatch API and print differentiated results."""
import requests
import json
import sys

API = "http://localhost:8000/api/v1"

# Test 1: Normal vitals
payload1 = {
    "triage": {
        "heart_rate": 85, "bp_systolic": 120, "bp_diastolic": 80,
        "spo2": 98, "respiratory_rate": 16, "temperature": 37.0,
        "gcs_score": 15, "pain_scale": 3, "age": 30,
        "has_chronic_condition": 0, "gender": "M",
        "chief_complaint": "headache"
    },
    "location": {"lat": 40.7128, "lon": -74.0060, "hour": 10, "day_of_week": 2, "month": 4}
}

# Test 2: Critical vitals
payload2 = {
    "triage": {
        "heart_rate": 150, "bp_systolic": 200, "bp_diastolic": 110,
        "spo2": 85, "respiratory_rate": 30, "temperature": 39.5,
        "gcs_score": 5, "pain_scale": 10, "age": 70,
        "has_chronic_condition": 1, "gender": "F",
        "chief_complaint": "cardiac_arrest"
    },
    "location": {"lat": 40.7128, "lon": -74.0060, "hour": 18, "day_of_week": 2, "month": 4}
}

lines = []
for label, payload in [("NORMAL VITALS", payload1), ("CRITICAL VITALS", payload2)]:
    try:
        r = requests.post(f"{API}/dispatch", json=payload, timeout=10)
    except Exception as e:
        lines.append(f"CONNECTION ERROR: {e}")
        continue

    if r.status_code != 200:
        lines.append(f"ERROR {r.status_code}: {r.text}")
        continue

    data = r.json()
    t = data['triage_result']
    lines.append(f"\n=== {label} ===")
    lines.append(f"Severity: {t['severity_level']} ({t['severity_label']}) | Dept: {t['recommended_department']} | Conf: {t['confidence']:.1%}")
    lines.append(f"{'Hospital':<40} {'ETA':>8} {'Beds':>6} {'Score':>7}")
    lines.append("-" * 65)
    for rec in data['recommendations'][:10]:
        lines.append(f"{rec['name']:<40} {rec['eta_min']:>6.1f}m {rec['beds_available']:>5}  {rec['score']:>6.3f}")

with open('test_dispatch_output.txt', 'w') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
