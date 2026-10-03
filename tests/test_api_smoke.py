from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "models_loaded" in payload


def test_nearby_hospitals_endpoint():
    response = client.get("/api/hospitals/nearby", params={"lat": 40.7589, "lon": -73.9851, "radius_km": 25})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)


def test_recommendation_endpoint():
    response = client.post(
        "/api/emergency/recommend",
        json={
            "location": {"latitude": 40.7589, "longitude": -73.9851},
            "patient_vitals": {
                "heart_rate": 88,
                "bp_systolic": 118,
                "bp_diastolic": 78,
                "spo2": 97,
                "respiratory_rate": 18,
                "temperature": 37.1,
                "gcs_score": 15,
                "pain_scale": 4,
                "age": 45,
                "gender": "Male",
                "has_chronic_condition": False,
                "chief_complaint": "chest_pain",
            },
            "use_ml_eta": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["severity_info"]["severity"] in {1, 2, 3, 4, 5}
    assert payload["optimal_hospital"]["hospital_id"] > 0
    assert payload["processing_time_ms"] >= 0
