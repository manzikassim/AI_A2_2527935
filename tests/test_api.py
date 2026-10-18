from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info_has_verification_code():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["verification_code"] == "AI-A2-7935"


def test_predict_valid(valid_record):
    r = client.post("/predict", json=valid_record)
    assert r.status_code == 200
    body = r.json()
    assert body["verification_code"] == "AI-A2-7935"
    assert {"yield_estimate_tons", "high_risk", "risk_probability", "cluster"} <= body.keys()


def test_predict_range_error_returns_4xx(valid_record):
    valid_record["rainfall_mm"] = 5000
    assert client.post("/predict", json=valid_record).status_code == 422


def test_predict_wrong_type_returns_4xx(valid_record):
    valid_record["avg_temp_c"] = "hot"
    assert client.post("/predict", json=valid_record).status_code == 422


def test_predict_missing_field_returns_4xx(valid_record):
    del valid_record["season"]
    assert client.post("/predict", json=valid_record).status_code == 422
