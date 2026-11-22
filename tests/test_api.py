from fastapi.testclient import TestClient
from src.api.main import app, load_champion_model


def test_predict_endpoint_contract() -> None:
    load_champion_model()
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_version"].startswith("v")

    sample_payload = {
        "timestamp": "2026-07-01 12:00:00",
        "ambient_temp": 78.5,
        "industrial_activity": 85.0,
        "grid_price_index": 65.5,
        "is_weekend": 0,
        "hour": 12,
        "demand_lag_1h": 275.0,
        "demand_lag_24h": 270.0,
        "demand_rolling_mean_24h": 268.5,
    }
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_demand_mwh" in body
    assert body["predicted_demand_mwh"] > 0
