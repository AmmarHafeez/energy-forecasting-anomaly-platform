from __future__ import annotations

from fastapi.testclient import TestClient

from energy_forecasting_anomaly.api.app import ModelStore, create_app


def test_health_reports_missing_models() -> None:
    client = TestClient(create_app(ModelStore()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "forecast_model_available": False,
        "anomaly_model_available": False,
    }


def test_forecast_returns_service_error_when_model_is_missing() -> None:
    client = TestClient(create_app(ModelStore()))

    response = client.post("/forecast", json={"records": [{"features": {"load_mw": 100.0}}]})

    assert response.status_code == 503
    assert "Forecast model is unavailable" in response.json()["detail"]


def test_detect_anomaly_returns_service_error_when_detector_is_missing() -> None:
    client = TestClient(create_app(ModelStore()))

    response = client.post(
        "/detect-anomaly",
        json={"records": [{"actual_load_mw": 110.0, "predicted_load_mw": 100.0}]},
    )

    assert response.status_code == 503
    assert "Anomaly detector is unavailable" in response.json()["detail"]
