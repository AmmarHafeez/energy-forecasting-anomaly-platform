from __future__ import annotations

import json

import pandas as pd

from energy_forecasting_anomaly.api.app import (
    AnomalyRequest,
    BatchPredictionRequest,
    ForecastRequest,
)
from energy_forecasting_anomaly.api.make_payload import (
    PayloadConfig,
    build_api_payload,
    save_api_payload,
)


def _synthetic_energy_frame(periods: int = 60) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h")
    hour = timestamps.hour
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "zone": ["DE_TEST"] * periods,
            "load_mw": [900.0 + index + int(value) * 2.0 for index, value in enumerate(hour)],
            "temperature_c": [5.0 + int(value) * 0.1 for value in hour],
            "wind_speed_mps": [3.0] * periods,
            "solar_radiation_wm2": [0.0 if value < 7 or value > 18 else 250.0 for value in hour],
        }
    )


def test_save_api_payload_writes_forecast_anomaly_and_batch_requests(tmp_path) -> None:
    input_path = tmp_path / "energy_weather.csv"
    output_path = tmp_path / "reports" / "artifacts" / "sample_api_payload.json"
    _synthetic_energy_frame().to_csv(input_path, index=False)
    config = PayloadConfig(
        input_path=input_path,
        output_path=output_path,
        forecast_horizon=1,
        records=3,
    )

    result_path = save_api_payload(config)
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert result_path == output_path
    assert output_path.exists()
    assert set(payload) == {"metadata", "forecast_request", "anomaly_request", "batch_request"}
    assert len(payload["forecast_request"]["records"]) == 3
    assert len(payload["batch_request"]["records"]) == 3
    assert len(payload["anomaly_request"]["records"]) == 3
    assert payload["metadata"]["record_count"] == 3


def test_api_payload_records_are_valid_feature_dictionaries(tmp_path) -> None:
    input_path = tmp_path / "energy_weather.csv"
    output_path = tmp_path / "sample_api_payload.json"
    _synthetic_energy_frame().to_csv(input_path, index=False)
    payload = build_api_payload(
        PayloadConfig(
            input_path=input_path,
            output_path=output_path,
            forecast_horizon=1,
            records=2,
        )
    )

    forecast_records = payload["forecast_request"]["records"]
    batch_records = payload["batch_request"]["records"]
    assert all(isinstance(record["features"], dict) for record in forecast_records)
    assert all(record["features"] for record in forecast_records)
    assert all(
        isinstance(value, float)
        for record in forecast_records
        for value in record["features"].values()
    )

    ForecastRequest(**payload["forecast_request"])
    BatchPredictionRequest(**payload["batch_request"])
    AnomalyRequest(**payload["anomaly_request"])
    assert len(batch_records) == 2


def test_payload_generation_does_not_require_model_files(tmp_path) -> None:
    input_path = tmp_path / "energy_weather.csv"
    output_path = tmp_path / "sample_api_payload.json"
    _synthetic_energy_frame().to_csv(input_path, index=False)

    payload = build_api_payload(
        PayloadConfig(
            input_path=input_path,
            output_path=output_path,
            forecast_horizon=1,
            records=1,
        )
    )

    assert payload["forecast_request"]["records"]
    assert not (tmp_path / "models").exists()
