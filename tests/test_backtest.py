from __future__ import annotations

import json

import pandas as pd
import pytest

from energy_forecasting_anomaly.evaluation.backtest import BacktestConfig, run_backtest


def _synthetic_energy_frame(periods: int = 80, zone: str = "DE_TEST") -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h")
    hour = timestamps.hour
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "zone": [zone] * periods,
            "load_mw": [1000.0 + int(value) * 3.0 + index for index, value in enumerate(hour)],
            "temperature_c": [5.0 + int(value) * 0.2 for value in hour],
            "wind_speed_mps": [4.0] * periods,
            "solar_radiation_wm2": [0.0 if value < 7 or value > 18 else 250.0 for value in hour],
        }
    )


def test_backtest_config_rejects_invalid_parameters(tmp_path) -> None:
    with pytest.raises(ValueError, match="initial_train_size"):
        BacktestConfig(input_path=tmp_path / "input.csv", initial_train_size=1)

    with pytest.raises(ValueError, match="fold_size"):
        BacktestConfig(input_path=tmp_path / "input.csv", fold_size=1)

    with pytest.raises(ValueError, match="step_size"):
        BacktestConfig(input_path=tmp_path / "input.csv", step_size=0)


def test_backtest_writes_metrics_json_with_fold_and_aggregate_metrics(tmp_path) -> None:
    input_path = tmp_path / "energy_weather.csv"
    metrics_dir = tmp_path / "metrics"
    _synthetic_energy_frame().to_csv(input_path, index=False)
    config = BacktestConfig(
        input_path=input_path,
        metrics_dir=metrics_dir,
        forecast_horizon=1,
        model="ridge",
        initial_train_size=25,
        fold_size=10,
        step_size=10,
        random_state=42,
    )

    result = run_backtest(config)
    metrics_path = metrics_dir / "backtest_ridge_h1.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metrics_path.exists()
    assert result["metrics_path"] == str(metrics_path)
    assert payload["fold_metrics"]
    assert payload["aggregate_metrics"]["fold_count"] == len(payload["fold_metrics"])
    assert {"mean_mae", "mean_rmse", "mean_mape", "mean_r2"}.issubset(
        payload["aggregate_metrics"]
    )


def test_backtest_rejects_multiple_zones_with_clear_error(tmp_path) -> None:
    frame = pd.concat(
        [
            _synthetic_energy_frame(periods=40, zone="north"),
            _synthetic_energy_frame(periods=40, zone="south"),
        ],
        ignore_index=True,
    )
    input_path = tmp_path / "multi_zone.csv"
    frame.to_csv(input_path, index=False)
    config = BacktestConfig(
        input_path=input_path,
        metrics_dir=tmp_path / "metrics",
        forecast_horizon=1,
        initial_train_size=20,
        fold_size=5,
        step_size=5,
    )

    with pytest.raises(ValueError, match="exactly one zone"):
        run_backtest(config)
