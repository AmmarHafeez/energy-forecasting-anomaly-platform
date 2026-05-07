from __future__ import annotations

import json

import pandas as pd

from energy_forecasting_anomaly.evaluation.evaluate_anomalies import (
    AnomalyEvaluationConfig,
    run_anomaly_evaluation,
)


def _synthetic_energy_frame(periods: int = 90, *, with_labels: bool = True) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h")
    hour = timestamps.hour
    load = [1000.0 + int(value) * 2.0 + index * 0.4 for index, value in enumerate(hour)]
    if with_labels:
        load[-5] += 180.0

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "zone": ["DE_TEST"] * periods,
            "load_mw": load,
            "temperature_c": [4.0 + int(value) * 0.15 for value in hour],
            "wind_speed_mps": [3.5] * periods,
            "solar_radiation_wm2": [0.0 if value < 7 or value > 18 else 200.0 for value in hour],
        }
    )
    if with_labels:
        frame["is_anomaly"] = False
        frame.loc[periods - 5, "is_anomaly"] = True
    return frame


def test_anomaly_evaluation_writes_labeled_method_metrics(tmp_path) -> None:
    input_path = tmp_path / "energy_weather.csv"
    metrics_dir = tmp_path / "metrics"
    _synthetic_energy_frame(with_labels=True).to_csv(input_path, index=False)
    config = AnomalyEvaluationConfig(
        input_path=input_path,
        metrics_dir=metrics_dir,
        forecast_horizon=1,
        model="ridge",
        split_method="chronological",
        test_size=0.3,
        methods=("residual_zscore", "robust_residual", "isolation_forest"),
        random_state=42,
    )

    result = run_anomaly_evaluation(config)
    metrics_path = metrics_dir / "anomaly_comparison_h1.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metrics_path.exists()
    assert result["metrics_path"] == str(metrics_path)
    assert set(payload["methods"]) == {"residual_zscore", "robust_residual", "isolation_forest"}
    for method_result in payload["methods"].values():
        assert "config" in method_result
        assert "detected_count" in method_result
        assert method_result["labels_available"] is True
        assert "labeled_anomaly_count" in method_result
        assert {"precision", "recall", "macro_f1", "confusion_matrix"}.issubset(method_result)


def test_anomaly_evaluation_handles_missing_labels_gracefully(tmp_path) -> None:
    input_path = tmp_path / "energy_weather.csv"
    metrics_dir = tmp_path / "metrics"
    _synthetic_energy_frame(with_labels=False).to_csv(input_path, index=False)
    config = AnomalyEvaluationConfig(
        input_path=input_path,
        metrics_dir=metrics_dir,
        forecast_horizon=1,
        model="ridge",
        split_method="chronological",
        test_size=0.3,
        methods=("robust_residual",),
        random_state=42,
    )

    run_anomaly_evaluation(config)
    payload = json.loads((metrics_dir / "anomaly_comparison_h1.json").read_text(encoding="utf-8"))
    method_result = payload["methods"]["robust_residual"]

    assert method_result["labels_available"] is False
    assert "detected_count" in method_result
    assert "config" in method_result
    assert "labeled_anomaly_count" not in method_result
    assert "precision" not in method_result
