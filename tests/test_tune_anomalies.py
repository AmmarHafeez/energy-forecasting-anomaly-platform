from __future__ import annotations

import json

import pandas as pd

from energy_forecasting_anomaly.evaluation.tune_anomalies import (
    AnomalyTuningConfig,
    run_anomaly_tuning,
    select_best_candidate,
    train_validation_test_split,
)


def _synthetic_energy_frame(periods: int = 100, *, with_labels: bool = True) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h")
    hour = timestamps.hour
    load = [900.0 + int(value) * 2.5 + index * 0.3 for index, value in enumerate(hour)]
    if with_labels:
        load[-8] += 220.0
        load[-4] -= 180.0

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "zone": ["DE_TEST"] * periods,
            "load_mw": load,
            "temperature_c": [6.0 + int(value) * 0.1 for value in hour],
            "wind_speed_mps": [3.0] * periods,
            "solar_radiation_wm2": [0.0 if value < 7 or value > 18 else 250.0 for value in hour],
        }
    )
    if with_labels:
        frame["is_anomaly"] = False
        frame.loc[[periods - 8, periods - 4], "is_anomaly"] = True
    return frame


def test_train_validation_test_split_is_chronological() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
            "zone": ["north"] * 10,
            "load_mw": range(10),
        }
    )

    train_frame, validation_frame, test_frame = train_validation_test_split(
        frame,
        validation_size=0.2,
        test_size=0.2,
        split_method="chronological",
    )

    assert len(train_frame) == 6
    assert len(validation_frame) == 2
    assert len(test_frame) == 2
    assert train_frame["timestamp"].max() < validation_frame["timestamp"].min()
    assert validation_frame["timestamp"].max() < test_frame["timestamp"].min()


def test_select_best_candidate_uses_macro_f1_when_labels_exist() -> None:
    candidates = [
        {"config": {"threshold": 1.5}, "macro_f1": 0.4},
        {"config": {"threshold": 3.0}, "macro_f1": 0.8},
    ]

    selected = select_best_candidate(candidates, labels_available=True)

    assert selected["config"]["threshold"] == 3.0


def test_anomaly_tuning_writes_json_with_selected_and_test_results(tmp_path) -> None:
    input_path = tmp_path / "energy_weather.csv"
    metrics_dir = tmp_path / "metrics"
    _synthetic_energy_frame(with_labels=True).to_csv(input_path, index=False)
    config = AnomalyTuningConfig(
        input_path=input_path,
        metrics_dir=metrics_dir,
        forecast_horizon=1,
        model="ridge",
        split_method="chronological",
        validation_size=0.2,
        test_size=0.2,
        methods=("residual_zscore", "robust_residual", "isolation_forest"),
        random_state=42,
    )

    result = run_anomaly_tuning(config)
    metrics_path = metrics_dir / "anomaly_tuning_h1.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metrics_path.exists()
    assert result["metrics_path"] == str(metrics_path)
    assert "selected_config_by_method" in payload
    assert "test_results_by_method" in payload
    assert set(payload["test_results_by_method"]) == {
        "residual_zscore",
        "robust_residual",
        "isolation_forest",
    }
    assert payload["selected_config_by_method"]["residual_zscore"]["threshold"] in [
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
    ]
    assert payload["selected_config_by_method"]["robust_residual"]["threshold"] in [
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
    ]
    assert payload["labels_available"] is True
    for method_result in payload["test_results_by_method"].values():
        assert "detected_count" in method_result
        assert "labeled_anomaly_count" in method_result
        assert {"precision", "recall", "macro_f1", "confusion_matrix"}.issubset(method_result)


def test_anomaly_tuning_handles_missing_labels_gracefully(tmp_path) -> None:
    input_path = tmp_path / "energy_weather.csv"
    metrics_dir = tmp_path / "metrics"
    _synthetic_energy_frame(with_labels=False).to_csv(input_path, index=False)
    config = AnomalyTuningConfig(
        input_path=input_path,
        metrics_dir=metrics_dir,
        forecast_horizon=1,
        model="ridge",
        split_method="chronological",
        validation_size=0.2,
        test_size=0.2,
        methods=("residual_zscore",),
        random_state=42,
    )

    run_anomaly_tuning(config)
    payload = json.loads((metrics_dir / "anomaly_tuning_h1.json").read_text(encoding="utf-8"))
    test_result = payload["test_results_by_method"]["residual_zscore"]

    assert payload["labels_available"] is False
    assert payload["selection_metric"] == "default_configuration"
    assert payload["selected_config_by_method"]["residual_zscore"]["threshold"] == 3.0
    assert test_result["precision"] is None
    assert test_result["confusion_matrix"] is None


def test_anomaly_tuning_zero_scale_residual_case_does_not_crash(tmp_path) -> None:
    input_path = tmp_path / "flat_energy_weather.csv"
    metrics_dir = tmp_path / "metrics"
    frame = _synthetic_energy_frame(periods=80, with_labels=True)
    frame["load_mw"] = 1000.0
    frame.to_csv(input_path, index=False)
    config = AnomalyTuningConfig(
        input_path=input_path,
        metrics_dir=metrics_dir,
        forecast_horizon=1,
        model="ridge",
        split_method="chronological",
        validation_size=0.2,
        test_size=0.2,
        methods=("robust_residual",),
        random_state=42,
    )

    run_anomaly_tuning(config)
    payload = json.loads((metrics_dir / "anomaly_tuning_h1.json").read_text(encoding="utf-8"))

    assert "robust_residual" in payload["test_results_by_method"]
