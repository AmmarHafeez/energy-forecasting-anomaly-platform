from __future__ import annotations

from energy_forecasting_anomaly.evaluation import anomaly_metrics, forecast_metrics, write_json_metrics


def test_forecast_metrics_for_perfect_predictions() -> None:
    metrics = forecast_metrics([1.0, 2.0, 4.0], [1.0, 2.0, 4.0])

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mape"] == 0.0
    assert metrics["r2"] == 1.0


def test_anomaly_metrics_with_labels() -> None:
    metrics = anomaly_metrics([0, 1, 1, 0], [0, 1, 0, 0])

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]


def test_write_json_metrics(tmp_path) -> None:
    path = write_json_metrics({"mae": 1.25}, tmp_path / "metrics.json")

    assert path.exists()
    assert '"mae": 1.25' in path.read_text(encoding="utf-8")
