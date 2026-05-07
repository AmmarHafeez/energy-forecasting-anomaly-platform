"""Command-line training pipeline for local energy forecasting."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from energy_forecasting_anomaly.anomaly import ResidualZScoreDetector, save_anomaly_detector
from energy_forecasting_anomaly.data import read_energy_weather_csv
from energy_forecasting_anomaly.evaluation import (
    anomaly_metrics,
    forecast_metrics,
    write_json_metrics,
)
from energy_forecasting_anomaly.features import (
    TARGET_COLUMN,
    FeatureConfig,
    build_feature_frame,
    build_supervised_frame,
)
from energy_forecasting_anomaly.models import predict_forecast, save_model_bundle, train_forecaster

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train local energy forecasting baselines.")
    parser.add_argument("--input", default="data/raw/energy_weather.csv", help="Input CSV path.")
    parser.add_argument("--models-dir", default="models", help="Directory for model files.")
    parser.add_argument("--metrics-dir", default="reports/metrics", help="Directory for metrics JSON.")
    parser.add_argument("--forecast-horizon", type=int, default=24, help="Forecast horizon in rows.")
    parser.add_argument(
        "--model",
        choices=["ridge", "random_forest"],
        default="ridge",
        help="Forecasting model type.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Ordered test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for supported models.")
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> dict[str, Path]:
    """Run local training, evaluation, and artifact writing."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw_frame = read_energy_weather_csv(args.input)
    feature_config = FeatureConfig(forecast_horizon=args.forecast_horizon)
    feature_frame = build_feature_frame(raw_frame, feature_config)
    supervised_frame, feature_names = build_supervised_frame(feature_frame, TARGET_COLUMN)
    train_frame, test_frame = temporal_train_test_split(supervised_frame, args.test_size)

    forecast_bundle = train_forecaster(
        train_frame[feature_names],
        train_frame[TARGET_COLUMN],
        model_type=args.model,
        feature_names=feature_names,
        target_column=TARGET_COLUMN,
        forecast_horizon=args.forecast_horizon,
        random_state=args.random_state,
    )

    models_dir = Path(args.models_dir)
    metrics_dir = Path(args.metrics_dir)
    forecast_model_path = models_dir / f"forecast_{args.model}_h{args.forecast_horizon}.joblib"
    forecast_metrics_path = metrics_dir / f"forecast_{args.model}_h{args.forecast_horizon}.json"
    anomaly_model_path = models_dir / "anomaly_residual_zscore.joblib"
    anomaly_metrics_path = metrics_dir / "anomaly_residual_zscore.json"

    save_model_bundle(forecast_bundle, forecast_model_path)

    train_predictions = predict_forecast(forecast_bundle, train_frame[feature_names])
    test_predictions = predict_forecast(forecast_bundle, test_frame[feature_names])

    forecast_metric_values = forecast_metrics(test_frame[TARGET_COLUMN].to_numpy(), test_predictions)
    write_json_metrics(
        {
            "model": args.model,
            "forecast_horizon": args.forecast_horizon,
            "test_size": args.test_size,
            "row_counts": {"train": len(train_frame), "test": len(test_frame)},
            "metrics": forecast_metric_values,
        },
        forecast_metrics_path,
    )

    detector = ResidualZScoreDetector(threshold=3.0).fit(
        train_frame[TARGET_COLUMN],
        train_predictions,
    )
    save_anomaly_detector(detector, anomaly_model_path)

    scored_anomalies = detector.score(test_frame[TARGET_COLUMN], test_predictions)
    anomaly_metric_values: dict[str, Any] = {
        "records_scored": int(len(scored_anomalies)),
        "anomalies_detected": int(scored_anomalies["is_anomaly"].sum()),
        "threshold": detector.threshold,
    }

    label_column = _find_label_column(test_frame)
    if label_column is not None:
        anomaly_metric_values["labeled_metrics"] = anomaly_metrics(
            test_frame[label_column].astype(int).to_numpy(),
            scored_anomalies["is_anomaly"].astype(int).to_numpy(),
        )

    write_json_metrics(anomaly_metric_values, anomaly_metrics_path)
    LOGGER.info("Training pipeline completed")
    return {
        "forecast_model": forecast_model_path,
        "forecast_metrics": forecast_metrics_path,
        "anomaly_model": anomaly_model_path,
        "anomaly_metrics": anomaly_metrics_path,
    }


def temporal_train_test_split(
    frame: pd.DataFrame,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows in timestamp order without shuffling."""

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    test_count = max(1, int(round(len(frame) * test_size)))
    train_count = len(frame) - test_count
    if train_count < 2:
        raise ValueError("Not enough rows for a train/test split.")
    train_frame = frame.iloc[:train_count].copy()
    test_frame = frame.iloc[train_count:].copy()
    return train_frame, test_frame


def _find_label_column(frame: pd.DataFrame) -> str | None:
    for column in ("is_anomaly", "anomaly_label", "label"):
        if column in frame.columns:
            return column
    return None


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    args = parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
