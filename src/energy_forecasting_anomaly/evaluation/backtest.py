"""Rolling-origin backtesting for local energy load forecasting."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from energy_forecasting_anomaly.data import read_energy_weather_csv
from energy_forecasting_anomaly.evaluation.metrics import forecast_metrics, write_json_metrics
from energy_forecasting_anomaly.features import (
    TARGET_COLUMN,
    FeatureConfig,
    build_feature_frame,
    build_supervised_frame,
)
from energy_forecasting_anomaly.models import predict_forecast, train_forecaster
from energy_forecasting_anomaly.training.pipeline import sort_chronologically

LOGGER = logging.getLogger(__name__)
BacktestModel = Literal["ridge", "random_forest"]


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for rolling-origin backtesting."""

    input_path: Path
    metrics_dir: Path = Path("reports/metrics")
    forecast_horizon: int = 24
    model: BacktestModel = "ridge"
    initial_train_size: int = 1000
    fold_size: int = 168
    step_size: int = 168
    random_state: int = 42
    timestamp_column: str = "timestamp"
    zone_column: str = "zone"

    def __post_init__(self) -> None:
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive.")
        if self.initial_train_size < 2:
            raise ValueError("initial_train_size must be at least 2.")
        if self.fold_size < 2:
            raise ValueError("fold_size must be at least 2.")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive.")
        if self.model not in {"ridge", "random_forest"}:
            raise ValueError("model must be 'ridge' or 'random_forest'.")


def run_backtest(config: BacktestConfig) -> dict[str, Any]:
    """Run rolling-origin backtesting and write metrics JSON."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw_frame = read_energy_weather_csv(config.input_path)
    _validate_single_zone(raw_frame, config.zone_column)

    feature_config = FeatureConfig(forecast_horizon=config.forecast_horizon)
    feature_frame = build_feature_frame(raw_frame, feature_config)
    supervised_frame, feature_names = build_supervised_frame(feature_frame, TARGET_COLUMN)
    supervised_frame = sort_chronologically(
        supervised_frame,
        timestamp_column=config.timestamp_column,
        zone_column=config.zone_column,
    )

    fold_metrics = _run_folds(supervised_frame, feature_names, config)
    aggregate_metrics = _aggregate_metrics(fold_metrics)
    output = {
        "model": config.model,
        "forecast_horizon": config.forecast_horizon,
        "initial_train_size": config.initial_train_size,
        "fold_size": config.fold_size,
        "step_size": config.step_size,
        "random_state": config.random_state,
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate_metrics,
    }
    metrics_path = backtest_metrics_path(config)
    write_json_metrics(output, metrics_path)
    output["metrics_path"] = str(metrics_path)
    LOGGER.info("Backtest completed with %d folds", aggregate_metrics["fold_count"])
    return output


def backtest_metrics_path(config: BacktestConfig) -> Path:
    """Return the standard backtest metrics path."""

    return config.metrics_dir / f"backtest_{config.model}_h{config.forecast_horizon}.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for rolling-origin backtesting."""

    parser = argparse.ArgumentParser(description="Run rolling-origin forecast backtesting.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    parser.add_argument("--metrics-dir", type=Path, default=Path("reports/metrics"))
    parser.add_argument("--forecast-horizon", type=int, default=24)
    parser.add_argument("--model", choices=["ridge", "random_forest"], default="ridge")
    parser.add_argument("--initial-train-size", type=int, default=1000)
    parser.add_argument("--fold-size", type=int, default=168)
    parser.add_argument("--step-size", type=int, default=168)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    args = parse_args(argv)
    config = BacktestConfig(
        input_path=args.input,
        metrics_dir=args.metrics_dir,
        forecast_horizon=args.forecast_horizon,
        model=args.model,
        initial_train_size=args.initial_train_size,
        fold_size=args.fold_size,
        step_size=args.step_size,
        random_state=args.random_state,
    )
    run_backtest(config)


def _run_folds(
    supervised_frame: pd.DataFrame,
    feature_names: list[str],
    config: BacktestConfig,
) -> list[dict[str, Any]]:
    if config.initial_train_size + config.fold_size > len(supervised_frame):
        raise ValueError(
            "Not enough supervised rows for one backtest fold. "
            "Reduce initial_train_size or fold_size."
        )

    folds: list[dict[str, Any]] = []
    cutoff = config.initial_train_size
    fold_index = 0
    while cutoff + config.fold_size <= len(supervised_frame):
        train_frame = supervised_frame.iloc[:cutoff].copy()
        test_frame = supervised_frame.iloc[cutoff : cutoff + config.fold_size].copy()
        bundle = train_forecaster(
            train_frame[feature_names],
            train_frame[TARGET_COLUMN],
            model_type=config.model,
            feature_names=feature_names,
            target_column=TARGET_COLUMN,
            forecast_horizon=config.forecast_horizon,
            random_state=config.random_state,
        )
        predictions = predict_forecast(bundle, test_frame[feature_names])
        metrics = forecast_metrics(test_frame[TARGET_COLUMN].to_numpy(), predictions)
        folds.append(
            {
                "fold": fold_index,
                "train_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
                "train_start": _timestamp_text(train_frame, config.timestamp_column, first=True),
                "train_end": _timestamp_text(train_frame, config.timestamp_column, first=False),
                "test_start": _timestamp_text(test_frame, config.timestamp_column, first=True),
                "test_end": _timestamp_text(test_frame, config.timestamp_column, first=False),
                "metrics": metrics,
            }
        )
        fold_index += 1
        cutoff += config.step_size

    if not folds:
        raise ValueError("Backtest produced no folds.")
    return folds


def _aggregate_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, float | int]:
    metric_names = ("mae", "rmse", "mape", "r2")
    aggregate: dict[str, float | int] = {"fold_count": len(fold_metrics)}
    for metric_name in metric_names:
        values = np.asarray(
            [fold["metrics"][metric_name] for fold in fold_metrics],
            dtype=float,
        )
        aggregate[f"mean_{metric_name}"] = float(np.nanmean(values))
    return aggregate


def _validate_single_zone(frame: pd.DataFrame, zone_column: str) -> None:
    if zone_column not in frame.columns:
        raise ValueError(f"Missing zone column required for backtesting: {zone_column}")
    zone_count = frame[zone_column].nunique()
    if zone_count != 1:
        raise ValueError(
            "Backtesting currently supports exactly one zone. "
            f"Found {zone_count} zones."
        )


def _timestamp_text(frame: pd.DataFrame, timestamp_column: str, *, first: bool) -> str:
    value = frame.iloc[0 if first else -1][timestamp_column]
    return pd.Timestamp(value).isoformat()


if __name__ == "__main__":
    main()
