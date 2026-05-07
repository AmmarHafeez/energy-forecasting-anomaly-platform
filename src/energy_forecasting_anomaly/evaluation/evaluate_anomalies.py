"""Compare anomaly detection methods on forecast residuals and features."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from energy_forecasting_anomaly.anomaly import (
    IsolationForestDetector,
    ResidualZScoreDetector,
    RobustResidualDetector,
)
from energy_forecasting_anomaly.data import read_energy_weather_csv
from energy_forecasting_anomaly.evaluation.metrics import anomaly_metrics, write_json_metrics
from energy_forecasting_anomaly.features import (
    TARGET_COLUMN,
    FeatureConfig,
    build_feature_frame,
    build_supervised_frame,
)
from energy_forecasting_anomaly.models import predict_forecast, train_forecaster
from energy_forecasting_anomaly.training.pipeline import SplitMethod, train_test_split_frame

LOGGER = logging.getLogger(__name__)
AnomalyMethod = Literal["residual_zscore", "robust_residual", "isolation_forest"]
ForecastModel = Literal["ridge", "random_forest"]


@dataclass(frozen=True)
class AnomalyEvaluationConfig:
    """Configuration for anomaly method comparison."""

    input_path: Path
    metrics_dir: Path = Path("reports/metrics")
    forecast_horizon: int = 24
    model: ForecastModel = "ridge"
    split_method: SplitMethod = "chronological"
    test_size: float = 0.2
    methods: tuple[AnomalyMethod, ...] = (
        "residual_zscore",
        "robust_residual",
        "isolation_forest",
    )
    random_state: int = 42
    residual_zscore_threshold: float = 3.0
    robust_residual_threshold: float = 3.5
    isolation_contamination: float | str = "auto"

    def __post_init__(self) -> None:
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive.")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1.")
        if self.model not in {"ridge", "random_forest"}:
            raise ValueError("model must be 'ridge' or 'random_forest'.")
        if self.split_method not in {"chronological", "random"}:
            raise ValueError("split_method must be 'chronological' or 'random'.")
        invalid_methods = [method for method in self.methods if method not in _allowed_methods()]
        if invalid_methods:
            raise ValueError(f"Unsupported anomaly methods: {', '.join(invalid_methods)}.")
        if not self.methods:
            raise ValueError("At least one anomaly method is required.")


def run_anomaly_evaluation(config: AnomalyEvaluationConfig) -> dict[str, Any]:
    """Run anomaly comparison and write metrics JSON."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw_frame = read_energy_weather_csv(config.input_path)
    feature_config = FeatureConfig(forecast_horizon=config.forecast_horizon)
    feature_frame = build_feature_frame(raw_frame, feature_config)
    supervised_frame, feature_names = build_supervised_frame(feature_frame, TARGET_COLUMN)
    train_frame, test_frame = train_test_split_frame(
        supervised_frame,
        config.test_size,
        split_method=config.split_method,
        random_state=config.random_state,
    )

    forecast_bundle = train_forecaster(
        train_frame[feature_names],
        train_frame[TARGET_COLUMN],
        model_type=config.model,
        feature_names=feature_names,
        target_column=TARGET_COLUMN,
        forecast_horizon=config.forecast_horizon,
        random_state=config.random_state,
    )
    train_predictions = predict_forecast(forecast_bundle, train_frame[feature_names])
    test_predictions = predict_forecast(forecast_bundle, test_frame[feature_names])

    method_results = {
        method: _evaluate_method(
            method,
            train_frame,
            test_frame,
            feature_names,
            train_predictions,
            test_predictions,
            config,
        )
        for method in config.methods
    }
    output = {
        "model": config.model,
        "forecast_horizon": config.forecast_horizon,
        "split_method": config.split_method,
        "test_size": config.test_size,
        "random_state": config.random_state,
        "row_counts": {"train": len(train_frame), "test": len(test_frame)},
        "methods": method_results,
    }
    metrics_path = anomaly_comparison_metrics_path(config)
    write_json_metrics(output, metrics_path)
    output["metrics_path"] = str(metrics_path)
    LOGGER.info("Anomaly comparison completed for %d methods", len(config.methods))
    return output


def anomaly_comparison_metrics_path(config: AnomalyEvaluationConfig) -> Path:
    """Return the standard anomaly comparison metrics path."""

    return config.metrics_dir / f"anomaly_comparison_h{config.forecast_horizon}.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for anomaly comparison."""

    parser = argparse.ArgumentParser(description="Compare anomaly detection methods.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    parser.add_argument("--metrics-dir", type=Path, default=Path("reports/metrics"))
    parser.add_argument("--forecast-horizon", type=int, default=24)
    parser.add_argument("--model", choices=["ridge", "random_forest"], default="ridge")
    parser.add_argument(
        "--split-method",
        choices=["chronological", "random"],
        default="chronological",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(_allowed_methods()),
        default=["residual_zscore", "robust_residual", "isolation_forest"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--residual-zscore-threshold", type=float, default=3.0)
    parser.add_argument("--robust-residual-threshold", type=float, default=3.5)
    parser.add_argument("--isolation-contamination", default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    args = parse_args(argv)
    config = AnomalyEvaluationConfig(
        input_path=args.input,
        metrics_dir=args.metrics_dir,
        forecast_horizon=args.forecast_horizon,
        model=args.model,
        split_method=args.split_method,
        test_size=args.test_size,
        methods=tuple(args.methods),
        random_state=args.random_state,
        residual_zscore_threshold=args.residual_zscore_threshold,
        robust_residual_threshold=args.robust_residual_threshold,
        isolation_contamination=_parse_contamination(args.isolation_contamination),
    )
    run_anomaly_evaluation(config)


def _evaluate_method(
    method: AnomalyMethod,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_names: list[str],
    train_predictions: Any,
    test_predictions: Any,
    config: AnomalyEvaluationConfig,
) -> dict[str, Any]:
    if method == "residual_zscore":
        detector = ResidualZScoreDetector(threshold=config.residual_zscore_threshold).fit(
            train_frame[TARGET_COLUMN],
            train_predictions,
        )
        scores = detector.score(test_frame[TARGET_COLUMN], test_predictions)
        method_config: dict[str, Any] = {"threshold": detector.threshold}
    elif method == "robust_residual":
        detector = RobustResidualDetector(threshold=config.robust_residual_threshold).fit(
            train_frame[TARGET_COLUMN],
            train_predictions,
        )
        scores = detector.score(test_frame[TARGET_COLUMN], test_predictions)
        method_config = {"threshold": detector.threshold}
    else:
        detector = IsolationForestDetector(
            contamination=config.isolation_contamination,
            random_state=config.random_state,
        ).fit(train_frame[feature_names], feature_names)
        scores = detector.score(test_frame[feature_names])
        method_config = {
            "contamination": config.isolation_contamination,
            "feature_count": len(feature_names),
        }

    return _build_method_result(method, method_config, scores, test_frame)


def _build_method_result(
    method: AnomalyMethod,
    method_config: dict[str, Any],
    scores: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> dict[str, Any]:
    predictions = scores["is_anomaly"].astype(int).to_numpy()
    result: dict[str, Any] = {
        "method": method,
        "config": method_config,
        "detected_count": int(predictions.sum()),
        "labels_available": "is_anomaly" in test_frame.columns,
    }

    if "is_anomaly" in test_frame.columns:
        labels = test_frame["is_anomaly"].astype(bool).astype(int).to_numpy()
        result["labeled_anomaly_count"] = int(labels.sum())
        result.update(anomaly_metrics(labels, predictions))

    return result


def _allowed_methods() -> set[str]:
    return {"residual_zscore", "robust_residual", "isolation_forest"}


def _parse_contamination(value: str) -> float | str:
    if value == "auto":
        return value
    contamination = float(value)
    if not 0.0 < contamination <= 0.5:
        raise ValueError("isolation-contamination must be 'auto' or a value in (0, 0.5].")
    return contamination


if __name__ == "__main__":
    main()
