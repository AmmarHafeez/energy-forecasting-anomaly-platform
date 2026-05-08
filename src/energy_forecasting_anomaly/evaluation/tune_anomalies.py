"""Tune anomaly detector thresholds on validation data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
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
from energy_forecasting_anomaly.training.pipeline import SplitMethod, sort_chronologically

LOGGER = logging.getLogger(__name__)
AnomalyMethod = Literal["residual_zscore", "robust_residual", "isolation_forest"]
ForecastModel = Literal["ridge", "random_forest"]
RESIDUAL_ZSCORE_GRID: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
ROBUST_RESIDUAL_GRID: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
ISOLATION_CONTAMINATION_GRID: tuple[str | float, ...] = ("auto", 0.01, 0.02, 0.05, 0.1)


@dataclass(frozen=True)
class AnomalyTuningConfig:
    """Configuration for anomaly threshold calibration."""

    input_path: Path
    metrics_dir: Path = Path("reports/metrics")
    forecast_horizon: int = 24
    model: ForecastModel = "ridge"
    split_method: SplitMethod = "chronological"
    validation_size: float = 0.2
    test_size: float = 0.2
    methods: tuple[AnomalyMethod, ...] = (
        "residual_zscore",
        "robust_residual",
        "isolation_forest",
    )
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive.")
        if self.model not in {"ridge", "random_forest"}:
            raise ValueError("model must be 'ridge' or 'random_forest'.")
        if self.split_method not in {"chronological", "random"}:
            raise ValueError("split_method must be 'chronological' or 'random'.")
        if not 0.0 < self.validation_size < 1.0:
            raise ValueError("validation_size must be between 0 and 1.")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1.")
        if self.validation_size + self.test_size >= 1.0:
            raise ValueError("validation_size and test_size must leave training rows.")
        invalid_methods = [method for method in self.methods if method not in _allowed_methods()]
        if invalid_methods:
            raise ValueError(f"Unsupported anomaly methods: {', '.join(invalid_methods)}.")
        if not self.methods:
            raise ValueError("At least one anomaly method is required.")


def run_anomaly_tuning(config: AnomalyTuningConfig) -> dict[str, Any]:
    """Tune anomaly methods on validation data and evaluate on test data."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw_frame = read_energy_weather_csv(config.input_path)
    _validate_single_zone(raw_frame)
    feature_config = FeatureConfig(forecast_horizon=config.forecast_horizon)
    feature_frame = build_feature_frame(raw_frame, feature_config)
    supervised_frame, feature_names = build_supervised_frame(feature_frame, TARGET_COLUMN)
    train_frame, validation_frame, test_frame = train_validation_test_split(
        supervised_frame,
        validation_size=config.validation_size,
        test_size=config.test_size,
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
    validation_predictions = predict_forecast(forecast_bundle, validation_frame[feature_names])
    test_predictions = predict_forecast(forecast_bundle, test_frame[feature_names])

    labels_available = (
        "is_anomaly" in validation_frame.columns and "is_anomaly" in test_frame.columns
    )
    method_outputs = {
        method: _tune_method(
            method,
            train_frame,
            validation_frame,
            test_frame,
            feature_names,
            train_predictions,
            validation_predictions,
            test_predictions,
            config,
            labels_available,
        )
        for method in config.methods
    }
    output = {
        "configuration": _configuration_payload(config),
        "row_counts": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        "labels_available": labels_available,
        "selection_metric": "macro_f1" if labels_available else "default_configuration",
        "validation_results": {
            method: result["validation_results"] for method, result in method_outputs.items()
        },
        "selected_config_by_method": {
            method: result["selected_config"] for method, result in method_outputs.items()
        },
        "test_results_by_method": {
            method: result["test_results"] for method, result in method_outputs.items()
        },
    }
    metrics_path = anomaly_tuning_metrics_path(config)
    write_json_metrics(output, metrics_path)
    output["metrics_path"] = str(metrics_path)
    LOGGER.info("Anomaly tuning completed for %d methods", len(config.methods))
    return output


def train_validation_test_split(
    frame: pd.DataFrame,
    *,
    validation_size: float,
    test_size: float,
    split_method: SplitMethod = "chronological",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows into train, validation, and test sets."""

    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size and test_size must leave training rows.")

    if split_method == "chronological":
        ordered = sort_chronologically(frame)
    elif split_method == "random":
        ordered = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    else:
        raise ValueError("split_method must be 'chronological' or 'random'.")

    validation_count = max(1, int(round(len(ordered) * validation_size)))
    test_count = max(1, int(round(len(ordered) * test_size)))
    train_count = len(ordered) - validation_count - test_count
    if train_count < 2:
        raise ValueError("Not enough rows for train/validation/test split.")

    train_frame = ordered.iloc[:train_count].copy()
    validation_frame = ordered.iloc[train_count : train_count + validation_count].copy()
    test_frame = ordered.iloc[train_count + validation_count :].copy()
    if validation_frame.empty or test_frame.empty:
        raise ValueError("Validation and test splits must not be empty.")
    return train_frame, validation_frame, test_frame


def select_best_candidate(
    candidates: list[dict[str, Any]],
    labels_available: bool,
) -> dict[str, Any]:
    """Select the best validation candidate."""

    if not candidates:
        raise ValueError("At least one candidate is required.")
    if not labels_available:
        return candidates[0]
    return max(candidates, key=lambda candidate: candidate.get("macro_f1", -1.0))


def anomaly_tuning_metrics_path(config: AnomalyTuningConfig) -> Path:
    """Return the standard anomaly tuning metrics path."""

    return config.metrics_dir / f"anomaly_tuning_h{config.forecast_horizon}.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for anomaly tuning."""

    parser = argparse.ArgumentParser(description="Tune anomaly detector thresholds.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    parser.add_argument("--metrics-dir", type=Path, default=Path("reports/metrics"))
    parser.add_argument("--forecast-horizon", type=int, default=24)
    parser.add_argument("--model", choices=["ridge", "random_forest"], default="ridge")
    parser.add_argument(
        "--split-method",
        choices=["chronological", "random"],
        default="chronological",
    )
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(_allowed_methods()),
        default=["residual_zscore", "robust_residual", "isolation_forest"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    args = parse_args(argv)
    config = AnomalyTuningConfig(
        input_path=args.input,
        metrics_dir=args.metrics_dir,
        forecast_horizon=args.forecast_horizon,
        model=args.model,
        split_method=args.split_method,
        validation_size=args.validation_size,
        test_size=args.test_size,
        methods=tuple(args.methods),
        random_state=args.random_state,
    )
    run_anomaly_tuning(config)


def _tune_method(
    method: AnomalyMethod,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_names: list[str],
    train_predictions: Any,
    validation_predictions: Any,
    test_predictions: Any,
    config: AnomalyTuningConfig,
    labels_available: bool,
) -> dict[str, Any]:
    if method == "residual_zscore":
        return _tune_residual_method(
            method,
            RESIDUAL_ZSCORE_GRID,
            train_frame,
            validation_frame,
            test_frame,
            train_predictions,
            validation_predictions,
            test_predictions,
            labels_available,
        )
    if method == "robust_residual":
        return _tune_residual_method(
            method,
            ROBUST_RESIDUAL_GRID,
            train_frame,
            validation_frame,
            test_frame,
            train_predictions,
            validation_predictions,
            test_predictions,
            labels_available,
        )
    return _tune_isolation_forest(
        train_frame,
        validation_frame,
        test_frame,
        feature_names,
        config,
        labels_available,
    )


def _tune_residual_method(
    method: AnomalyMethod,
    threshold_grid: tuple[float, ...],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    train_predictions: Any,
    validation_predictions: Any,
    test_predictions: Any,
    labels_available: bool,
) -> dict[str, Any]:
    validation_results = [
        _score_residual_candidate(
            method,
            threshold,
            train_frame,
            validation_frame,
            train_predictions,
            validation_predictions,
            labels_available,
        )
        for threshold in threshold_grid
    ]
    selected_threshold = (
        float(select_best_candidate(validation_results, labels_available)["config"]["threshold"])
        if labels_available
        else _default_threshold(method)
    )
    test_scores = _score_residuals_with_threshold(
        method,
        selected_threshold,
        train_frame,
        test_frame,
        train_predictions,
        test_predictions,
    )
    return {
        "validation_results": validation_results,
        "selected_config": {
            "threshold": selected_threshold,
            "label_based_selection": labels_available,
        },
        "test_results": _result_payload(test_scores, test_frame, labels_available),
    }


def _tune_isolation_forest(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_names: list[str],
    config: AnomalyTuningConfig,
    labels_available: bool,
) -> dict[str, Any]:
    validation_results = [
        _score_isolation_candidate(
            contamination,
            train_frame,
            validation_frame,
            feature_names,
            config.random_state,
            labels_available,
        )
        for contamination in ISOLATION_CONTAMINATION_GRID
    ]
    selected = select_best_candidate(validation_results, labels_available)
    selected_contamination = selected["config"]["contamination"]
    detector = IsolationForestDetector(
        contamination=selected_contamination,
        random_state=config.random_state,
    ).fit(train_frame[feature_names], feature_names)
    test_scores = detector.score(test_frame[feature_names])
    return {
        "validation_results": validation_results,
        "selected_config": {
            "contamination": selected_contamination,
            "label_based_selection": labels_available,
        },
        "test_results": _result_payload(test_scores, test_frame, labels_available),
    }


def _score_residual_candidate(
    method: AnomalyMethod,
    threshold: float,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    train_predictions: Any,
    validation_predictions: Any,
    labels_available: bool,
) -> dict[str, Any]:
    scores = _score_residuals_with_threshold(
        method,
        threshold,
        train_frame,
        validation_frame,
        train_predictions,
        validation_predictions,
    )
    payload = _result_payload(scores, validation_frame, labels_available)
    payload["config"] = {"threshold": threshold}
    return payload


def _score_residuals_with_threshold(
    method: AnomalyMethod,
    threshold: float,
    train_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    train_predictions: Any,
    score_predictions: Any,
) -> pd.DataFrame:
    if method == "residual_zscore":
        detector = ResidualZScoreDetector(threshold=threshold).fit(
            train_frame[TARGET_COLUMN],
            train_predictions,
        )
    else:
        detector = RobustResidualDetector(threshold=threshold).fit(
            train_frame[TARGET_COLUMN],
            train_predictions,
        )
    return detector.score(score_frame[TARGET_COLUMN], score_predictions)


def _score_isolation_candidate(
    contamination: str | float,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_names: list[str],
    random_state: int,
    labels_available: bool,
) -> dict[str, Any]:
    detector = IsolationForestDetector(
        contamination=contamination,
        random_state=random_state,
    ).fit(train_frame[feature_names], feature_names)
    scores = detector.score(validation_frame[feature_names])
    payload = _result_payload(scores, validation_frame, labels_available)
    payload["config"] = {"contamination": contamination}
    return payload


def _result_payload(
    scores: pd.DataFrame,
    frame: pd.DataFrame,
    labels_available: bool,
) -> dict[str, Any]:
    predictions = scores["is_anomaly"].astype(int).to_numpy()
    payload: dict[str, Any] = {
        "detected_count": int(predictions.sum()),
        "labeled_anomaly_count": None,
        "precision": None,
        "recall": None,
        "macro_f1": None,
        "confusion_matrix": None,
    }
    if labels_available:
        labels = frame["is_anomaly"].astype(bool).astype(int).to_numpy()
        payload["labeled_anomaly_count"] = int(labels.sum())
        payload.update(anomaly_metrics(labels, predictions))
    return payload


def _validate_single_zone(frame: pd.DataFrame) -> None:
    zone_count = frame["zone"].nunique()
    if zone_count != 1:
        raise ValueError(
            "Anomaly tuning currently supports exactly one zone. "
            f"Found {zone_count} zones."
        )


def _configuration_payload(config: AnomalyTuningConfig) -> dict[str, Any]:
    return {
        "input": str(config.input_path),
        "forecast_horizon": config.forecast_horizon,
        "model": config.model,
        "split_method": config.split_method,
        "validation_size": config.validation_size,
        "test_size": config.test_size,
        "methods": list(config.methods),
        "random_state": config.random_state,
    }


def _allowed_methods() -> set[str]:
    return {"residual_zscore", "robust_residual", "isolation_forest"}


def _default_threshold(method: AnomalyMethod) -> float:
    return 3.0 if method == "residual_zscore" else 3.5


if __name__ == "__main__":
    main()
