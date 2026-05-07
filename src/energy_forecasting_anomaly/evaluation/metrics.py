"""Forecast and anomaly evaluation metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)

LOGGER = logging.getLogger(__name__)


def forecast_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> dict[str, float]:
    """Compute standard regression metrics for load forecasts."""

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if actual.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    errors = actual - predicted
    nonzero_mask = np.abs(actual) > 1e-12
    mape = (
        float(np.mean(np.abs(errors[nonzero_mask] / actual[nonzero_mask])) * 100.0)
        if nonzero_mask.any()
        else float("nan")
    )
    r2 = float(r2_score(actual, predicted)) if len(actual) >= 2 else float("nan")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": mape,
        "r2": r2,
    }


def anomaly_metrics(
    labels: np.ndarray | list[int] | list[bool],
    predictions: np.ndarray | list[int] | list[bool],
) -> dict[str, Any]:
    """Compute anomaly classification metrics when labels are available."""

    actual = np.asarray(labels).astype(int)
    predicted = np.asarray(predictions).astype(int)
    if actual.shape != predicted.shape:
        raise ValueError("labels and predictions must have the same shape.")
    if actual.size == 0:
        raise ValueError("labels and predictions must not be empty.")
    precision, recall, _, _ = precision_recall_fscore_support(
        actual,
        predicted,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
    }


def write_json_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    """Write metrics to a local JSON file."""

    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    LOGGER.info("Wrote metrics to %s", metrics_path)
    return metrics_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        converted = float(value)
        return converted if np.isfinite(converted) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
