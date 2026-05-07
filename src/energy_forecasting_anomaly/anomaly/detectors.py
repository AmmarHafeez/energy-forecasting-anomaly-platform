"""Lightweight anomaly detection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

LOGGER = logging.getLogger(__name__)


def score_residuals(
    actual: pd.Series | np.ndarray | list[float],
    predicted: pd.Series | np.ndarray | list[float],
    *,
    threshold: float = 3.0,
    residual_mean: float | None = None,
    residual_std: float | None = None,
) -> pd.DataFrame:
    """Score anomalies from forecast residual z-scores."""

    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted values must have the same shape.")
    if actual_values.size == 0:
        raise ValueError("actual and predicted values must not be empty.")
    residuals = actual_values - predicted_values

    center = float(np.mean(residuals)) if residual_mean is None else residual_mean
    scale = float(np.std(residuals, ddof=0)) if residual_std is None else residual_std
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    z_scores = np.abs((residuals - center) / scale)
    return pd.DataFrame(
        {
            "anomaly_score": z_scores,
            "is_anomaly": z_scores >= threshold,
        }
    )


@dataclass
class ResidualZScoreDetector:
    """Residual z-score detector fitted on reference residuals."""

    threshold: float = 3.0
    residual_mean_: float | None = None
    residual_std_: float | None = None

    def fit(
        self,
        actual: pd.Series | np.ndarray | list[float],
        predicted: pd.Series | np.ndarray | list[float],
    ) -> ResidualZScoreDetector:
        """Fit detector statistics from actual and predicted values."""

        actual_values = np.asarray(actual, dtype=float)
        predicted_values = np.asarray(predicted, dtype=float)
        if actual_values.shape != predicted_values.shape:
            raise ValueError("actual and predicted values must have the same shape.")
        return self.fit_residuals(actual_values - predicted_values)

    def fit_residuals(
        self,
        residuals: pd.Series | np.ndarray | list[float],
    ) -> ResidualZScoreDetector:
        """Fit detector statistics directly from residuals."""

        residual_values = np.asarray(residuals, dtype=float)
        if residual_values.size == 0:
            raise ValueError("residuals must not be empty.")
        self.residual_mean_ = float(np.mean(residual_values))
        residual_std = float(np.std(residual_values, ddof=0))
        self.residual_std_ = residual_std if residual_std > 0.0 else 1.0
        LOGGER.info("Fitted residual z-score detector")
        return self

    def score(
        self,
        actual: pd.Series | np.ndarray | list[float],
        predicted: pd.Series | np.ndarray | list[float],
    ) -> pd.DataFrame:
        """Score actual and predicted values with fitted residual statistics."""

        if self.residual_mean_ is None or self.residual_std_ is None:
            raise ValueError("ResidualZScoreDetector must be fitted before scoring.")
        return score_residuals(
            actual,
            predicted,
            threshold=self.threshold,
            residual_mean=self.residual_mean_,
            residual_std=self.residual_std_,
        )


@dataclass
class IsolationForestDetector:
    """Small wrapper around sklearn IsolationForest."""

    contamination: float | str = "auto"
    random_state: int = 42
    feature_names: list[str] | None = None
    _imputer: SimpleImputer | None = field(default=None, init=False, repr=False)
    _model: IsolationForest | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        features: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> IsolationForestDetector:
        """Fit an IsolationForest detector on numeric features."""

        selected_features = feature_names or list(features.columns)
        if not selected_features:
            raise ValueError("feature_names must not be empty.")
        self.feature_names = selected_features
        self._imputer = SimpleImputer(strategy="median")
        matrix = self._imputer.fit_transform(features[selected_features])
        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._model.fit(matrix)
        LOGGER.info("Fitted IsolationForest detector with %d features", len(selected_features))
        return self

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score feature rows with a fitted IsolationForest detector."""

        if self._model is None or self._imputer is None or self.feature_names is None:
            raise ValueError("IsolationForestDetector must be fitted before scoring.")
        missing = [name for name in self.feature_names if name not in features.columns]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Missing required detector features: {missing_text}.")

        matrix = self._imputer.transform(features[self.feature_names])
        decision_scores = self._model.decision_function(matrix)
        predictions = self._model.predict(matrix)
        return pd.DataFrame(
            {
                "anomaly_score": -decision_scores,
                "is_anomaly": predictions == -1,
            }
        )


def save_anomaly_detector(detector: Any, path: str | Path) -> Path:
    """Persist an anomaly detector with joblib."""

    detector_path = Path(path)
    detector_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, detector_path)
    LOGGER.info("Saved anomaly detector to %s", detector_path)
    return detector_path


def load_anomaly_detector(path: str | Path) -> Any:
    """Load an anomaly detector from a joblib file."""

    detector_path = Path(path)
    detector = joblib.load(detector_path)
    LOGGER.info("Loaded anomaly detector from %s", detector_path)
    return detector
