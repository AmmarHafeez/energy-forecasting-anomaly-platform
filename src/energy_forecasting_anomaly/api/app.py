"""FastAPI service for forecasts and anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from energy_forecasting_anomaly.anomaly import load_anomaly_detector
from energy_forecasting_anomaly.models import ModelBundle, load_model_bundle, predict_forecast

LOGGER = logging.getLogger(__name__)
DEFAULT_FORECAST_MODEL_PATH = Path("models/forecast_ridge_h24.joblib")
DEFAULT_ANOMALY_MODEL_PATH = Path("models/anomaly_residual_zscore.joblib")


@dataclass
class ModelStore:
    """Runtime model registry for the API service."""

    forecast_bundle: ModelBundle | None = None
    anomaly_detector: Any | None = None

    @classmethod
    def from_environment(cls) -> ModelStore:
        forecast_path = Path(os.getenv("ENERGY_MODEL_PATH", str(DEFAULT_FORECAST_MODEL_PATH)))
        anomaly_path = Path(os.getenv("ENERGY_ANOMALY_MODEL_PATH", str(DEFAULT_ANOMALY_MODEL_PATH)))
        return cls(
            forecast_bundle=_load_forecast_if_available(forecast_path),
            anomaly_detector=_load_anomaly_if_available(anomaly_path),
        )


class FeatureRecord(BaseModel):
    """Feature record for one forecast row."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(..., min_length=1)

    @field_validator("features")
    @classmethod
    def validate_features(cls, features: dict[str, float]) -> dict[str, float]:
        for name, value in features.items():
            if not name or not name.strip():
                raise ValueError("Feature names must be non-empty.")
            if not np.isfinite(value):
                raise ValueError(f"Feature value must be finite: {name}.")
        return features


class ForecastRequest(BaseModel):
    """Forecast request containing one or more feature records."""

    model_config = ConfigDict(extra="forbid")

    records: list[FeatureRecord] = Field(..., min_length=1, max_length=1000)


class ResidualRecord(BaseModel):
    """Actual and predicted values for residual anomaly scoring."""

    model_config = ConfigDict(extra="forbid")

    actual_load_mw: float
    predicted_load_mw: float

    @field_validator("actual_load_mw", "predicted_load_mw")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("Load values must be finite.")
        return value


class AnomalyRequest(BaseModel):
    """Residual anomaly detection request."""

    model_config = ConfigDict(extra="forbid")

    records: list[ResidualRecord] = Field(..., min_length=1, max_length=1000)


class BatchPredictionRecord(BaseModel):
    """Forecast features with an optional actual load for anomaly scoring."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(..., min_length=1)
    actual_load_mw: float | None = None

    @field_validator("features")
    @classmethod
    def validate_features(cls, features: dict[str, float]) -> dict[str, float]:
        return FeatureRecord(features=features).features

    @field_validator("actual_load_mw")
    @classmethod
    def validate_actual(cls, value: float | None) -> float | None:
        if value is not None and not np.isfinite(value):
            raise ValueError("actual_load_mw must be finite.")
        return value


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    model_config = ConfigDict(extra="forbid")

    records: list[BatchPredictionRecord] = Field(..., min_length=1, max_length=1000)


def create_app(model_store: ModelStore | None = None) -> FastAPI:
    """Create a FastAPI app that starts even when model files are missing."""

    api = FastAPI(
        title="Energy Forecasting and Anomaly Detection",
        version="0.1.0",
    )
    api.state.model_store = model_store or ModelStore.from_environment()

    @api.get("/health")
    def health() -> dict[str, bool | str]:
        store: ModelStore = api.state.model_store
        return {
            "status": "ok",
            "forecast_model_available": store.forecast_bundle is not None,
            "anomaly_model_available": store.anomaly_detector is not None,
        }

    @api.post("/forecast")
    def forecast(request: ForecastRequest) -> dict[str, Any]:
        store: ModelStore = api.state.model_store
        bundle = _require_forecast_bundle(store)
        features = _feature_frame([record.features for record in request.records], bundle)
        predictions = predict_forecast(bundle, features)
        return {
            "forecast_horizon": bundle.metadata.forecast_horizon,
            "model_type": bundle.metadata.model_type,
            "predictions": [float(value) for value in predictions],
        }

    @api.post("/detect-anomaly")
    def detect_anomaly(request: AnomalyRequest) -> dict[str, Any]:
        store: ModelStore = api.state.model_store
        detector = _require_anomaly_detector(store)
        actual = [record.actual_load_mw for record in request.records]
        predicted = [record.predicted_load_mw for record in request.records]
        scores = detector.score(actual, predicted)
        return {"results": _score_records(scores)}

    @api.post("/predict/batch")
    def predict_batch(request: BatchPredictionRequest) -> dict[str, Any]:
        store: ModelStore = api.state.model_store
        bundle = _require_forecast_bundle(store)
        features = _feature_frame([record.features for record in request.records], bundle)
        predictions = predict_forecast(bundle, features)

        response: dict[str, Any] = {
            "forecast_horizon": bundle.metadata.forecast_horizon,
            "model_type": bundle.metadata.model_type,
            "predictions": [float(value) for value in predictions],
        }

        actual_values = [record.actual_load_mw for record in request.records]
        if any(value is not None for value in actual_values):
            if not all(value is not None for value in actual_values):
                raise HTTPException(
                    status_code=422,
                    detail="actual_load_mw must be provided for all records or no records.",
                )
            detector = _require_anomaly_detector(store)
            scores = detector.score(actual_values, predictions)
            response["anomalies"] = _score_records(scores)

        return response

    return api


def _load_forecast_if_available(path: Path) -> ModelBundle | None:
    if not path.exists():
        LOGGER.info("Forecast model not found at %s", path)
        return None
    try:
        return load_model_bundle(path)
    except Exception:
        LOGGER.exception("Failed to load forecast model from %s", path)
        return None


def _load_anomaly_if_available(path: Path) -> Any | None:
    if not path.exists():
        LOGGER.info("Anomaly model not found at %s", path)
        return None
    try:
        return load_anomaly_detector(path)
    except Exception:
        LOGGER.exception("Failed to load anomaly model from %s", path)
        return None


def _require_forecast_bundle(store: ModelStore) -> ModelBundle:
    if store.forecast_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Forecast model is unavailable. Train a model and configure ENERGY_MODEL_PATH.",
        )
    return store.forecast_bundle


def _require_anomaly_detector(store: ModelStore) -> Any:
    if store.anomaly_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Anomaly detector is unavailable. Train a detector and configure ENERGY_ANOMALY_MODEL_PATH.",
        )
    return store.anomaly_detector


def _feature_frame(records: list[dict[str, float]], bundle: ModelBundle) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    feature_names = bundle.metadata.feature_names
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features: {', '.join(missing)}.",
        )
    return frame[feature_names]


def _score_records(scores: pd.DataFrame) -> list[dict[str, float | bool]]:
    return [
        {
            "anomaly_score": float(row["anomaly_score"]),
            "is_anomaly": bool(row["is_anomaly"]),
        }
        for _, row in scores.iterrows()
    ]


app = create_app()
