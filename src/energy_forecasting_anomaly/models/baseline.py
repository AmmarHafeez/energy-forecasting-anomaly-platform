"""Baseline forecasting models and model bundle persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_forecasting_anomaly import __version__

LOGGER = logging.getLogger(__name__)

ModelName = Literal["ridge", "random_forest"]


@dataclass
class ModelMetadata:
    """Versioned model metadata stored with every forecasting model."""

    model_type: str
    version: str
    created_at_utc: str
    feature_names: list[str]
    target_column: str
    forecast_horizon: int
    random_state: int | None = None


@dataclass
class ModelBundle:
    """Forecasting estimator plus the metadata required for inference."""

    model: Any
    metadata: ModelMetadata


def create_regressor(model_name: ModelName, random_state: int = 42) -> Pipeline:
    """Create a supported baseline regressor pipeline."""

    if model_name == "ridge":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model type: {model_name}")


def train_forecaster(
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    *,
    model_type: ModelName = "ridge",
    feature_names: list[str] | None = None,
    target_column: str = "target_load_mw",
    forecast_horizon: int = 24,
    random_state: int = 42,
) -> ModelBundle:
    """Fit a baseline forecasting model and return a persistable bundle."""

    if features.empty:
        raise ValueError("features must contain at least one row.")

    selected_features = feature_names or list(features.columns)
    if not selected_features:
        raise ValueError("feature_names must not be empty.")

    estimator = create_regressor(model_type, random_state=random_state)
    estimator.fit(features[selected_features], target)

    metadata = ModelMetadata(
        model_type=model_type,
        version=__version__,
        created_at_utc=datetime.now(tz=UTC).isoformat(),
        feature_names=selected_features,
        target_column=target_column,
        forecast_horizon=forecast_horizon,
        random_state=random_state,
    )
    LOGGER.info("Trained %s forecaster with %d features", model_type, len(selected_features))
    return ModelBundle(model=estimator, metadata=metadata)


def predict_forecast(bundle: ModelBundle, features: pd.DataFrame) -> np.ndarray:
    """Predict forecast targets from a model bundle."""

    missing = [name for name in bundle.metadata.feature_names if name not in features.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required model features: {missing_text}.")
    return bundle.model.predict(features[bundle.metadata.feature_names])


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    """Persist a model bundle with joblib."""

    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": bundle.model, "metadata": asdict(bundle.metadata)}, model_path)
    LOGGER.info("Saved model bundle to %s", model_path)
    return model_path


def load_model_bundle(path: str | Path) -> ModelBundle:
    """Load a joblib model bundle."""

    model_path = Path(path)
    payload = joblib.load(model_path)
    metadata = payload["metadata"]
    if isinstance(metadata, dict):
        metadata = ModelMetadata(**metadata)
    LOGGER.info("Loaded model bundle from %s", model_path)
    return ModelBundle(model=payload["model"], metadata=metadata)
