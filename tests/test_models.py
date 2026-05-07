from __future__ import annotations

import pandas as pd

from energy_forecasting_anomaly.models import (
    load_model_bundle,
    predict_forecast,
    save_model_bundle,
    train_forecaster,
)


def test_train_predict_and_persist_ridge_forecaster(tmp_path) -> None:
    features = pd.DataFrame(
        {
            "load_mw": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "temperature_c": [1.0, 1.2, 1.1, 1.4, 1.5, 1.6],
            "hour": [0, 1, 2, 3, 4, 5],
        }
    )
    target = pd.Series([11.0, 12.0, 13.0, 14.0, 15.0, 16.0])

    bundle = train_forecaster(
        features,
        target,
        model_type="ridge",
        feature_names=list(features.columns),
        forecast_horizon=1,
    )
    predictions = predict_forecast(bundle, features.tail(2))

    assert len(predictions) == 2
    assert bundle.metadata.model_type == "ridge"

    model_path = tmp_path / "forecast.joblib"
    save_model_bundle(bundle, model_path)
    loaded = load_model_bundle(model_path)

    loaded_predictions = predict_forecast(loaded, features.tail(2))
    assert loaded.metadata.feature_names == list(features.columns)
    assert loaded_predictions.shape == predictions.shape
