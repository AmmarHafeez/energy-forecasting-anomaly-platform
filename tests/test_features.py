from __future__ import annotations

import pandas as pd

from energy_forecasting_anomaly.features import (
    TARGET_COLUMN,
    FeatureConfig,
    build_feature_frame,
    build_supervised_frame,
)


def _series_frame(periods: int = 40) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "zone": ["north"] * periods,
            "load_mw": [100.0 + index for index in range(periods)],
            "temperature_c": [5.0 + index * 0.1 for index in range(periods)],
            "wind_speed_mps": [3.0] * periods,
            "solar_radiation_wm2": [0.0] * periods,
        }
    )


def test_build_feature_frame_adds_calendar_lag_rolling_and_target_columns() -> None:
    config = FeatureConfig(forecast_horizon=2, lag_periods=(1, 24), rolling_windows=(3,))

    result = build_feature_frame(_series_frame(), config)

    assert {"hour", "day_of_week", "month", "is_weekend"}.issubset(result.columns)
    assert {"lag_1", "lag_24", "rolling_mean_3", "rolling_std_3"}.issubset(result.columns)
    assert TARGET_COLUMN in result.columns
    assert result.loc[0, TARGET_COLUMN] == result.loc[2, "load_mw"]
    assert pd.isna(result.loc[len(result) - 1, TARGET_COLUMN])


def test_build_supervised_frame_drops_missing_targets_and_unusable_features() -> None:
    config = FeatureConfig(forecast_horizon=2, lag_periods=(1, 168), rolling_windows=(3,))
    feature_frame = build_feature_frame(_series_frame(), config)

    supervised, feature_names = build_supervised_frame(feature_frame)

    assert TARGET_COLUMN not in feature_names
    assert "lag_168" not in feature_names
    assert not supervised[feature_names + [TARGET_COLUMN]].isna().any().any()
