"""Feature generation for short-term energy load forecasting."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import pandas as pd
from pandas.api.types import is_numeric_dtype

LOGGER = logging.getLogger(__name__)

TARGET_COLUMN = "target_load_mw"
DEFAULT_LAGS: tuple[int, ...] = (1, 24, 168)
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (24, 168)
WEATHER_COLUMNS: tuple[str, ...] = (
    "temperature_c",
    "wind_speed_mps",
    "solar_radiation_wm2",
)
DEFAULT_EXCLUDED_FEATURES: frozenset[str] = frozenset(
    {"is_anomaly", "anomaly_label", "label", TARGET_COLUMN}
)


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for deterministic time-series feature generation."""

    timestamp_column: str = "timestamp"
    zone_column: str = "zone"
    load_column: str = "load_mw"
    forecast_horizon: int = 24
    lag_periods: tuple[int, ...] = DEFAULT_LAGS
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS
    rolling_min_periods: int = 1
    target_column: str = TARGET_COLUMN

    def __post_init__(self) -> None:
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive.")
        if any(period <= 0 for period in self.lag_periods):
            raise ValueError("lag periods must be positive.")
        if any(window <= 1 for window in self.rolling_windows):
            raise ValueError("rolling windows must be greater than one.")
        if self.rolling_min_periods <= 0:
            raise ValueError("rolling_min_periods must be positive.")


def build_feature_frame(frame: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Create calendar, lag, rolling, weather, and target columns."""

    cfg = config or FeatureConfig()
    featured = _sort_frame(frame, cfg)
    featured = add_calendar_features(featured, cfg.timestamp_column)
    featured = add_lag_features(featured, cfg)
    featured = add_rolling_features(featured, cfg)
    featured = add_forecast_target(featured, cfg)
    LOGGER.info("Built feature frame with %d rows and %d columns", *featured.shape)
    return featured


def add_calendar_features(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    """Add calendar features from a timestamp column."""

    if timestamp_column not in frame.columns:
        raise KeyError(f"Missing timestamp column: {timestamp_column}")

    result = frame.copy()
    timestamps = pd.to_datetime(result[timestamp_column], errors="raise")
    result["hour"] = timestamps.dt.hour
    result["day_of_week"] = timestamps.dt.dayofweek
    result["month"] = timestamps.dt.month
    result["is_weekend"] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)
    return result


def add_lag_features(frame: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Add load lag features grouped by zone when a zone column exists."""

    cfg = config or FeatureConfig()
    if cfg.load_column not in frame.columns:
        raise KeyError(f"Missing load column: {cfg.load_column}")

    result = _sort_frame(frame, cfg)
    grouped_load = _grouped_load(result, cfg)
    for lag in cfg.lag_periods:
        result[f"lag_{lag}"] = grouped_load.shift(lag)
    return result


def add_rolling_features(frame: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Add leakage-safe rolling load statistics based on prior observations."""

    cfg = config or FeatureConfig()
    if cfg.load_column not in frame.columns:
        raise KeyError(f"Missing load column: {cfg.load_column}")

    result = _sort_frame(frame, cfg)
    grouped_load = _grouped_load(result, cfg)
    for window in cfg.rolling_windows:
        min_periods = min(cfg.rolling_min_periods, window)
        result[f"rolling_mean_{window}"] = grouped_load.transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=min_periods).mean()
        )
        result[f"rolling_std_{window}"] = grouped_load.transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=min_periods).std()
        )
    return result


def add_forecast_target(frame: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Add a future load target for the configured forecast horizon."""

    cfg = config or FeatureConfig()
    if cfg.load_column not in frame.columns:
        raise KeyError(f"Missing load column: {cfg.load_column}")

    result = _sort_frame(frame, cfg)
    grouped_load = _grouped_load(result, cfg)
    result[cfg.target_column] = grouped_load.shift(-cfg.forecast_horizon)
    return result


def select_feature_columns(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    excluded_columns: set[str] | None = None,
) -> list[str]:
    """Select numeric model feature columns while excluding labels and timestamps."""

    excluded = set(DEFAULT_EXCLUDED_FEATURES)
    excluded.add(target_column)
    if excluded_columns:
        excluded.update(excluded_columns)

    feature_columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if column == "timestamp":
            continue
        if is_numeric_dtype(frame[column]):
            feature_columns.append(column)
    return feature_columns


def build_supervised_frame(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    excluded_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return rows and numeric feature names suitable for model training."""

    feature_columns = select_feature_columns(frame, target_column, excluded_columns)
    feature_columns = [column for column in feature_columns if not frame[column].isna().all()]
    if not feature_columns:
        raise ValueError("No usable numeric feature columns were found.")

    required_columns = [*feature_columns, target_column]
    supervised = frame.dropna(subset=required_columns).copy()
    if supervised.empty:
        raise ValueError("No rows remain after dropping missing features and targets.")
    return supervised, feature_columns


def _sort_frame(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    sort_columns = [config.timestamp_column]
    if config.zone_column in frame.columns:
        sort_columns = [config.zone_column, config.timestamp_column]
    return frame.sort_values(sort_columns).reset_index(drop=True).copy()


def _grouped_load(frame: pd.DataFrame, config: FeatureConfig) -> pd.core.groupby.SeriesGroupBy:
    if config.zone_column in frame.columns:
        return frame.groupby(config.zone_column, sort=False)[config.load_column]
    return frame.groupby(lambda _: 0, sort=False)[config.load_column]
