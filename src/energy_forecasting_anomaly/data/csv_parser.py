"""CSV parsing and validation for local energy and weather time series."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)

EXPECTED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "zone",
    "load_mw",
    "temperature_c",
    "wind_speed_mps",
    "solar_radiation_wm2",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "load_mw",
    "temperature_c",
    "wind_speed_mps",
    "solar_radiation_wm2",
)


class DataValidationError(ValueError):
    """Raised when local input data fails validation."""


@dataclass(frozen=True)
class CsvReadOptions:
    """Validation options for energy and weather CSV files."""

    timestamp_column: str = "timestamp"
    zone_column: str = "zone"
    required_columns: tuple[str, ...] = EXPECTED_COLUMNS
    numeric_columns: tuple[str, ...] = NUMERIC_COLUMNS
    duplicate_subset: tuple[str, ...] = ("zone", "timestamp")


def read_energy_weather_csv(path: str | Path, options: CsvReadOptions | None = None) -> pd.DataFrame:
    """Read, validate, and sort a local energy and weather CSV file."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    LOGGER.info("Reading energy weather CSV from %s", csv_path)
    frame = pd.read_csv(csv_path)
    return validate_energy_weather_frame(frame, options)


def validate_energy_weather_frame(
    frame: pd.DataFrame,
    options: CsvReadOptions | None = None,
) -> pd.DataFrame:
    """Validate an in-memory energy and weather frame and return a sorted copy."""

    opts = options or CsvReadOptions()
    if frame.empty:
        raise DataValidationError("Input data is empty.")

    _validate_columns(frame.columns, opts.required_columns)

    validated = frame.copy()
    timestamp_column = opts.timestamp_column
    zone_column = opts.zone_column

    validated[timestamp_column] = pd.to_datetime(
        validated[timestamp_column],
        errors="coerce",
    )
    if validated[timestamp_column].isna().any():
        raise DataValidationError("One or more timestamp values could not be parsed.")

    for column in opts.numeric_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")

    if validated[list(opts.required_columns)].isna().any().any():
        raise DataValidationError("Required columns contain missing or non-numeric values.")

    empty_zones = validated[zone_column].astype(str).str.strip().eq("")
    if empty_zones.any():
        raise DataValidationError("Zone values must be non-empty.")

    duplicate_subset = [column for column in opts.duplicate_subset if column in validated.columns]
    if duplicate_subset and validated.duplicated(subset=duplicate_subset).any():
        joined_subset = ", ".join(duplicate_subset)
        raise DataValidationError(f"Duplicate records found for columns: {joined_subset}.")

    validated = validated.sort_values([zone_column, timestamp_column]).reset_index(drop=True)
    LOGGER.info("Validated %d rows across %d zones", len(validated), validated[zone_column].nunique())
    return validated


def _validate_columns(actual_columns: Iterable[str], required_columns: Iterable[str]) -> None:
    actual = set(actual_columns)
    missing = [column for column in required_columns if column not in actual]
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(f"Missing required columns: {missing_text}.")
