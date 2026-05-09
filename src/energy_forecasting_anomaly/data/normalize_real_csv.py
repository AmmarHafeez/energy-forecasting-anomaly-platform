"""Normalize local energy and weather CSV exports to the canonical schema."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd

from energy_forecasting_anomaly.data.csv_parser import (
    DataValidationError,
    EXPECTED_COLUMNS,
    read_energy_weather_csv,
    validate_energy_weather_frame,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizeCsvConfig:
    """Configuration for real CSV schema normalization."""

    input_path: Path
    output_path: Path
    timestamp_column: str
    load_column: str
    temperature_column: str
    wind_column: str
    solar_column: str
    zone_column: str | None = None
    zone_value: str | None = None

    def __post_init__(self) -> None:
        source_names = {
            "timestamp_column": self.timestamp_column,
            "load_column": self.load_column,
            "temperature_column": self.temperature_column,
            "wind_column": self.wind_column,
            "solar_column": self.solar_column,
        }
        for label, value in source_names.items():
            if not value.strip():
                raise ValueError(f"{label} must be non-empty.")
        if self.zone_column is not None and not self.zone_column.strip():
            raise ValueError("zone_column must be non-empty when provided.")
        if self.zone_value is not None and not self.zone_value.strip():
            raise ValueError("zone_value must be non-empty when provided.")
        if self.zone_column is None and self.zone_value is None:
            raise ValueError("Either zone_column or zone_value must be provided.")


def normalize_real_csv_frame(frame: pd.DataFrame, config: NormalizeCsvConfig) -> pd.DataFrame:
    """Normalize a source frame to canonical energy and weather columns."""

    if frame.empty:
        raise DataValidationError("Input data is empty.")

    _validate_required_source_columns(frame, config)
    normalized = pd.DataFrame(
        {
            "timestamp": _parse_timestamp(frame[config.timestamp_column]),
            "zone": _build_zone_series(frame, config),
            "load_mw": _coerce_numeric(frame[config.load_column], config.load_column),
            "temperature_c": _coerce_numeric(
                frame[config.temperature_column],
                config.temperature_column,
            ),
            "wind_speed_mps": _coerce_numeric(frame[config.wind_column], config.wind_column),
            "solar_radiation_wm2": _coerce_numeric(frame[config.solar_column], config.solar_column),
        }
    )
    validated = validate_energy_weather_frame(normalized)
    return validated.sort_values(["timestamp", "zone"]).reset_index(drop=True)


def normalize_real_csv_file(config: NormalizeCsvConfig) -> Path:
    """Read, normalize, validate, and write a local CSV export."""

    if not config.input_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {config.input_path}")

    LOGGER.info("Reading source CSV from %s", config.input_path)
    source = pd.read_csv(config.input_path)
    normalized = normalize_real_csv_frame(source, config)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.loc[:, list(EXPECTED_COLUMNS)].to_csv(config.output_path, index=False)
    read_energy_weather_csv(config.output_path)
    LOGGER.info("Wrote normalized CSV to %s", config.output_path)
    return config.output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for real CSV normalization."""

    parser = argparse.ArgumentParser(
        description="Normalize a local energy/weather CSV export to the canonical schema."
    )
    parser.add_argument("--input", dest="input_path", type=Path, required=True)
    parser.add_argument("--output", dest="output_path", type=Path, required=True)
    parser.add_argument("--timestamp-column", required=True)
    parser.add_argument("--zone-column")
    parser.add_argument("--load-column", required=True)
    parser.add_argument("--temperature-column", required=True)
    parser.add_argument("--wind-column", required=True)
    parser.add_argument("--solar-column", required=True)
    parser.add_argument("--zone-value")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    config = NormalizeCsvConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        timestamp_column=args.timestamp_column,
        zone_column=args.zone_column,
        load_column=args.load_column,
        temperature_column=args.temperature_column,
        wind_column=args.wind_column,
        solar_column=args.solar_column,
        zone_value=args.zone_value,
    )
    normalize_real_csv_file(config)


def _validate_required_source_columns(frame: pd.DataFrame, config: NormalizeCsvConfig) -> None:
    required_columns = [
        config.timestamp_column,
        config.load_column,
        config.temperature_column,
        config.wind_column,
        config.solar_column,
    ]
    if config.zone_column is not None:
        if config.zone_column in frame.columns:
            required_columns.append(config.zone_column)
        elif config.zone_value is None:
            raise DataValidationError(f"Missing required source columns: {config.zone_column}.")

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise DataValidationError(f"Missing required source columns: {missing_text}.")


def _parse_timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise DataValidationError("One or more source timestamp values could not be parsed.")
    return parsed


def _build_zone_series(frame: pd.DataFrame, config: NormalizeCsvConfig) -> pd.Series:
    if config.zone_column is not None and config.zone_column in frame.columns:
        return frame[config.zone_column]
    if config.zone_value is not None:
        return pd.Series([config.zone_value] * len(frame), index=frame.index)
    raise DataValidationError("A zone source column or zone_value is required.")


def _coerce_numeric(series: pd.Series, source_column: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise DataValidationError(
            f"Source column '{source_column}' contains missing or non-numeric values."
        )
    return numeric


if __name__ == "__main__":
    main()
