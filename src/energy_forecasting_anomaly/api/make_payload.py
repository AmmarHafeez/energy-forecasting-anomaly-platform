"""Build sample API request payloads from local energy and weather CSV data."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy_forecasting_anomaly.data import read_energy_weather_csv
from energy_forecasting_anomaly.features import (
    TARGET_COLUMN,
    FeatureConfig,
    build_feature_frame,
    build_supervised_frame,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PayloadConfig:
    """Configuration for sample API payload generation."""

    input_path: Path
    output_path: Path
    forecast_horizon: int = 24
    records: int = 3

    def __post_init__(self) -> None:
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive.")
        if self.records <= 0:
            raise ValueError("records must be positive.")


def build_api_payload(config: PayloadConfig) -> dict[str, Any]:
    """Build forecast, anomaly, and batch request payloads."""

    raw_frame = read_energy_weather_csv(config.input_path)
    feature_config = FeatureConfig(forecast_horizon=config.forecast_horizon)
    feature_frame = build_feature_frame(raw_frame, feature_config)
    supervised_frame, feature_names = build_supervised_frame(feature_frame, TARGET_COLUMN)
    selected_rows = supervised_frame.tail(config.records).copy()
    if selected_rows.empty:
        raise ValueError("No valid rows are available for API payload generation.")

    feature_records = [_feature_record(row, feature_names) for _, row in selected_rows.iterrows()]
    anomaly_records = [_residual_record(row) for _, row in selected_rows.iterrows()]
    payload = {
        "metadata": {
            "input": str(config.input_path),
            "forecast_horizon": config.forecast_horizon,
            "requested_records": config.records,
            "record_count": len(feature_records),
            "feature_columns": feature_names,
            "residual_example": (
                "actual_load_mw uses the forecast target; predicted_load_mw uses current load_mw "
                "as a naive residual example."
            ),
        },
        "forecast_request": {
            "records": [{"features": features} for features in feature_records],
        },
        "anomaly_request": {
            "records": anomaly_records,
        },
        "batch_request": {
            "records": [{"features": features} for features in feature_records],
        },
    }
    return payload


def save_api_payload(config: PayloadConfig) -> Path:
    """Build and write an API payload JSON file."""

    payload = build_api_payload(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    LOGGER.info("Wrote sample API payload to %s", config.output_path)
    return config.output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for sample payload generation."""

    parser = argparse.ArgumentParser(description="Build sample API request payload JSON.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--forecast-horizon", type=int, default=24)
    parser.add_argument("--records", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    config = PayloadConfig(
        input_path=args.input,
        output_path=args.output,
        forecast_horizon=args.forecast_horizon,
        records=args.records,
    )
    save_api_payload(config)


def _feature_record(row: pd.Series, feature_names: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    for feature_name in feature_names:
        value = float(row[feature_name])
        if not np.isfinite(value):
            raise ValueError(f"Feature value must be finite: {feature_name}")
        features[feature_name] = value
    return features


def _residual_record(row: pd.Series) -> dict[str, float]:
    return {
        "actual_load_mw": float(row[TARGET_COLUMN]),
        "predicted_load_mw": float(row["load_mw"]),
    }


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


if __name__ == "__main__":
    main()
