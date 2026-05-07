"""Generate deterministic demo energy and weather time-series data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
ANOMALY_TYPES: tuple[str, ...] = ("spike", "drop", "level_shift")


@dataclass(frozen=True)
class DemoDataConfig:
    """Configuration for synthetic demo data generation."""

    output: Path
    start: str = "2025-01-01"
    periods: int = 2160
    freq: str = "h"
    zone: str = "DE_DEMO"
    random_state: int = 42
    anomaly_fraction: float = 0.02

    def __post_init__(self) -> None:
        if self.periods <= 0:
            raise ValueError("periods must be positive.")
        if not self.zone.strip():
            raise ValueError("zone must be non-empty.")
        if not 0.0 <= self.anomaly_fraction < 1.0:
            raise ValueError("anomaly_fraction must be greater than or equal to 0 and less than 1.")


def generate_demo_frame(config: DemoDataConfig) -> pd.DataFrame:
    """Generate a deterministic hourly demo energy and weather data frame."""

    rng = np.random.default_rng(config.random_state)
    timestamps = pd.date_range(start=config.start, periods=config.periods, freq=config.freq)
    if len(timestamps) != config.periods:
        raise ValueError("Could not generate the requested number of timestamps.")

    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()

    annual_temperature = 10.0 + 12.0 * np.sin(2.0 * np.pi * (day_of_year - 80.0) / 365.25)
    daily_temperature = 4.0 * np.sin(2.0 * np.pi * (hour - 14.0) / 24.0)
    temperature_c = annual_temperature + daily_temperature + rng.normal(0.0, 1.8, config.periods)

    wind_speed_mps = np.clip(rng.gamma(shape=2.3, scale=2.0, size=config.periods), 0.2, 18.0)

    daylight = np.sin(np.pi * (hour - 6.0) / 12.0)
    seasonal_solar = 0.65 + 0.35 * np.sin(2.0 * np.pi * (day_of_year - 80.0) / 365.25)
    solar_radiation_wm2 = np.clip(
        850.0 * np.maximum(daylight, 0.0) * seasonal_solar + rng.normal(0.0, 35.0, config.periods),
        0.0,
        1000.0,
    )

    evening_peak = 180.0 * np.exp(-0.5 * ((hour - 19.0) / 3.2) ** 2)
    morning_peak = 85.0 * np.exp(-0.5 * ((hour - 8.0) / 2.5) ** 2)
    weekend_adjustment = np.where(day_of_week >= 5, -95.0, 0.0)
    cooling_load = 11.0 * np.maximum(temperature_c - 18.0, 0.0)
    heating_load = 7.0 * np.maximum(8.0 - temperature_c, 0.0)
    random_noise = rng.normal(0.0, 22.0, config.periods)

    load_mw = (
        950.0
        + evening_peak
        + morning_peak
        + weekend_adjustment
        + cooling_load
        + heating_load
        + random_noise
    )
    load_mw = np.clip(load_mw, 100.0, None)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "zone": config.zone,
            "load_mw": load_mw.round(3),
            "temperature_c": temperature_c.round(3),
            "wind_speed_mps": wind_speed_mps.round(3),
            "solar_radiation_wm2": solar_radiation_wm2.round(3),
        }
    )

    if config.anomaly_fraction > 0.0:
        frame = inject_anomalies(frame, config.anomaly_fraction, rng)

    return frame


def inject_anomalies(
    frame: pd.DataFrame,
    anomaly_fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject labeled synthetic load anomalies into a generated frame."""

    result = frame.copy()
    result["is_anomaly"] = False
    result["anomaly_type"] = "normal"

    anomaly_count = max(1, int(round(len(result) * anomaly_fraction)))
    anomaly_count = min(anomaly_count, len(result))
    anomaly_indices = rng.choice(result.index.to_numpy(), size=anomaly_count, replace=False)

    for index in anomaly_indices:
        anomaly_type = str(rng.choice(ANOMALY_TYPES))
        if anomaly_type == "spike":
            multiplier = rng.uniform(1.35, 1.8)
            result.loc[index, "load_mw"] = result.loc[index, "load_mw"] * multiplier
            _mark_anomaly(result, [int(index)], anomaly_type)
        elif anomaly_type == "drop":
            multiplier = rng.uniform(0.45, 0.7)
            result.loc[index, "load_mw"] = result.loc[index, "load_mw"] * multiplier
            _mark_anomaly(result, [int(index)], anomaly_type)
        else:
            span = range(int(index), min(int(index) + 6, len(result)))
            shift = rng.choice([-1.0, 1.0]) * rng.uniform(90.0, 160.0)
            span_indices = list(span)
            result.loc[span_indices, "load_mw"] = result.loc[span_indices, "load_mw"] + shift
            _mark_anomaly(result, span_indices, anomaly_type)

    result["load_mw"] = result["load_mw"].clip(lower=100.0).round(3)
    return result


def save_demo_csv(config: DemoDataConfig) -> Path:
    """Generate and write a demo CSV to the configured output path."""

    frame = generate_demo_frame(config)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.output, index=False)
    LOGGER.info("Wrote %d demo rows to %s", len(frame), config.output)
    return config.output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for demo data generation."""

    parser = argparse.ArgumentParser(description="Generate local demo energy and weather CSV data.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--start", default="2025-01-01", help="First timestamp.")
    parser.add_argument("--periods", type=int, default=2160, help="Number of periods to generate.")
    parser.add_argument("--freq", default="h", help="Pandas frequency string.")
    parser.add_argument("--zone", default="DE_DEMO", help="Zone identifier.")
    parser.add_argument("--random-state", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument(
        "--anomaly-fraction",
        type=float,
        default=0.02,
        help="Approximate fraction of rows affected by synthetic anomalies.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    config = DemoDataConfig(
        output=args.output,
        start=args.start,
        periods=args.periods,
        freq=args.freq,
        zone=args.zone,
        random_state=args.random_state,
        anomaly_fraction=args.anomaly_fraction,
    )
    save_demo_csv(config)


def _mark_anomaly(result: pd.DataFrame, indices: list[int], anomaly_type: str) -> None:
    result.loc[indices, "is_anomaly"] = True
    result.loc[indices, "anomaly_type"] = anomaly_type


if __name__ == "__main__":
    main()
