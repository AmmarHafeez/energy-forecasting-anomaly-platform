from __future__ import annotations

import pandas as pd
import pytest

from energy_forecasting_anomaly.data import DataValidationError, read_energy_weather_csv


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 01:00:00",
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
            ],
            "zone": ["north", "north", "south"],
            "load_mw": [101.0, 100.0, 120.0],
            "temperature_c": [5.0, 4.5, 7.0],
            "wind_speed_mps": [3.0, 2.8, 4.0],
            "solar_radiation_wm2": [0.0, 0.0, 0.0],
        }
    )


def test_read_energy_weather_csv_sorts_and_parses(tmp_path) -> None:
    csv_path = tmp_path / "energy_weather.csv"
    _valid_frame().to_csv(csv_path, index=False)

    result = read_energy_weather_csv(csv_path)

    assert list(result["zone"]) == ["north", "north", "south"]
    assert result.loc[0, "timestamp"] < result.loc[1, "timestamp"]
    assert result["load_mw"].dtype.kind in {"f", "i"}


def test_read_energy_weather_csv_rejects_invalid_timestamp(tmp_path) -> None:
    frame = _valid_frame()
    frame.loc[0, "timestamp"] = "not-a-time"
    csv_path = tmp_path / "bad_timestamp.csv"
    frame.to_csv(csv_path, index=False)

    with pytest.raises(DataValidationError, match="timestamp"):
        read_energy_weather_csv(csv_path)


def test_read_energy_weather_csv_rejects_duplicate_zone_timestamp(tmp_path) -> None:
    frame = _valid_frame()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    csv_path = tmp_path / "duplicates.csv"
    frame.to_csv(csv_path, index=False)

    with pytest.raises(DataValidationError, match="Duplicate"):
        read_energy_weather_csv(csv_path)


def test_read_energy_weather_csv_rejects_non_numeric_measurement(tmp_path) -> None:
    frame = _valid_frame()
    frame["load_mw"] = frame["load_mw"].astype(object)
    frame.loc[0, "load_mw"] = "missing"
    csv_path = tmp_path / "bad_numeric.csv"
    frame.to_csv(csv_path, index=False)

    with pytest.raises(DataValidationError, match="missing or non-numeric"):
        read_energy_weather_csv(csv_path)
