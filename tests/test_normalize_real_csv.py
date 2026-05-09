from __future__ import annotations

import pandas as pd
import pytest

from energy_forecasting_anomaly.data import (
    DataValidationError,
    EXPECTED_COLUMNS,
    read_energy_weather_csv,
)
from energy_forecasting_anomaly.data.normalize_real_csv import (
    NormalizeCsvConfig,
    normalize_real_csv_file,
)


def _source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [
                "2026-01-01 02:00:00",
                "2026-01-01 00:00:00",
                "2026-01-01 01:00:00",
            ],
            "bidding_zone": ["DE_LU", "DE_LU", "DE_LU"],
            "load": [930.0, 900.0, 915.0],
            "temperature": [5.0, 4.8, 4.9],
            "wind_speed": [6.0, 5.5, 5.8],
            "solar_radiation": [0.0, 0.0, 0.0],
        }
    )


def _config(input_path, output_path, zone_value: str | None = None) -> NormalizeCsvConfig:
    return NormalizeCsvConfig(
        input_path=input_path,
        output_path=output_path,
        timestamp_column="time",
        zone_column="bidding_zone",
        load_column="load",
        temperature_column="temperature",
        wind_column="wind_speed",
        solar_column="solar_radiation",
        zone_value=zone_value,
    )


def test_normalize_real_csv_with_explicit_source_columns(tmp_path) -> None:
    input_path = tmp_path / "real_export.csv"
    output_path = tmp_path / "processed" / "normalized.csv"
    _source_frame().to_csv(input_path, index=False)

    result_path = normalize_real_csv_file(_config(input_path, output_path))
    output_frame = pd.read_csv(result_path)

    assert result_path == output_path
    assert list(output_frame.columns) == list(EXPECTED_COLUMNS)
    assert list(output_frame["timestamp"]) == sorted(output_frame["timestamp"])
    assert output_frame["zone"].unique().tolist() == ["DE_LU"]


def test_normalize_real_csv_uses_zone_value_when_source_zone_is_missing(tmp_path) -> None:
    input_path = tmp_path / "real_export.csv"
    output_path = tmp_path / "processed" / "normalized.csv"
    _source_frame().drop(columns=["bidding_zone"]).to_csv(input_path, index=False)

    normalize_real_csv_file(_config(input_path, output_path, zone_value="DE_LU"))
    output_frame = pd.read_csv(output_path)

    assert output_frame["zone"].unique().tolist() == ["DE_LU"]


def test_normalize_real_csv_rejects_missing_required_source_column(tmp_path) -> None:
    input_path = tmp_path / "real_export.csv"
    output_path = tmp_path / "processed" / "normalized.csv"
    _source_frame().drop(columns=["wind_speed"]).to_csv(input_path, index=False)

    with pytest.raises(DataValidationError, match="wind_speed"):
        normalize_real_csv_file(_config(input_path, output_path))


def test_normalize_real_csv_rejects_non_numeric_measurement(tmp_path) -> None:
    input_path = tmp_path / "real_export.csv"
    output_path = tmp_path / "processed" / "normalized.csv"
    frame = _source_frame()
    frame["load"] = frame["load"].astype(object)
    frame.loc[0, "load"] = "missing"
    frame.to_csv(input_path, index=False)

    with pytest.raises(DataValidationError, match="non-numeric"):
        normalize_real_csv_file(_config(input_path, output_path))


def test_normalize_real_csv_rejects_duplicate_timestamp_zone_rows(tmp_path) -> None:
    input_path = tmp_path / "real_export.csv"
    output_path = tmp_path / "processed" / "normalized.csv"
    frame = _source_frame()
    frame.loc[1, "time"] = frame.loc[0, "time"]
    frame.to_csv(input_path, index=False)

    with pytest.raises(DataValidationError, match="Duplicate"):
        normalize_real_csv_file(_config(input_path, output_path))


def test_normalized_output_passes_existing_parser(tmp_path) -> None:
    input_path = tmp_path / "real_export.csv"
    output_path = tmp_path / "processed" / "normalized.csv"
    _source_frame().to_csv(input_path, index=False)

    normalize_real_csv_file(_config(input_path, output_path))
    parsed = read_energy_weather_csv(output_path)

    assert len(parsed) == 3
    assert parsed["load_mw"].dtype.kind in {"f", "i"}
