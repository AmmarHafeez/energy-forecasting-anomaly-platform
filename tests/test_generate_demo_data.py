from __future__ import annotations

import pandas as pd

from energy_forecasting_anomaly.data import read_energy_weather_csv
from energy_forecasting_anomaly.data.generate_demo_data import (
    DemoDataConfig,
    generate_demo_frame,
    save_demo_csv,
)


EXPECTED_COLUMNS = {
    "timestamp",
    "zone",
    "load_mw",
    "temperature_c",
    "wind_speed_mps",
    "solar_radiation_wm2",
}


def test_save_demo_csv_writes_expected_rows_and_columns(tmp_path) -> None:
    output_path = tmp_path / "data" / "raw" / "demo_energy_weather.csv"
    config = DemoDataConfig(output=output_path, periods=48, anomaly_fraction=0.0)

    result_path = save_demo_csv(config)
    frame = pd.read_csv(result_path)

    assert result_path == output_path
    assert output_path.exists()
    assert len(frame) == 48
    assert EXPECTED_COLUMNS.issubset(frame.columns)


def test_generate_demo_frame_is_deterministic_for_same_random_state(tmp_path) -> None:
    config = DemoDataConfig(
        output=tmp_path / "demo.csv",
        periods=72,
        random_state=123,
        anomaly_fraction=0.02,
    )

    first = generate_demo_frame(config)
    second = generate_demo_frame(config)

    pd.testing.assert_frame_equal(first, second)


def test_generate_demo_frame_includes_anomaly_labels_when_fraction_is_positive(tmp_path) -> None:
    config = DemoDataConfig(
        output=tmp_path / "demo.csv",
        periods=120,
        random_state=42,
        anomaly_fraction=0.05,
    )

    frame = generate_demo_frame(config)

    assert {"is_anomaly", "anomaly_type"}.issubset(frame.columns)
    assert frame["is_anomaly"].any()
    assert set(frame.loc[frame["is_anomaly"], "anomaly_type"]).issubset(
        {"spike", "drop", "level_shift"}
    )


def test_generated_demo_csv_passes_existing_parser(tmp_path) -> None:
    output_path = tmp_path / "demo_energy_weather.csv"
    config = DemoDataConfig(output=output_path, periods=96, anomaly_fraction=0.03)

    save_demo_csv(config)
    parsed = read_energy_weather_csv(output_path)

    assert len(parsed) == 96
    assert EXPECTED_COLUMNS.issubset(parsed.columns)
