from __future__ import annotations

import pandas as pd

from energy_forecasting_anomaly.training.pipeline import train_test_split_frame


def test_chronological_split_uses_earlier_rows_for_train_and_later_rows_for_test() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 04:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 03:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 02:00:00",
                ]
            ),
            "zone": ["north"] * 5,
            "load_mw": [104.0, 100.0, 103.0, 101.0, 102.0],
        }
    )

    train_frame, test_frame = train_test_split_frame(
        frame,
        test_size=0.4,
        split_method="chronological",
    )

    assert train_frame["timestamp"].tolist() == list(
        pd.date_range("2026-01-01", periods=3, freq="h")
    )
    assert test_frame["timestamp"].tolist() == list(
        pd.date_range("2026-01-01 03:00:00", periods=2, freq="h")
    )
    assert train_frame["timestamp"].max() < test_frame["timestamp"].min()
