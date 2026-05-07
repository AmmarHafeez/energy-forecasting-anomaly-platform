"""Lightweight reference statistics and drift summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def compute_reference_stats(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute simple reference statistics for numeric columns."""

    selected_columns = columns or [
        column for column in frame.columns if is_numeric_dtype(frame[column])
    ]
    stats: dict[str, dict[str, float]] = {}
    for column in selected_columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        stats[column] = {
            "count": float(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return stats


def compare_to_reference_stats(
    current: pd.DataFrame,
    reference_stats: dict[str, dict[str, Any]],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compare current numeric statistics to a reference profile."""

    current_stats = compute_reference_stats(current, columns)
    rows: list[dict[str, float | str]] = []
    for column, current_column_stats in current_stats.items():
        reference_column_stats = reference_stats.get(column)
        if not reference_column_stats:
            continue
        reference_std = float(reference_column_stats.get("std", 0.0))
        current_mean = float(current_column_stats["mean"])
        reference_mean = float(reference_column_stats["mean"])
        mean_delta = current_mean - reference_mean
        normalized_mean_delta = mean_delta / reference_std if reference_std > 0.0 else np.nan
        rows.append(
            {
                "column": column,
                "current_mean": current_mean,
                "reference_mean": reference_mean,
                "mean_delta": mean_delta,
                "normalized_mean_delta": float(normalized_mean_delta),
                "current_std": float(current_column_stats["std"]),
                "reference_std": reference_std,
            }
        )
    return pd.DataFrame(rows)
