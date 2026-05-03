"""Memory optimization helpers for chunked ingestion."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype, is_object_dtype


def estimate_frame_memory_mb(frame: pd.DataFrame) -> float:
    """Estimate a frame's memory footprint in megabytes."""
    return float(frame.memory_usage(deep=True).sum() / (1024 ** 2))


def _convert_object_series(series: pd.Series) -> pd.Series:
    """Convert object series to numeric when nearly all values are numeric."""
    non_null_count = int(series.notna().sum())
    if non_null_count == 0:
        return series.astype("string")

    numeric_candidate = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(numeric_candidate.notna().sum() / non_null_count)
    if numeric_ratio >= 0.99:
        return numeric_candidate

    return series.astype("string")


def optimize_dtypes(frame: pd.DataFrame, protected_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Downcast numeric columns and standardize text columns."""
    optimized_frame = frame.copy()
    protected = set(protected_columns or [])

    for column in optimized_frame.columns:
        if column in protected:
            optimized_frame[column] = optimized_frame[column].astype("string")
            continue

        series = optimized_frame[column]

        if is_object_dtype(series):
            series = _convert_object_series(series)

        if is_integer_dtype(series):
            optimized_frame[column] = pd.to_numeric(series, downcast="integer")
        elif is_float_dtype(series):
            optimized_frame[column] = pd.to_numeric(series, downcast="float")
        else:
            optimized_frame[column] = series

    return optimized_frame

