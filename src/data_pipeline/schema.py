"""Schema normalization and alignment utilities for ingestion."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import pandas as pd

_NON_ALNUM_PATTERN = re.compile(r"[^0-9a-zA-Z]+")


def normalize_column_name(column_name: str) -> str:
    """Normalize a raw column name into a stable snake_case identifier."""
    cleaned = column_name.strip().lower()
    cleaned = cleaned.replace("%", "pct")
    cleaned = cleaned.replace("/", "_per_")
    cleaned = _NON_ALNUM_PATTERN.sub("_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned or "unnamed_column"


def normalize_columns(columns: Iterable[str]) -> list[str]:
    """Normalize columns while keeping names unique and order-preserving."""
    normalized_columns: list[str] = []
    seen: dict[str, int] = {}

    for raw_column in columns:
        base_name = normalize_column_name(str(raw_column))
        suffix_index = seen.get(base_name, 0)
        normalized_name = base_name if suffix_index == 0 else f"{base_name}__dup{suffix_index}"
        seen[base_name] = suffix_index + 1
        normalized_columns.append(normalized_name)

    return normalized_columns


def ordered_union(column_sets: Iterable[Sequence[str]]) -> list[str]:
    """Build an ordered union of schema columns."""
    canonical_columns: list[str] = []
    seen: set[str] = set()

    for columns in column_sets:
        for column in columns:
            if column not in seen:
                seen.add(column)
                canonical_columns.append(column)

    return canonical_columns


def schema_difference(reference_columns: Sequence[str], current_columns: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return columns missing from and extra to the reference schema."""
    reference_set = set(reference_columns)
    current_set = set(current_columns)
    missing_columns = [column for column in reference_columns if column not in current_set]
    extra_columns = [column for column in current_columns if column not in reference_set]
    return missing_columns, extra_columns


def align_frame_to_schema(frame: pd.DataFrame, canonical_columns: Sequence[str]) -> pd.DataFrame:
    """Align a frame to the canonical schema by adding missing columns and reordering."""
    aligned_frame = frame.copy()

    for column in canonical_columns:
        if column not in aligned_frame.columns:
            aligned_frame[column] = pd.NA

    return aligned_frame.reindex(columns=list(canonical_columns))

