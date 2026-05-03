"""Phase 2 data cleaning pipeline for SentinelNet v2."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CleaningConfig
from .schema import align_frame_to_schema, normalize_columns

LOGGER = logging.getLogger("sentinelnet.phase2")

_NON_LABEL_TOKEN_PATTERN = re.compile(r"[^0-9a-zA-Z]+")


@dataclass(slots=True)
class CleaningStatistics:
    """Intermediate statistics collected during the Phase 2 profiling pass."""

    normalized_columns: list[str]
    feature_columns: list[str]
    text_columns: list[str]
    rows_read: int
    missing_before: dict[str, int]
    infinite_counts: dict[str, int]
    numeric_fill_values: dict[str, float]
    text_fill_values: dict[str, str]
    raw_label_distribution: dict[str, int]
    normalized_label_distribution: dict[str, int]
    unknown_labels: dict[str, int]


@dataclass(slots=True)
class CleaningReport:
    """Serializable report returned after the Phase 2 cleaning pipeline completes."""

    created_at_utc: str
    input_path: str
    output_path: str
    report_path: str
    rows_read: int
    rows_written: int
    duplicate_rows_removed: int
    rows_dropped_missing_critical: int
    infinite_values_replaced: int
    missing_values_imputed: int
    critical_columns: list[str]
    critical_missing_after: dict[str, int]
    missing_before: dict[str, int]
    missing_after: dict[str, int]
    infinite_counts_by_column: dict[str, int]
    numeric_imputation_values: dict[str, float]
    text_imputation_values: dict[str, str]
    raw_label_distribution: dict[str, int]
    cleaned_label_distribution: dict[str, int]
    unknown_labels: dict[str, int]
    columns: list[str]
    validation_passed: bool
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "created_at_utc": self.created_at_utc,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "report_path": self.report_path,
            "rows_read": self.rows_read,
            "rows_written": self.rows_written,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "rows_dropped_missing_critical": self.rows_dropped_missing_critical,
            "infinite_values_replaced": self.infinite_values_replaced,
            "missing_values_imputed": self.missing_values_imputed,
            "critical_columns": self.critical_columns,
            "critical_missing_after": self.critical_missing_after,
            "missing_before": self.missing_before,
            "missing_after": self.missing_after,
            "infinite_counts_by_column": self.infinite_counts_by_column,
            "numeric_imputation_values": self.numeric_imputation_values,
            "text_imputation_values": self.text_imputation_values,
            "raw_label_distribution": self.raw_label_distribution,
            "cleaned_label_distribution": self.cleaned_label_distribution,
            "unknown_labels": self.unknown_labels,
            "columns": self.columns,
            "validation_passed": self.validation_passed,
            "config": self.config,
        }


def normalize_string_series(series: pd.Series) -> pd.Series:
    """Trim and standardize whitespace in a text column."""
    normalized = series.astype("string").str.strip()
    normalized = normalized.str.replace(r"\s+", " ", regex=True)
    return normalized.mask(normalized == "", pd.NA)


def sanitize_label_text(label: str) -> str:
    """Normalize label punctuation and whitespace before canonicalization."""
    sanitized = unicodedata.normalize("NFKC", label)
    sanitized = sanitized.replace("\ufffd", "-")
    sanitized = re.sub(r"[\u2010-\u2015\u2212]", "-", sanitized)
    sanitized = re.sub(r"\s*-\s*", " - ", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized.strip(" -")


def build_label_lookup_key(label: str) -> str:
    """Create a stable lookup key for label alias matching."""
    sanitized = sanitize_label_text(label)
    key = _NON_LABEL_TOKEN_PATTERN.sub(" ", sanitized.lower())
    return re.sub(r"\s+", " ", key).strip()


def canonicalize_label(value: Any, label_aliases: dict[str, str]) -> str | pd.NA:
    """Standardize a raw label value into a canonical label."""
    if value is None or pd.isna(value):
        return pd.NA

    stripped = str(value).strip()
    if not stripped:
        return pd.NA

    lookup_key = build_label_lookup_key(stripped)
    sanitized = sanitize_label_text(stripped)
    return label_aliases.get(lookup_key, sanitized)


def should_zero_fill(column_name: str, zero_fill_suffixes: tuple[str, ...]) -> bool:
    """Decide whether a numeric column should be filled with zero."""
    return any(column_name.endswith(suffix) for suffix in zero_fill_suffixes)


def _read_normalized_header(input_data_path: Path) -> list[str]:
    """Read and normalize the CSV header once for schema planning."""
    header_frame = pd.read_csv(input_data_path, nrows=0)
    return normalize_columns(list(header_frame.columns))


def _validate_required_columns(normalized_columns: list[str], config: CleaningConfig) -> None:
    """Validate that the cleaning input contains the required columns."""
    required_columns = set(config.critical_columns) | {config.label_column, config.source_file_column}
    missing_columns = sorted(required_columns.difference(normalized_columns))
    if missing_columns:
        raise ValueError(f"Cleaning input is missing required columns: {missing_columns}")


def _load_and_standardize_chunk(
    chunk: pd.DataFrame,
    normalized_columns: list[str],
    config: CleaningConfig,
) -> pd.DataFrame:
    """Normalize headers, align schema, standardize text fields, and coerce features to numeric."""
    chunk.columns = normalize_columns(list(chunk.columns))
    standardized = align_frame_to_schema(chunk, normalized_columns)

    metadata_columns = {config.label_column, config.source_file_column}
    for column in metadata_columns:
        standardized[column] = normalize_string_series(standardized[column])

    feature_columns = [column for column in normalized_columns if column not in metadata_columns]
    if feature_columns:
        standardized.loc[:, feature_columns] = standardized.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
        standardized.loc[:, feature_columns] = standardized.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan)

    standardized[config.label_column] = standardized[config.label_column].map(lambda value: canonicalize_label(value, config.label_aliases))
    standardized[config.source_file_column] = normalize_string_series(standardized[config.source_file_column])
    return standardized


def collect_cleaning_statistics(config: CleaningConfig, logger: logging.Logger | None = None) -> CleaningStatistics:
    """Collect dataset-wide statistics required for deterministic Phase 2 cleaning."""
    active_logger = logger or LOGGER
    normalized_columns = _read_normalized_header(config.input_data_path)
    _validate_required_columns(normalized_columns, config)

    metadata_columns = [config.label_column, config.source_file_column]
    feature_columns = [column for column in normalized_columns if column not in metadata_columns]

    rows_read = 0
    missing_before: Counter[str] = Counter()
    infinite_counts: Counter[str] = Counter()
    numeric_sum: defaultdict[str, float] = defaultdict(float)
    numeric_non_null: Counter[str] = Counter()
    text_value_counters: dict[str, Counter[str]] = {column: Counter() for column in metadata_columns}
    raw_label_distribution: Counter[str] = Counter()
    normalized_label_distribution: Counter[str] = Counter()
    unknown_labels: Counter[str] = Counter()

    for chunk_index, chunk in enumerate(
        pd.read_csv(config.input_data_path, chunksize=config.chunk_size, low_memory=False),
        start=1,
    ):
        normalized_chunk = chunk.copy()
        normalized_chunk.columns = normalize_columns(list(normalized_chunk.columns))
        normalized_chunk = align_frame_to_schema(normalized_chunk, normalized_columns)
        normalized_chunk[config.label_column] = normalize_string_series(normalized_chunk[config.label_column])
        normalized_chunk[config.source_file_column] = normalize_string_series(normalized_chunk[config.source_file_column])

        standardized = _load_and_standardize_chunk(chunk, normalized_columns, config)
        rows_read += len(standardized)

        missing_before.update(
            {
                column: int(value)
                for column, value in standardized.isna().sum().to_dict().items()
                if int(value)
            }
        )
        raw_label_distribution.update(normalized_chunk[config.label_column].fillna("<NA>").tolist())

        normalized_labels_current = standardized[config.label_column]
        normalized_label_distribution.update(normalized_labels_current.fillna("<NA>").tolist())

        unknown_mask = normalized_labels_current.notna() & ~normalized_labels_current.isin(list(config.label_aliases.values()))
        if unknown_mask.any():
            unknown_labels.update(normalized_labels_current.loc[unknown_mask].astype(str).tolist())

        feature_frame = standardized.loc[:, feature_columns]
        if not feature_frame.empty:
            raw_numeric = normalized_chunk.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
            inf_mask = np.isinf(raw_numeric.to_numpy())
            if inf_mask.any():
                counts = inf_mask.sum(axis=0)
                infinite_counts.update(
                    {
                        column: int(count)
                        for column, count in zip(feature_columns, counts)
                        if int(count)
                    }
                )

            for column in feature_columns:
                valid_values = feature_frame[column].dropna()
                if not valid_values.empty:
                    numeric_sum[column] += float(valid_values.sum())
                    numeric_non_null[column] += int(valid_values.count())

        for column in metadata_columns:
            valid_values = standardized[column].dropna()
            if not valid_values.empty:
                text_value_counters[column].update(valid_values.astype(str).tolist())

        active_logger.info(
            "Profiled cleaning chunk %d | rows=%d | cumulative_rows=%d",
            chunk_index,
            len(standardized),
            rows_read,
        )

    numeric_fill_values: dict[str, float] = {}
    for column in feature_columns:
        if should_zero_fill(column, config.zero_fill_suffixes):
            numeric_fill_values[column] = 0.0
        elif numeric_non_null[column]:
            numeric_fill_values[column] = float(numeric_sum[column] / numeric_non_null[column])
        else:
            numeric_fill_values[column] = 0.0

    text_fill_values = {
        column: counter.most_common(1)[0][0] if counter else "unknown"
        for column, counter in text_value_counters.items()
    }

    return CleaningStatistics(
        normalized_columns=normalized_columns,
        feature_columns=feature_columns,
        text_columns=metadata_columns,
        rows_read=rows_read,
        missing_before=dict(sorted(missing_before.items())),
        infinite_counts=dict(sorted(infinite_counts.items())),
        numeric_fill_values=numeric_fill_values,
        text_fill_values=text_fill_values,
        raw_label_distribution=dict(sorted(raw_label_distribution.items())),
        normalized_label_distribution=dict(sorted(normalized_label_distribution.items())),
        unknown_labels=dict(sorted(unknown_labels.items())),
    )


def run_cleaning_pipeline(
    config: CleaningConfig,
    logger: logging.Logger | None = None,
) -> CleaningReport:
    """Execute the complete Phase 2 cleaning workflow."""
    active_logger = logger or LOGGER
    config.ensure_directories()

    if not config.input_data_path.exists():
        raise FileNotFoundError(f"Combined dataset not found at {config.input_data_path}")

    statistics = collect_cleaning_statistics(config, active_logger)
    if config.output_path.exists():
        config.output_path.unlink()

    rows_written = 0
    duplicate_rows_removed = 0
    rows_dropped_missing_critical = 0
    missing_values_imputed = 0
    missing_after: Counter[str] = Counter()
    critical_missing_after: Counter[str] = Counter()
    cleaned_label_distribution: Counter[str] = Counter()
    dedupe_columns = [
        column
        for column in statistics.normalized_columns
        if column not in set(config.deduplication_excluded_columns)
    ]
    seen_hashes: set[int] = set()
    header_written = False

    for chunk_index, chunk in enumerate(
        pd.read_csv(config.input_data_path, chunksize=config.chunk_size, low_memory=False),
        start=1,
    ):
        standardized = _load_and_standardize_chunk(chunk, statistics.normalized_columns, config)
        critical_mask = standardized.loc[:, list(config.critical_columns)].notna().all(axis=1)
        dropped_rows = int((~critical_mask).sum())
        rows_dropped_missing_critical += dropped_rows
        cleaned = standardized.loc[critical_mask].copy()

        for column in statistics.feature_columns:
            missing_count = int(cleaned[column].isna().sum())
            if missing_count:
                cleaned[column] = cleaned[column].fillna(statistics.numeric_fill_values[column])
                missing_values_imputed += missing_count

        for column in statistics.text_columns:
            if column in config.critical_columns:
                continue
            missing_count = int(cleaned[column].isna().sum())
            if missing_count:
                cleaned[column] = cleaned[column].fillna(statistics.text_fill_values[column])
                missing_values_imputed += missing_count

        row_hashes = pd.util.hash_pandas_object(cleaned.loc[:, dedupe_columns], index=False).tolist()
        keep_mask: list[bool] = []
        for row_hash in row_hashes:
            if row_hash in seen_hashes:
                keep_mask.append(False)
            else:
                seen_hashes.add(int(row_hash))
                keep_mask.append(True)

        retained = cleaned.loc[keep_mask, statistics.normalized_columns].copy()
        duplicate_rows_removed += len(cleaned) - len(retained)
        rows_written += len(retained)

        missing_after.update(
            {
                column: int(value)
                for column, value in retained.isna().sum().to_dict().items()
                if int(value)
            }
        )
        critical_missing_after.update(
            {
                column: int(value)
                for column, value in retained.loc[:, list(config.critical_columns)].isna().sum().to_dict().items()
                if int(value)
            }
        )
        cleaned_label_distribution.update(retained[config.label_column].fillna("<NA>").tolist())

        retained.to_csv(
            config.output_path,
            mode="a" if header_written else "w",
            header=not header_written,
            index=False,
        )
        header_written = True

        active_logger.info(
            "Cleaned chunk %d | input_rows=%d | dropped_missing=%d | duplicates_removed=%d | written=%d",
            chunk_index,
            len(standardized),
            dropped_rows,
            len(cleaned) - len(retained),
            len(retained),
        )

    validation_passed = not critical_missing_after and not missing_after and not statistics.unknown_labels
    report = CleaningReport(
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        input_path=str(config.input_data_path),
        output_path=str(config.output_path),
        report_path=str(config.report_path),
        rows_read=statistics.rows_read,
        rows_written=rows_written,
        duplicate_rows_removed=duplicate_rows_removed,
        rows_dropped_missing_critical=rows_dropped_missing_critical,
        infinite_values_replaced=int(sum(statistics.infinite_counts.values())),
        missing_values_imputed=missing_values_imputed,
        critical_columns=list(config.critical_columns),
        critical_missing_after=dict(sorted(critical_missing_after.items())),
        missing_before=statistics.missing_before,
        missing_after=dict(sorted(missing_after.items())),
        infinite_counts_by_column=statistics.infinite_counts,
        numeric_imputation_values=statistics.numeric_fill_values,
        text_imputation_values=statistics.text_fill_values,
        raw_label_distribution=statistics.raw_label_distribution,
        cleaned_label_distribution=dict(sorted(cleaned_label_distribution.items())),
        unknown_labels=statistics.unknown_labels,
        columns=statistics.normalized_columns,
        validation_passed=validation_passed,
        config=config.to_dict(),
    )

    config.report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if not validation_passed:
        raise ValueError(
            f"Phase 2 validation failed. Unknown labels={statistics.unknown_labels}, "
            f"critical_missing_after={dict(critical_missing_after)}, missing_after={dict(missing_after)}"
        )

    active_logger.info(
        "Completed Phase 2 cleaning | rows_read=%d | rows_written=%d | duplicates_removed=%d | dropped_missing=%d",
        report.rows_read,
        report.rows_written,
        report.duplicate_rows_removed,
        report.rows_dropped_missing_critical,
    )
    return report
