"""Phase 3 label engineering pipeline for SentinelNet v2."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .cleaning import canonicalize_label, normalize_string_series
from .config import LabelHandlingConfig
from .schema import normalize_columns

LOGGER = logging.getLogger("sentinelnet.phase3")


@dataclass(slots=True)
class LabelHandlingReport:
    """Serializable report returned after Phase 3 label engineering completes."""

    created_at_utc: str
    input_path: str
    output_path: str
    mapping_path: str
    report_path: str
    rows_read: int
    rows_written: int
    missing_labels: int
    unknown_labels: dict[str, int]
    label_distribution: dict[str, int]
    binary_target_distribution: dict[int, int]
    multiclass_target_distribution: dict[int, int]
    binary_label_mapping: dict[str, int]
    multiclass_label_mapping: dict[str, int]
    validation_passed: bool
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "created_at_utc": self.created_at_utc,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "mapping_path": self.mapping_path,
            "report_path": self.report_path,
            "rows_read": self.rows_read,
            "rows_written": self.rows_written,
            "missing_labels": self.missing_labels,
            "unknown_labels": self.unknown_labels,
            "label_distribution": self.label_distribution,
            "binary_target_distribution": self.binary_target_distribution,
            "multiclass_target_distribution": self.multiclass_target_distribution,
            "binary_label_mapping": self.binary_label_mapping,
            "multiclass_label_mapping": self.multiclass_label_mapping,
            "validation_passed": self.validation_passed,
            "config": self.config,
        }


def build_binary_label_mapping(allowed_labels: tuple[str, ...], benign_label: str) -> dict[str, int]:
    """Build the binary target mapping with BENIGN fixed to 0."""
    return {
        label: 0 if label == benign_label else 1
        for label in allowed_labels
    }


def build_multiclass_label_mapping(allowed_labels: tuple[str, ...]) -> dict[str, int]:
    """Build the deterministic multiclass mapping."""
    return {
        label: class_id
        for class_id, label in enumerate(allowed_labels)
    }


def _read_normalized_header(config: LabelHandlingConfig) -> list[str]:
    """Read and normalize the input header once for validation."""
    header_frame = pd.read_csv(config.input_data_path, nrows=0)
    return normalize_columns(list(header_frame.columns))


def _canonicalize_label_series(labels: pd.Series, config: LabelHandlingConfig) -> pd.Series:
    """Canonicalize a label series using the shared Phase 2 normalizer."""
    normalized = normalize_string_series(labels)
    return normalized.map(lambda value: canonicalize_label(value, config.label_aliases))


def _validate_chunk_labels(
    canonical_labels: pd.Series,
    allowed_labels: set[str],
) -> tuple[int, Counter[str]]:
    """Validate label completeness and vocabulary membership for a chunk."""
    missing_labels = int(canonical_labels.isna().sum())
    unknown_labels: Counter[str] = Counter()
    if not canonical_labels.empty:
        unknown_mask = canonical_labels.notna() & ~canonical_labels.isin(allowed_labels)
        if unknown_mask.any():
            unknown_labels.update(canonical_labels.loc[unknown_mask].astype(str).tolist())
    return missing_labels, unknown_labels


def run_label_handling_pipeline(
    config: LabelHandlingConfig,
    logger: logging.Logger | None = None,
) -> LabelHandlingReport:
    """Execute the complete Phase 3 label engineering workflow."""
    active_logger = logger or LOGGER
    config.ensure_directories()

    if not config.input_data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {config.input_data_path}")

    normalized_columns = _read_normalized_header(config)
    if config.label_column not in normalized_columns:
        raise ValueError(f"Required label column {config.label_column!r} not found in {config.input_data_path}")

    if config.output_path.exists():
        config.output_path.unlink()

    binary_mapping = build_binary_label_mapping(config.allowed_labels, config.benign_label)
    multiclass_mapping = build_multiclass_label_mapping(config.allowed_labels)
    allowed_label_set = set(config.allowed_labels)

    rows_read = 0
    rows_written = 0
    missing_labels = 0
    unknown_labels: Counter[str] = Counter()
    label_distribution: Counter[str] = Counter()
    binary_distribution: Counter[int] = Counter()
    multiclass_distribution: Counter[int] = Counter()
    header_written = False

    try:
        for chunk_index, chunk in enumerate(
            pd.read_csv(config.input_data_path, chunksize=config.chunk_size, low_memory=False),
            start=1,
        ):
            chunk.columns = normalize_columns(list(chunk.columns))
            rows_read += len(chunk)

            canonical_labels = _canonicalize_label_series(chunk[config.label_column], config)
            chunk_missing_labels, chunk_unknown_labels = _validate_chunk_labels(canonical_labels, allowed_label_set)
            if chunk_missing_labels or chunk_unknown_labels:
                missing_labels += chunk_missing_labels
                unknown_labels.update(chunk_unknown_labels)
                raise ValueError(
                    f"Phase 3 label validation failed in chunk {chunk_index}. "
                    f"missing_labels={chunk_missing_labels}, unknown_labels={dict(chunk_unknown_labels)}"
                )

            chunk[config.label_column] = canonical_labels
            chunk[config.binary_target_column] = canonical_labels.map(binary_mapping).astype("int8")
            chunk[config.multiclass_target_column] = canonical_labels.map(multiclass_mapping).astype("int16")

            label_distribution.update(canonical_labels.astype(str).tolist())
            binary_distribution.update(chunk[config.binary_target_column].astype(int).tolist())
            multiclass_distribution.update(chunk[config.multiclass_target_column].astype(int).tolist())

            chunk.to_csv(
                config.output_path,
                mode="a" if header_written else "w",
                header=not header_written,
                index=False,
            )
            header_written = True
            rows_written += len(chunk)

            active_logger.info(
                "Encoded chunk %d | rows=%d | cumulative_rows=%d",
                chunk_index,
                len(chunk),
                rows_written,
            )
    except Exception:
        if config.output_path.exists():
            config.output_path.unlink()
        raise

    mapping_payload = {
        "label_column": config.label_column,
        "binary_target_column": config.binary_target_column,
        "multiclass_target_column": config.multiclass_target_column,
        "benign_label": config.benign_label,
        "allowed_labels": list(config.allowed_labels),
        "binary_label_mapping": binary_mapping,
        "multiclass_label_mapping": multiclass_mapping,
        "inverse_multiclass_mapping": {str(class_id): label for label, class_id in multiclass_mapping.items()},
    }
    config.mapping_path.write_text(json.dumps(mapping_payload, indent=2), encoding="utf-8")

    validation_passed = (
        rows_read == rows_written
        and missing_labels == 0
        and not unknown_labels
        and binary_distribution.get(0, 0) == label_distribution.get(config.benign_label, 0)
        and sum(binary_distribution.values()) == rows_written
        and sum(multiclass_distribution.values()) == rows_written
    )

    report = LabelHandlingReport(
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        input_path=str(config.input_data_path),
        output_path=str(config.output_path),
        mapping_path=str(config.mapping_path),
        report_path=str(config.report_path),
        rows_read=rows_read,
        rows_written=rows_written,
        missing_labels=missing_labels,
        unknown_labels=dict(sorted(unknown_labels.items())),
        label_distribution=dict(sorted(label_distribution.items())),
        binary_target_distribution=dict(sorted(binary_distribution.items())),
        multiclass_target_distribution=dict(sorted(multiclass_distribution.items())),
        binary_label_mapping=binary_mapping,
        multiclass_label_mapping=multiclass_mapping,
        validation_passed=validation_passed,
        config=config.to_dict(),
    )
    config.report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if not validation_passed:
        raise ValueError(
            f"Phase 3 validation failed. rows_read={rows_read}, rows_written={rows_written}, "
            f"missing_labels={missing_labels}, unknown_labels={dict(unknown_labels)}"
        )

    active_logger.info(
        "Completed Phase 3 label engineering | rows=%d | classes=%d | output=%s",
        rows_written,
        len(multiclass_mapping),
        config.output_path,
    )
    return report

