"""Multi-file CSV ingestion pipeline for SentinelNet Phase 1."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import IngestionConfig
from .optimizer import estimate_frame_memory_mb, optimize_dtypes
from .schema import align_frame_to_schema, normalize_columns, ordered_union, schema_difference

LOGGER = logging.getLogger("sentinelnet.phase1")


@dataclass(slots=True)
class SourceFileProfile:
    """Metadata captured for each discovered source file."""

    file_name: str
    file_path: str
    encoding: str
    delimiter: str
    raw_columns: list[str]
    normalized_columns: list[str]
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    rows_ingested: int = 0
    chunks_processed: int = 0


@dataclass(slots=True)
class IngestionReport:
    """Serializable report returned after pipeline completion."""

    created_at_utc: str
    output_path: str
    report_path: str
    files_processed: int
    total_rows: int
    total_columns: int
    canonical_columns: list[str]
    schema_mismatch_detected: bool
    profiles: list[SourceFileProfile]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report into a JSON-serializable dictionary."""
        return {
            "created_at_utc": self.created_at_utc,
            "output_path": self.output_path,
            "report_path": self.report_path,
            "files_processed": self.files_processed,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "canonical_columns": self.canonical_columns,
            "schema_mismatch_detected": self.schema_mismatch_detected,
            "profiles": [asdict(profile) for profile in self.profiles],
            "config": self.config,
        }


def discover_source_files(raw_data_dir: Path) -> list[Path]:
    """Find all CSV files beneath the raw data directory."""
    csv_files = sorted(raw_data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_data_dir}")
    return csv_files


def detect_encoding(file_path: Path, candidates: tuple[str, ...]) -> str:
    """Detect a workable file encoding from a prioritized list."""
    for encoding in candidates:
        try:
            with file_path.open("r", encoding=encoding) as handle:
                handle.read(16384)
            return encoding
        except UnicodeDecodeError:
            continue

    return candidates[-1]


def detect_delimiter(file_path: Path, encoding: str, candidates: tuple[str, ...]) -> str:
    """Detect the CSV delimiter using a sample of the file."""
    with file_path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        sample = handle.read(16384)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=list(candidates))
        return dialect.delimiter
    except csv.Error:
        return ","


def read_header_columns(file_path: Path, delimiter: str, encoding: str) -> list[str]:
    """Read header columns using the same parser as the chunk loader."""
    header_frame = pd.read_csv(
        file_path,
        sep=delimiter,
        encoding=encoding,
        encoding_errors="replace",
        nrows=0,
    )
    return [str(column) for column in header_frame.columns]


def profile_source_files(config: IngestionConfig, logger: logging.Logger | None = None) -> list[SourceFileProfile]:
    """Profile source files before chunked ingestion begins."""
    active_logger = logger or LOGGER
    profiles: list[SourceFileProfile] = []

    for file_path in discover_source_files(config.raw_data_dir):
        encoding = detect_encoding(file_path, config.encoding_candidates)
        delimiter = detect_delimiter(file_path, encoding, config.delimiter_candidates)
        raw_columns = read_header_columns(file_path, delimiter, encoding)
        normalized_columns = normalize_columns(raw_columns)
        profiles.append(
            SourceFileProfile(
                file_name=file_path.name,
                file_path=str(file_path),
                encoding=encoding,
                delimiter=delimiter,
                raw_columns=raw_columns,
                normalized_columns=normalized_columns,
            )
        )
        active_logger.info(
            "Profiled %s | encoding=%s | delimiter=%r | columns=%d",
            file_path.name,
            encoding,
            delimiter,
            len(raw_columns),
        )

    if profiles:
        reference_columns = profiles[0].normalized_columns
        for profile in profiles[1:]:
            missing_columns, extra_columns = schema_difference(reference_columns, profile.normalized_columns)
            profile.missing_columns = missing_columns
            profile.extra_columns = extra_columns
            if missing_columns or extra_columns:
                active_logger.warning(
                    "Schema mismatch in %s | missing=%s | extra=%s",
                    profile.file_name,
                    missing_columns,
                    extra_columns,
                )

    return profiles


def _iter_aligned_chunks(
    profile: SourceFileProfile,
    canonical_columns: list[str],
    config: IngestionConfig,
) -> pd.io.parsers.TextFileReader:
    """Yield aligned chunks from a source file."""
    reader = pd.read_csv(
        profile.file_path,
        sep=profile.delimiter,
        encoding=profile.encoding,
        encoding_errors="replace",
        chunksize=config.chunk_size,
        low_memory=False,
        on_bad_lines="warn",
    )

    for raw_chunk in reader:
        raw_chunk.columns = normalize_columns(list(raw_chunk.columns))
        aligned_chunk = align_frame_to_schema(raw_chunk, canonical_columns)
        aligned_chunk["source_file"] = profile.file_name
        yield optimize_dtypes(aligned_chunk, protected_columns=("label", "source_file"))


def run_ingestion_pipeline(
    config: IngestionConfig,
    logger: logging.Logger | None = None,
) -> IngestionReport:
    """Run the complete Phase 1 ingestion workflow."""
    active_logger = logger or LOGGER
    config.ensure_directories()
    profiles = profile_source_files(config, active_logger)
    canonical_columns = ordered_union(profile.normalized_columns for profile in profiles)
    output_columns = canonical_columns + ["source_file"]

    active_logger.info("Canonical schema resolved with %d columns", len(canonical_columns))
    active_logger.info("Writing combined dataset to %s", config.output_path)

    total_rows = 0
    header_written = False

    for profile in profiles:
        active_logger.info("Loading %s", profile.file_name)
        for chunk_index, chunk in enumerate(_iter_aligned_chunks(profile, canonical_columns, config), start=1):
            chunk = chunk.reindex(columns=output_columns)
            memory_mb = estimate_frame_memory_mb(chunk)
            chunk.to_csv(
                config.output_path,
                mode="a" if header_written else "w",
                header=not header_written,
                index=False,
            )
            header_written = True
            row_count = len(chunk)
            total_rows += row_count
            profile.rows_ingested += row_count
            profile.chunks_processed = chunk_index
            active_logger.info(
                "Wrote %s chunk %d | rows=%d | memory_mb=%.2f",
                profile.file_name,
                chunk_index,
                row_count,
                memory_mb,
            )

    report = IngestionReport(
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        output_path=str(config.output_path),
        report_path=str(config.report_path),
        files_processed=len(profiles),
        total_rows=total_rows,
        total_columns=len(output_columns),
        canonical_columns=canonical_columns,
        schema_mismatch_detected=any(profile.missing_columns or profile.extra_columns for profile in profiles),
        profiles=profiles,
        config=config.to_dict(),
    )

    config.report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    active_logger.info(
        "Completed Phase 1 ingestion | files=%d | rows=%d | output=%s | report=%s",
        report.files_processed,
        report.total_rows,
        report.output_path,
        report.report_path,
    )
    return report

