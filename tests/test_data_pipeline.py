"""Tests for the SentinelNet Phase 1 ingestion pipeline."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from src.data_pipeline.config import IngestionConfig
from src.data_pipeline.ingestion import run_ingestion_pipeline
from src.data_pipeline.optimizer import optimize_dtypes
from src.data_pipeline.schema import align_frame_to_schema, normalize_columns


def test_normalize_columns_disambiguates_duplicates() -> None:
    columns = [" Flow Duration", "Flow Duration", "Label", "Label"]
    assert normalize_columns(columns) == [
        "flow_duration",
        "flow_duration__dup1",
        "label",
        "label__dup1",
    ]


def test_align_frame_to_schema_preserves_order_and_adds_missing_columns() -> None:
    frame = pd.DataFrame({"label": ["BENIGN"], "flow_duration": [5]})
    aligned = align_frame_to_schema(frame, ["flow_duration", "destination_port", "label"])

    assert list(aligned.columns) == ["flow_duration", "destination_port", "label"]
    assert pd.isna(aligned.loc[0, "destination_port"])


def test_optimize_dtypes_downcasts_numeric_columns() -> None:
    frame = pd.DataFrame(
        {
            "flow_duration": [1, 2, 3],
            "flow_bytes_s": [1.0, 2.5, 3.5],
            "label": ["BENIGN", "DDoS", "BENIGN"],
        }
    )

    optimized = optimize_dtypes(frame, protected_columns=("label",))

    assert str(optimized["label"].dtype) == "string"
    assert str(optimized["flow_duration"].dtype) in {"int8", "int16", "int32", "int64"}
    assert str(optimized["flow_bytes_s"].dtype) in {"float32", "float64"}


def test_run_ingestion_pipeline_merges_mismatched_files() -> None:
    local_temp_root = Path(__file__).resolve().parents[1] / ".tmp_tests"
    local_temp_root.mkdir(parents=True, exist_ok=True)
    temp_project_root = local_temp_root / "ingestion_test_workspace"

    try:
        shutil.rmtree(temp_project_root, ignore_errors=True)
        temp_project_root.mkdir(parents=True, exist_ok=True)
        raw_dir = temp_project_root / "data" / "raw"
        interim_dir = temp_project_root / "data" / "interim"
        logs_dir = temp_project_root / "logs"
        raw_dir.mkdir(parents=True)
        interim_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)

        (raw_dir / "part_a.csv").write_text(
            " Flow Duration, Label\n"
            "10,BENIGN\n",
            encoding="utf-8",
        )
        (raw_dir / "part_b.csv").write_text(
            "Flow Duration ,Destination Port,Label\n"
            "20,443,ATTACK\n",
            encoding="utf-8",
        )

        config = IngestionConfig(
            project_root=temp_project_root,
            raw_data_dir=raw_dir,
            interim_data_dir=interim_dir,
            logs_dir=logs_dir,
            chunk_size=1,
        )
        logger = logging.getLogger("sentinelnet.test")
        report = run_ingestion_pipeline(config, logger)

        combined = pd.read_csv(interim_dir / "combined.csv")

        assert report.files_processed == 2
        assert report.total_rows == 2
        assert report.schema_mismatch_detected is True
        assert "source_file" in combined.columns
        assert set(combined["source_file"]) == {"part_a.csv", "part_b.csv"}
        assert "destination_port" in combined.columns
    finally:
        shutil.rmtree(temp_project_root, ignore_errors=True)
