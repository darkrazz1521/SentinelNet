"""Tests for the SentinelNet Phase 2 cleaning pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.data_pipeline.cleaning import build_label_lookup_key, canonicalize_label, run_cleaning_pipeline
from src.data_pipeline.config import CleaningConfig, default_label_aliases


def test_canonicalize_label_normalizes_web_attack_variants() -> None:
    aliases = default_label_aliases()
    assert build_label_lookup_key("Web Attack \ufffd Brute Force") == "web attack brute force"
    assert canonicalize_label("Web Attack \ufffd Brute Force", aliases) == "Web Attack - Brute Force"
    assert canonicalize_label("  DDoS  ", aliases) == "DDoS"


def test_run_cleaning_pipeline_removes_duplicates_and_repairs_missing_values() -> None:
    project_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "cleaning_test_workspace"
    shutil.rmtree(project_root, ignore_errors=True)
    (project_root / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    combined_path = project_root / "data" / "interim" / "combined.csv"
    combined_path.write_text(
        " destination_port, flow_duration,total_fwd_packets,total_backward_packets,flow_bytes_per_s,flow_packets_per_s,label,source_file\n"
        "80,10,1,1,inf,5,BENIGN,file_a.csv\n"
        "80,10,1,1,inf,5,BENIGN,file_b.csv\n"
        "443,20,2,2,,7,Web Attack \ufffd Brute Force,file_c.csv\n"
        "53,30,3,3,11,, ,file_d.csv\n",
        encoding="utf-8",
    )

    config = CleaningConfig(
        project_root=project_root,
        input_data_path=combined_path,
        interim_data_dir=project_root / "data" / "interim",
        logs_dir=project_root / "logs",
        chunk_size=2,
    )

    try:
        report = run_cleaning_pipeline(config)
        cleaned = pd.read_csv(project_root / "data" / "interim" / "cleaned.csv")

        assert report.rows_read == 4
        assert report.rows_written == 2
        assert report.duplicate_rows_removed == 1
        assert report.rows_dropped_missing_critical == 1
        assert report.validation_passed is True
        assert cleaned.isna().sum().sum() == 0
        assert set(cleaned["label"]) == {"BENIGN", "Web Attack - Brute Force"}
        assert cleaned.loc[cleaned["label"] == "BENIGN", "flow_bytes_per_s"].iloc[0] == 0.0
    finally:
        shutil.rmtree(project_root, ignore_errors=True)

