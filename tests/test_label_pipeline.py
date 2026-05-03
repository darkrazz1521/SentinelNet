"""Tests for the SentinelNet Phase 3 label engineering pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline.config import LabelHandlingConfig, default_allowed_labels
from src.data_pipeline.label_handling import (
    build_binary_label_mapping,
    build_multiclass_label_mapping,
    run_label_handling_pipeline,
)


def test_label_mappings_fix_benign_to_zero() -> None:
    allowed_labels = default_allowed_labels()
    binary_mapping = build_binary_label_mapping(allowed_labels, "BENIGN")
    multiclass_mapping = build_multiclass_label_mapping(allowed_labels)

    assert binary_mapping["BENIGN"] == 0
    assert binary_mapping["DDoS"] == 1
    assert multiclass_mapping["BENIGN"] == 0
    assert multiclass_mapping["Web Attack - XSS"] > 0


def test_run_label_handling_pipeline_creates_targets_and_mappings() -> None:
    project_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "label_test_workspace"
    shutil.rmtree(project_root, ignore_errors=True)
    (project_root / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    cleaned_path = project_root / "data" / "interim" / "cleaned.csv"
    cleaned_path.write_text(
        "destination_port,flow_duration,label,source_file\n"
        "80,10,BENIGN,file_a.csv\n"
        "443,20,DDoS,file_b.csv\n"
        "8080,30,Web Attack \ufffd Brute Force,file_c.csv\n",
        encoding="utf-8",
    )

    config = LabelHandlingConfig(
        project_root=project_root,
        input_data_path=cleaned_path,
        processed_data_dir=project_root / "data" / "processed",
        logs_dir=project_root / "logs",
        chunk_size=2,
    )

    try:
        report = run_label_handling_pipeline(config)
        labeled = pd.read_csv(project_root / "data" / "processed" / "labeled_dataset.csv")
        mappings = json.loads((project_root / "data" / "processed" / "label_mappings.json").read_text(encoding="utf-8"))

        assert report.rows_read == 3
        assert report.rows_written == 3
        assert report.missing_labels == 0
        assert report.validation_passed is True
        assert set(labeled["label"]) == {"BENIGN", "DDoS", "Web Attack - Brute Force"}
        assert labeled.loc[labeled["label"] == "BENIGN", "label_binary"].iloc[0] == 0
        assert labeled.loc[labeled["label"] == "DDoS", "label_binary"].iloc[0] == 1
        assert labeled.loc[labeled["label"] == "BENIGN", "label_multiclass"].iloc[0] == 0
        assert mappings["multiclass_label_mapping"]["BENIGN"] == 0
    finally:
        shutil.rmtree(project_root, ignore_errors=True)


def test_run_label_handling_pipeline_rejects_unknown_labels() -> None:
    project_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "label_unknown_test_workspace"
    shutil.rmtree(project_root, ignore_errors=True)
    (project_root / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    cleaned_path = project_root / "data" / "interim" / "cleaned.csv"
    cleaned_path.write_text(
        "destination_port,flow_duration,label,source_file\n"
        "80,10,Totally New Attack,file_a.csv\n",
        encoding="utf-8",
    )

    config = LabelHandlingConfig(
        project_root=project_root,
        input_data_path=cleaned_path,
        processed_data_dir=project_root / "data" / "processed",
        logs_dir=project_root / "logs",
        chunk_size=1,
    )

    try:
        with pytest.raises(ValueError):
            run_label_handling_pipeline(config)
        assert not (project_root / "data" / "processed" / "labeled_dataset.csv").exists()
    finally:
        shutil.rmtree(project_root, ignore_errors=True)

