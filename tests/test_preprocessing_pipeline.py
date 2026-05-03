"""Tests for the SentinelNet Phase 4 preprocessing pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.config import PreprocessingConfig
from src.data_pipeline.preprocessing import run_preprocessing_pipeline


def test_run_preprocessing_pipeline_creates_split_and_resampled_artifacts() -> None:
    project_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "preprocessing_test_workspace"
    shutil.rmtree(project_root, ignore_errors=True)
    (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    benign_protocols = ["tcp", "udp"]
    attack_labels = [
        ("Bot", 1, 12),
        ("DDoS", 2, 12),
        ("PortScan", 10, 12),
    ]

    for index in range(36):
        rows.append(
            {
                "feature_numeric": float(index),
                "feature_rate": float(index) / 10.0,
                "protocol": benign_protocols[index % len(benign_protocols)],
                "source_file": f"file_{index % 2}.csv",
                "label": "BENIGN",
                "label_binary": 0,
                "label_multiclass": 0,
            }
        )

    for label, class_id, count in attack_labels:
        for offset in range(count):
            rows.append(
                {
                    "feature_numeric": float(100 + class_id * 10 + offset),
                    "feature_rate": float(offset) / 5.0,
                    "protocol": "icmp" if offset % 2 == 0 else "tcp",
                    "source_file": f"{label.lower()}.csv",
                    "label": label,
                    "label_binary": 1,
                    "label_multiclass": class_id,
                }
            )

    input_frame = pd.DataFrame(rows)
    input_path = project_root / "data" / "processed" / "labeled_dataset.csv"
    input_frame.to_csv(input_path, index=False)

    config = PreprocessingConfig(
        project_root=project_root,
        input_data_path=input_path,
        output_dir=project_root / "data" / "processed" / "preprocessed",
        logs_dir=project_root / "logs",
        chunk_size=20,
        binary_majority_cap=20,
        multiclass_benign_cap=20,
        multiclass_attack_cap=15,
        multiclass_min_target_count=10,
    )

    try:
        report = run_preprocessing_pipeline(config)
        manifest = json.loads((config.feature_manifest_path).read_text(encoding="utf-8"))
        binary_train = np.load(config.binary_dir / "X_train_resampled.npy")
        binary_targets = np.load(config.binary_dir / "y_train_resampled.npy")
        binary_test = np.load(config.binary_dir / "X_test.npy")
        multiclass_train = np.load(config.multiclass_dir / "X_train_resampled.npy")
        multiclass_targets = np.load(config.multiclass_dir / "y_train_resampled.npy")

        assert report.validation_passed is True
        assert report.rows_read == len(input_frame)
        assert report.transformed_feature_count == len(manifest["feature_names"])
        assert "protocol_icmp" in manifest["feature_names"]
        assert "protocol_tcp" in manifest["feature_names"]
        assert binary_train.shape[1] == report.transformed_feature_count
        assert binary_test.shape[1] == report.transformed_feature_count
        assert multiclass_train.shape[1] == report.transformed_feature_count
        assert report.binary_train_distribution_after[0] == report.binary_train_distribution_after[1]
        assert set(np.unique(multiclass_targets)).issuperset({0, 1, 2, 10})
        assert not np.isnan(binary_train).any()
        assert not np.isnan(multiclass_train).any()
        assert set(np.unique(binary_targets)).issubset({0, 1})
    finally:
        shutil.rmtree(project_root, ignore_errors=True)

