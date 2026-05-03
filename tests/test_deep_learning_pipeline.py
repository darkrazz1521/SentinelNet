"""Tests for the SentinelNet Phase 7 deep-learning pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.deep_learning.config import DeepLearningConfig
from src.models.deep_learning.training import run_deep_learning_pipeline


def _build_phase7_fixture(rows_per_block: int = 24) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    class_specs = [
        ("BENIGN", 0, 0, 0.0),
        ("Bot", 1, 1, 4.0),
        ("DDoS", 1, 2, 8.0),
        ("PortScan", 1, 10, 12.0),
    ]
    for class_index, (label, binary_target, multiclass_target, offset) in enumerate(class_specs):
        for file_number in range(2):
            source_file = f"{label.lower().replace(' ', '_')}_{file_number}.csv"
            for index in range(rows_per_block):
                drift = class_index * 0.5 + file_number * 0.25
                rows.append(
                    {
                        "flow_duration": 1000 + offset * 100 + index * 11 + drift,
                        "flow_bytes_per_s": 50 + offset * 3 + index * 0.8 + drift,
                        "packet_length_variance": 10 + offset + index * 0.3,
                        "fwd_header_length": 20 + offset + index * 0.2,
                        "destination_port": 80 + class_index * 100 + (index % 12),
                        "rolling_unique_destination_ports_w20": 1 + (index % 5) + offset / 6.0,
                        "burstiness_score": 0.1 + index / 200.0 + offset / 50.0,
                        "forward_payload_efficiency": 0.2 + index / 150.0 + offset / 40.0,
                        "label": label,
                        "source_file": source_file,
                        "label_binary": binary_target,
                        "label_multiclass": multiclass_target,
                    }
                )
    return pd.DataFrame(rows)


def test_run_deep_learning_pipeline_trains_phase7_models() -> None:
    project_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "deep_learning_test_workspace"
    shutil.rmtree(project_root, ignore_errors=True)
    (project_root / "data" / "processed" / "feature_engineered").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (project_root / "models" / "saved_models").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    frame = _build_phase7_fixture()
    input_path = project_root / "data" / "processed" / "feature_engineered" / "selected_dataset.csv"
    manifest_path = project_root / "data" / "processed" / "feature_engineered" / "selected_feature_manifest.json"
    mapping_path = project_root / "data" / "processed" / "label_mappings.json"
    train_indices_path = project_root / "data" / "processed" / "feature_engineered" / "train_indices.npy"
    test_indices_path = project_root / "data" / "processed" / "feature_engineered" / "test_indices.npy"
    frame.to_csv(input_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "selected_features": [
                    "flow_duration",
                    "flow_bytes_per_s",
                    "packet_length_variance",
                    "fwd_header_length",
                    "destination_port",
                    "rolling_unique_destination_ports_w20",
                    "burstiness_score",
                    "forward_payload_efficiency",
                ]
            }
        ),
        encoding="utf-8",
    )
    mapping_path.write_text(
        json.dumps(
            {
                "inverse_multiclass_mapping": {
                    "0": "BENIGN",
                    "1": "Bot",
                    "2": "DDoS",
                    "10": "PortScan",
                }
            }
        ),
        encoding="utf-8",
    )

    train_indices = []
    test_indices = []
    for _, group in frame.groupby("source_file", sort=False):
        group_indices = group.index.to_numpy(dtype=np.int64)
        train_indices.extend(group_indices[::2].tolist())
        test_indices.extend(group_indices[1::2].tolist())
    np.save(train_indices_path, np.asarray(train_indices, dtype=np.int64))
    np.save(test_indices_path, np.asarray(test_indices, dtype=np.int64))

    config = DeepLearningConfig(
        project_root=project_root,
        input_data_path=input_path,
        feature_manifest_path=manifest_path,
        label_mapping_path=mapping_path,
        train_indices_path=train_indices_path,
        test_indices_path=test_indices_path,
        output_dir=project_root / "models" / "saved_models" / "phase7_deep_learning",
        logs_dir=project_root / "logs",
        batch_size=16,
        epochs=2,
        patience=1,
        validation_size=0.2,
        learning_rate=0.001,
        tf_intra_op_threads=1,
        tf_inter_op_threads=1,
        dnn_hidden_units=(32, 16),
        dnn_dropout=0.1,
        lstm_units=(16,),
        lstm_dropout=0.1,
        lstm_sequence_length=4,
        lstm_stride=2,
        autoencoder_hidden_units=(32, 16),
        autoencoder_latent_dim=8,
        binary_train_cap=64,
        binary_test_cap=32,
        multiclass_train_cap=96,
        multiclass_test_cap=48,
        lstm_train_sequence_cap=32,
        lstm_test_sequence_cap=16,
        autoencoder_train_cap=48,
        autoencoder_test_cap=32,
    )

    try:
        report = run_deep_learning_pipeline(config)
        metrics_summary = pd.read_csv(config.metrics_path)
        scaler = joblib.load(config.scaler_path)

        assert report.validation_passed is True
        assert len(report.models) == 5
        assert len(metrics_summary) == 5
        assert scaler is not None
        assert (config.dnn_dir / "binary.keras").exists()
        assert (config.dnn_dir / "multiclass.keras").exists()
        assert (config.lstm_dir / "binary.keras").exists()
        assert (config.lstm_dir / "multiclass.keras").exists()
        assert (config.autoencoder_dir / "autoencoder.keras").exists()
        assert set(metrics_summary["model"]) == {"dnn", "lstm", "autoencoder"}
        assert set(metrics_summary["task"]) == {"binary", "multiclass", "binary_anomaly"}
    finally:
        shutil.rmtree(project_root, ignore_errors=True)
