"""Tests for the SentinelNet Phase 16 advanced response pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.deployment.phase16 import run_phase16_pipeline
from src.deployment.phase16_config import Phase16Config


class FakePhase16AnomalyScorer:
    """Deterministic anomaly scorer used to validate the Phase 16 workflow."""

    def score_batch(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        unique_ports = feature_frame["rolling_unique_destination_ports_w20"].to_numpy(dtype=float)
        burstiness = feature_frame["burstiness_score"].to_numpy(dtype=float)
        flow_bytes = feature_frame["flow_bytes_per_s"].to_numpy(dtype=float)

        high_novelty = (unique_ports >= 6.0) | (burstiness >= 2.5)
        medium_novelty = flow_bytes >= 40000.0

        detector_base = np.where(high_novelty, 0.93, np.where(medium_novelty, 0.72, 0.08))
        payload = {
            "phase8_isolation_forest_attack_probability": detector_base,
            "phase8_one_class_svm_attack_probability": np.clip(detector_base + 0.01, 0.0, 1.0),
            "phase8_lof_attack_probability": np.clip(detector_base - 0.01, 0.0, 1.0),
            "phase8_autoencoder_attack_probability": np.clip(detector_base + 0.02, 0.0, 1.0),
        }
        result = pd.DataFrame(payload)
        result["mean_anomaly_attack_probability"] = result.mean(axis=1)
        result["max_anomaly_attack_probability"] = result.max(axis=1)
        return result

    def close(self) -> None:
        """Match the production scorer interface."""


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_phase16_pipeline_writes_zero_day_actions_and_learning_queue() -> None:
    project_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "phase16_test_workspace"
    shutil.rmtree(project_root, ignore_errors=True)

    config_dir = project_root / "config"
    data_dir = project_root / "data" / "processed" / "feature_engineered"
    streaming_dir = project_root / "data" / "streaming"
    phase8_dir = project_root / "models" / "saved_models" / "phase8_anomaly_detection"
    logs_dir = project_root / "logs"
    for path in (config_dir, data_dir, streaming_dir, phase8_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    feature_names = [
        "destination_port",
        "flow_duration",
        "total_fwd_packets",
        "flow_bytes_per_s",
        "rolling_unique_destination_ports_w20",
        "forward_payload_efficiency",
        "burstiness_score",
    ]
    dataset = pd.DataFrame(
        [
            {
                "destination_port": 80,
                "flow_duration": 10.0,
                "total_fwd_packets": 8.0,
                "flow_bytes_per_s": 1500.0,
                "rolling_unique_destination_ports_w20": 1.0,
                "forward_payload_efficiency": 2.0,
                "burstiness_score": 0.2,
                "source_file": "train_a.csv",
                "label_binary": 0,
                "label_multiclass": 0,
            },
            {
                "destination_port": 443,
                "flow_duration": 12.0,
                "total_fwd_packets": 10.0,
                "flow_bytes_per_s": 2200.0,
                "rolling_unique_destination_ports_w20": 1.0,
                "forward_payload_efficiency": 2.5,
                "burstiness_score": 0.3,
                "source_file": "train_a.csv",
                "label_binary": 0,
                "label_multiclass": 0,
            },
            {
                "destination_port": 53,
                "flow_duration": 15.0,
                "total_fwd_packets": 6.0,
                "flow_bytes_per_s": 1800.0,
                "rolling_unique_destination_ports_w20": 1.0,
                "forward_payload_efficiency": 2.1,
                "burstiness_score": 0.25,
                "source_file": "train_b.csv",
                "label_binary": 0,
                "label_multiclass": 0,
            },
            {
                "destination_port": 22,
                "flow_duration": 18.0,
                "total_fwd_packets": 7.0,
                "flow_bytes_per_s": 1600.0,
                "rolling_unique_destination_ports_w20": 1.0,
                "forward_payload_efficiency": 2.3,
                "burstiness_score": 0.35,
                "source_file": "train_b.csv",
                "label_binary": 0,
                "label_multiclass": 0,
            },
            {
                "destination_port": 443,
                "flow_duration": 4.0,
                "total_fwd_packets": 240.0,
                "flow_bytes_per_s": 150000.0,
                "rolling_unique_destination_ports_w20": 8.0,
                "forward_payload_efficiency": 18.0,
                "burstiness_score": 3.0,
                "source_file": "test_novel.csv",
                "label_binary": 1,
                "label_multiclass": 2,
            },
            {
                "destination_port": 443,
                "flow_duration": 4.5,
                "total_fwd_packets": 255.0,
                "flow_bytes_per_s": 152000.0,
                "rolling_unique_destination_ports_w20": 8.0,
                "forward_payload_efficiency": 17.0,
                "burstiness_score": 2.8,
                "source_file": "test_novel.csv",
                "label_binary": 1,
                "label_multiclass": 2,
            },
            {
                "destination_port": 80,
                "flow_duration": 30.0,
                "total_fwd_packets": 90.0,
                "flow_bytes_per_s": 50000.0,
                "rolling_unique_destination_ports_w20": 2.0,
                "forward_payload_efficiency": 5.0,
                "burstiness_score": 0.9,
                "source_file": "test_known.csv",
                "label_binary": 1,
                "label_multiclass": 13,
            },
            {
                "destination_port": 53,
                "flow_duration": 20.0,
                "total_fwd_packets": 12.0,
                "flow_bytes_per_s": 2100.0,
                "rolling_unique_destination_ports_w20": 1.0,
                "forward_payload_efficiency": 2.0,
                "burstiness_score": 0.2,
                "source_file": "test_normal.csv",
                "label_binary": 0,
                "label_multiclass": 0,
            },
        ]
    )
    selected_dataset_path = data_dir / "selected_dataset.csv"
    dataset.to_csv(selected_dataset_path, index=False)

    _write_json(data_dir / "selected_feature_manifest.json", {"selected_features": feature_names})
    _write_json(
        project_root / "data" / "processed" / "label_mappings.json",
        {"inverse_multiclass_mapping": {"0": "BENIGN", "2": "DDoS", "13": "Web Attack - Sql Injection"}},
    )
    np.save(data_dir / "train_indices.npy", np.asarray([0, 1, 2, 3], dtype=np.int64))
    np.save(data_dir / "test_indices.npy", np.asarray([4, 5, 6, 7], dtype=np.int64))

    enriched_predictions = pd.DataFrame(
        [
            {
                "stream_order": 0,
                "original_index": 4,
                "event_time_utc": "2026-01-01T00:00:00+00:00",
                "source_file": "test_novel.csv",
                "true_binary_label": 1,
                "true_binary_label_name": "ATTACK",
                "true_multiclass_label_name": "DDoS",
                "predicted_binary_label": 0,
                "predicted_binary_label_name": "BENIGN",
                "binary_attack_probability": 0.40,
                "predicted_multiclass_label_name": "BENIGN",
                "multiclass_confidence": 0.30,
                "risk_score": 68.0,
                "alert_level": "Suspicious",
                "is_alert": True,
                "selected_binary_variant": "weighted_scoring",
                "selected_multiclass_variant": "stacking",
            },
            {
                "stream_order": 1,
                "original_index": 5,
                "event_time_utc": "2026-01-01T00:00:01+00:00",
                "source_file": "test_novel.csv",
                "true_binary_label": 1,
                "true_binary_label_name": "ATTACK",
                "true_multiclass_label_name": "DDoS",
                "predicted_binary_label": 0,
                "predicted_binary_label_name": "BENIGN",
                "binary_attack_probability": 0.45,
                "predicted_multiclass_label_name": "BENIGN",
                "multiclass_confidence": 0.28,
                "risk_score": 82.0,
                "alert_level": "Attack",
                "is_alert": True,
                "selected_binary_variant": "weighted_scoring",
                "selected_multiclass_variant": "stacking",
            },
            {
                "stream_order": 2,
                "original_index": 6,
                "event_time_utc": "2026-01-01T00:00:02+00:00",
                "source_file": "test_known.csv",
                "true_binary_label": 1,
                "true_binary_label_name": "ATTACK",
                "true_multiclass_label_name": "Web Attack - Sql Injection",
                "predicted_binary_label": 1,
                "predicted_binary_label_name": "ATTACK",
                "binary_attack_probability": 0.92,
                "predicted_multiclass_label_name": "Web Attack - Sql Injection",
                "multiclass_confidence": 0.94,
                "risk_score": 96.0,
                "alert_level": "Attack",
                "is_alert": True,
                "selected_binary_variant": "weighted_scoring",
                "selected_multiclass_variant": "stacking",
            },
            {
                "stream_order": 3,
                "original_index": 7,
                "event_time_utc": "2026-01-01T00:00:03+00:00",
                "source_file": "test_normal.csv",
                "true_binary_label": 0,
                "true_binary_label_name": "BENIGN",
                "true_multiclass_label_name": "BENIGN",
                "predicted_binary_label": 0,
                "predicted_binary_label_name": "BENIGN",
                "binary_attack_probability": 0.03,
                "predicted_multiclass_label_name": "BENIGN",
                "multiclass_confidence": 0.98,
                "risk_score": 4.0,
                "alert_level": "Normal",
                "is_alert": False,
                "selected_binary_variant": "weighted_scoring",
                "selected_multiclass_variant": "stacking",
            },
        ]
    )
    predictions_path = streaming_dir / "stream_predictions_with_alerts.csv"
    enriched_predictions.to_csv(predictions_path, index=False)

    _write_json(phase8_dir / "anomaly_detection_report.json", {"validation_passed": True})

    _write_json(
        config_dir / "streaming_config.json",
        {
            "input_data_path": str(selected_dataset_path),
            "feature_manifest_path": str(data_dir / "selected_feature_manifest.json"),
            "label_mapping_path": str(project_root / "data" / "processed" / "label_mappings.json"),
            "train_indices_path": str(data_dir / "train_indices.npy"),
            "test_indices_path": str(data_dir / "test_indices.npy"),
            "phase6_output_dir": str(project_root / "models" / "saved_models" / "phase6_ml"),
            "phase7_output_dir": str(project_root / "models" / "saved_models" / "phase7_deep_learning"),
            "phase8_output_dir": str(phase8_dir),
            "phase9_output_dir": str(project_root / "models" / "saved_models" / "phase9_ensemble"),
            "output_dir": str(streaming_dir),
            "logs_dir": str(logs_dir),
            "stream_split": "test",
            "chunk_size": 4,
            "inference_batch_size": 2,
        },
    )

    config = Phase16Config(
        project_root=project_root,
        streaming_config_path=config_dir / "streaming_config.json",
        input_predictions_path=predictions_path,
        output_dir=streaming_dir / "phase16_advanced",
        logs_dir=logs_dir,
        processing_batch_size=2,
        continuous_learning_max_queue=10,
        drift_reference_max_rows=4,
        retraining_zero_day_trigger=1,
        retraining_queue_trigger=2,
        retraining_drift_feature_trigger=1,
    )

    report = run_phase16_pipeline(
        config,
        anomaly_scorer_factory=lambda *_args, **_kwargs: FakePhase16AnomalyScorer(),
    )

    assert report.validation_passed is True
    assert report.total_rows_processed == 4
    assert report.zero_day_candidate_rows == 2
    assert report.auto_block_rows == 1
    assert report.continuous_learning_queue_rows >= 2
    assert report.retraining_recommended is True

    classified = pd.read_csv(config.classified_predictions_path)
    zero_day = pd.read_csv(config.zero_day_candidates_path)
    autoblock = pd.read_csv(config.autoblock_actions_path)
    learning_queue = pd.read_csv(config.continuous_learning_queue_path)
    drift_frame = pd.read_csv(config.feature_drift_path)
    retraining_manifest = json.loads(config.retraining_manifest_path.read_text(encoding="utf-8"))

    assert len(classified) == 4
    assert len(zero_day) == 2
    assert len(autoblock) == 1
    assert len(learning_queue) >= 2
    assert drift_frame["drifted"].any()
    assert retraining_manifest["retraining_recommended"] is True
    assert "UNKNOWN-NOVELTY" in zero_day["operational_attack_label_name"].tolist()
    assert autoblock.iloc[0]["auto_block_action"] == "SIMULATED_DESTINATION_PORT_QUARANTINE"
