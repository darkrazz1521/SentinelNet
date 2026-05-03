"""Tests for the SentinelNet Phase 14 FastAPI application."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.config import ApiConfig
from api.fastapi_app import create_app
from api.service import SentinelNetApiService


class FakePredictor:
    """Lightweight predictor used to validate the Phase 14 API contract."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.selected_variants = SimpleNamespace(binary_variant="weighted_scoring", multiclass_variant="stacking")

    def predict_batch(self, batch):
        rows = len(batch.feature_frame)
        attack_probability = np.clip(batch.feature_frame.iloc[:, 1].to_numpy(dtype=float), 0.0, 1.0)
        binary_probabilities = np.column_stack([1.0 - attack_probability, attack_probability])
        binary_predicted_labels = (attack_probability >= 0.5).astype(np.int32)

        multiclass_probabilities = np.zeros((rows, 3), dtype=np.float64)
        multiclass_predicted_labels = np.zeros(rows, dtype=np.int32)
        for index, probability in enumerate(attack_probability):
            if probability >= 0.8:
                multiclass_probabilities[index] = [0.02, 0.03, 0.95]
                multiclass_predicted_labels[index] = 2
            elif probability >= 0.5:
                multiclass_probabilities[index] = [0.10, 0.85, 0.05]
                multiclass_predicted_labels[index] = 10
            else:
                multiclass_probabilities[index] = [0.97, 0.02, 0.01]
                multiclass_predicted_labels[index] = 0

        return SimpleNamespace(
            binary_variant="weighted_scoring",
            multiclass_variant="stacking",
            binary_probabilities=binary_probabilities,
            binary_predicted_labels=binary_predicted_labels,
            multiclass_probabilities=multiclass_probabilities,
            multiclass_predicted_labels=multiclass_predicted_labels,
        )

    def close(self) -> None:
        """Match the real predictor interface."""


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_phase14_fastapi_app_serves_predict_stream_alerts_and_metrics() -> None:
    project_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "api_test_workspace"
    shutil.rmtree(project_root, ignore_errors=True)

    config_dir = project_root / "config"
    streaming_dir = project_root / "data" / "streaming"
    processed_dir = project_root / "data" / "processed" / "feature_engineered"
    explainability_dir = project_root / "models" / "saved_models" / "phase10_explainability"
    phase9_dir = project_root / "models" / "saved_models" / "phase9_ensemble"
    logs_dir = project_root / "logs"
    for path in [
        config_dir,
        streaming_dir,
        processed_dir,
        explainability_dir / "phase6" / "shap_values",
        explainability_dir / "phase9",
        phase9_dir,
        logs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    feature_manifest_path = processed_dir / "selected_feature_manifest.json"
    label_mapping_path = project_root / "data" / "processed" / "label_mappings.json"
    train_indices_path = processed_dir / "train_indices.npy"
    test_indices_path = processed_dir / "test_indices.npy"

    _write_json(feature_manifest_path, {"selected_features": ["flow_duration", "burstiness_score"]})
    _write_json(
        label_mapping_path,
        {
            "inverse_multiclass_mapping": {
                "0": "BENIGN",
                "2": "DDoS",
                "10": "PortScan",
            }
        },
    )
    np.save(train_indices_path, np.asarray([0, 1], dtype=np.int64))
    np.save(test_indices_path, np.asarray([2, 3], dtype=np.int64))

    predictions = pd.DataFrame(
        [
            {
                "stream_order": 0,
                "original_index": 0,
                "event_time_utc": "2026-01-01T00:00:00+00:00",
                "source_file": "monday.csv",
                "true_binary_label": 0,
                "true_binary_label_name": "BENIGN",
                "true_multiclass_label_name": "BENIGN",
                "predicted_binary_label": 0,
                "predicted_binary_label_name": "BENIGN",
                "binary_attack_probability": 0.04,
                "predicted_multiclass_label_name": "BENIGN",
                "multiclass_confidence": 0.99,
            },
            {
                "stream_order": 1,
                "original_index": 1,
                "event_time_utc": "2026-01-01T00:00:10+00:00",
                "source_file": "tuesday.csv",
                "true_binary_label": 1,
                "true_binary_label_name": "ATTACK",
                "true_multiclass_label_name": "PortScan",
                "predicted_binary_label": 1,
                "predicted_binary_label_name": "ATTACK",
                "binary_attack_probability": 0.62,
                "predicted_multiclass_label_name": "PortScan",
                "multiclass_confidence": 0.88,
            },
            {
                "stream_order": 2,
                "original_index": 2,
                "event_time_utc": "2026-01-01T00:00:20+00:00",
                "source_file": "wednesday.csv",
                "true_binary_label": 1,
                "true_binary_label_name": "ATTACK",
                "true_multiclass_label_name": "DDoS",
                "predicted_binary_label": 1,
                "predicted_binary_label_name": "ATTACK",
                "binary_attack_probability": 0.96,
                "predicted_multiclass_label_name": "DDoS",
                "multiclass_confidence": 0.97,
            },
        ]
    )
    predictions.to_csv(streaming_dir / "stream_predictions.csv", index=False)

    enriched = predictions.assign(
        risk_score=[2.5, 62.0, 84.0],
        alert_level=["Normal", "Suspicious", "Attack"],
        is_alert=[False, True, True],
        alert_id=["", "SNT-SUSPICIOUS-00000001", "SNT-ATTACK-00000002"],
        alert_timestamp_utc=predictions["event_time_utc"],
        alert_generated_at_utc=predictions["event_time_utc"],
        recommended_action=[
            "Continue monitoring.",
            "Increase monitoring, capture packet context, and validate against baseline.",
            "Trigger containment workflow and rate-limit suspected sources.",
        ],
        alert_message=[
            "Normal traffic confidence within acceptable range.",
            "Suspicious alert for PortScan | risk=62.00 | binary_attack_probability=0.6200",
            "Attack alert for DDoS | risk=84.00 | binary_attack_probability=0.9600",
        ],
        alert_rule_version="phase12_v1",
        selected_binary_variant="weighted_scoring",
        selected_multiclass_variant="stacking",
    )
    enriched.to_csv(streaming_dir / "stream_predictions_with_alerts.csv", index=False)
    enriched.loc[enriched["is_alert"]].to_csv(streaming_dir / "alerts.csv", index=False)

    _write_json(
        streaming_dir / "streaming_report.json",
        {
            "rows_streamed": 3,
            "selected_binary_variant": "weighted_scoring",
            "selected_multiclass_variant": "stacking",
            "throughput_rows_per_second": 111.1,
            "average_batch_latency_ms": 75.0,
        },
    )
    _write_json(
        streaming_dir / "alerting_report.json",
        {
            "alert_rows_written": 2,
            "level_counts": {"Normal": 1, "Suspicious": 1, "Attack": 1},
            "predicted_attack_counts": {"PortScan": 1, "DDoS": 1},
            "average_risk_score": 49.5,
            "max_risk_score": 84.0,
        },
    )

    pd.DataFrame(
        [
            {"task": "binary", "ensemble": "weighted_scoring", "status": "trained", "accuracy": 0.99, "precision": 0.99, "recall": 0.99, "f1_score": 0.99, "roc_auc": 0.999},
            {"task": "multiclass", "ensemble": "stacking", "status": "trained", "accuracy": 0.98, "precision": 0.98, "recall": 0.98, "f1_score": 0.98, "roc_auc": 0.997},
        ]
    ).to_csv(phase9_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(
        [{"feature_name": "flow_duration", "mean_abs_contribution": 0.41, "mean_contribution": -0.03}]
    ).to_csv(explainability_dir / "phase6" / "shap_values" / "binary_lightgbm_summary.csv", index=False)
    pd.DataFrame(
        [{"feature_name": "burstiness_score", "mean_abs_contribution": 0.37, "mean_contribution": 0.02}]
    ).to_csv(explainability_dir / "phase6" / "shap_values" / "multiclass_random_forest_summary.csv", index=False)
    pd.DataFrame(
        [{"feature_name": "phase6_lightgbm", "mean_abs_contribution": 0.29, "mean_contribution": -0.01}]
    ).to_csv(explainability_dir / "phase9" / "binary_weighted_scoring_summary.csv", index=False)
    pd.DataFrame(
        [{"feature_name": "phase6_random_forest", "mean_abs_contribution": 0.25, "mean_contribution": 0.01}]
    ).to_csv(explainability_dir / "phase9" / "multiclass_stacking_summary.csv", index=False)

    _write_json(
        config_dir / "streaming_config.json",
        {
            "feature_manifest_path": str(feature_manifest_path),
            "label_mapping_path": str(label_mapping_path),
            "train_indices_path": str(train_indices_path),
            "test_indices_path": str(test_indices_path),
            "phase6_output_dir": str(project_root / "models" / "saved_models" / "phase6_ml"),
            "phase7_output_dir": str(project_root / "models" / "saved_models" / "phase7_deep_learning"),
            "phase8_output_dir": str(project_root / "models" / "saved_models" / "phase8_anomaly_detection"),
            "phase9_output_dir": str(phase9_dir),
            "output_dir": str(streaming_dir),
            "logs_dir": str(logs_dir),
            "stream_split": "full",
            "chunk_size": 2,
            "inference_batch_size": 2,
        },
    )
    _write_json(
        config_dir / "alerting_config.json",
        {
            "input_predictions_path": str(streaming_dir / "stream_predictions.csv"),
            "streaming_report_path": str(streaming_dir / "streaming_report.json"),
            "output_dir": str(streaming_dir),
            "logs_dir": str(logs_dir),
            "chunk_size": 2,
            "suspicious_threshold": 40.0,
            "attack_threshold": 70.0,
            "binary_probability_weight": 0.55,
            "attack_severity_weight": 0.20,
            "class_confidence_weight": 0.15,
            "disagreement_weight": 0.10,
        },
    )
    _write_json(
        config_dir / "dashboard_config.json",
        {
            "streaming_report_path": str(streaming_dir / "streaming_report.json"),
            "alerting_report_path": str(streaming_dir / "alerting_report.json"),
            "predictions_path": str(streaming_dir / "stream_predictions.csv"),
            "enriched_predictions_path": str(streaming_dir / "stream_predictions_with_alerts.csv"),
            "alerts_path": str(streaming_dir / "alerts.csv"),
            "phase9_metrics_path": str(phase9_dir / "metrics_summary.csv"),
            "binary_shap_summary_path": str(explainability_dir / "phase6" / "shap_values" / "binary_lightgbm_summary.csv"),
            "multiclass_shap_summary_path": str(explainability_dir / "phase6" / "shap_values" / "multiclass_random_forest_summary.csv"),
            "binary_ensemble_summary_path": str(explainability_dir / "phase9" / "binary_weighted_scoring_summary.csv"),
            "multiclass_ensemble_summary_path": str(explainability_dir / "phase9" / "multiclass_stacking_summary.csv"),
            "chunk_size": 2,
        },
    )
    _write_json(
        config_dir / "api_config.json",
        {
            "streaming_config_path": str(config_dir / "streaming_config.json"),
            "alerting_config_path": str(config_dir / "alerting_config.json"),
            "dashboard_config_path": str(config_dir / "dashboard_config.json"),
            "logs_dir": str(logs_dir),
            "default_alert_limit": 10,
            "default_stream_limit": 10,
            "max_page_size": 50,
        },
    )

    api_config = ApiConfig.from_json(config_dir / "api_config.json", project_root=project_root)
    service = SentinelNetApiService(api_config, predictor_factory=FakePredictor)
    client = TestClient(create_app(api_config, service))

    try:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "ready"

        predict_response = client.post(
            "/predict",
            json={
                "records": [
                    {
                        "source_file": "api_0.csv",
                        "features": {"flow_duration": 0.12, "burstiness_score": 0.10},
                    },
                    {
                        "source_file": "api_1.csv",
                        "features": {"flow_duration": 0.42, "burstiness_score": 0.92},
                    },
                ]
            },
        )
        assert predict_response.status_code == 200
        predict_payload = predict_response.json()
        assert predict_payload["record_count"] == 2
        assert predict_payload["results"][0]["alert_level"] == "Normal"
        assert predict_payload["results"][1]["alert_level"] == "Attack"

        alerts_response = client.get("/alerts", params={"limit": 1, "alert_level": "Attack"})
        assert alerts_response.status_code == 200
        alerts_payload = alerts_response.json()
        assert alerts_payload["count"] == 1
        assert alerts_payload["records"][0]["alert_level"] == "Attack"

        stream_response = client.get("/stream", params={"limit": 2})
        assert stream_response.status_code == 200
        assert stream_response.json()["count"] == 2

        ndjson_response = client.get("/stream", params={"limit": 2, "response_format": "ndjson"})
        assert ndjson_response.status_code == 200
        assert ndjson_response.headers["content-type"].startswith("application/x-ndjson")
        assert len([line for line in ndjson_response.text.splitlines() if line.strip()]) == 2

        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        metrics_payload = metrics_response.json()
        assert metrics_payload["overview"]["rows_streamed"] == 3
        assert metrics_payload["binary_roc_auc"] > 0.5
        assert len(metrics_payload["recent_alerts"]) >= 1
    finally:
        client.close()
        service.close()
        shutil.rmtree(project_root, ignore_errors=True)
