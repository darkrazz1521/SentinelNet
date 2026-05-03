"""Tests for the SentinelNet Phase 12 alerting pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.deployment.alerting import run_alerting_pipeline
from src.deployment.alerting_config import AlertingConfig


def test_run_alerting_pipeline_generates_risk_scored_alerts() -> None:
    project_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "alerting_test_workspace"
    shutil.rmtree(project_root, ignore_errors=True)
    (project_root / "data" / "streaming").mkdir(parents=True, exist_ok=True)
    (project_root / "logs").mkdir(parents=True, exist_ok=True)

    predictions_path = project_root / "data" / "streaming" / "stream_predictions.csv"
    streaming_report_path = project_root / "data" / "streaming" / "streaming_report.json"

    frame = pd.DataFrame(
        [
            {
                "stream_order": 0,
                "original_index": 8,
                "event_time_utc": "2026-01-01T00:00:00+00:00",
                "source_file": "benign_0.csv",
                "true_binary_label": 0,
                "true_binary_label_name": "BENIGN",
                "true_multiclass_label": 0,
                "true_multiclass_label_name": "BENIGN",
                "predicted_binary_label": 0,
                "predicted_binary_label_name": "BENIGN",
                "binary_attack_probability": 0.05,
                "predicted_multiclass_label": 0,
                "predicted_multiclass_label_name": "BENIGN",
                "multiclass_confidence": 0.99,
                "selected_binary_variant": "weighted_scoring",
                "selected_multiclass_variant": "stacking",
                "batch_latency_ms": 100.0,
            },
            {
                "stream_order": 1,
                "original_index": 16,
                "event_time_utc": "2026-01-01T00:00:00.100000+00:00",
                "source_file": "suspicious_0.csv",
                "true_binary_label": 1,
                "true_binary_label_name": "ATTACK",
                "true_multiclass_label": 10,
                "true_multiclass_label_name": "PortScan",
                "predicted_binary_label": 0,
                "predicted_binary_label_name": "BENIGN",
                "binary_attack_probability": 0.25,
                "predicted_multiclass_label": 10,
                "predicted_multiclass_label_name": "PortScan",
                "multiclass_confidence": 0.85,
                "selected_binary_variant": "weighted_scoring",
                "selected_multiclass_variant": "stacking",
                "batch_latency_ms": 100.0,
            },
            {
                "stream_order": 2,
                "original_index": 21,
                "event_time_utc": "2026-01-01T00:00:00.200000+00:00",
                "source_file": "attack_0.csv",
                "true_binary_label": 1,
                "true_binary_label_name": "ATTACK",
                "true_multiclass_label": 2,
                "true_multiclass_label_name": "DDoS",
                "predicted_binary_label": 1,
                "predicted_binary_label_name": "ATTACK",
                "binary_attack_probability": 0.96,
                "predicted_multiclass_label": 2,
                "predicted_multiclass_label_name": "DDoS",
                "multiclass_confidence": 0.97,
                "selected_binary_variant": "weighted_scoring",
                "selected_multiclass_variant": "stacking",
                "batch_latency_ms": 100.0,
            },
        ]
    )
    frame.to_csv(predictions_path, index=False)
    streaming_report_path.write_text(
        json.dumps(
            {
                "selected_binary_variant": "weighted_scoring",
                "selected_multiclass_variant": "stacking",
            }
        ),
        encoding="utf-8",
    )

    config = AlertingConfig(
        project_root=project_root,
        input_predictions_path=predictions_path,
        streaming_report_path=streaming_report_path,
        output_dir=project_root / "data" / "streaming",
        logs_dir=project_root / "logs",
        chunk_size=2,
    )

    try:
        report = run_alerting_pipeline(config)
        enriched = pd.read_csv(config.enriched_predictions_path)
        alerts = pd.read_csv(config.alerts_path)

        assert report.validation_passed is True
        assert report.total_rows_processed == 3
        assert report.alert_rows_written == 2
        assert report.level_counts == {"Normal": 1, "Suspicious": 1, "Attack": 1}
        assert report.selected_binary_variant == "weighted_scoring"
        assert report.selected_multiclass_variant == "stacking"
        assert set(enriched["alert_level"]) == {"Normal", "Suspicious", "Attack"}
        assert set(alerts["alert_level"]) == {"Suspicious", "Attack"}
        assert "risk_score" in enriched.columns
        assert "recommended_action" in alerts.columns
        assert enriched.loc[enriched["stream_order"] == 2, "risk_score"].iloc[0] >= 70.0
    finally:
        shutil.rmtree(project_root, ignore_errors=True)
