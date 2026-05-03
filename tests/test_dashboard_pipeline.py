"""Tests for the SentinelNet Phase 13 dashboard data pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from dashboard.config import DashboardConfig
from dashboard.dashboard_data import build_dashboard_snapshot


def test_build_dashboard_snapshot_aggregates_streaming_alerts_and_explanations() -> None:
    project_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "dashboard_test_workspace"
    shutil.rmtree(project_root, ignore_errors=True)

    streaming_dir = project_root / "data" / "streaming"
    explainability_dir = project_root / "models" / "saved_models" / "phase10_explainability"
    phase9_dir = project_root / "models" / "saved_models" / "phase9_ensemble"
    (streaming_dir).mkdir(parents=True, exist_ok=True)
    (explainability_dir / "phase6" / "shap_values").mkdir(parents=True, exist_ok=True)
    (explainability_dir / "phase9").mkdir(parents=True, exist_ok=True)
    phase9_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame(
        [
            {
                "stream_order": 0,
                "event_time_utc": "2026-01-01T00:00:00+00:00",
                "source_file": "monday.csv",
                "true_binary_label": 0,
                "true_binary_label_name": "BENIGN",
                "true_multiclass_label_name": "BENIGN",
                "predicted_binary_label": 0,
                "predicted_binary_label_name": "BENIGN",
                "binary_attack_probability": 0.03,
                "predicted_multiclass_label_name": "BENIGN",
                "multiclass_confidence": 0.99,
            },
            {
                "stream_order": 1,
                "event_time_utc": "2026-01-01T00:00:10+00:00",
                "source_file": "tuesday.csv",
                "true_binary_label": 1,
                "true_binary_label_name": "ATTACK",
                "true_multiclass_label_name": "PortScan",
                "predicted_binary_label": 1,
                "predicted_binary_label_name": "ATTACK",
                "binary_attack_probability": 0.91,
                "predicted_multiclass_label_name": "PortScan",
                "multiclass_confidence": 0.88,
            },
            {
                "stream_order": 2,
                "event_time_utc": "2026-01-01T00:00:20+00:00",
                "source_file": "wednesday.csv",
                "true_binary_label": 1,
                "true_binary_label_name": "ATTACK",
                "true_multiclass_label_name": "DDoS",
                "predicted_binary_label": 1,
                "predicted_binary_label_name": "ATTACK",
                "binary_attack_probability": 0.97,
                "predicted_multiclass_label_name": "DDoS",
                "multiclass_confidence": 0.96,
            },
            {
                "stream_order": 3,
                "event_time_utc": "2026-01-01T00:00:30+00:00",
                "source_file": "thursday.csv",
                "true_binary_label": 0,
                "true_binary_label_name": "BENIGN",
                "true_multiclass_label_name": "BENIGN",
                "predicted_binary_label": 1,
                "predicted_binary_label_name": "ATTACK",
                "binary_attack_probability": 0.76,
                "predicted_multiclass_label_name": "Bot",
                "multiclass_confidence": 0.84,
            },
        ]
    )
    predictions.to_csv(streaming_dir / "stream_predictions.csv", index=False)

    enriched = predictions.assign(
        risk_score=[2.5, 61.0, 84.0, 52.0],
        alert_level=["Normal", "Suspicious", "Attack", "Suspicious"],
        recommended_action=[
            "Continue monitoring.",
            "Increase monitoring, capture packet context, and validate against baseline.",
            "Trigger containment workflow and rate-limit suspected sources.",
            "Increase monitoring, capture packet context, and validate against baseline.",
        ],
        alert_id=["", "SNT-SUSPICIOUS-00000001", "SNT-ATTACK-00000002", "SNT-SUSPICIOUS-00000003"],
        alert_message=[
            "Normal traffic confidence within acceptable range.",
            "Suspicious alert for PortScan | risk=61.00 | binary_attack_probability=0.9100",
            "Attack alert for DDoS | risk=84.00 | binary_attack_probability=0.9700",
            "Suspicious alert for Bot | risk=52.00 | binary_attack_probability=0.7600",
        ],
    )
    enriched.to_csv(streaming_dir / "stream_predictions_with_alerts.csv", index=False)
    enriched.loc[enriched["alert_level"] != "Normal"].to_csv(streaming_dir / "alerts.csv", index=False)

    (streaming_dir / "streaming_report.json").write_text(
        json.dumps(
            {
                "rows_streamed": 4,
                "selected_binary_variant": "weighted_scoring",
                "selected_multiclass_variant": "stacking",
                "throughput_rows_per_second": 125.5,
                "average_batch_latency_ms": 83.2,
            }
        ),
        encoding="utf-8",
    )
    (streaming_dir / "alerting_report.json").write_text(
        json.dumps(
            {
                "alert_rows_written": 3,
                "level_counts": {"Normal": 1, "Suspicious": 2, "Attack": 1},
                "predicted_attack_counts": {"PortScan": 1, "DDoS": 1, "Bot": 1},
                "average_risk_score": 49.875,
                "max_risk_score": 84.0,
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {"task": "binary", "ensemble": "weighted_scoring", "f1_score": 0.99, "roc_auc": 0.999, "accuracy": 0.99, "precision": 0.99, "recall": 0.99},
            {"task": "multiclass", "ensemble": "stacking", "f1_score": 0.98, "roc_auc": 0.997, "accuracy": 0.98, "precision": 0.98, "recall": 0.98},
        ]
    ).to_csv(phase9_dir / "metrics_summary.csv", index=False)

    pd.DataFrame(
        [
            {"feature_name": "destination_port", "mean_abs_contribution": 0.42, "mean_contribution": -0.03},
            {"feature_name": "flow_iat_min", "mean_abs_contribution": 0.21, "mean_contribution": 0.01},
        ]
    ).to_csv(explainability_dir / "phase6" / "shap_values" / "binary_lightgbm_summary.csv", index=False)
    pd.DataFrame(
        [
            {"feature_name": "rolling_unique_destination_ports_w20", "mean_abs_contribution": 0.51, "mean_contribution": 0.02},
            {"feature_name": "forward_payload_efficiency", "mean_abs_contribution": 0.17, "mean_contribution": -0.01},
        ]
    ).to_csv(explainability_dir / "phase6" / "shap_values" / "multiclass_random_forest_summary.csv", index=False)
    pd.DataFrame(
        [
            {"feature_name": "phase6_lightgbm", "mean_abs_contribution": 0.31, "mean_contribution": -0.01},
            {"feature_name": "phase7_dnn", "mean_abs_contribution": 0.27, "mean_contribution": -0.02},
        ]
    ).to_csv(explainability_dir / "phase9" / "binary_weighted_scoring_summary.csv", index=False)
    pd.DataFrame(
        [
            {"feature_name": "phase6_random_forest", "mean_abs_contribution": 0.29, "mean_contribution": 0.01},
            {"feature_name": "phase7_lstm", "mean_abs_contribution": 0.22, "mean_contribution": -0.02},
        ]
    ).to_csv(explainability_dir / "phase9" / "multiclass_stacking_summary.csv", index=False)

    config = DashboardConfig(
        project_root=project_root,
        streaming_dir=streaming_dir,
        explainability_dir=explainability_dir,
        phase9_output_dir=phase9_dir,
        chunk_size=2,
    )

    try:
        snapshot = build_dashboard_snapshot(config, recent_rows=2, explanation_top_n=2, multiclass_top_k=3)

        assert snapshot.overview_metrics["rows_streamed"] == 4
        assert snapshot.overview_metrics["alert_rows_written"] == 3
        assert len(snapshot.recent_predictions) == 2
        assert len(snapshot.recent_alerts) == 2
        assert snapshot.alert_level_counts["count"].tolist() == [1, 2, 1]
        assert snapshot.attack_distribution["attack_type"].tolist() == ["Bot", "DDoS", "PortScan"]
        assert snapshot.binary_confusion_matrix.loc["Actual BENIGN", "Predicted ATTACK"] == 1
        assert "Actual BENIGN" in snapshot.multiclass_confusion_matrix.index
        assert snapshot.binary_roc_auc > 0.5
        assert "timestamp" in snapshot.alert_timeline.columns
        assert snapshot.binary_shap_summary.iloc[0]["feature_name"] == "destination_port"
        assert snapshot.binary_ensemble_summary.iloc[0]["feature_name"] == "phase6_lightgbm"
    finally:
        shutil.rmtree(project_root, ignore_errors=True)
