"""Tests for the SentinelNet Phase 15 performance optimization pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.deployment.performance import run_performance_optimization_pipeline
from src.deployment.performance_config import PerformanceConfig


class FakePredictor:
    """Deterministic predictor used to validate the Phase 15 optimization flow."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.reset_calls = 0

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

        return type(
            "BatchPredictionResult",
            (),
            {
                "binary_variant": "weighted_scoring",
                "multiclass_variant": "stacking",
                "binary_probabilities": binary_probabilities,
                "binary_predicted_labels": binary_predicted_labels,
                "multiclass_probabilities": multiclass_probabilities,
                "multiclass_predicted_labels": multiclass_predicted_labels,
            },
        )()

    def reset_state(self) -> None:
        """Match the production predictor interface used by Phase 15."""
        self.reset_calls += 1

    def close(self) -> None:
        """Match the production predictor interface used by Phase 15."""


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_performance_optimization_pipeline_writes_benchmarks_and_tuned_configs() -> None:
    project_root = Path(__file__).resolve().parents[1] / ".tmp_tests" / "performance_test_workspace"
    shutil.rmtree(project_root, ignore_errors=True)

    config_dir = project_root / "config"
    streaming_dir = project_root / "data" / "streaming"
    processed_dir = project_root / "data" / "processed" / "feature_engineered"
    explainability_dir = project_root / "models" / "saved_models" / "phase10_explainability"
    phase9_dir = project_root / "models" / "saved_models" / "phase9_ensemble"
    logs_dir = project_root / "logs"
    output_dir = project_root / "models" / "saved_models" / "phase15_performance"
    for path in [
        config_dir,
        streaming_dir,
        processed_dir,
        explainability_dir / "phase6" / "shap_values",
        explainability_dir / "phase9",
        phase9_dir,
        logs_dir,
        output_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    selected_dataset_path = processed_dir / "selected_dataset.csv"
    feature_manifest_path = processed_dir / "selected_feature_manifest.json"
    label_mapping_path = project_root / "data" / "processed" / "label_mappings.json"
    train_indices_path = processed_dir / "train_indices.npy"
    test_indices_path = processed_dir / "test_indices.npy"

    frame = pd.DataFrame(
        [
            {
                "flow_duration": 0.10 + index * 0.01,
                "burstiness_score": 0.20 + (index % 6) * 0.12,
                "source_file": f"source_{index // 4}.csv",
                "label_binary": 0 if index % 3 == 0 else 1,
                "label_multiclass": 0 if index % 3 == 0 else (10 if index % 2 == 0 else 2),
            }
            for index in range(16)
        ]
    )
    frame.to_csv(selected_dataset_path, index=False)
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
    np.save(train_indices_path, np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64))
    np.save(test_indices_path, np.asarray([8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int64))

    predictions = pd.DataFrame(
        [
            {
                "stream_order": index,
                "original_index": index,
                "event_time_utc": f"2026-01-01T00:00:{index:02d}+00:00",
                "source_file": f"source_{index // 4}.csv",
                "true_binary_label": int(frame.iloc[index]["label_binary"]),
                "true_binary_label_name": "ATTACK" if int(frame.iloc[index]["label_binary"]) == 1 else "BENIGN",
                "true_multiclass_label_name": "BENIGN" if int(frame.iloc[index]["label_multiclass"]) == 0 else ("PortScan" if int(frame.iloc[index]["label_multiclass"]) == 10 else "DDoS"),
                "predicted_binary_label": 1 if float(frame.iloc[index]["burstiness_score"]) >= 0.5 else 0,
                "predicted_binary_label_name": "ATTACK" if float(frame.iloc[index]["burstiness_score"]) >= 0.5 else "BENIGN",
                "binary_attack_probability": min(0.98, max(0.02, float(frame.iloc[index]["burstiness_score"]))),
                "predicted_multiclass_label_name": "BENIGN" if float(frame.iloc[index]["burstiness_score"]) < 0.5 else ("PortScan" if index % 2 == 0 else "DDoS"),
                "multiclass_confidence": min(0.99, 0.55 + float(frame.iloc[index]["burstiness_score"]) * 0.35),
            }
            for index in range(len(frame))
        ]
    )
    predictions.to_csv(streaming_dir / "stream_predictions.csv", index=False)
    enriched = predictions.assign(
        risk_score=np.where(predictions["binary_attack_probability"] >= 0.8, 82.0, np.where(predictions["binary_attack_probability"] >= 0.5, 58.0, 5.0)),
        alert_level=np.where(predictions["binary_attack_probability"] >= 0.8, "Attack", np.where(predictions["binary_attack_probability"] >= 0.5, "Suspicious", "Normal")),
        is_alert=np.where(predictions["binary_attack_probability"] >= 0.5, True, False),
        alert_id=[f"SNT-{index:08d}" if float(prob) >= 0.5 else "" for index, prob in enumerate(predictions["binary_attack_probability"])],
        alert_timestamp_utc=predictions["event_time_utc"],
        alert_generated_at_utc=predictions["event_time_utc"],
        recommended_action=np.where(predictions["binary_attack_probability"] >= 0.8, "Escalate to incident response and isolate the affected flow or host.", np.where(predictions["binary_attack_probability"] >= 0.5, "Increase monitoring, capture packet context, and validate against baseline.", "Continue monitoring.")),
        alert_message=np.where(predictions["binary_attack_probability"] >= 0.5, "Alert generated.", "Normal traffic confidence within acceptable range."),
        alert_rule_version="phase12_v1",
        selected_binary_variant="weighted_scoring",
        selected_multiclass_variant="stacking",
    )
    enriched.to_csv(streaming_dir / "stream_predictions_with_alerts.csv", index=False)
    enriched.loc[enriched["is_alert"]].to_csv(streaming_dir / "alerts.csv", index=False)

    _write_json(
        streaming_dir / "streaming_report.json",
        {
            "rows_streamed": len(frame),
            "selected_binary_variant": "weighted_scoring",
            "selected_multiclass_variant": "stacking",
            "throughput_rows_per_second": 150.0,
            "average_batch_latency_ms": 60.0,
        },
    )
    _write_json(
        streaming_dir / "alerting_report.json",
        {
            "alert_rows_written": int(enriched["is_alert"].sum()),
            "level_counts": {"Normal": int((enriched["alert_level"] == "Normal").sum()), "Suspicious": int((enriched["alert_level"] == "Suspicious").sum()), "Attack": int((enriched["alert_level"] == "Attack").sum())},
            "predicted_attack_counts": {"PortScan": int((enriched["predicted_multiclass_label_name"] == "PortScan").sum()), "DDoS": int((enriched["predicted_multiclass_label_name"] == "DDoS").sum())},
            "average_risk_score": float(enriched["risk_score"].mean()),
            "max_risk_score": float(enriched["risk_score"].max()),
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
            "input_data_path": str(selected_dataset_path),
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
            "chunk_size": 4,
            "inference_batch_size": 4,
        },
    )
    _write_json(
        config_dir / "alerting_config.json",
        {
            "input_predictions_path": str(streaming_dir / "stream_predictions.csv"),
            "streaming_report_path": str(streaming_dir / "streaming_report.json"),
            "output_dir": str(streaming_dir),
            "logs_dir": str(logs_dir),
            "chunk_size": 4,
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
            "chunk_size": 4,
        },
    )
    _write_json(
        config_dir / "api_config.json",
        {
            "streaming_config_path": str(config_dir / "streaming_config.json"),
            "alerting_config_path": str(config_dir / "alerting_config.json"),
            "dashboard_config_path": str(config_dir / "dashboard_config.json"),
            "logs_dir": str(logs_dir),
            "default_alert_limit": 5,
            "default_stream_limit": 5,
            "max_page_size": 20,
            "preload_predictor_on_startup": False,
        },
    )
    _write_json(
        config_dir / "performance_config.json",
        {
            "streaming_config_path": str(config_dir / "streaming_config.json"),
            "alerting_config_path": str(config_dir / "alerting_config.json"),
            "dashboard_config_path": str(config_dir / "dashboard_config.json"),
            "api_config_path": str(config_dir / "api_config.json"),
            "output_dir": str(output_dir),
            "logs_dir": str(logs_dir),
            "benchmark_rows": 8,
            "warmup_rows": 4,
            "warmup_iterations": 1,
            "streaming_repetitions": 2,
            "api_predict_repetitions": 2,
            "api_read_repetitions": 2,
            "metrics_cached_repetitions": 2,
            "candidate_inference_batch_sizes": [2, 4, 8],
            "candidate_api_predict_batch_sizes": [1, 2, 4],
            "candidate_stream_page_sizes": [1, 2, 4],
            "metrics_recent_rows": 4,
            "metrics_explanation_top_n": 1,
            "metrics_multiclass_top_k": 3,
        },
    )

    config = PerformanceConfig.from_json(config_dir / "performance_config.json", project_root=project_root)

    try:
        report = run_performance_optimization_pipeline(config, predictor_factory=FakePredictor)
        streaming_benchmarks = pd.read_csv(config.streaming_benchmarks_path)
        api_predict_benchmarks = pd.read_csv(config.api_predict_benchmarks_path)
        api_read_benchmarks = pd.read_csv(config.api_read_benchmarks_path)
        optimized_streaming_config = json.loads(config.optimized_streaming_config_path.read_text(encoding="utf-8"))
        optimized_api_config = json.loads(config.optimized_api_config_path.read_text(encoding="utf-8"))

        assert report.validation_passed is True
        assert report.benchmark_rows == 8
        assert report.recommended_inference_batch_size in {2, 4, 8}
        assert report.recommended_api_predict_batch_size in {1, 2, 4}
        assert report.recommended_stream_page_limit in {1, 2, 4}
        assert report.recommended_alert_page_limit in {1, 2, 4}
        assert set(streaming_benchmarks["candidate_value"]) == {2, 4, 8}
        assert set(api_predict_benchmarks["candidate_value"]) == {1, 2, 4}
        assert {"stream_page", "alerts_page", "metrics_refresh", "metrics_cached"} <= set(api_read_benchmarks["benchmark"])
        assert optimized_streaming_config["inference_batch_size"] == report.recommended_inference_batch_size
        assert optimized_api_config["preload_predictor_on_startup"] is True
        assert optimized_api_config["default_stream_limit"] == report.recommended_stream_page_limit
        assert optimized_api_config["default_alert_limit"] == report.recommended_alert_page_limit
    finally:
        shutil.rmtree(project_root, ignore_errors=True)
