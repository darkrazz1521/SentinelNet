"""SentinelNet Phase 16 advanced response pipeline."""

from __future__ import annotations

import gc
import heapq
import json
import logging
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_pipeline.logging_utils import configure_logging

from .config import StreamingConfig
from .data import StreamingMetadata, iter_stream_batches, load_streaming_metadata
from .phase16_config import Phase16Config
from .predictor import _format_features_for_artifact, _logistic_transform, _score_as_anomaly

LOGGER = logging.getLogger("sentinelnet.phase16")

REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "stream_order",
    "original_index",
    "event_time_utc",
    "source_file",
    "binary_attack_probability",
    "predicted_multiclass_label_name",
    "multiclass_confidence",
    "risk_score",
    "alert_level",
)

CONTEXT_FEATURES: tuple[str, ...] = (
    "destination_port",
    "flow_duration",
    "total_fwd_packets",
    "flow_bytes_per_s",
    "rolling_unique_destination_ports_w20",
    "forward_payload_efficiency",
    "burstiness_score",
)

QUEUE_COLUMNS: tuple[str, ...] = (
    "stream_order",
    "original_index",
    "event_time_utc",
    "source_file",
    "destination_port",
    "predicted_multiclass_label_name",
    "operational_attack_label_name",
    "threat_family",
    "alert_level",
    "risk_score",
    "binary_attack_probability",
    "multiclass_confidence",
    "mean_anomaly_attack_probability",
    "max_anomaly_attack_probability",
    "anomaly_disagreement",
    "novelty_score",
    "is_zero_day_candidate",
    "feedback_priority",
    "feedback_reason_codes",
    "suggested_feedback_action",
)


@dataclass(slots=True)
class ThreatDescriptor:
    """Operational classification metadata for a predicted attack label."""

    family: str
    tactic: str
    severity: str
    playbook: str
    response_scope: str


@dataclass(slots=True)
class LoadedPhase16AnomalyModel:
    """Persisted anomaly detector and its calibrated threshold."""

    name: str
    estimator: Any
    threshold: float


@dataclass(slots=True)
class RunningFeatureStats:
    """Incremental feature statistics used for drift monitoring."""

    feature_names: list[str]
    rows: int = 0
    sum_vector: np.ndarray | None = None
    sum_squares: np.ndarray | None = None

    def __post_init__(self) -> None:
        feature_count = len(self.feature_names)
        self.sum_vector = np.zeros(feature_count, dtype=np.float64)
        self.sum_squares = np.zeros(feature_count, dtype=np.float64)

    def update(self, feature_frame: pd.DataFrame) -> None:
        """Accumulate statistics from one feature batch."""
        matrix = np.nan_to_num(feature_frame.loc[:, self.feature_names].to_numpy(dtype=np.float64, copy=False))
        self.rows += int(len(matrix))
        if len(matrix) == 0:
            return
        self.sum_vector += matrix.sum(axis=0)
        self.sum_squares += np.square(matrix).sum(axis=0)

    def to_frame(self, *, prefix: str) -> pd.DataFrame:
        """Convert accumulated statistics into a feature summary frame."""
        rows = max(self.rows, 1)
        mean_vector = self.sum_vector / rows
        variance = np.maximum((self.sum_squares / rows) - np.square(mean_vector), 0.0)
        std_vector = np.sqrt(variance)
        return pd.DataFrame(
            {
                "feature_name": self.feature_names,
                f"{prefix}_mean": mean_vector,
                f"{prefix}_std": std_vector,
                f"{prefix}_rows": int(self.rows),
            }
        )


@dataclass(slots=True)
class Phase16Report:
    """Serializable report for the Phase 16 advanced response stage."""

    created_at_utc: str
    input_predictions_path: str
    output_dir: str
    classified_predictions_path: str
    zero_day_candidates_path: str
    autoblock_actions_path: str
    continuous_learning_queue_path: str
    feature_drift_path: str
    retraining_manifest_path: str
    report_path: str
    feature_count: int
    drift_reference_rows: int
    total_rows_processed: int
    benign_rows: int
    known_attack_rows: int
    zero_day_candidate_rows: int
    auto_block_rows: int
    continuous_learning_queue_rows: int
    drifted_feature_count: int
    average_novelty_score: float
    top_drift_features: list[str]
    attack_family_counts: dict[str, int]
    zero_day_family_counts: dict[str, int]
    auto_block_action_counts: dict[str, int]
    retraining_recommended: bool
    retraining_reasons: list[str]
    validation_passed: bool
    notes: list[str]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "created_at_utc": self.created_at_utc,
            "input_predictions_path": self.input_predictions_path,
            "output_dir": self.output_dir,
            "classified_predictions_path": self.classified_predictions_path,
            "zero_day_candidates_path": self.zero_day_candidates_path,
            "autoblock_actions_path": self.autoblock_actions_path,
            "continuous_learning_queue_path": self.continuous_learning_queue_path,
            "feature_drift_path": self.feature_drift_path,
            "retraining_manifest_path": self.retraining_manifest_path,
            "report_path": self.report_path,
            "feature_count": self.feature_count,
            "drift_reference_rows": self.drift_reference_rows,
            "total_rows_processed": self.total_rows_processed,
            "benign_rows": self.benign_rows,
            "known_attack_rows": self.known_attack_rows,
            "zero_day_candidate_rows": self.zero_day_candidate_rows,
            "auto_block_rows": self.auto_block_rows,
            "continuous_learning_queue_rows": self.continuous_learning_queue_rows,
            "drifted_feature_count": self.drifted_feature_count,
            "average_novelty_score": self.average_novelty_score,
            "top_drift_features": self.top_drift_features,
            "attack_family_counts": self.attack_family_counts,
            "zero_day_family_counts": self.zero_day_family_counts,
            "auto_block_action_counts": self.auto_block_action_counts,
            "retraining_recommended": self.retraining_recommended,
            "retraining_reasons": self.retraining_reasons,
            "validation_passed": self.validation_passed,
            "notes": self.notes,
            "config": self.config,
        }


THREAT_DESCRIPTOR_MAP: dict[str, ThreatDescriptor] = {
    "BENIGN": ThreatDescriptor(
        family="Normal Traffic",
        tactic="None",
        severity="Informational",
        playbook="Continue monitoring with normal telemetry retention.",
        response_scope="none",
    ),
    "Bot": ThreatDescriptor(
        family="Botnet Activity",
        tactic="Command and Control",
        severity="High",
        playbook="Quarantine the suspected source and inspect lateral communication paths.",
        response_scope="source_file",
    ),
    "DDoS": ThreatDescriptor(
        family="Availability Disruption",
        tactic="Impact",
        severity="Critical",
        playbook="Trigger containment workflow and rate-limit the targeted service or port.",
        response_scope="destination_port",
    ),
    "DoS GoldenEye": ThreatDescriptor(
        family="Availability Disruption",
        tactic="Impact",
        severity="High",
        playbook="Rate-limit the impacted service and inspect connection saturation behavior.",
        response_scope="destination_port",
    ),
    "DoS Hulk": ThreatDescriptor(
        family="Availability Disruption",
        tactic="Impact",
        severity="Critical",
        playbook="Throttle aggressive flows and isolate the affected service tier.",
        response_scope="destination_port",
    ),
    "DoS Slowhttptest": ThreatDescriptor(
        family="Availability Disruption",
        tactic="Impact",
        severity="High",
        playbook="Apply slow-connection mitigations and investigate application timeouts.",
        response_scope="destination_port",
    ),
    "DoS slowloris": ThreatDescriptor(
        family="Availability Disruption",
        tactic="Impact",
        severity="High",
        playbook="Block abusive session patterns and inspect reverse proxy protections.",
        response_scope="destination_port",
    ),
    "FTP-Patator": ThreatDescriptor(
        family="Credential Abuse",
        tactic="Brute Force",
        severity="High",
        playbook="Lock down the targeted authentication surface and rotate exposed credentials.",
        response_scope="source_file",
    ),
    "Heartbleed": ThreatDescriptor(
        family="Memory Disclosure",
        tactic="Collection",
        severity="Critical",
        playbook="Escalate immediately, patch vulnerable services, and capture memory-forensics artifacts.",
        response_scope="flow_signature",
    ),
    "Infiltration": ThreatDescriptor(
        family="Lateral Movement / Exfiltration",
        tactic="Exfiltration",
        severity="Critical",
        playbook="Isolate the suspected host path, preserve traffic captures, and inspect outbound data movement.",
        response_scope="source_file",
    ),
    "PortScan": ThreatDescriptor(
        family="Reconnaissance",
        tactic="Discovery",
        severity="Medium",
        playbook="Escalate to perimeter monitoring and suppress repeated probing against exposed ports.",
        response_scope="destination_port",
    ),
    "SSH-Patator": ThreatDescriptor(
        family="Credential Abuse",
        tactic="Brute Force",
        severity="High",
        playbook="Investigate repeated SSH failures and apply temporary access controls to the source.",
        response_scope="source_file",
    ),
    "Web Attack - Brute Force": ThreatDescriptor(
        family="Web Exploitation",
        tactic="Initial Access",
        severity="High",
        playbook="Throttle the targeted endpoint, preserve HTTP context, and inspect auth controls.",
        response_scope="flow_signature",
    ),
    "Web Attack - Sql Injection": ThreatDescriptor(
        family="Web Exploitation",
        tactic="Initial Access",
        severity="Critical",
        playbook="Isolate the affected application path and inspect database query surfaces immediately.",
        response_scope="flow_signature",
    ),
    "Web Attack - XSS": ThreatDescriptor(
        family="Web Exploitation",
        tactic="Initial Access",
        severity="High",
        playbook="Inspect the targeted route, sanitize payload handling, and validate session integrity.",
        response_scope="flow_signature",
    ),
}


class RealPhase16AnomalyScorer:
    """Real Phase 8-backed anomaly scorer used during Phase 16."""

    def __init__(self, streaming_config: StreamingConfig, metadata: StreamingMetadata, batch_size: int) -> None:
        self.feature_names = metadata.feature_names
        self.batch_size = int(batch_size)
        self.scaler = joblib.load(streaming_config.phase8_output_dir / "common" / "feature_scaler.joblib")
        isolation_payload = joblib.load(streaming_config.phase8_output_dir / "isolation_forest" / "isolation_forest.joblib")
        one_class_payload = joblib.load(streaming_config.phase8_output_dir / "one_class_svm" / "one_class_svm.joblib")
        lof_payload = joblib.load(streaming_config.phase8_output_dir / "lof" / "lof.joblib")
        self.detectors = {
            "isolation_forest": LoadedPhase16AnomalyModel(
                name="isolation_forest",
                estimator=isolation_payload["estimator"],
                threshold=float(isolation_payload["threshold"]),
            ),
            "one_class_svm": LoadedPhase16AnomalyModel(
                name="one_class_svm",
                estimator=one_class_payload["estimator"],
                threshold=float(one_class_payload["threshold"]),
            ),
            "lof": LoadedPhase16AnomalyModel(
                name="lof",
                estimator=lof_payload["estimator"],
                threshold=float(lof_payload["threshold"]),
            ),
        }
        self.autoencoder = tf.keras.models.load_model(streaming_config.phase8_output_dir / "autoencoder" / "autoencoder.keras")
        autoencoder_metadata = json.loads(
            (streaming_config.phase8_output_dir / "autoencoder" / "autoencoder_metadata.json").read_text(encoding="utf-8")
        )
        self.autoencoder_threshold = float(autoencoder_metadata["threshold"])

    def score_batch(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Score one feature batch with the Phase 8 detector ensemble."""
        X_raw = feature_frame.to_numpy(dtype=np.float32, copy=False)
        X_scaler = _format_features_for_artifact(self.scaler, X_raw, self.feature_names)
        X_scaled = self.scaler.transform(X_scaler).astype(np.float32, copy=False)

        result: dict[str, np.ndarray] = {}
        per_detector_probabilities: list[np.ndarray] = []
        for name, model in self.detectors.items():
            X_estimator = _format_features_for_artifact(model.estimator, X_scaled, self.feature_names)
            scores = _score_as_anomaly(model.estimator, X_estimator)
            probabilities = _logistic_transform(scores, model.threshold)
            result[f"phase8_{name}_attack_probability"] = probabilities.astype(np.float64, copy=False)
            per_detector_probabilities.append(probabilities.astype(np.float64, copy=False))

        reconstructed = self.autoencoder.predict(X_scaled, batch_size=self.batch_size, verbose=0)
        reconstruction_error = np.mean(np.square(X_scaled - reconstructed), axis=1)
        autoencoder_probability = _logistic_transform(reconstruction_error, self.autoencoder_threshold)
        result["phase8_autoencoder_attack_probability"] = autoencoder_probability.astype(np.float64, copy=False)
        per_detector_probabilities.append(autoencoder_probability.astype(np.float64, copy=False))

        stacked = np.column_stack(per_detector_probabilities).astype(np.float64, copy=False)
        result["mean_anomaly_attack_probability"] = stacked.mean(axis=1)
        result["max_anomaly_attack_probability"] = stacked.max(axis=1)
        return pd.DataFrame(result)

    def close(self) -> None:
        """Release TensorFlow resources when scoring completes."""
        tf.keras.backend.clear_session()


def build_phase16_logger(config: Phase16Config) -> logging.Logger:
    """Create the dedicated Phase 16 logger."""
    return configure_logging(config.log_path, config.log_level, logger_name="sentinelnet.phase16")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with indentation."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_frame(path: Path, frame: pd.DataFrame, *, include_header: bool) -> None:
    """Append a frame to CSV with optional header emission."""
    frame.to_csv(path, mode="a", index=False, header=include_header)


def _remove_existing_outputs(config: Phase16Config) -> None:
    """Delete prior Phase 16 artifacts before a fresh run."""
    for path in (
        config.classified_predictions_path,
        config.zero_day_candidates_path,
        config.autoblock_actions_path,
        config.continuous_learning_queue_path,
        config.feature_drift_path,
        config.retraining_manifest_path,
        config.report_path,
    ):
        if path.exists():
            path.unlink()


def _validate_prediction_chunk(chunk: pd.DataFrame) -> None:
    """Fail fast when the Phase 12 output is missing required columns."""
    missing_columns = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in chunk.columns]
    if missing_columns:
        raise ValueError(
            "Phase 16 requires enriched prediction columns that were not found: "
            + ", ".join(sorted(missing_columns))
        )


def _descriptor_for_label(label_name: str) -> ThreatDescriptor:
    """Look up the operational descriptor for one predicted label."""
    return THREAT_DESCRIPTOR_MAP.get(
        label_name,
        ThreatDescriptor(
            family="Unknown Attack",
            tactic="Unknown",
            severity="High",
            playbook="Escalate to analyst review and preserve full packet context for triage.",
            response_scope="flow_signature",
        ),
    )


def _infer_novel_descriptor(feature_row: dict[str, float]) -> ThreatDescriptor:
    """Infer a novelty-family hint for zero-day candidates that lack a reliable known label."""
    unique_ports = float(feature_row.get("rolling_unique_destination_ports_w20", 0.0))
    flow_bytes_per_s = float(feature_row.get("flow_bytes_per_s", 0.0))
    total_fwd_packets = float(feature_row.get("total_fwd_packets", 0.0))
    forward_payload_efficiency = float(feature_row.get("forward_payload_efficiency", 0.0))
    burstiness_score = float(feature_row.get("burstiness_score", 0.0))

    if unique_ports >= 5.0:
        return ThreatDescriptor(
            family="Novel Reconnaissance",
            tactic="Discovery",
            severity="High",
            playbook="Preserve packet context and suppress the scanning surface with temporary access controls.",
            response_scope="destination_port",
        )
    if flow_bytes_per_s >= 100000.0 or total_fwd_packets >= 200.0 or burstiness_score >= 2.5:
        return ThreatDescriptor(
            family="Novel Availability Disruption",
            tactic="Impact",
            severity="Critical",
            playbook="Throttle the suspected service path and inspect flood-style burst behavior immediately.",
            response_scope="destination_port",
        )
    if forward_payload_efficiency >= 20.0:
        return ThreatDescriptor(
            family="Novel Exploitation",
            tactic="Initial Access",
            severity="High",
            playbook="Escalate to application triage, capture payload context, and isolate the affected flow signature.",
            response_scope="flow_signature",
        )
    return ThreatDescriptor(
        family="Novel Suspicious Activity",
        tactic="Execution",
        severity="High",
        playbook="Escalate to analyst review and quarantine the flow signature until classified.",
        response_scope="flow_signature",
    )


def _build_block_target(scope: str, source_file: str, destination_port: float | None, attack_label: str, original_index: int) -> str:
    """Build a stable simulated blocking key for repeat-offender tracking."""
    if scope == "destination_port" and destination_port is not None and np.isfinite(destination_port):
        return f"destination_port:{int(destination_port)}"
    if scope == "source_file":
        return f"source_file:{source_file}"
    port_component = "unknown_port" if destination_port is None or not np.isfinite(destination_port) else f"port:{int(destination_port)}"
    return f"flow_signature:{source_file}:{port_component}:{attack_label}:{original_index}"


def _auto_block_action(scope: str, *, is_zero_day: bool) -> str:
    """Resolve the simulated blocking action name for a target scope."""
    if is_zero_day:
        if scope == "destination_port":
            return "SIMULATED_DESTINATION_PORT_QUARANTINE"
        if scope == "source_file":
            return "SIMULATED_SOURCE_CONTAINMENT"
        return "SIMULATED_FLOW_SIGNATURE_QUARANTINE"
    if scope == "destination_port":
        return "SIMULATED_PORT_BLOCK"
    if scope == "source_file":
        return "SIMULATED_SOURCE_RATE_LIMIT"
    return "SIMULATED_FLOW_SIGNATURE_BLOCK"


def _feedback_priority(
    *,
    risk_score: float,
    novelty_score: float,
    binary_attack_probability: float,
    multiclass_confidence: float,
    is_zero_day_candidate: bool,
) -> float:
    """Build a bounded analyst-review priority score."""
    boundary_score = 1.0 - min(abs(binary_attack_probability - 0.5) / 0.5, 1.0)
    priority = (
        0.50 * (novelty_score / 100.0)
        + 0.20 * (risk_score / 100.0)
        + 0.20 * (1.0 - multiclass_confidence)
        + 0.10 * boundary_score
    )
    if is_zero_day_candidate:
        priority += 0.10
    return float(min(priority * 100.0, 100.0))


def _push_queue_candidate(
    heap: list[tuple[float, int, dict[str, Any]]],
    *,
    priority: float,
    sequence_number: int,
    record: dict[str, Any],
    max_size: int,
) -> None:
    """Maintain a bounded max-priority analyst queue using a min-heap."""
    payload = (float(priority), int(sequence_number), record)
    if len(heap) < max_size:
        heapq.heappush(heap, payload)
        return
    if payload[0] > heap[0][0]:
        heapq.heapreplace(heap, payload)


def _build_reference_statistics(
    streaming_config: StreamingConfig,
    phase16_config: Phase16Config,
    feature_names: list[str],
) -> RunningFeatureStats:
    """Build baseline train-split feature statistics for drift detection."""
    reference_config = deepcopy(streaming_config)
    reference_config.stream_split = "train"
    reference_config.max_rows = phase16_config.drift_reference_max_rows
    reference_config.inference_batch_size = phase16_config.processing_batch_size or streaming_config.inference_batch_size
    reference_metadata = load_streaming_metadata(reference_config)
    stats = RunningFeatureStats(feature_names)
    for batch, _ in iter_stream_batches(reference_config, reference_metadata):
        stats.update(batch.feature_frame)
    return stats


def _build_feature_drift_frame(
    reference_stats: RunningFeatureStats,
    stream_stats: RunningFeatureStats,
    *,
    zscore_threshold: float,
) -> pd.DataFrame:
    """Build a sorted feature drift frame from reference and stream statistics."""
    reference_frame = reference_stats.to_frame(prefix="reference")
    stream_frame = stream_stats.to_frame(prefix="stream")
    merged = reference_frame.merge(stream_frame, on="feature_name", how="inner")
    merged["absolute_mean_shift"] = (merged["stream_mean"] - merged["reference_mean"]).abs()
    merged["z_shift"] = merged["absolute_mean_shift"] / merged["reference_std"].clip(lower=1e-6)
    merged["drifted"] = merged["z_shift"] >= float(zscore_threshold)
    return merged.sort_values(["drifted", "z_shift"], ascending=[False, False]).reset_index(drop=True)


def _build_retraining_manifest(
    config: Phase16Config,
    *,
    zero_day_candidate_rows: int,
    continuous_learning_queue_rows: int,
    drifted_feature_count: int,
    top_drift_features: list[str],
) -> dict[str, Any]:
    """Build a continuous-learning manifest and its retraining recommendation."""
    reasons: list[str] = []
    recommended_actions: list[str] = []
    recommended_phases: list[int] = []

    if zero_day_candidate_rows >= config.retraining_zero_day_trigger:
        reasons.append(f"zero_day_candidate_rows>={config.retraining_zero_day_trigger}")
        recommended_actions.append("Review the novelty queue, validate analyst labels, and expand the attack taxonomy where needed.")
        recommended_phases.extend([7, 8, 9, 10, 11, 12])
    if continuous_learning_queue_rows >= config.retraining_queue_trigger:
        reasons.append(f"continuous_learning_queue_rows>={config.retraining_queue_trigger}")
        recommended_actions.append("Refresh supervised classifiers with the latest analyst-reviewed edge cases near the decision boundary.")
        recommended_phases.extend([6, 7, 9, 10, 11, 12])
    if drifted_feature_count >= config.retraining_drift_feature_trigger:
        reasons.append(f"drifted_feature_count>={config.retraining_drift_feature_trigger}")
        recommended_actions.append("Recompute preprocessing and feature-engineering baselines before retraining downstream detectors.")
        recommended_phases.extend([4, 5, 6, 7, 8, 9, 10, 11, 12])

    retraining_recommended = bool(reasons)
    unique_phases = sorted({int(phase) for phase in recommended_phases})
    return {
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "retraining_recommended": retraining_recommended,
        "retraining_reasons": reasons,
        "recommended_actions": recommended_actions,
        "recommended_phases": unique_phases,
        "continuous_learning_queue_path": str(config.continuous_learning_queue_path),
        "zero_day_candidates_path": str(config.zero_day_candidates_path),
        "feature_drift_path": str(config.feature_drift_path),
        "top_drift_features": top_drift_features,
        "queue_target_rows": int(continuous_learning_queue_rows),
        "zero_day_target_rows": int(zero_day_candidate_rows),
        "drifted_feature_count": int(drifted_feature_count),
    }


def _build_notes(config: Phase16Config) -> list[str]:
    """Build runtime notes for the Phase 16 report."""
    return [
        "Phase 16 zero-day candidates combine Phase 8 anomaly consensus with disagreement against the deployed ensemble outputs.",
        "Auto-block actions are simulated only; they produce operational recommendations and blocking scopes without modifying live network state.",
        f"Continuous-learning queues are capped at the top {config.continuous_learning_max_queue} analyst-priority samples.",
    ]


def run_phase16_pipeline(
    config: Phase16Config,
    logger: logging.Logger | None = None,
    *,
    anomaly_scorer_factory: Callable[[StreamingConfig, StreamingMetadata], Any] | None = None,
) -> Phase16Report:
    """Execute the complete SentinelNet Phase 16 advanced response workflow."""
    active_logger = logger or LOGGER
    config.ensure_directories()
    _remove_existing_outputs(config)

    if not config.streaming_config_path.exists():
        raise FileNotFoundError(f"Phase 16 streaming config not found at {config.streaming_config_path}")
    if not config.input_predictions_path.exists():
        raise FileNotFoundError(f"Phase 16 input predictions not found at {config.input_predictions_path}")

    streaming_config = StreamingConfig.from_json(config.streaming_config_path, project_root=config.project_root)
    processing_streaming_config = deepcopy(streaming_config)
    processing_streaming_config.inference_batch_size = (
        config.processing_batch_size or streaming_config.inference_batch_size
    )
    metadata = load_streaming_metadata(processing_streaming_config)

    for required_path in (
        processing_streaming_config.input_data_path,
        processing_streaming_config.feature_manifest_path,
        processing_streaming_config.label_mapping_path,
        processing_streaming_config.phase8_output_dir / "anomaly_detection_report.json",
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Required Phase 16 dependency not found at {required_path}")

    active_logger.info(
        "Starting Phase 16 advanced response | processing_batch_size=%d | feature_count=%d",
        processing_streaming_config.inference_batch_size,
        len(metadata.feature_names),
    )

    reference_stats = _build_reference_statistics(streaming_config, config, metadata.feature_names)
    active_logger.info("Built Phase 16 drift baseline | reference_rows=%d", reference_stats.rows)
    stream_stats = RunningFeatureStats(metadata.feature_names)

    scorer_builder = anomaly_scorer_factory or (
        lambda stream_config, stream_metadata: RealPhase16AnomalyScorer(
            stream_config,
            stream_metadata,
            batch_size=config.anomaly_batch_size,
        )
    )
    anomaly_scorer = scorer_builder(processing_streaming_config, metadata)

    attack_family_counts: Counter[str] = Counter()
    zero_day_family_counts: Counter[str] = Counter()
    auto_block_action_counts: Counter[str] = Counter()
    offense_counter: Counter[str] = Counter()

    total_rows_processed = 0
    benign_rows = 0
    known_attack_rows = 0
    zero_day_candidate_rows = 0
    auto_block_rows = 0
    novelty_score_sum = 0.0
    queue_heap: list[tuple[float, int, dict[str, Any]]] = []
    queue_sequence = 0
    classified_columns: list[str] | None = None

    prediction_iterator = pd.read_csv(
        config.input_predictions_path,
        chunksize=processing_streaming_config.inference_batch_size,
        low_memory=False,
    )

    context_feature_names = [name for name in CONTEXT_FEATURES if name in metadata.feature_names]
    include_header = {
        "classified": True,
        "zero_day": True,
        "autoblock": True,
    }

    try:
        for batch_index, (stream_batch, _) in enumerate(iter_stream_batches(processing_streaming_config, metadata), start=1):
            prediction_chunk = next(prediction_iterator, None)
            if prediction_chunk is None:
                raise ValueError("Phase 16 stream replay exhausted before the enriched prediction file ended.")

            prediction_chunk = prediction_chunk.reset_index(drop=True).copy()
            _validate_prediction_chunk(prediction_chunk)
            if len(prediction_chunk) != len(stream_batch.original_indices):
                raise ValueError("Phase 16 stream replay and enriched prediction chunks are misaligned in row count.")

            prediction_original_indices = prediction_chunk["original_index"].to_numpy(dtype=np.int64, copy=False)
            if not np.array_equal(prediction_original_indices, stream_batch.original_indices):
                raise ValueError("Phase 16 stream replay and enriched predictions diverged on original_index ordering.")

            prediction_source_files = prediction_chunk["source_file"].astype(str).to_numpy(copy=False)
            if not np.array_equal(prediction_source_files, stream_batch.source_files):
                raise ValueError("Phase 16 stream replay and enriched predictions diverged on source_file ordering.")

            anomaly_scores = anomaly_scorer.score_batch(stream_batch.feature_frame).reset_index(drop=True)
            context_frame = stream_batch.feature_frame.loc[:, context_feature_names].reset_index(drop=True).copy()
            stream_stats.update(stream_batch.feature_frame)

            binary_attack_probability = prediction_chunk["binary_attack_probability"].astype(float).to_numpy(copy=False)
            multiclass_confidence = prediction_chunk["multiclass_confidence"].astype(float).to_numpy(copy=False)
            risk_score = prediction_chunk["risk_score"].astype(float).to_numpy(copy=False)
            predicted_labels = prediction_chunk["predicted_multiclass_label_name"].astype(str).tolist()
            alert_levels = prediction_chunk["alert_level"].astype(str).tolist()
            mean_anomaly = anomaly_scores["mean_anomaly_attack_probability"].to_numpy(dtype=np.float64, copy=False)
            max_anomaly = anomaly_scores["max_anomaly_attack_probability"].to_numpy(dtype=np.float64, copy=False)
            anomaly_disagreement_raw = mean_anomaly - binary_attack_probability
            anomaly_disagreement = np.clip(anomaly_disagreement_raw, a_min=0.0, a_max=None)
            novelty_score = 100.0 * (
                0.45 * mean_anomaly
                + 0.20 * max_anomaly
                + 0.20 * anomaly_disagreement.clip(max=1.0)
                + 0.15 * (1.0 - multiclass_confidence)
            )
            novelty_score = novelty_score.clip(min=0.0, max=100.0)

            zero_day_candidate_mask = (
                (risk_score >= config.zero_day_min_risk_score)
                & (mean_anomaly >= config.zero_day_min_mean_anomaly_probability)
                & (max_anomaly >= config.zero_day_min_max_anomaly_probability)
                & (anomaly_disagreement >= config.zero_day_min_disagreement)
                & (
                    (prediction_chunk["predicted_multiclass_label_name"].astype(str) == "BENIGN").to_numpy(copy=False)
                    | (multiclass_confidence <= config.zero_day_max_multiclass_confidence)
                )
            )

            operational_attack_labels: list[str] = []
            threat_families: list[str] = []
            mitre_tactics: list[str] = []
            severity_tiers: list[str] = []
            response_playbooks: list[str] = []
            recommended_block_scopes: list[str] = []
            classification_confidence_tiers: list[str] = []
            zero_day_reasons: list[str] = []
            block_targets: list[str] = []
            repeat_offense_counts: list[int] = []
            auto_block_decisions: list[bool] = []
            auto_block_actions: list[str] = []
            auto_block_reasons: list[str] = []
            auto_block_ttls: list[int] = []
            feedback_priorities: list[float] = []
            feedback_reason_codes: list[str] = []
            feedback_actions: list[str] = []

            for position, predicted_label in enumerate(predicted_labels):
                descriptor = _descriptor_for_label(predicted_label)
                context_row = {
                    name: float(context_frame.iloc[position][name])
                    for name in context_feature_names
                    if pd.notna(context_frame.iloc[position][name])
                }
                is_zero_day_candidate = bool(zero_day_candidate_mask[position])
                candidate_reasons: list[str] = []
                if mean_anomaly[position] >= config.zero_day_min_mean_anomaly_probability:
                    candidate_reasons.append("high_anomaly_consensus")
                if max_anomaly[position] >= config.zero_day_min_max_anomaly_probability:
                    candidate_reasons.append("detector_peak_alarm")
                if anomaly_disagreement[position] >= config.zero_day_min_disagreement:
                    candidate_reasons.append("anomaly_ensemble_disagreement")
                if predicted_label == "BENIGN":
                    candidate_reasons.append("benign_prediction_conflict")
                if multiclass_confidence[position] <= config.zero_day_max_multiclass_confidence:
                    candidate_reasons.append("low_multiclass_confidence")

                if is_zero_day_candidate and predicted_label == "BENIGN":
                    descriptor = _infer_novel_descriptor(context_row)
                    operational_label = "UNKNOWN-NOVELTY"
                elif is_zero_day_candidate and predicted_label != "BENIGN":
                    operational_label = f"Novel Variant - {predicted_label}"
                else:
                    operational_label = predicted_label

                if multiclass_confidence[position] >= 0.85:
                    confidence_tier = "High"
                elif multiclass_confidence[position] >= 0.65:
                    confidence_tier = "Medium"
                else:
                    confidence_tier = "Low"

                destination_port = context_row.get("destination_port")
                block_target = _build_block_target(
                    descriptor.response_scope,
                    prediction_source_files[position],
                    destination_port,
                    operational_label,
                    int(prediction_original_indices[position]),
                )

                offense_event = alert_levels[position] != "Normal" or is_zero_day_candidate
                if offense_event:
                    offense_counter[block_target] += 1
                repeat_offense_count = int(offense_counter.get(block_target, 0))

                auto_block = False
                auto_block_reason = "No automatic containment trigger satisfied."
                auto_block_action_name = "MONITOR_ONLY"
                if is_zero_day_candidate and novelty_score[position] >= config.auto_block_zero_day_threshold and repeat_offense_count >= config.auto_block_min_repeat_hits:
                    auto_block = True
                    auto_block_reason = "Zero-day novelty score exceeded the containment threshold with repeated offenses."
                    auto_block_action_name = _auto_block_action(descriptor.response_scope, is_zero_day=True)
                elif alert_levels[position] == "Attack" and risk_score[position] >= config.auto_block_attack_threshold and repeat_offense_count >= config.auto_block_min_repeat_hits:
                    auto_block = True
                    auto_block_reason = "High-risk repeated attack pattern triggered simulated containment."
                    auto_block_action_name = _auto_block_action(descriptor.response_scope, is_zero_day=False)

                review_reasons: list[str] = []
                if is_zero_day_candidate:
                    review_reasons.extend(candidate_reasons)
                if abs(binary_attack_probability[position] - 0.5) <= config.low_confidence_binary_margin:
                    review_reasons.append("binary_decision_boundary")
                if multiclass_confidence[position] <= config.low_confidence_multiclass_confidence:
                    review_reasons.append("multiclass_decision_boundary")
                if anomaly_disagreement_raw[position] >= 0.05:
                    review_reasons.append("classifier_anomaly_divergence")

                feedback_priority = _feedback_priority(
                    risk_score=float(risk_score[position]),
                    novelty_score=float(novelty_score[position]),
                    binary_attack_probability=float(binary_attack_probability[position]),
                    multiclass_confidence=float(multiclass_confidence[position]),
                    is_zero_day_candidate=is_zero_day_candidate,
                )

                if review_reasons:
                    feedback_action = (
                        "Prioritize analyst triage, label the sample, and schedule it for the next supervised retraining window."
                        if is_zero_day_candidate
                        else "Review the boundary-case sample and feed the labeled result into the active-learning buffer."
                    )
                    record = {
                        "stream_order": int(prediction_chunk.iloc[position]["stream_order"]),
                        "original_index": int(prediction_original_indices[position]),
                        "event_time_utc": str(prediction_chunk.iloc[position]["event_time_utc"]),
                        "source_file": str(prediction_source_files[position]),
                        "destination_port": None if destination_port is None or not np.isfinite(destination_port) else int(destination_port),
                        "predicted_multiclass_label_name": predicted_label,
                        "operational_attack_label_name": operational_label,
                        "threat_family": descriptor.family,
                        "alert_level": alert_levels[position],
                        "risk_score": float(risk_score[position]),
                        "binary_attack_probability": float(binary_attack_probability[position]),
                        "multiclass_confidence": float(multiclass_confidence[position]),
                        "mean_anomaly_attack_probability": float(mean_anomaly[position]),
                        "max_anomaly_attack_probability": float(max_anomaly[position]),
                        "anomaly_disagreement": float(anomaly_disagreement[position]),
                        "novelty_score": float(novelty_score[position]),
                        "is_zero_day_candidate": bool(is_zero_day_candidate),
                        "feedback_priority": float(feedback_priority),
                        "feedback_reason_codes": "|".join(sorted(set(review_reasons))),
                        "suggested_feedback_action": feedback_action,
                    }
                    _push_queue_candidate(
                        queue_heap,
                        priority=feedback_priority,
                        sequence_number=queue_sequence,
                        record=record,
                        max_size=config.continuous_learning_max_queue,
                    )
                    queue_sequence += 1
                else:
                    feedback_action = "No analyst review required."

                operational_attack_labels.append(operational_label)
                threat_families.append(descriptor.family)
                mitre_tactics.append(descriptor.tactic)
                severity_tiers.append(descriptor.severity)
                response_playbooks.append(descriptor.playbook)
                recommended_block_scopes.append(descriptor.response_scope)
                classification_confidence_tiers.append(confidence_tier)
                zero_day_reasons.append("|".join(sorted(set(candidate_reasons))) if is_zero_day_candidate else "")
                block_targets.append(block_target)
                repeat_offense_counts.append(repeat_offense_count)
                auto_block_decisions.append(auto_block)
                auto_block_actions.append(auto_block_action_name)
                auto_block_reasons.append(auto_block_reason)
                auto_block_ttls.append(config.auto_block_ttl_minutes if auto_block else 0)
                feedback_priorities.append(float(feedback_priority))
                feedback_reason_codes.append(record["feedback_reason_codes"] if review_reasons else "")
                feedback_actions.append(feedback_action)

                attack_family_counts.update([descriptor.family])
                if is_zero_day_candidate:
                    zero_day_family_counts.update([descriptor.family])
                if auto_block:
                    auto_block_action_counts.update([auto_block_action_name])

            classified_chunk = pd.concat([prediction_chunk, context_frame, anomaly_scores], axis=1)
            classified_chunk["anomaly_disagreement"] = anomaly_disagreement.astype(float)
            classified_chunk["novelty_score"] = novelty_score.astype(float)
            classified_chunk["is_zero_day_candidate"] = zero_day_candidate_mask.astype(bool)
            classified_chunk["zero_day_reason_codes"] = zero_day_reasons
            classified_chunk["operational_attack_label_name"] = operational_attack_labels
            classified_chunk["threat_family"] = threat_families
            classified_chunk["mitre_tactic"] = mitre_tactics
            classified_chunk["severity_tier"] = severity_tiers
            classified_chunk["response_playbook"] = response_playbooks
            classified_chunk["recommended_block_scope"] = recommended_block_scopes
            classified_chunk["classification_confidence_tier"] = classification_confidence_tiers
            classified_chunk["block_target"] = block_targets
            classified_chunk["repeat_offense_count"] = repeat_offense_counts
            classified_chunk["auto_block_decision"] = auto_block_decisions
            classified_chunk["auto_block_action"] = auto_block_actions
            classified_chunk["auto_block_reason"] = auto_block_reasons
            classified_chunk["auto_block_ttl_minutes"] = auto_block_ttls
            classified_chunk["feedback_priority"] = feedback_priorities
            classified_chunk["feedback_reason_codes"] = feedback_reason_codes
            classified_chunk["suggested_feedback_action"] = feedback_actions

            classified_columns = classified_chunk.columns.tolist()
            _append_frame(
                config.classified_predictions_path,
                classified_chunk,
                include_header=include_header["classified"],
            )
            include_header["classified"] = False

            zero_day_chunk = classified_chunk.loc[classified_chunk["is_zero_day_candidate"]].copy()
            if not zero_day_chunk.empty:
                _append_frame(
                    config.zero_day_candidates_path,
                    zero_day_chunk,
                    include_header=include_header["zero_day"],
                )
                include_header["zero_day"] = False

            auto_block_chunk = classified_chunk.loc[classified_chunk["auto_block_decision"]].copy()
            if not auto_block_chunk.empty:
                _append_frame(
                    config.autoblock_actions_path,
                    auto_block_chunk,
                    include_header=include_header["autoblock"],
                )
                include_header["autoblock"] = False

            batch_rows = len(classified_chunk)
            total_rows_processed += batch_rows
            benign_rows += int((prediction_chunk["predicted_multiclass_label_name"].astype(str) == "BENIGN").sum())
            known_attack_rows += int((prediction_chunk["predicted_multiclass_label_name"].astype(str) != "BENIGN").sum())
            zero_day_candidate_rows += int(zero_day_candidate_mask.sum())
            auto_block_rows += int(np.asarray(auto_block_decisions, dtype=bool).sum())
            novelty_score_sum += float(novelty_score.sum())

            active_logger.info(
                "Processed Phase 16 batch %d | rows=%d | cumulative_rows=%d | zero_day_candidates=%d | autoblocks=%d",
                batch_index,
                batch_rows,
                total_rows_processed,
                zero_day_candidate_rows,
                auto_block_rows,
            )
            del classified_chunk, zero_day_chunk, auto_block_chunk, anomaly_scores, context_frame
            gc.collect()

        if next(prediction_iterator, None) is not None:
            raise ValueError("Phase 16 enriched predictions contain more rows than the replayed stream split.")
    finally:
        if hasattr(anomaly_scorer, "close"):
            anomaly_scorer.close()

    if total_rows_processed == 0 or classified_columns is None:
        raise ValueError("Phase 16 did not process any rows.")

    if include_header["zero_day"]:
        pd.DataFrame(columns=classified_columns).to_csv(config.zero_day_candidates_path, index=False)
    if include_header["autoblock"]:
        pd.DataFrame(columns=classified_columns).to_csv(config.autoblock_actions_path, index=False)

    drift_frame = _build_feature_drift_frame(
        reference_stats,
        stream_stats,
        zscore_threshold=config.drift_zscore_threshold,
    )
    drift_frame.to_csv(config.feature_drift_path, index=False)
    drifted_feature_count = int(drift_frame["drifted"].sum())
    top_drift_features = drift_frame.loc[:, "feature_name"].head(config.drift_top_n).astype(str).tolist()

    queue_records = [payload[2] for payload in sorted(queue_heap, key=lambda item: (item[0], item[1]), reverse=True)]
    queue_frame = pd.DataFrame(queue_records, columns=list(QUEUE_COLUMNS))
    queue_frame.to_csv(config.continuous_learning_queue_path, index=False)

    retraining_manifest = _build_retraining_manifest(
        config,
        zero_day_candidate_rows=zero_day_candidate_rows,
        continuous_learning_queue_rows=len(queue_frame),
        drifted_feature_count=drifted_feature_count,
        top_drift_features=top_drift_features,
    )
    _write_json(config.retraining_manifest_path, retraining_manifest)

    report = Phase16Report(
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        input_predictions_path=str(config.input_predictions_path),
        output_dir=str(config.output_dir),
        classified_predictions_path=str(config.classified_predictions_path),
        zero_day_candidates_path=str(config.zero_day_candidates_path),
        autoblock_actions_path=str(config.autoblock_actions_path),
        continuous_learning_queue_path=str(config.continuous_learning_queue_path),
        feature_drift_path=str(config.feature_drift_path),
        retraining_manifest_path=str(config.retraining_manifest_path),
        report_path=str(config.report_path),
        feature_count=len(metadata.feature_names),
        drift_reference_rows=reference_stats.rows,
        total_rows_processed=total_rows_processed,
        benign_rows=benign_rows,
        known_attack_rows=known_attack_rows,
        zero_day_candidate_rows=zero_day_candidate_rows,
        auto_block_rows=auto_block_rows,
        continuous_learning_queue_rows=len(queue_frame),
        drifted_feature_count=drifted_feature_count,
        average_novelty_score=float(novelty_score_sum / total_rows_processed),
        top_drift_features=top_drift_features,
        attack_family_counts={label: int(count) for label, count in attack_family_counts.most_common()},
        zero_day_family_counts={label: int(count) for label, count in zero_day_family_counts.most_common()},
        auto_block_action_counts={label: int(count) for label, count in auto_block_action_counts.most_common()},
        retraining_recommended=bool(retraining_manifest["retraining_recommended"]),
        retraining_reasons=[str(reason) for reason in retraining_manifest["retraining_reasons"]],
        validation_passed=all(
            path.exists()
            for path in (
                config.classified_predictions_path,
                config.zero_day_candidates_path,
                config.autoblock_actions_path,
                config.continuous_learning_queue_path,
                config.feature_drift_path,
                config.retraining_manifest_path,
            )
        ),
        notes=_build_notes(config),
        config=config.to_dict(),
    )
    _write_json(config.report_path, report.to_dict())
    active_logger.info(
        "Completed Phase 16 advanced response | rows=%d | zero_day_candidates=%d | autoblocks=%d | retraining_recommended=%s",
        total_rows_processed,
        zero_day_candidate_rows,
        auto_block_rows,
        report.retraining_recommended,
    )
    return report
