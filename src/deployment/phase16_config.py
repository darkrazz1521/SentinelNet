"""Configuration for SentinelNet Phase 16 advanced response features."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data_pipeline.config import default_project_root


def _resolve_path(project_root: Path, value: str | Path | None, fallback: str) -> Path:
    """Resolve a path against the project root when it is not absolute."""
    candidate = Path(value) if value is not None else Path(fallback)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


@dataclass(slots=True)
class Phase16Config:
    """Runtime configuration for the Phase 16 advanced response stage."""

    project_root: Path = field(default_factory=default_project_root)
    streaming_config_path: Path | None = None
    input_predictions_path: Path | None = None
    output_dir: Path | None = None
    logs_dir: Path | None = None
    classified_predictions_filename: str = "phase16_classified_predictions.csv"
    zero_day_candidates_filename: str = "phase16_zero_day_candidates.csv"
    autoblock_actions_filename: str = "phase16_autoblock_actions.csv"
    continuous_learning_queue_filename: str = "phase16_continuous_learning_queue.csv"
    feature_drift_filename: str = "phase16_feature_drift_summary.csv"
    retraining_manifest_filename: str = "phase16_retraining_manifest.json"
    report_filename: str = "phase16_advanced_features_report.json"
    log_filename: str = "phase16_advanced_features.log"
    processing_batch_size: int | None = None
    anomaly_batch_size: int = 256
    zero_day_min_mean_anomaly_probability: float = 0.70
    zero_day_min_max_anomaly_probability: float = 0.85
    zero_day_min_disagreement: float = 0.15
    zero_day_max_multiclass_confidence: float = 0.75
    zero_day_min_risk_score: float = 40.0
    low_confidence_binary_margin: float = 0.10
    low_confidence_multiclass_confidence: float = 0.60
    auto_block_attack_threshold: float = 85.0
    auto_block_zero_day_threshold: float = 70.0
    auto_block_min_repeat_hits: int = 2
    auto_block_ttl_minutes: int = 60
    continuous_learning_max_queue: int = 5000
    drift_reference_max_rows: int = 100000
    drift_zscore_threshold: float = 2.50
    drift_top_n: int = 15
    retraining_zero_day_trigger: int = 250
    retraining_queue_trigger: int = 1500
    retraining_drift_feature_trigger: int = 5
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.streaming_config_path = _resolve_path(self.project_root, self.streaming_config_path, "config/streaming_config.json")
        self.input_predictions_path = _resolve_path(
            self.project_root,
            self.input_predictions_path,
            "data/streaming/stream_predictions_with_alerts.csv",
        )
        self.output_dir = _resolve_path(self.project_root, self.output_dir, "data/streaming/phase16_advanced")
        self.logs_dir = _resolve_path(self.project_root, self.logs_dir, "logs")
        self.processing_batch_size = None if self.processing_batch_size is None else int(self.processing_batch_size)
        self.anomaly_batch_size = int(self.anomaly_batch_size)
        self.auto_block_min_repeat_hits = int(self.auto_block_min_repeat_hits)
        self.auto_block_ttl_minutes = int(self.auto_block_ttl_minutes)
        self.continuous_learning_max_queue = int(self.continuous_learning_max_queue)
        self.drift_reference_max_rows = int(self.drift_reference_max_rows)
        self.drift_top_n = int(self.drift_top_n)
        self.retraining_zero_day_trigger = int(self.retraining_zero_day_trigger)
        self.retraining_queue_trigger = int(self.retraining_queue_trigger)
        self.retraining_drift_feature_trigger = int(self.retraining_drift_feature_trigger)
        self.log_level = self.log_level.upper()
        if self.processing_batch_size is not None and self.processing_batch_size <= 0:
            raise ValueError("processing_batch_size must be positive when provided.")
        if self.anomaly_batch_size <= 0:
            raise ValueError("anomaly_batch_size must be a positive integer.")
        if self.auto_block_min_repeat_hits <= 0:
            raise ValueError("auto_block_min_repeat_hits must be positive.")
        if self.continuous_learning_max_queue <= 0:
            raise ValueError("continuous_learning_max_queue must be positive.")
        if self.drift_reference_max_rows <= 0:
            raise ValueError("drift_reference_max_rows must be positive.")

    @property
    def classified_predictions_path(self) -> Path:
        """Return the classified prediction output path."""
        return self.output_dir / self.classified_predictions_filename

    @property
    def zero_day_candidates_path(self) -> Path:
        """Return the zero-day candidate output path."""
        return self.output_dir / self.zero_day_candidates_filename

    @property
    def autoblock_actions_path(self) -> Path:
        """Return the auto-block action output path."""
        return self.output_dir / self.autoblock_actions_filename

    @property
    def continuous_learning_queue_path(self) -> Path:
        """Return the continuous-learning queue output path."""
        return self.output_dir / self.continuous_learning_queue_filename

    @property
    def feature_drift_path(self) -> Path:
        """Return the feature drift summary output path."""
        return self.output_dir / self.feature_drift_filename

    @property
    def retraining_manifest_path(self) -> Path:
        """Return the retraining manifest output path."""
        return self.output_dir / self.retraining_manifest_filename

    @property
    def report_path(self) -> Path:
        """Return the Phase 16 report path."""
        return self.output_dir / self.report_filename

    @property
    def log_path(self) -> Path:
        """Return the Phase 16 log path."""
        return self.logs_dir / self.log_filename

    def ensure_directories(self) -> None:
        """Create runtime directories when they do not already exist."""
        for directory in (self.output_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json(
        cls,
        config_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "Phase16Config":
        """Create a Phase 16 config from a JSON file."""
        root = Path(project_root).resolve() if project_root is not None else default_project_root()
        resolved_config_path = Path(config_path) if config_path is not None else root / "config" / "phase16_config.json"
        if not resolved_config_path.is_absolute():
            resolved_config_path = root / resolved_config_path
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        return cls(
            project_root=root,
            streaming_config_path=payload.get("streaming_config_path"),
            input_predictions_path=payload.get("input_predictions_path"),
            output_dir=payload.get("output_dir"),
            logs_dir=payload.get("logs_dir"),
            classified_predictions_filename=payload.get(
                "classified_predictions_filename",
                "phase16_classified_predictions.csv",
            ),
            zero_day_candidates_filename=payload.get(
                "zero_day_candidates_filename",
                "phase16_zero_day_candidates.csv",
            ),
            autoblock_actions_filename=payload.get(
                "autoblock_actions_filename",
                "phase16_autoblock_actions.csv",
            ),
            continuous_learning_queue_filename=payload.get(
                "continuous_learning_queue_filename",
                "phase16_continuous_learning_queue.csv",
            ),
            feature_drift_filename=payload.get(
                "feature_drift_filename",
                "phase16_feature_drift_summary.csv",
            ),
            retraining_manifest_filename=payload.get(
                "retraining_manifest_filename",
                "phase16_retraining_manifest.json",
            ),
            report_filename=payload.get("report_filename", "phase16_advanced_features_report.json"),
            log_filename=payload.get("log_filename", "phase16_advanced_features.log"),
            processing_batch_size=payload.get("processing_batch_size"),
            anomaly_batch_size=payload.get("anomaly_batch_size", 256),
            zero_day_min_mean_anomaly_probability=payload.get("zero_day_min_mean_anomaly_probability", 0.70),
            zero_day_min_max_anomaly_probability=payload.get("zero_day_min_max_anomaly_probability", 0.85),
            zero_day_min_disagreement=payload.get("zero_day_min_disagreement", 0.15),
            zero_day_max_multiclass_confidence=payload.get("zero_day_max_multiclass_confidence", 0.75),
            zero_day_min_risk_score=payload.get("zero_day_min_risk_score", 40.0),
            low_confidence_binary_margin=payload.get("low_confidence_binary_margin", 0.10),
            low_confidence_multiclass_confidence=payload.get("low_confidence_multiclass_confidence", 0.60),
            auto_block_attack_threshold=payload.get("auto_block_attack_threshold", 85.0),
            auto_block_zero_day_threshold=payload.get("auto_block_zero_day_threshold", 70.0),
            auto_block_min_repeat_hits=payload.get("auto_block_min_repeat_hits", 2),
            auto_block_ttl_minutes=payload.get("auto_block_ttl_minutes", 60),
            continuous_learning_max_queue=payload.get("continuous_learning_max_queue", 5000),
            drift_reference_max_rows=payload.get("drift_reference_max_rows", 100000),
            drift_zscore_threshold=payload.get("drift_zscore_threshold", 2.50),
            drift_top_n=payload.get("drift_top_n", 15),
            retraining_zero_day_trigger=payload.get("retraining_zero_day_trigger", 250),
            retraining_queue_trigger=payload.get("retraining_queue_trigger", 1500),
            retraining_drift_feature_trigger=payload.get("retraining_drift_feature_trigger", 5),
            log_level=payload.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config for diagnostics."""
        return {
            "project_root": str(self.project_root),
            "streaming_config_path": str(self.streaming_config_path),
            "input_predictions_path": str(self.input_predictions_path),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            "classified_predictions_filename": self.classified_predictions_filename,
            "zero_day_candidates_filename": self.zero_day_candidates_filename,
            "autoblock_actions_filename": self.autoblock_actions_filename,
            "continuous_learning_queue_filename": self.continuous_learning_queue_filename,
            "feature_drift_filename": self.feature_drift_filename,
            "retraining_manifest_filename": self.retraining_manifest_filename,
            "report_filename": self.report_filename,
            "log_filename": self.log_filename,
            "processing_batch_size": self.processing_batch_size,
            "anomaly_batch_size": self.anomaly_batch_size,
            "zero_day_min_mean_anomaly_probability": self.zero_day_min_mean_anomaly_probability,
            "zero_day_min_max_anomaly_probability": self.zero_day_min_max_anomaly_probability,
            "zero_day_min_disagreement": self.zero_day_min_disagreement,
            "zero_day_max_multiclass_confidence": self.zero_day_max_multiclass_confidence,
            "zero_day_min_risk_score": self.zero_day_min_risk_score,
            "low_confidence_binary_margin": self.low_confidence_binary_margin,
            "low_confidence_multiclass_confidence": self.low_confidence_multiclass_confidence,
            "auto_block_attack_threshold": self.auto_block_attack_threshold,
            "auto_block_zero_day_threshold": self.auto_block_zero_day_threshold,
            "auto_block_min_repeat_hits": self.auto_block_min_repeat_hits,
            "auto_block_ttl_minutes": self.auto_block_ttl_minutes,
            "continuous_learning_max_queue": self.continuous_learning_max_queue,
            "drift_reference_max_rows": self.drift_reference_max_rows,
            "drift_zscore_threshold": self.drift_zscore_threshold,
            "drift_top_n": self.drift_top_n,
            "retraining_zero_day_trigger": self.retraining_zero_day_trigger,
            "retraining_queue_trigger": self.retraining_queue_trigger,
            "retraining_drift_feature_trigger": self.retraining_drift_feature_trigger,
            "log_level": self.log_level,
        }
