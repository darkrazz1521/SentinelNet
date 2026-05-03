"""Configuration for SentinelNet Phase 12 alerting."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data_pipeline.config import default_project_root


def _resolve_path(project_root: Path, value: str | Path | None, fallback: str) -> Path:
    """Resolve a possibly relative path against the project root."""
    candidate = Path(value) if value is not None else Path(fallback)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _optional_int(value: int | None) -> int | None:
    """Normalize an optional integer value."""
    if value is None:
        return None
    return int(value)


@dataclass(slots=True)
class AlertingConfig:
    """Runtime configuration for Phase 12 alerting."""

    project_root: Path = field(default_factory=default_project_root)
    input_predictions_path: Path | None = None
    streaming_report_path: Path | None = None
    output_dir: Path | None = None
    logs_dir: Path | None = None
    enriched_predictions_filename: str = "stream_predictions_with_alerts.csv"
    alerts_filename: str = "alerts.csv"
    report_filename: str = "alerting_report.json"
    log_filename: str = "phase12_alerting.log"
    chunk_size: int = 50_000
    max_rows: int | None = None
    suspicious_threshold: float = 40.0
    attack_threshold: float = 70.0
    binary_probability_weight: float = 0.55
    attack_severity_weight: float = 0.20
    class_confidence_weight: float = 0.15
    disagreement_weight: float = 0.10
    emit_normal_events: bool = False
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.input_predictions_path = _resolve_path(
            self.project_root,
            self.input_predictions_path,
            "data/streaming/stream_predictions.csv",
        )
        self.streaming_report_path = _resolve_path(
            self.project_root,
            self.streaming_report_path,
            "data/streaming/streaming_report.json",
        )
        self.output_dir = _resolve_path(self.project_root, self.output_dir, "data/streaming")
        self.logs_dir = _resolve_path(self.project_root, self.logs_dir, "logs")
        self.chunk_size = int(self.chunk_size)
        self.max_rows = _optional_int(self.max_rows)
        self.suspicious_threshold = float(self.suspicious_threshold)
        self.attack_threshold = float(self.attack_threshold)
        self.binary_probability_weight = float(self.binary_probability_weight)
        self.attack_severity_weight = float(self.attack_severity_weight)
        self.class_confidence_weight = float(self.class_confidence_weight)
        self.disagreement_weight = float(self.disagreement_weight)
        self.emit_normal_events = bool(self.emit_normal_events)
        self.log_level = self.log_level.upper()
        if not 0.0 <= self.suspicious_threshold <= 100.0:
            raise ValueError("suspicious_threshold must be between 0 and 100.")
        if not 0.0 <= self.attack_threshold <= 100.0:
            raise ValueError("attack_threshold must be between 0 and 100.")
        if self.attack_threshold < self.suspicious_threshold:
            raise ValueError("attack_threshold must be greater than or equal to suspicious_threshold.")

    @property
    def enriched_predictions_path(self) -> Path:
        """Return the enriched stream output path."""
        return self.output_dir / self.enriched_predictions_filename

    @property
    def alerts_path(self) -> Path:
        """Return the filtered alerts output path."""
        return self.output_dir / self.alerts_filename

    @property
    def report_path(self) -> Path:
        """Return the Phase 12 report destination."""
        return self.output_dir / self.report_filename

    @property
    def log_path(self) -> Path:
        """Return the Phase 12 log destination."""
        return self.logs_dir / self.log_filename

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not already exist."""
        for directory in (self.output_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json(
        cls,
        config_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "AlertingConfig":
        """Create a Phase 12 configuration from a JSON file."""
        root = Path(project_root).resolve() if project_root is not None else default_project_root()
        resolved_config_path = Path(config_path) if config_path is not None else root / "config" / "alerting_config.json"
        if not resolved_config_path.is_absolute():
            resolved_config_path = root / resolved_config_path
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        return cls(
            project_root=root,
            input_predictions_path=payload.get("input_predictions_path"),
            streaming_report_path=payload.get("streaming_report_path"),
            output_dir=payload.get("output_dir"),
            logs_dir=payload.get("logs_dir"),
            enriched_predictions_filename=payload.get("enriched_predictions_filename", "stream_predictions_with_alerts.csv"),
            alerts_filename=payload.get("alerts_filename", "alerts.csv"),
            report_filename=payload.get("report_filename", "alerting_report.json"),
            log_filename=payload.get("log_filename", "phase12_alerting.log"),
            chunk_size=payload.get("chunk_size", 50_000),
            max_rows=payload.get("max_rows"),
            suspicious_threshold=payload.get("suspicious_threshold", 40.0),
            attack_threshold=payload.get("attack_threshold", 70.0),
            binary_probability_weight=payload.get("binary_probability_weight", 0.55),
            attack_severity_weight=payload.get("attack_severity_weight", 0.20),
            class_confidence_weight=payload.get("class_confidence_weight", 0.15),
            disagreement_weight=payload.get("disagreement_weight", 0.10),
            emit_normal_events=payload.get("emit_normal_events", False),
            log_level=payload.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration for reports."""
        return {
            "project_root": str(self.project_root),
            "input_predictions_path": str(self.input_predictions_path),
            "streaming_report_path": str(self.streaming_report_path),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            "enriched_predictions_filename": self.enriched_predictions_filename,
            "alerts_filename": self.alerts_filename,
            "report_filename": self.report_filename,
            "log_filename": self.log_filename,
            "chunk_size": self.chunk_size,
            "max_rows": self.max_rows,
            "suspicious_threshold": self.suspicious_threshold,
            "attack_threshold": self.attack_threshold,
            "binary_probability_weight": self.binary_probability_weight,
            "attack_severity_weight": self.attack_severity_weight,
            "class_confidence_weight": self.class_confidence_weight,
            "disagreement_weight": self.disagreement_weight,
            "emit_normal_events": self.emit_normal_events,
            "log_level": self.log_level,
        }
