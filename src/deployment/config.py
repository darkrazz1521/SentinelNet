"""Configuration for SentinelNet Phase 11 real-time simulation."""

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
class StreamingConfig:
    """Runtime configuration for Phase 11 streaming simulation."""

    project_root: Path = field(default_factory=default_project_root)
    input_data_path: Path | None = None
    feature_manifest_path: Path | None = None
    label_mapping_path: Path | None = None
    train_indices_path: Path | None = None
    test_indices_path: Path | None = None
    phase6_output_dir: Path | None = None
    phase7_output_dir: Path | None = None
    phase8_output_dir: Path | None = None
    phase9_output_dir: Path | None = None
    output_dir: Path | None = None
    logs_dir: Path | None = None
    predictions_filename: str = "stream_predictions.csv"
    report_filename: str = "streaming_report.json"
    log_filename: str = "phase11_streaming.log"
    source_file_column: str = "source_file"
    binary_target_column: str = "label_binary"
    multiclass_target_column: str = "label_multiclass"
    stream_split: str = "test"
    chunk_size: int = 50_000
    inference_batch_size: int = 512
    max_rows: int | None = None
    event_interval_ms: int = 100
    simulation_start_utc: str | None = None
    binary_selection_metric: str = "f1_score"
    multiclass_selection_metric: str = "f1_score"
    binary_variant_override: str | None = None
    multiclass_variant_override: str | None = None
    random_state: int = 42
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.input_data_path = _resolve_path(
            self.project_root,
            self.input_data_path,
            "data/processed/feature_engineered/selected_dataset.csv",
        )
        self.feature_manifest_path = _resolve_path(
            self.project_root,
            self.feature_manifest_path,
            "data/processed/feature_engineered/selected_feature_manifest.json",
        )
        self.label_mapping_path = _resolve_path(
            self.project_root,
            self.label_mapping_path,
            "data/processed/label_mappings.json",
        )
        self.train_indices_path = _resolve_path(
            self.project_root,
            self.train_indices_path,
            "data/processed/feature_engineered/train_indices.npy",
        )
        self.test_indices_path = _resolve_path(
            self.project_root,
            self.test_indices_path,
            "data/processed/feature_engineered/test_indices.npy",
        )
        self.phase6_output_dir = _resolve_path(self.project_root, self.phase6_output_dir, "models/saved_models/phase6_ml")
        self.phase7_output_dir = _resolve_path(self.project_root, self.phase7_output_dir, "models/saved_models/phase7_deep_learning")
        self.phase8_output_dir = _resolve_path(self.project_root, self.phase8_output_dir, "models/saved_models/phase8_anomaly_detection")
        self.phase9_output_dir = _resolve_path(self.project_root, self.phase9_output_dir, "models/saved_models/phase9_ensemble")
        self.output_dir = _resolve_path(self.project_root, self.output_dir, "data/streaming")
        self.logs_dir = _resolve_path(self.project_root, self.logs_dir, "logs")
        self.stream_split = self.stream_split.lower()
        self.chunk_size = int(self.chunk_size)
        self.inference_batch_size = int(self.inference_batch_size)
        self.max_rows = _optional_int(self.max_rows)
        self.event_interval_ms = int(self.event_interval_ms)
        self.binary_selection_metric = str(self.binary_selection_metric)
        self.multiclass_selection_metric = str(self.multiclass_selection_metric)
        self.binary_variant_override = None if self.binary_variant_override is None else str(self.binary_variant_override)
        self.multiclass_variant_override = None if self.multiclass_variant_override is None else str(self.multiclass_variant_override)
        self.random_state = int(self.random_state)
        self.log_level = self.log_level.upper()

    @property
    def predictions_path(self) -> Path:
        """Return the streaming predictions destination."""
        return self.output_dir / self.predictions_filename

    @property
    def report_path(self) -> Path:
        """Return the Phase 11 report destination."""
        return self.output_dir / self.report_filename

    @property
    def log_path(self) -> Path:
        """Return the Phase 11 log destination."""
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
    ) -> "StreamingConfig":
        """Create a Phase 11 configuration from a JSON file."""
        root = Path(project_root).resolve() if project_root is not None else default_project_root()
        resolved_config_path = Path(config_path) if config_path is not None else root / "config" / "streaming_config.json"
        if not resolved_config_path.is_absolute():
            resolved_config_path = root / resolved_config_path
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        return cls(
            project_root=root,
            input_data_path=payload.get("input_data_path"),
            feature_manifest_path=payload.get("feature_manifest_path"),
            label_mapping_path=payload.get("label_mapping_path"),
            train_indices_path=payload.get("train_indices_path"),
            test_indices_path=payload.get("test_indices_path"),
            phase6_output_dir=payload.get("phase6_output_dir"),
            phase7_output_dir=payload.get("phase7_output_dir"),
            phase8_output_dir=payload.get("phase8_output_dir"),
            phase9_output_dir=payload.get("phase9_output_dir"),
            output_dir=payload.get("output_dir"),
            logs_dir=payload.get("logs_dir"),
            predictions_filename=payload.get("predictions_filename", "stream_predictions.csv"),
            report_filename=payload.get("report_filename", "streaming_report.json"),
            log_filename=payload.get("log_filename", "phase11_streaming.log"),
            source_file_column=payload.get("source_file_column", "source_file"),
            binary_target_column=payload.get("binary_target_column", "label_binary"),
            multiclass_target_column=payload.get("multiclass_target_column", "label_multiclass"),
            stream_split=payload.get("stream_split", "test"),
            chunk_size=payload.get("chunk_size", 50_000),
            inference_batch_size=payload.get("inference_batch_size", 512),
            max_rows=payload.get("max_rows"),
            event_interval_ms=payload.get("event_interval_ms", 100),
            simulation_start_utc=payload.get("simulation_start_utc"),
            binary_selection_metric=payload.get("binary_selection_metric", "f1_score"),
            multiclass_selection_metric=payload.get("multiclass_selection_metric", "f1_score"),
            binary_variant_override=payload.get("binary_variant_override"),
            multiclass_variant_override=payload.get("multiclass_variant_override"),
            random_state=payload.get("random_state", 42),
            log_level=payload.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration for reports."""
        return {
            "project_root": str(self.project_root),
            "input_data_path": str(self.input_data_path),
            "feature_manifest_path": str(self.feature_manifest_path),
            "label_mapping_path": str(self.label_mapping_path),
            "train_indices_path": str(self.train_indices_path),
            "test_indices_path": str(self.test_indices_path),
            "phase6_output_dir": str(self.phase6_output_dir),
            "phase7_output_dir": str(self.phase7_output_dir),
            "phase8_output_dir": str(self.phase8_output_dir),
            "phase9_output_dir": str(self.phase9_output_dir),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            "predictions_filename": self.predictions_filename,
            "report_filename": self.report_filename,
            "log_filename": self.log_filename,
            "source_file_column": self.source_file_column,
            "binary_target_column": self.binary_target_column,
            "multiclass_target_column": self.multiclass_target_column,
            "stream_split": self.stream_split,
            "chunk_size": self.chunk_size,
            "inference_batch_size": self.inference_batch_size,
            "max_rows": self.max_rows,
            "event_interval_ms": self.event_interval_ms,
            "simulation_start_utc": self.simulation_start_utc,
            "binary_selection_metric": self.binary_selection_metric,
            "multiclass_selection_metric": self.multiclass_selection_metric,
            "binary_variant_override": self.binary_variant_override,
            "multiclass_variant_override": self.multiclass_variant_override,
            "random_state": self.random_state,
            "log_level": self.log_level,
        }
