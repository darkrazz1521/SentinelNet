"""Configuration for SentinelNet Phase 5 feature engineering."""

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


@dataclass(slots=True)
class FeatureEngineeringConfig:
    """Runtime configuration for Phase 5 feature engineering and selection."""

    project_root: Path = field(default_factory=default_project_root)
    input_data_path: Path | None = None
    output_dir: Path | None = None
    logs_dir: Path | None = None
    engineered_output_filename: str = "engineered_dataset.csv"
    selected_output_filename: str = "selected_dataset.csv"
    report_filename: str = "feature_engineering_report.json"
    feature_manifest_filename: str = "selected_feature_manifest.json"
    selection_report_filename: str = "feature_selection_report.json"
    mi_scores_filename: str = "mutual_information_scores.csv"
    pca_filename: str = "pca.joblib"
    train_indices_filename: str = "train_indices.npy"
    test_indices_filename: str = "test_indices.npy"
    log_filename: str = "phase5_feature_engineering.log"
    chunk_size: int = 100_000
    test_size: float = 0.2
    random_state: int = 42
    label_column: str = "label"
    source_file_column: str = "source_file"
    binary_target_column: str = "label_binary"
    multiclass_target_column: str = "label_multiclass"
    rolling_windows: tuple[int, ...] = (5, 20)
    short_flow_duration_threshold: float = 1_000_000.0
    correlation_threshold: float = 0.98
    selection_sample_size: int = 200_000
    rfe_sample_size: int = 50_000
    mutual_information_top_k: int = 60
    rfe_n_features_to_select: int = 30
    pca_enabled: bool = True
    pca_variance_threshold: float = 0.95
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.input_data_path = _resolve_path(self.project_root, self.input_data_path, "data/processed/labeled_dataset.csv")
        self.output_dir = _resolve_path(self.project_root, self.output_dir, "data/processed/feature_engineered")
        self.logs_dir = _resolve_path(self.project_root, self.logs_dir, "logs")
        self.chunk_size = int(self.chunk_size)
        self.test_size = float(self.test_size)
        self.random_state = int(self.random_state)
        self.rolling_windows = tuple(sorted(set(int(window) for window in self.rolling_windows)))
        self.selection_sample_size = int(self.selection_sample_size)
        self.rfe_sample_size = int(self.rfe_sample_size)
        self.mutual_information_top_k = int(self.mutual_information_top_k)
        self.rfe_n_features_to_select = int(self.rfe_n_features_to_select)
        self.short_flow_duration_threshold = float(self.short_flow_duration_threshold)
        self.correlation_threshold = float(self.correlation_threshold)
        self.pca_variance_threshold = float(self.pca_variance_threshold)
        self.log_level = self.log_level.upper()

        if not self.rolling_windows:
            raise ValueError("rolling_windows must contain at least one window size.")
        if self.selection_sample_size < 1 or self.rfe_sample_size < 1:
            raise ValueError("selection sample sizes must be positive.")

    @property
    def engineered_output_path(self) -> Path:
        """Return the full engineered dataset path."""
        return self.output_dir / self.engineered_output_filename

    @property
    def selected_output_path(self) -> Path:
        """Return the selected-feature dataset path."""
        return self.output_dir / self.selected_output_filename

    @property
    def report_path(self) -> Path:
        """Return the main Phase 5 report path."""
        return self.output_dir / self.report_filename

    @property
    def feature_manifest_path(self) -> Path:
        """Return the selected-feature manifest path."""
        return self.output_dir / self.feature_manifest_filename

    @property
    def selection_report_path(self) -> Path:
        """Return the feature-selection report path."""
        return self.output_dir / self.selection_report_filename

    @property
    def mi_scores_path(self) -> Path:
        """Return the mutual-information score artifact path."""
        return self.output_dir / self.mi_scores_filename

    @property
    def pca_path(self) -> Path:
        """Return the optional PCA artifact path."""
        return self.output_dir / self.pca_filename

    @property
    def train_indices_path(self) -> Path:
        """Return the saved train-index path."""
        return self.output_dir / self.train_indices_filename

    @property
    def test_indices_path(self) -> Path:
        """Return the saved test-index path."""
        return self.output_dir / self.test_indices_filename

    @property
    def log_path(self) -> Path:
        """Return the Phase 5 log destination."""
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
    ) -> "FeatureEngineeringConfig":
        """Create a Phase 5 configuration from a JSON file."""
        root = Path(project_root).resolve() if project_root is not None else default_project_root()
        resolved_config_path = Path(config_path) if config_path is not None else root / "config" / "feature_engineering_config.json"
        if not resolved_config_path.is_absolute():
            resolved_config_path = root / resolved_config_path
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        return cls(
            project_root=root,
            input_data_path=payload.get("input_data_path"),
            output_dir=payload.get("output_dir"),
            logs_dir=payload.get("logs_dir"),
            engineered_output_filename=payload.get("engineered_output_filename", "engineered_dataset.csv"),
            selected_output_filename=payload.get("selected_output_filename", "selected_dataset.csv"),
            report_filename=payload.get("report_filename", "feature_engineering_report.json"),
            feature_manifest_filename=payload.get("feature_manifest_filename", "selected_feature_manifest.json"),
            selection_report_filename=payload.get("selection_report_filename", "feature_selection_report.json"),
            mi_scores_filename=payload.get("mi_scores_filename", "mutual_information_scores.csv"),
            pca_filename=payload.get("pca_filename", "pca.joblib"),
            train_indices_filename=payload.get("train_indices_filename", "train_indices.npy"),
            test_indices_filename=payload.get("test_indices_filename", "test_indices.npy"),
            log_filename=payload.get("log_filename", "phase5_feature_engineering.log"),
            chunk_size=payload.get("chunk_size", 100_000),
            test_size=payload.get("test_size", 0.2),
            random_state=payload.get("random_state", 42),
            label_column=payload.get("label_column", "label"),
            source_file_column=payload.get("source_file_column", "source_file"),
            binary_target_column=payload.get("binary_target_column", "label_binary"),
            multiclass_target_column=payload.get("multiclass_target_column", "label_multiclass"),
            rolling_windows=tuple(payload.get("rolling_windows", (5, 20))),
            short_flow_duration_threshold=payload.get("short_flow_duration_threshold", 1_000_000.0),
            correlation_threshold=payload.get("correlation_threshold", 0.98),
            selection_sample_size=payload.get("selection_sample_size", 200_000),
            rfe_sample_size=payload.get("rfe_sample_size", 50_000),
            mutual_information_top_k=payload.get("mutual_information_top_k", 60),
            rfe_n_features_to_select=payload.get("rfe_n_features_to_select", 30),
            pca_enabled=payload.get("pca_enabled", True),
            pca_variance_threshold=payload.get("pca_variance_threshold", 0.95),
            log_level=payload.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration for logs and reports."""
        return {
            "project_root": str(self.project_root),
            "input_data_path": str(self.input_data_path),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            "engineered_output_filename": self.engineered_output_filename,
            "selected_output_filename": self.selected_output_filename,
            "report_filename": self.report_filename,
            "feature_manifest_filename": self.feature_manifest_filename,
            "selection_report_filename": self.selection_report_filename,
            "mi_scores_filename": self.mi_scores_filename,
            "pca_filename": self.pca_filename,
            "train_indices_filename": self.train_indices_filename,
            "test_indices_filename": self.test_indices_filename,
            "log_filename": self.log_filename,
            "chunk_size": self.chunk_size,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "label_column": self.label_column,
            "source_file_column": self.source_file_column,
            "binary_target_column": self.binary_target_column,
            "multiclass_target_column": self.multiclass_target_column,
            "rolling_windows": list(self.rolling_windows),
            "short_flow_duration_threshold": self.short_flow_duration_threshold,
            "correlation_threshold": self.correlation_threshold,
            "selection_sample_size": self.selection_sample_size,
            "rfe_sample_size": self.rfe_sample_size,
            "mutual_information_top_k": self.mutual_information_top_k,
            "rfe_n_features_to_select": self.rfe_n_features_to_select,
            "pca_enabled": self.pca_enabled,
            "pca_variance_threshold": self.pca_variance_threshold,
            "log_level": self.log_level,
        }

