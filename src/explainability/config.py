"""Configuration for SentinelNet Phase 10 explainability."""

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
class ExplainabilityConfig:
    """Runtime configuration for Phase 10 explainability."""

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
    report_filename: str = "explainability_report.json"
    summary_filename: str = "artifact_summary.csv"
    log_filename: str = "phase10_explainability.log"
    source_file_column: str = "source_file"
    binary_target_column: str = "label_binary"
    multiclass_target_column: str = "label_multiclass"
    random_state: int = 42
    background_sample_cap: int | None = 64
    binary_explain_cap: int | None = 24
    multiclass_explain_cap: int | None = 24
    ensemble_binary_explain_cap: int | None = 24
    ensemble_multiclass_explain_cap: int | None = 24
    shap_permutations: int = 12
    top_features_per_explanation: int = 8
    binary_selection_metric: str = "roc_auc"
    multiclass_selection_metric: str = "f1_score"
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
        self.output_dir = _resolve_path(self.project_root, self.output_dir, "models/saved_models/phase10_explainability")
        self.logs_dir = _resolve_path(self.project_root, self.logs_dir, "logs")
        self.random_state = int(self.random_state)
        self.background_sample_cap = _optional_int(self.background_sample_cap)
        self.binary_explain_cap = _optional_int(self.binary_explain_cap)
        self.multiclass_explain_cap = _optional_int(self.multiclass_explain_cap)
        self.ensemble_binary_explain_cap = _optional_int(self.ensemble_binary_explain_cap)
        self.ensemble_multiclass_explain_cap = _optional_int(self.ensemble_multiclass_explain_cap)
        self.shap_permutations = int(self.shap_permutations)
        self.top_features_per_explanation = int(self.top_features_per_explanation)
        self.binary_selection_metric = str(self.binary_selection_metric)
        self.multiclass_selection_metric = str(self.multiclass_selection_metric)
        self.log_level = self.log_level.upper()

    @property
    def phase6_dir(self) -> Path:
        """Return the Phase 10 raw-feature explanation directory."""
        return self.output_dir / "phase6"

    @property
    def phase6_native_dir(self) -> Path:
        """Return the directory for Phase 6 native importances."""
        return self.phase6_dir / "native_importance"

    @property
    def phase6_shap_dir(self) -> Path:
        """Return the directory for raw-feature SHAP-style explanations."""
        return self.phase6_dir / "shap_values"

    @property
    def phase9_dir(self) -> Path:
        """Return the directory for Phase 9 ensemble explanations."""
        return self.output_dir / "phase9"

    @property
    def report_path(self) -> Path:
        """Return the Phase 10 report destination."""
        return self.output_dir / self.report_filename

    @property
    def summary_path(self) -> Path:
        """Return the artifact summary CSV path."""
        return self.output_dir / self.summary_filename

    @property
    def log_path(self) -> Path:
        """Return the Phase 10 log destination."""
        return self.logs_dir / self.log_filename

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not already exist."""
        for directory in (
            self.output_dir,
            self.phase6_dir,
            self.phase6_native_dir,
            self.phase6_shap_dir,
            self.phase9_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json(
        cls,
        config_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "ExplainabilityConfig":
        """Create a Phase 10 configuration from a JSON file."""
        root = Path(project_root).resolve() if project_root is not None else default_project_root()
        resolved_config_path = Path(config_path) if config_path is not None else root / "config" / "explainability_config.json"
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
            report_filename=payload.get("report_filename", "explainability_report.json"),
            summary_filename=payload.get("summary_filename", "artifact_summary.csv"),
            log_filename=payload.get("log_filename", "phase10_explainability.log"),
            source_file_column=payload.get("source_file_column", "source_file"),
            binary_target_column=payload.get("binary_target_column", "label_binary"),
            multiclass_target_column=payload.get("multiclass_target_column", "label_multiclass"),
            random_state=payload.get("random_state", 42),
            background_sample_cap=payload.get("background_sample_cap", 64),
            binary_explain_cap=payload.get("binary_explain_cap", 24),
            multiclass_explain_cap=payload.get("multiclass_explain_cap", 24),
            ensemble_binary_explain_cap=payload.get("ensemble_binary_explain_cap", 24),
            ensemble_multiclass_explain_cap=payload.get("ensemble_multiclass_explain_cap", 24),
            shap_permutations=payload.get("shap_permutations", 12),
            top_features_per_explanation=payload.get("top_features_per_explanation", 8),
            binary_selection_metric=payload.get("binary_selection_metric", "roc_auc"),
            multiclass_selection_metric=payload.get("multiclass_selection_metric", "f1_score"),
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
            "report_filename": self.report_filename,
            "summary_filename": self.summary_filename,
            "log_filename": self.log_filename,
            "source_file_column": self.source_file_column,
            "binary_target_column": self.binary_target_column,
            "multiclass_target_column": self.multiclass_target_column,
            "random_state": self.random_state,
            "background_sample_cap": self.background_sample_cap,
            "binary_explain_cap": self.binary_explain_cap,
            "multiclass_explain_cap": self.multiclass_explain_cap,
            "ensemble_binary_explain_cap": self.ensemble_binary_explain_cap,
            "ensemble_multiclass_explain_cap": self.ensemble_multiclass_explain_cap,
            "shap_permutations": self.shap_permutations,
            "top_features_per_explanation": self.top_features_per_explanation,
            "binary_selection_metric": self.binary_selection_metric,
            "multiclass_selection_metric": self.multiclass_selection_metric,
            "log_level": self.log_level,
        }
