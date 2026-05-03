"""Configuration for the SentinelNet Phase 13 dashboard."""

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
class DashboardConfig:
    """Runtime configuration for the Phase 13 Streamlit dashboard."""

    project_root: Path = field(default_factory=default_project_root)
    streaming_dir: Path | None = None
    explainability_dir: Path | None = None
    phase9_output_dir: Path | None = None
    streaming_report_path: Path | None = None
    alerting_report_path: Path | None = None
    predictions_path: Path | None = None
    enriched_predictions_path: Path | None = None
    alerts_path: Path | None = None
    phase9_metrics_path: Path | None = None
    binary_shap_summary_path: Path | None = None
    multiclass_shap_summary_path: Path | None = None
    binary_ensemble_summary_path: Path | None = None
    multiclass_ensemble_summary_path: Path | None = None
    chunk_size: int = 100_000

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.streaming_dir = _resolve_path(self.project_root, self.streaming_dir, "data/streaming")
        self.explainability_dir = _resolve_path(
            self.project_root,
            self.explainability_dir,
            "models/saved_models/phase10_explainability",
        )
        self.phase9_output_dir = _resolve_path(
            self.project_root,
            self.phase9_output_dir,
            "models/saved_models/phase9_ensemble",
        )
        self.streaming_report_path = _resolve_path(
            self.project_root,
            self.streaming_report_path,
            "data/streaming/streaming_report.json",
        )
        self.alerting_report_path = _resolve_path(
            self.project_root,
            self.alerting_report_path,
            "data/streaming/alerting_report.json",
        )
        self.predictions_path = _resolve_path(
            self.project_root,
            self.predictions_path,
            "data/streaming/stream_predictions.csv",
        )
        self.enriched_predictions_path = _resolve_path(
            self.project_root,
            self.enriched_predictions_path,
            "data/streaming/stream_predictions_with_alerts.csv",
        )
        self.alerts_path = _resolve_path(
            self.project_root,
            self.alerts_path,
            "data/streaming/alerts.csv",
        )
        self.phase9_metrics_path = _resolve_path(
            self.project_root,
            self.phase9_metrics_path,
            "models/saved_models/phase9_ensemble/metrics_summary.csv",
        )
        self.binary_shap_summary_path = _resolve_path(
            self.project_root,
            self.binary_shap_summary_path,
            "models/saved_models/phase10_explainability/phase6/shap_values/binary_lightgbm_summary.csv",
        )
        self.multiclass_shap_summary_path = _resolve_path(
            self.project_root,
            self.multiclass_shap_summary_path,
            "models/saved_models/phase10_explainability/phase6/shap_values/multiclass_random_forest_summary.csv",
        )
        self.binary_ensemble_summary_path = _resolve_path(
            self.project_root,
            self.binary_ensemble_summary_path,
            "models/saved_models/phase10_explainability/phase9/binary_weighted_scoring_summary.csv",
        )
        self.multiclass_ensemble_summary_path = _resolve_path(
            self.project_root,
            self.multiclass_ensemble_summary_path,
            "models/saved_models/phase10_explainability/phase9/multiclass_stacking_summary.csv",
        )
        self.chunk_size = int(self.chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")

    @classmethod
    def from_json(
        cls,
        config_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "DashboardConfig":
        """Create a dashboard configuration from a JSON file."""
        root = Path(project_root).resolve() if project_root is not None else default_project_root()
        resolved_config_path = Path(config_path) if config_path is not None else root / "config" / "dashboard_config.json"
        if not resolved_config_path.is_absolute():
            resolved_config_path = root / resolved_config_path
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        return cls(
            project_root=root,
            streaming_dir=payload.get("streaming_dir"),
            explainability_dir=payload.get("explainability_dir"),
            phase9_output_dir=payload.get("phase9_output_dir"),
            streaming_report_path=payload.get("streaming_report_path"),
            alerting_report_path=payload.get("alerting_report_path"),
            predictions_path=payload.get("predictions_path"),
            enriched_predictions_path=payload.get("enriched_predictions_path"),
            alerts_path=payload.get("alerts_path"),
            phase9_metrics_path=payload.get("phase9_metrics_path"),
            binary_shap_summary_path=payload.get("binary_shap_summary_path"),
            multiclass_shap_summary_path=payload.get("multiclass_shap_summary_path"),
            binary_ensemble_summary_path=payload.get("binary_ensemble_summary_path"),
            multiclass_ensemble_summary_path=payload.get("multiclass_ensemble_summary_path"),
            chunk_size=payload.get("chunk_size", 100_000),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration for debugging and UI display."""
        return {
            "project_root": str(self.project_root),
            "streaming_dir": str(self.streaming_dir),
            "explainability_dir": str(self.explainability_dir),
            "phase9_output_dir": str(self.phase9_output_dir),
            "streaming_report_path": str(self.streaming_report_path),
            "alerting_report_path": str(self.alerting_report_path),
            "predictions_path": str(self.predictions_path),
            "enriched_predictions_path": str(self.enriched_predictions_path),
            "alerts_path": str(self.alerts_path),
            "phase9_metrics_path": str(self.phase9_metrics_path),
            "binary_shap_summary_path": str(self.binary_shap_summary_path),
            "multiclass_shap_summary_path": str(self.multiclass_shap_summary_path),
            "binary_ensemble_summary_path": str(self.binary_ensemble_summary_path),
            "multiclass_ensemble_summary_path": str(self.multiclass_ensemble_summary_path),
            "chunk_size": self.chunk_size,
        }
