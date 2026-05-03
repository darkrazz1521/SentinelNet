"""Configuration for the SentinelNet Phase 14 FastAPI service."""

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
class ApiConfig:
    """Runtime configuration for the Phase 14 FastAPI layer."""

    project_root: Path = field(default_factory=default_project_root)
    streaming_config_path: Path | None = None
    alerting_config_path: Path | None = None
    dashboard_config_path: Path | None = None
    logs_dir: Path | None = None
    log_filename: str = "phase14_api.log"
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "SentinelNet v2 API"
    version: str = "0.14.0"
    default_alert_limit: int = 100
    default_stream_limit: int = 100
    max_page_size: int = 1000
    metrics_recent_rows: int = 20
    metrics_explanation_top_n: int = 10
    metrics_multiclass_top_k: int = 8
    preload_predictor_on_startup: bool = False
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.streaming_config_path = _resolve_path(self.project_root, self.streaming_config_path, "config/streaming_config.json")
        self.alerting_config_path = _resolve_path(self.project_root, self.alerting_config_path, "config/alerting_config.json")
        self.dashboard_config_path = _resolve_path(self.project_root, self.dashboard_config_path, "config/dashboard_config.json")
        self.logs_dir = _resolve_path(self.project_root, self.logs_dir, "logs")
        self.port = int(self.port)
        self.default_alert_limit = int(self.default_alert_limit)
        self.default_stream_limit = int(self.default_stream_limit)
        self.max_page_size = int(self.max_page_size)
        self.metrics_recent_rows = int(self.metrics_recent_rows)
        self.metrics_explanation_top_n = int(self.metrics_explanation_top_n)
        self.metrics_multiclass_top_k = int(self.metrics_multiclass_top_k)
        self.preload_predictor_on_startup = bool(self.preload_predictor_on_startup)
        self.log_level = self.log_level.upper()
        if self.default_alert_limit <= 0 or self.default_stream_limit <= 0 or self.max_page_size <= 0:
            raise ValueError("API pagination settings must be positive integers.")

    @property
    def log_path(self) -> Path:
        """Return the Phase 14 API log path."""
        return self.logs_dir / self.log_filename

    @classmethod
    def from_json(
        cls,
        config_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "ApiConfig":
        """Create an API configuration from a JSON file."""
        root = Path(project_root).resolve() if project_root is not None else default_project_root()
        resolved_config_path = Path(config_path) if config_path is not None else root / "config" / "api_config.json"
        if not resolved_config_path.is_absolute():
            resolved_config_path = root / resolved_config_path
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        return cls(
            project_root=root,
            streaming_config_path=payload.get("streaming_config_path"),
            alerting_config_path=payload.get("alerting_config_path"),
            dashboard_config_path=payload.get("dashboard_config_path"),
            logs_dir=payload.get("logs_dir"),
            log_filename=payload.get("log_filename", "phase14_api.log"),
            host=payload.get("host", "0.0.0.0"),
            port=payload.get("port", 8000),
            title=payload.get("title", "SentinelNet v2 API"),
            version=payload.get("version", "0.14.0"),
            default_alert_limit=payload.get("default_alert_limit", 100),
            default_stream_limit=payload.get("default_stream_limit", 100),
            max_page_size=payload.get("max_page_size", 1000),
            metrics_recent_rows=payload.get("metrics_recent_rows", 20),
            metrics_explanation_top_n=payload.get("metrics_explanation_top_n", 10),
            metrics_multiclass_top_k=payload.get("metrics_multiclass_top_k", 8),
            preload_predictor_on_startup=payload.get("preload_predictor_on_startup", False),
            log_level=payload.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration for diagnostics."""
        return {
            "project_root": str(self.project_root),
            "streaming_config_path": str(self.streaming_config_path),
            "alerting_config_path": str(self.alerting_config_path),
            "dashboard_config_path": str(self.dashboard_config_path),
            "logs_dir": str(self.logs_dir),
            "log_filename": self.log_filename,
            "host": self.host,
            "port": self.port,
            "title": self.title,
            "version": self.version,
            "default_alert_limit": self.default_alert_limit,
            "default_stream_limit": self.default_stream_limit,
            "max_page_size": self.max_page_size,
            "metrics_recent_rows": self.metrics_recent_rows,
            "metrics_explanation_top_n": self.metrics_explanation_top_n,
            "metrics_multiclass_top_k": self.metrics_multiclass_top_k,
            "preload_predictor_on_startup": self.preload_predictor_on_startup,
            "log_level": self.log_level,
        }
