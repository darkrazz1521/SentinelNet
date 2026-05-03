"""Configuration for SentinelNet Phase 15 performance optimization."""

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


def _positive_int_list(values: list[int] | tuple[int, ...] | None, fallback: list[int]) -> list[int]:
    """Normalize a candidate integer list and enforce positivity."""
    raw_values = fallback if values is None else list(values)
    normalized = sorted({int(value) for value in raw_values if int(value) > 0})
    if not normalized:
        raise ValueError("Performance candidate lists must contain at least one positive integer.")
    return normalized


@dataclass(slots=True)
class PerformanceConfig:
    """Runtime configuration for the Phase 15 performance benchmark stage."""

    project_root: Path = field(default_factory=default_project_root)
    streaming_config_path: Path | None = None
    alerting_config_path: Path | None = None
    dashboard_config_path: Path | None = None
    api_config_path: Path | None = None
    output_dir: Path | None = None
    logs_dir: Path | None = None
    report_filename: str = "performance_report.json"
    streaming_benchmarks_filename: str = "streaming_batch_benchmarks.csv"
    api_predict_benchmarks_filename: str = "api_predict_benchmarks.csv"
    api_read_benchmarks_filename: str = "api_read_benchmarks.csv"
    optimized_streaming_config_filename: str = "optimized_streaming_config.json"
    optimized_api_config_filename: str = "optimized_api_config.json"
    log_filename: str = "phase15_performance.log"
    benchmark_rows: int = 2048
    warmup_rows: int = 256
    warmup_iterations: int = 1
    streaming_repetitions: int = 3
    api_predict_repetitions: int = 3
    api_read_repetitions: int = 3
    metrics_cached_repetitions: int = 3
    candidate_inference_batch_sizes: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    candidate_api_predict_batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 32, 64, 128])
    candidate_stream_page_sizes: list[int] = field(default_factory=lambda: [50, 100, 250, 500, 1000])
    max_streaming_p95_latency_ms: float = 1000.0
    max_api_predict_p95_latency_ms: float = 750.0
    max_stream_page_p95_latency_ms: float = 250.0
    max_alert_page_p95_latency_ms: float = 250.0
    metrics_recent_rows: int = 20
    metrics_explanation_top_n: int = 10
    metrics_multiclass_top_k: int = 8
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.streaming_config_path = _resolve_path(self.project_root, self.streaming_config_path, "config/streaming_config.json")
        self.alerting_config_path = _resolve_path(self.project_root, self.alerting_config_path, "config/alerting_config.json")
        self.dashboard_config_path = _resolve_path(self.project_root, self.dashboard_config_path, "config/dashboard_config.json")
        self.api_config_path = _resolve_path(self.project_root, self.api_config_path, "config/api_config.json")
        self.output_dir = _resolve_path(self.project_root, self.output_dir, "models/saved_models/phase15_performance")
        self.logs_dir = _resolve_path(self.project_root, self.logs_dir, "logs")
        self.benchmark_rows = int(self.benchmark_rows)
        self.warmup_rows = int(self.warmup_rows)
        self.warmup_iterations = int(self.warmup_iterations)
        self.streaming_repetitions = int(self.streaming_repetitions)
        self.api_predict_repetitions = int(self.api_predict_repetitions)
        self.api_read_repetitions = int(self.api_read_repetitions)
        self.metrics_cached_repetitions = int(self.metrics_cached_repetitions)
        self.metrics_recent_rows = int(self.metrics_recent_rows)
        self.metrics_explanation_top_n = int(self.metrics_explanation_top_n)
        self.metrics_multiclass_top_k = int(self.metrics_multiclass_top_k)
        self.candidate_inference_batch_sizes = _positive_int_list(self.candidate_inference_batch_sizes, [64, 128, 256, 512, 1024])
        self.candidate_api_predict_batch_sizes = _positive_int_list(self.candidate_api_predict_batch_sizes, [1, 8, 32, 64, 128])
        self.candidate_stream_page_sizes = _positive_int_list(self.candidate_stream_page_sizes, [50, 100, 250, 500, 1000])
        self.max_streaming_p95_latency_ms = float(self.max_streaming_p95_latency_ms)
        self.max_api_predict_p95_latency_ms = float(self.max_api_predict_p95_latency_ms)
        self.max_stream_page_p95_latency_ms = float(self.max_stream_page_p95_latency_ms)
        self.max_alert_page_p95_latency_ms = float(self.max_alert_page_p95_latency_ms)
        self.log_level = self.log_level.upper()
        if self.benchmark_rows <= 0 or self.warmup_rows <= 0:
            raise ValueError("benchmark_rows and warmup_rows must be positive integers.")

    @property
    def report_path(self) -> Path:
        """Return the main Phase 15 report path."""
        return self.output_dir / self.report_filename

    @property
    def streaming_benchmarks_path(self) -> Path:
        """Return the streaming benchmark CSV path."""
        return self.output_dir / self.streaming_benchmarks_filename

    @property
    def api_predict_benchmarks_path(self) -> Path:
        """Return the API predict benchmark CSV path."""
        return self.output_dir / self.api_predict_benchmarks_filename

    @property
    def api_read_benchmarks_path(self) -> Path:
        """Return the API read benchmark CSV path."""
        return self.output_dir / self.api_read_benchmarks_filename

    @property
    def optimized_streaming_config_path(self) -> Path:
        """Return the optimized streaming config output path."""
        return self.output_dir / self.optimized_streaming_config_filename

    @property
    def optimized_api_config_path(self) -> Path:
        """Return the optimized API config output path."""
        return self.output_dir / self.optimized_api_config_filename

    @property
    def log_path(self) -> Path:
        """Return the Phase 15 log path."""
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
    ) -> "PerformanceConfig":
        """Create a Phase 15 config from a JSON file."""
        root = Path(project_root).resolve() if project_root is not None else default_project_root()
        resolved_config_path = Path(config_path) if config_path is not None else root / "config" / "performance_config.json"
        if not resolved_config_path.is_absolute():
            resolved_config_path = root / resolved_config_path
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        return cls(
            project_root=root,
            streaming_config_path=payload.get("streaming_config_path"),
            alerting_config_path=payload.get("alerting_config_path"),
            dashboard_config_path=payload.get("dashboard_config_path"),
            api_config_path=payload.get("api_config_path"),
            output_dir=payload.get("output_dir"),
            logs_dir=payload.get("logs_dir"),
            report_filename=payload.get("report_filename", "performance_report.json"),
            streaming_benchmarks_filename=payload.get("streaming_benchmarks_filename", "streaming_batch_benchmarks.csv"),
            api_predict_benchmarks_filename=payload.get("api_predict_benchmarks_filename", "api_predict_benchmarks.csv"),
            api_read_benchmarks_filename=payload.get("api_read_benchmarks_filename", "api_read_benchmarks.csv"),
            optimized_streaming_config_filename=payload.get("optimized_streaming_config_filename", "optimized_streaming_config.json"),
            optimized_api_config_filename=payload.get("optimized_api_config_filename", "optimized_api_config.json"),
            log_filename=payload.get("log_filename", "phase15_performance.log"),
            benchmark_rows=payload.get("benchmark_rows", 2048),
            warmup_rows=payload.get("warmup_rows", 256),
            warmup_iterations=payload.get("warmup_iterations", 1),
            streaming_repetitions=payload.get("streaming_repetitions", 3),
            api_predict_repetitions=payload.get("api_predict_repetitions", 3),
            api_read_repetitions=payload.get("api_read_repetitions", 3),
            metrics_cached_repetitions=payload.get("metrics_cached_repetitions", 3),
            candidate_inference_batch_sizes=payload.get("candidate_inference_batch_sizes"),
            candidate_api_predict_batch_sizes=payload.get("candidate_api_predict_batch_sizes"),
            candidate_stream_page_sizes=payload.get("candidate_stream_page_sizes"),
            max_streaming_p95_latency_ms=payload.get("max_streaming_p95_latency_ms", 1000.0),
            max_api_predict_p95_latency_ms=payload.get("max_api_predict_p95_latency_ms", 750.0),
            max_stream_page_p95_latency_ms=payload.get("max_stream_page_p95_latency_ms", 250.0),
            max_alert_page_p95_latency_ms=payload.get("max_alert_page_p95_latency_ms", 250.0),
            metrics_recent_rows=payload.get("metrics_recent_rows", 20),
            metrics_explanation_top_n=payload.get("metrics_explanation_top_n", 10),
            metrics_multiclass_top_k=payload.get("metrics_multiclass_top_k", 8),
            log_level=payload.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config for reports."""
        return {
            "project_root": str(self.project_root),
            "streaming_config_path": str(self.streaming_config_path),
            "alerting_config_path": str(self.alerting_config_path),
            "dashboard_config_path": str(self.dashboard_config_path),
            "api_config_path": str(self.api_config_path),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            "report_filename": self.report_filename,
            "streaming_benchmarks_filename": self.streaming_benchmarks_filename,
            "api_predict_benchmarks_filename": self.api_predict_benchmarks_filename,
            "api_read_benchmarks_filename": self.api_read_benchmarks_filename,
            "optimized_streaming_config_filename": self.optimized_streaming_config_filename,
            "optimized_api_config_filename": self.optimized_api_config_filename,
            "log_filename": self.log_filename,
            "benchmark_rows": self.benchmark_rows,
            "warmup_rows": self.warmup_rows,
            "warmup_iterations": self.warmup_iterations,
            "streaming_repetitions": self.streaming_repetitions,
            "api_predict_repetitions": self.api_predict_repetitions,
            "api_read_repetitions": self.api_read_repetitions,
            "metrics_cached_repetitions": self.metrics_cached_repetitions,
            "candidate_inference_batch_sizes": self.candidate_inference_batch_sizes,
            "candidate_api_predict_batch_sizes": self.candidate_api_predict_batch_sizes,
            "candidate_stream_page_sizes": self.candidate_stream_page_sizes,
            "max_streaming_p95_latency_ms": self.max_streaming_p95_latency_ms,
            "max_api_predict_p95_latency_ms": self.max_api_predict_p95_latency_ms,
            "max_stream_page_p95_latency_ms": self.max_stream_page_p95_latency_ms,
            "max_alert_page_p95_latency_ms": self.max_alert_page_p95_latency_ms,
            "metrics_recent_rows": self.metrics_recent_rows,
            "metrics_explanation_top_n": self.metrics_explanation_top_n,
            "metrics_multiclass_top_k": self.metrics_multiclass_top_k,
            "log_level": self.log_level,
        }
