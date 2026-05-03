"""SentinelNet Phase 15 performance optimization pipeline."""

from __future__ import annotations

import gc
import json
import logging
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from api.config import ApiConfig
from api.service import SentinelNetApiService
from src.data_pipeline.logging_utils import configure_logging

from .config import StreamingConfig
from .data import StreamBatch, StreamingMetadata, load_streaming_metadata
from .performance_config import PerformanceConfig
from .predictor import StreamingEnsemblePredictor

LOGGER = logging.getLogger("sentinelnet.phase15")


@dataclass(slots=True)
class PerformanceOptimizationReport:
    """Serializable report for the Phase 15 optimization stage."""

    created_at_utc: str
    output_dir: str
    report_path: str
    streaming_benchmarks_path: str
    api_predict_benchmarks_path: str
    api_read_benchmarks_path: str
    optimized_streaming_config_path: str
    optimized_api_config_path: str
    feature_count: int
    benchmark_rows: int
    predictor_load_seconds: float
    api_predictor_warmup_seconds: float
    recommended_inference_batch_size: int
    recommended_api_predict_batch_size: int
    recommended_stream_page_limit: int
    recommended_alert_page_limit: int
    metrics_refresh_latency_ms: float
    metrics_cached_latency_ms: float
    metrics_cache_speedup: float
    validation_passed: bool
    notes: list[str]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "created_at_utc": self.created_at_utc,
            "output_dir": self.output_dir,
            "report_path": self.report_path,
            "streaming_benchmarks_path": self.streaming_benchmarks_path,
            "api_predict_benchmarks_path": self.api_predict_benchmarks_path,
            "api_read_benchmarks_path": self.api_read_benchmarks_path,
            "optimized_streaming_config_path": self.optimized_streaming_config_path,
            "optimized_api_config_path": self.optimized_api_config_path,
            "feature_count": self.feature_count,
            "benchmark_rows": self.benchmark_rows,
            "predictor_load_seconds": self.predictor_load_seconds,
            "api_predictor_warmup_seconds": self.api_predictor_warmup_seconds,
            "recommended_inference_batch_size": self.recommended_inference_batch_size,
            "recommended_api_predict_batch_size": self.recommended_api_predict_batch_size,
            "recommended_stream_page_limit": self.recommended_stream_page_limit,
            "recommended_alert_page_limit": self.recommended_alert_page_limit,
            "metrics_refresh_latency_ms": self.metrics_refresh_latency_ms,
            "metrics_cached_latency_ms": self.metrics_cached_latency_ms,
            "metrics_cache_speedup": self.metrics_cache_speedup,
            "validation_passed": self.validation_passed,
            "notes": self.notes,
            "config": self.config,
        }


def build_phase15_logger(config: PerformanceConfig) -> logging.Logger:
    """Create the dedicated Phase 15 logger."""
    return configure_logging(config.log_path, config.log_level, logger_name="sentinelnet.phase15")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with indentation."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _collect_benchmark_frame(
    streaming_config: StreamingConfig,
    metadata: StreamingMetadata,
    max_rows: int,
) -> pd.DataFrame:
    """Collect a bounded selected-row frame for predictor and API performance tests."""
    usecols = [
        *metadata.feature_names,
        streaming_config.source_file_column,
        streaming_config.binary_target_column,
        streaming_config.multiclass_target_column,
    ]
    chunk_iterator = pd.read_csv(
        streaming_config.input_data_path,
        usecols=usecols,
        chunksize=streaming_config.chunk_size,
        low_memory=False,
    )
    selected_indices = metadata.selected_indices
    collected: list[pd.DataFrame] = []
    collected_rows = 0
    global_start = 0
    selected_pointer = 0

    for chunk in chunk_iterator:
        chunk_length = len(chunk)
        global_end = global_start + chunk_length
        if selected_indices is None:
            selected_chunk = chunk.copy()
            selected_chunk["_original_index"] = np.arange(global_start, global_end, dtype=np.int64)
        else:
            left = selected_pointer
            right = int(np.searchsorted(selected_indices, global_end, side="left"))
            if right <= left:
                global_start = global_end
                continue
            retained_indices = selected_indices[left:right]
            local_positions = retained_indices - global_start
            selected_chunk = chunk.iloc[local_positions].copy()
            selected_chunk["_original_index"] = retained_indices
            selected_pointer = right

        if selected_chunk.empty:
            global_start = global_end
            continue
        remaining = max_rows - collected_rows
        if remaining <= 0:
            break
        selected_chunk = selected_chunk.iloc[:remaining].copy()
        collected.append(selected_chunk)
        collected_rows += len(selected_chunk)

        global_start = global_end
        if collected_rows >= max_rows:
            break
        if selected_indices is not None and selected_pointer >= len(selected_indices):
            break

    if not collected:
        raise ValueError("Unable to collect any benchmark rows for Phase 15.")
    return pd.concat(collected, ignore_index=True)


def _frame_to_batch(
    frame: pd.DataFrame,
    streaming_config: StreamingConfig,
    metadata: StreamingMetadata,
) -> StreamBatch:
    """Convert a benchmark frame slice into a typed stream batch."""
    feature_frame = frame.loc[:, metadata.feature_names].astype(np.float32)
    return StreamBatch(
        feature_frame=feature_frame,
        original_indices=frame["_original_index"].to_numpy(dtype=np.int64, copy=False),
        source_files=frame[streaming_config.source_file_column].astype(str).to_numpy(copy=False),
        true_binary_labels=frame[streaming_config.binary_target_column].to_numpy(dtype=np.int32, copy=False),
        true_multiclass_labels=frame[streaming_config.multiclass_target_column].to_numpy(dtype=np.int32, copy=False),
    )


def _iter_frame_batches(frame: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    """Split a frame into deterministic contiguous batches."""
    return [frame.iloc[start : start + batch_size].copy() for start in range(0, len(frame), batch_size)]


def _summarize_latency_measurements(
    latencies_ms: list[float],
    *,
    total_rows: int,
    total_calls: int,
    elapsed_seconds: float,
    peak_memory_bytes: int,
) -> dict[str, float | int]:
    """Summarize latency and throughput measurements."""
    latencies = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "total_rows": int(total_rows),
        "total_calls": int(total_calls),
        "mean_batch_latency_ms": float(latencies.mean()) if latencies.size else 0.0,
        "p50_batch_latency_ms": float(np.percentile(latencies, 50)) if latencies.size else 0.0,
        "p95_batch_latency_ms": float(np.percentile(latencies, 95)) if latencies.size else 0.0,
        "p99_batch_latency_ms": float(np.percentile(latencies, 99)) if latencies.size else 0.0,
        "max_batch_latency_ms": float(latencies.max()) if latencies.size else 0.0,
        "mean_row_latency_ms": float(latencies.sum() / max(total_rows, 1)) if latencies.size else 0.0,
        "throughput_rows_per_second": float(total_rows / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
        "peak_memory_mb": float(peak_memory_bytes / (1024 * 1024)),
    }


def _warmup_predictor(
    predictor: Any,
    frame: pd.DataFrame,
    streaming_config: StreamingConfig,
    metadata: StreamingMetadata,
    warmup_rows: int,
    warmup_iterations: int,
) -> None:
    """Warm up predictor graphs and reset state before timed measurements."""
    warmup_slice = frame.iloc[: min(len(frame), warmup_rows)].copy()
    if warmup_slice.empty:
        return
    warmup_batch = _frame_to_batch(warmup_slice, streaming_config, metadata)
    for _ in range(max(warmup_iterations, 0)):
        if hasattr(predictor, "reset_state"):
            predictor.reset_state()
        predictor.predict_batch(warmup_batch)
    if hasattr(predictor, "reset_state"):
        predictor.reset_state()


def _benchmark_streaming_predictor(
    predictor: Any,
    frame: pd.DataFrame,
    phase15_config: PerformanceConfig,
    streaming_config: StreamingConfig,
    metadata: StreamingMetadata,
) -> pd.DataFrame:
    """Benchmark direct predictor throughput and latency for candidate batch sizes."""
    benchmark_rows: list[dict[str, Any]] = []
    _warmup_predictor(
        predictor,
        frame,
        streaming_config,
        metadata,
        warmup_rows=phase15_config.warmup_rows,
        warmup_iterations=phase15_config.warmup_iterations,
    )

    for batch_size in phase15_config.candidate_inference_batch_sizes:
        latencies_ms: list[float] = []
        total_rows = 0
        total_calls = 0
        tracemalloc.start()
        candidate_started = perf_counter()
        for _ in range(phase15_config.streaming_repetitions):
            if hasattr(predictor, "reset_state"):
                predictor.reset_state()
            for batch_frame in _iter_frame_batches(frame, batch_size):
                batch = _frame_to_batch(batch_frame, streaming_config, metadata)
                started = perf_counter()
                predictor.predict_batch(batch)
                latencies_ms.append((perf_counter() - started) * 1000.0)
                total_rows += len(batch_frame)
                total_calls += 1
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        summary = _summarize_latency_measurements(
            latencies_ms,
            total_rows=total_rows,
            total_calls=total_calls,
            elapsed_seconds=perf_counter() - candidate_started,
            peak_memory_bytes=peak_memory_bytes,
        )
        benchmark_rows.append(
            {
                "benchmark": "streaming_predictor",
                "candidate_value": int(batch_size),
                "candidate_label": f"batch_size={batch_size}",
                "repetitions": int(phase15_config.streaming_repetitions),
                **summary,
            }
        )

    return pd.DataFrame(benchmark_rows)


def _build_api_payloads(frame: pd.DataFrame, feature_names: list[str], rows: int) -> list[dict[str, Any]]:
    """Build model-ready API payload records from benchmark rows."""
    limited = frame.iloc[: min(len(frame), rows)].copy()
    payload: list[dict[str, Any]] = []
    for row in limited.itertuples(index=False):
        row_dict = row._asdict()
        payload.append(
            {
                "source_file": str(row_dict["source_file"]),
                "event_time_utc": datetime.now(tz=timezone.utc),
                "features": {feature_name: float(row_dict[feature_name]) for feature_name in feature_names},
            }
        )
    return payload


def _benchmark_api_predict(
    service: SentinelNetApiService,
    frame: pd.DataFrame,
    phase15_config: PerformanceConfig,
) -> pd.DataFrame:
    """Benchmark the API prediction path across candidate request batch sizes."""
    benchmark_rows: list[dict[str, Any]] = []
    feature_names = service.feature_names

    for batch_size in phase15_config.candidate_api_predict_batch_sizes:
        payload = _build_api_payloads(frame, feature_names, batch_size)
        if not payload:
            continue
        service.predict(payload)
        latencies_ms: list[float] = []
        tracemalloc.start()
        started = perf_counter()
        for _ in range(phase15_config.api_predict_repetitions):
            request_started = perf_counter()
            service.predict(payload)
            latencies_ms.append((perf_counter() - request_started) * 1000.0)
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        summary = _summarize_latency_measurements(
            latencies_ms,
            total_rows=len(payload) * phase15_config.api_predict_repetitions,
            total_calls=phase15_config.api_predict_repetitions,
            elapsed_seconds=perf_counter() - started,
            peak_memory_bytes=peak_memory_bytes,
        )
        benchmark_rows.append(
            {
                "benchmark": "api_predict",
                "candidate_value": int(len(payload)),
                "candidate_label": f"request_rows={len(payload)}",
                "repetitions": int(phase15_config.api_predict_repetitions),
                **summary,
            }
        )

    return pd.DataFrame(benchmark_rows)


def _benchmark_api_reads(
    service: SentinelNetApiService,
    phase15_config: PerformanceConfig,
) -> tuple[pd.DataFrame, float, float]:
    """Benchmark stream page, alert page, and metrics retrieval latencies."""
    benchmark_rows: list[dict[str, Any]] = []

    for operation_name, call in (
        ("stream_page", lambda size: service.get_stream_page(limit=size, offset=0, alerts_only=False, alert_level=None)),
        ("alerts_page", lambda size: service.get_alerts_page(limit=size, offset=0, alert_level=None, attack_type=None, min_risk_score=None)),
    ):
        for page_size in phase15_config.candidate_stream_page_sizes:
            latencies_ms: list[float] = []
            tracemalloc.start()
            started = perf_counter()
            for _ in range(phase15_config.api_read_repetitions):
                call_started = perf_counter()
                call(page_size)
                latencies_ms.append((perf_counter() - call_started) * 1000.0)
            _, peak_memory_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            summary = _summarize_latency_measurements(
                latencies_ms,
                total_rows=page_size * phase15_config.api_read_repetitions,
                total_calls=phase15_config.api_read_repetitions,
                elapsed_seconds=perf_counter() - started,
                peak_memory_bytes=peak_memory_bytes,
            )
            benchmark_rows.append(
                {
                    "benchmark": operation_name,
                    "candidate_value": int(page_size),
                    "candidate_label": f"page_size={page_size}",
                    "repetitions": int(phase15_config.api_read_repetitions),
                    **summary,
                }
            )

    refresh_started = perf_counter()
    service.get_metrics(
        recent_rows=phase15_config.metrics_recent_rows,
        explanation_top_n=phase15_config.metrics_explanation_top_n,
        multiclass_top_k=phase15_config.metrics_multiclass_top_k,
        refresh=True,
    )
    metrics_refresh_latency_ms = (perf_counter() - refresh_started) * 1000.0

    cached_latencies_ms: list[float] = []
    for _ in range(phase15_config.metrics_cached_repetitions):
        cached_started = perf_counter()
        service.get_metrics(
            recent_rows=phase15_config.metrics_recent_rows,
            explanation_top_n=phase15_config.metrics_explanation_top_n,
            multiclass_top_k=phase15_config.metrics_multiclass_top_k,
            refresh=False,
        )
        cached_latencies_ms.append((perf_counter() - cached_started) * 1000.0)
    metrics_cached_latency_ms = float(np.mean(cached_latencies_ms)) if cached_latencies_ms else 0.0

    benchmark_rows.append(
        {
            "benchmark": "metrics_refresh",
            "candidate_value": int(phase15_config.metrics_recent_rows),
            "candidate_label": f"recent_rows={phase15_config.metrics_recent_rows}",
            "repetitions": 1,
            "total_rows": int(phase15_config.metrics_recent_rows),
            "total_calls": 1,
            "mean_batch_latency_ms": float(metrics_refresh_latency_ms),
            "p50_batch_latency_ms": float(metrics_refresh_latency_ms),
            "p95_batch_latency_ms": float(metrics_refresh_latency_ms),
            "p99_batch_latency_ms": float(metrics_refresh_latency_ms),
            "max_batch_latency_ms": float(metrics_refresh_latency_ms),
            "mean_row_latency_ms": float(metrics_refresh_latency_ms / max(phase15_config.metrics_recent_rows, 1)),
            "throughput_rows_per_second": float(phase15_config.metrics_recent_rows / max(metrics_refresh_latency_ms / 1000.0, 1e-9)),
            "peak_memory_mb": 0.0,
        }
    )
    benchmark_rows.append(
        {
            "benchmark": "metrics_cached",
            "candidate_value": int(phase15_config.metrics_recent_rows),
            "candidate_label": f"recent_rows={phase15_config.metrics_recent_rows}",
            "repetitions": int(phase15_config.metrics_cached_repetitions),
            "total_rows": int(phase15_config.metrics_recent_rows * phase15_config.metrics_cached_repetitions),
            "total_calls": int(phase15_config.metrics_cached_repetitions),
            "mean_batch_latency_ms": float(metrics_cached_latency_ms),
            "p50_batch_latency_ms": float(np.percentile(cached_latencies_ms, 50)) if cached_latencies_ms else 0.0,
            "p95_batch_latency_ms": float(np.percentile(cached_latencies_ms, 95)) if cached_latencies_ms else 0.0,
            "p99_batch_latency_ms": float(np.percentile(cached_latencies_ms, 99)) if cached_latencies_ms else 0.0,
            "max_batch_latency_ms": float(max(cached_latencies_ms)) if cached_latencies_ms else 0.0,
            "mean_row_latency_ms": float(metrics_cached_latency_ms / max(phase15_config.metrics_recent_rows, 1)),
            "throughput_rows_per_second": float(phase15_config.metrics_recent_rows / max(metrics_cached_latency_ms / 1000.0, 1e-9)),
            "peak_memory_mb": 0.0,
        }
    )

    return pd.DataFrame(benchmark_rows), float(metrics_refresh_latency_ms), float(metrics_cached_latency_ms)


def _select_best_candidate(
    frame: pd.DataFrame,
    *,
    benchmark_name: str,
    latency_threshold_ms: float,
) -> int:
    """Select the highest-throughput candidate that stays within the target p95 latency."""
    benchmark_frame = frame.loc[frame["benchmark"] == benchmark_name].copy()
    if benchmark_frame.empty:
        raise ValueError(f"No benchmark rows found for {benchmark_name!r}.")
    within_target = benchmark_frame.loc[benchmark_frame["p95_batch_latency_ms"] <= latency_threshold_ms].copy()
    selected = within_target if not within_target.empty else benchmark_frame
    selected = selected.sort_values(
        ["throughput_rows_per_second", "p95_batch_latency_ms", "candidate_value"],
        ascending=[False, True, True],
    )
    return int(selected.iloc[0]["candidate_value"])


def _write_optimized_configs(
    phase15_config: PerformanceConfig,
    *,
    recommended_inference_batch_size: int,
    recommended_stream_page_limit: int,
    recommended_alert_page_limit: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write optimized streaming and API config overlays."""
    streaming_payload = _read_json(phase15_config.streaming_config_path)
    api_payload = _read_json(phase15_config.api_config_path)

    streaming_payload["inference_batch_size"] = int(recommended_inference_batch_size)
    api_payload["preload_predictor_on_startup"] = True
    api_payload["default_stream_limit"] = int(recommended_stream_page_limit)
    api_payload["default_alert_limit"] = int(recommended_alert_page_limit)
    api_payload["max_page_size"] = int(max(api_payload.get("max_page_size", 0), recommended_stream_page_limit, recommended_alert_page_limit))

    _write_json(phase15_config.optimized_streaming_config_path, streaming_payload)
    _write_json(phase15_config.optimized_api_config_path, api_payload)
    return streaming_payload, api_payload


def run_performance_optimization_pipeline(
    config: PerformanceConfig,
    logger: logging.Logger | None = None,
    *,
    predictor_factory: type[StreamingEnsemblePredictor] | Any = StreamingEnsemblePredictor,
    service_class: type[SentinelNetApiService] = SentinelNetApiService,
) -> PerformanceOptimizationReport:
    """Execute the full Phase 15 benchmarking and optimization workflow."""
    active_logger = logger or LOGGER
    config.ensure_directories()
    required_paths = (
        config.streaming_config_path,
        config.alerting_config_path,
        config.dashboard_config_path,
        config.api_config_path,
    )
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Phase 15 dependency not found at {required_path}")

    streaming_config = StreamingConfig.from_json(config.streaming_config_path, project_root=config.project_root)
    api_config = ApiConfig.from_json(config.api_config_path, project_root=config.project_root)
    metadata = load_streaming_metadata(streaming_config)
    benchmark_frame = _collect_benchmark_frame(streaming_config, metadata, config.benchmark_rows)
    benchmark_rows = len(benchmark_frame)
    notes = [
        "Phase 15 benchmarks reuse the persisted Phase 11 predictor and Phase 14 API service to produce deployment recommendations.",
        "Predictor state is reset between timed runs so LSTM history does not leak across benchmark candidates.",
    ]

    active_logger.info(
        "Starting Phase 15 performance optimization | benchmark_rows=%d | feature_count=%d",
        benchmark_rows,
        len(metadata.feature_names),
    )

    predictor_started = perf_counter()
    predictor = predictor_factory(streaming_config, metadata)
    predictor_load_seconds = float(perf_counter() - predictor_started)
    active_logger.info("Loaded Phase 15 predictor | load_seconds=%.3f", predictor_load_seconds)

    try:
        streaming_benchmarks = _benchmark_streaming_predictor(
            predictor,
            benchmark_frame,
            config,
            streaming_config,
            metadata,
        )
    finally:
        if hasattr(predictor, "close"):
            predictor.close()

    service = service_class(api_config, predictor_factory=predictor_factory)
    try:
        api_predictor_warmup_seconds = service.warmup_predictor()
        active_logger.info("Warmed API predictor | load_seconds=%.3f", api_predictor_warmup_seconds)
        api_predict_benchmarks = _benchmark_api_predict(service, benchmark_frame, config)
        api_read_benchmarks, metrics_refresh_latency_ms, metrics_cached_latency_ms = _benchmark_api_reads(service, config)
    finally:
        service.close()
        gc.collect()

    recommended_inference_batch_size = _select_best_candidate(
        streaming_benchmarks,
        benchmark_name="streaming_predictor",
        latency_threshold_ms=config.max_streaming_p95_latency_ms,
    )
    recommended_api_predict_batch_size = _select_best_candidate(
        api_predict_benchmarks,
        benchmark_name="api_predict",
        latency_threshold_ms=config.max_api_predict_p95_latency_ms,
    )
    recommended_stream_page_limit = _select_best_candidate(
        api_read_benchmarks,
        benchmark_name="stream_page",
        latency_threshold_ms=config.max_stream_page_p95_latency_ms,
    )
    recommended_alert_page_limit = _select_best_candidate(
        api_read_benchmarks,
        benchmark_name="alerts_page",
        latency_threshold_ms=config.max_alert_page_p95_latency_ms,
    )
    _write_optimized_configs(
        config,
        recommended_inference_batch_size=recommended_inference_batch_size,
        recommended_stream_page_limit=recommended_stream_page_limit,
        recommended_alert_page_limit=recommended_alert_page_limit,
    )

    streaming_benchmarks.to_csv(config.streaming_benchmarks_path, index=False)
    api_predict_benchmarks.to_csv(config.api_predict_benchmarks_path, index=False)
    api_read_benchmarks.to_csv(config.api_read_benchmarks_path, index=False)

    metrics_cache_speedup = float(metrics_refresh_latency_ms / metrics_cached_latency_ms) if metrics_cached_latency_ms > 0 else 0.0
    report = PerformanceOptimizationReport(
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        output_dir=str(config.output_dir),
        report_path=str(config.report_path),
        streaming_benchmarks_path=str(config.streaming_benchmarks_path),
        api_predict_benchmarks_path=str(config.api_predict_benchmarks_path),
        api_read_benchmarks_path=str(config.api_read_benchmarks_path),
        optimized_streaming_config_path=str(config.optimized_streaming_config_path),
        optimized_api_config_path=str(config.optimized_api_config_path),
        feature_count=len(metadata.feature_names),
        benchmark_rows=benchmark_rows,
        predictor_load_seconds=predictor_load_seconds,
        api_predictor_warmup_seconds=api_predictor_warmup_seconds,
        recommended_inference_batch_size=recommended_inference_batch_size,
        recommended_api_predict_batch_size=recommended_api_predict_batch_size,
        recommended_stream_page_limit=recommended_stream_page_limit,
        recommended_alert_page_limit=recommended_alert_page_limit,
        metrics_refresh_latency_ms=metrics_refresh_latency_ms,
        metrics_cached_latency_ms=metrics_cached_latency_ms,
        metrics_cache_speedup=metrics_cache_speedup,
        validation_passed=all(
            path.exists()
            for path in (
                config.streaming_benchmarks_path,
                config.api_predict_benchmarks_path,
                config.api_read_benchmarks_path,
                config.optimized_streaming_config_path,
                config.optimized_api_config_path,
            )
        ),
        notes=notes,
        config=config.to_dict(),
    )
    _write_json(config.report_path, report.to_dict())

    if not report.validation_passed:
        raise ValueError("Phase 15 validation failed because benchmark artifacts were not written.")

    active_logger.info(
        "Completed Phase 15 optimization | recommended_inference_batch_size=%d | recommended_api_predict_batch_size=%d",
        report.recommended_inference_batch_size,
        report.recommended_api_predict_batch_size,
    )
    return report
