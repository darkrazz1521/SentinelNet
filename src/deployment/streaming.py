"""SentinelNet Phase 11 real-time streaming simulation."""

from __future__ import annotations

import csv
import gc
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.data_pipeline.logging_utils import configure_logging

from .config import StreamingConfig
from .data import StreamingMetadata, iter_stream_batches, load_streaming_metadata
from .predictor import BatchPredictionResult, StreamingEnsemblePredictor

LOGGER = logging.getLogger("sentinelnet.phase11")


@dataclass(slots=True)
class StreamingReport:
    """Serializable report for Phase 11 streaming simulation."""

    created_at_utc: str
    input_path: str
    output_dir: str
    predictions_path: str
    report_path: str
    stream_split: str
    feature_count: int
    rows_streamed: int
    selected_binary_variant: str
    selected_binary_metric_name: str
    selected_binary_metric_value: float
    selected_multiclass_variant: str
    selected_multiclass_metric_name: str
    selected_multiclass_metric_value: float
    average_batch_latency_ms: float
    p95_batch_latency_ms: float
    throughput_rows_per_second: float
    simulation_start_utc: str
    simulation_end_utc: str
    validation_passed: bool
    notes: list[str]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "created_at_utc": self.created_at_utc,
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "predictions_path": self.predictions_path,
            "report_path": self.report_path,
            "stream_split": self.stream_split,
            "feature_count": self.feature_count,
            "rows_streamed": self.rows_streamed,
            "selected_binary_variant": self.selected_binary_variant,
            "selected_binary_metric_name": self.selected_binary_metric_name,
            "selected_binary_metric_value": self.selected_binary_metric_value,
            "selected_multiclass_variant": self.selected_multiclass_variant,
            "selected_multiclass_metric_name": self.selected_multiclass_metric_name,
            "selected_multiclass_metric_value": self.selected_multiclass_metric_value,
            "average_batch_latency_ms": self.average_batch_latency_ms,
            "p95_batch_latency_ms": self.p95_batch_latency_ms,
            "throughput_rows_per_second": self.throughput_rows_per_second,
            "simulation_start_utc": self.simulation_start_utc,
            "simulation_end_utc": self.simulation_end_utc,
            "validation_passed": self.validation_passed,
            "notes": self.notes,
            "config": self.config,
        }


def build_phase11_logger(config: StreamingConfig) -> logging.Logger:
    """Create the dedicated Phase 11 logger."""
    return configure_logging(config.log_path, config.log_level, logger_name="sentinelnet.phase11")


def _resolve_simulation_start(config: StreamingConfig) -> datetime:
    """Resolve the configured simulation start timestamp."""
    if config.simulation_start_utc is None:
        return datetime.now(tz=timezone.utc)

    candidate = datetime.fromisoformat(config.simulation_start_utc)
    if candidate.tzinfo is None:
        return candidate.replace(tzinfo=timezone.utc)
    return candidate.astimezone(timezone.utc)


def _build_notes(config: StreamingConfig) -> list[str]:
    """Build runtime notes for the report."""
    notes: list[str] = []
    if config.stream_split != "full":
        notes.append(
            "LSTM context is stateful across the streamed split, but omitted rows outside the selected split "
            "do not contribute to sequence history."
        )
    if config.max_rows is not None:
        notes.append(f"Streaming run was capped at {config.max_rows} rows by configuration.")
    return notes


def _write_prediction_rows(
    writer: csv.DictWriter,
    result: BatchPredictionResult,
    batch_latency_ms: float,
    metadata: StreamingMetadata,
    batch_rows_written_before: int,
    simulation_start: datetime,
    event_interval_ms: int,
    batch,
) -> int:
    """Write one streamed batch of predictions to CSV."""
    rows_written = 0
    multiclass_label_to_position = {label: index for index, label in enumerate(sorted(metadata.inverse_multiclass_mapping))}

    for index in range(len(batch.original_indices)):
        global_stream_order = batch_rows_written_before + index
        event_time = simulation_start + timedelta(milliseconds=event_interval_ms * global_stream_order)
        predicted_binary_label = int(result.binary_predicted_labels[index])
        predicted_multiclass_label = int(result.multiclass_predicted_labels[index])
        predicted_multiclass_position = multiclass_label_to_position[predicted_multiclass_label]

        writer.writerow(
            {
                "stream_order": global_stream_order,
                "original_index": int(batch.original_indices[index]),
                "event_time_utc": event_time.isoformat(),
                "source_file": str(batch.source_files[index]),
                "true_binary_label": int(batch.true_binary_labels[index]),
                "true_binary_label_name": "ATTACK" if int(batch.true_binary_labels[index]) == 1 else "BENIGN",
                "true_multiclass_label": int(batch.true_multiclass_labels[index]),
                "true_multiclass_label_name": metadata.inverse_multiclass_mapping.get(int(batch.true_multiclass_labels[index]), str(int(batch.true_multiclass_labels[index]))),
                "predicted_binary_label": predicted_binary_label,
                "predicted_binary_label_name": "ATTACK" if predicted_binary_label == 1 else "BENIGN",
                "binary_attack_probability": float(result.binary_probabilities[index, 1]),
                "predicted_multiclass_label": predicted_multiclass_label,
                "predicted_multiclass_label_name": metadata.inverse_multiclass_mapping.get(predicted_multiclass_label, str(predicted_multiclass_label)),
                "multiclass_confidence": float(result.multiclass_probabilities[index, predicted_multiclass_position]),
                "selected_binary_variant": result.binary_variant,
                "selected_multiclass_variant": result.multiclass_variant,
                "batch_latency_ms": float(batch_latency_ms),
            }
        )
        rows_written += 1

    return rows_written


def _build_report(
    config: StreamingConfig,
    metadata: StreamingMetadata,
    predictor: StreamingEnsemblePredictor,
    rows_streamed: int,
    batch_latencies_ms: list[float],
    wall_clock_seconds: float,
    simulation_start: datetime,
    notes: list[str],
) -> StreamingReport:
    """Build the final Phase 11 report."""
    average_latency = float(np.mean(batch_latencies_ms)) if batch_latencies_ms else 0.0
    p95_latency = float(np.percentile(batch_latencies_ms, 95)) if batch_latencies_ms else 0.0
    throughput = float(rows_streamed / wall_clock_seconds) if wall_clock_seconds > 0 else 0.0
    simulation_end = simulation_start + timedelta(milliseconds=config.event_interval_ms * max(rows_streamed - 1, 0))

    return StreamingReport(
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        input_path=str(config.input_data_path),
        output_dir=str(config.output_dir),
        predictions_path=str(config.predictions_path),
        report_path=str(config.report_path),
        stream_split=config.stream_split,
        feature_count=len(metadata.feature_names),
        rows_streamed=rows_streamed,
        selected_binary_variant=predictor.selected_variants.binary_variant,
        selected_binary_metric_name=predictor.selected_variants.binary_metric_name,
        selected_binary_metric_value=predictor.selected_variants.binary_metric_value,
        selected_multiclass_variant=predictor.selected_variants.multiclass_variant,
        selected_multiclass_metric_name=predictor.selected_variants.multiclass_metric_name,
        selected_multiclass_metric_value=predictor.selected_variants.multiclass_metric_value,
        average_batch_latency_ms=average_latency,
        p95_batch_latency_ms=p95_latency,
        throughput_rows_per_second=throughput,
        simulation_start_utc=simulation_start.isoformat(),
        simulation_end_utc=simulation_end.isoformat(),
        validation_passed=rows_streamed > 0 and config.predictions_path.exists(),
        notes=notes,
        config=config.to_dict(),
    )


def run_streaming_pipeline(
    config: StreamingConfig,
    logger: logging.Logger | None = None,
) -> StreamingReport:
    """Execute the complete SentinelNet Phase 11 streaming simulation."""
    active_logger = logger or LOGGER
    config.ensure_directories()

    required_paths = (
        config.input_data_path,
        config.feature_manifest_path,
        config.label_mapping_path,
        config.phase6_output_dir / "metrics_summary.csv",
        config.phase7_output_dir / "deep_learning_report.json",
        config.phase8_output_dir / "anomaly_detection_report.json",
        config.phase9_output_dir / "metrics_summary.csv",
    )
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Phase 11 dependency not found at {required_path}")

    metadata = load_streaming_metadata(config)
    predictor = StreamingEnsemblePredictor(config, metadata)
    simulation_start = _resolve_simulation_start(config)
    notes = _build_notes(config)
    active_logger.info(
        "Starting streaming simulation | split=%s | feature_count=%d | selected_binary=%s | selected_multiclass=%s",
        config.stream_split,
        len(metadata.feature_names),
        predictor.selected_variants.binary_variant,
        predictor.selected_variants.multiclass_variant,
    )

    fieldnames = [
        "stream_order",
        "original_index",
        "event_time_utc",
        "source_file",
        "true_binary_label",
        "true_binary_label_name",
        "true_multiclass_label",
        "true_multiclass_label_name",
        "predicted_binary_label",
        "predicted_binary_label_name",
        "binary_attack_probability",
        "predicted_multiclass_label",
        "predicted_multiclass_label_name",
        "multiclass_confidence",
        "selected_binary_variant",
        "selected_multiclass_variant",
        "batch_latency_ms",
    ]

    batch_latencies_ms: list[float] = []
    rows_streamed = 0
    simulation_wall_start = perf_counter()
    try:
        with config.predictions_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for batch, _ in iter_stream_batches(config, metadata):
                batch_start = perf_counter()
                result = predictor.predict_batch(batch)
                batch_latency_ms = (perf_counter() - batch_start) * 1000.0
                batch_latencies_ms.append(batch_latency_ms)
                rows_streamed += _write_prediction_rows(
                    writer=writer,
                    result=result,
                    batch_latency_ms=batch_latency_ms,
                    metadata=metadata,
                    batch_rows_written_before=rows_streamed,
                    simulation_start=simulation_start,
                    event_interval_ms=config.event_interval_ms,
                    batch=batch,
                )

                if rows_streamed % max(config.inference_batch_size * 10, 1) == 0 or (
                    metadata.total_rows > 0 and rows_streamed == metadata.total_rows
                ):
                    active_logger.info(
                        "Streamed %d rows | last_batch_latency_ms=%.3f",
                        rows_streamed,
                        batch_latency_ms,
                    )

        wall_clock_seconds = perf_counter() - simulation_wall_start
        report = _build_report(
            config=config,
            metadata=metadata,
            predictor=predictor,
            rows_streamed=rows_streamed,
            batch_latencies_ms=batch_latencies_ms,
            wall_clock_seconds=wall_clock_seconds,
            simulation_start=simulation_start,
            notes=notes,
        )
        config.report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    finally:
        predictor.close()
        gc.collect()

    if not report.validation_passed:
        raise ValueError("Phase 11 validation failed because no streaming predictions were written.")

    active_logger.info(
        "Completed Phase 11 streaming simulation | rows=%d | avg_batch_latency_ms=%.3f | throughput_rows_per_second=%.3f",
        report.rows_streamed,
        report.average_batch_latency_ms,
        report.throughput_rows_per_second,
    )
    return report
