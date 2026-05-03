"""SentinelNet Phase 12 alerting pipeline."""

from __future__ import annotations

import gc
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_pipeline.logging_utils import configure_logging

from .alerting_config import AlertingConfig

LOGGER = logging.getLogger("sentinelnet.phase12")

REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "stream_order",
    "event_time_utc",
    "binary_attack_probability",
    "predicted_multiclass_label_name",
    "multiclass_confidence",
)

ATTACK_SEVERITY_MAP: dict[str, float] = {
    "BENIGN": 0.0,
    "Bot": 0.72,
    "DDoS": 0.98,
    "DoS GoldenEye": 0.92,
    "DoS Hulk": 0.95,
    "DoS Slowhttptest": 0.85,
    "DoS slowloris": 0.84,
    "FTP-Patator": 0.70,
    "Heartbleed": 1.00,
    "Infiltration": 0.97,
    "PortScan": 0.68,
    "SSH-Patator": 0.74,
    "Web Attack - Brute Force": 0.78,
    "Web Attack - Sql Injection": 0.99,
    "Web Attack - XSS": 0.80,
}


@dataclass(slots=True)
class AlertingReport:
    """Serializable report for Phase 12 alerting."""

    created_at_utc: str
    input_predictions_path: str
    output_dir: str
    enriched_predictions_path: str
    alerts_path: str
    report_path: str
    total_rows_processed: int
    alert_rows_written: int
    level_counts: dict[str, int]
    predicted_attack_counts: dict[str, int]
    average_risk_score: float
    max_risk_score: float
    selected_binary_variant: str
    selected_multiclass_variant: str
    validation_passed: bool
    notes: list[str]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "created_at_utc": self.created_at_utc,
            "input_predictions_path": self.input_predictions_path,
            "output_dir": self.output_dir,
            "enriched_predictions_path": self.enriched_predictions_path,
            "alerts_path": self.alerts_path,
            "report_path": self.report_path,
            "total_rows_processed": self.total_rows_processed,
            "alert_rows_written": self.alert_rows_written,
            "level_counts": self.level_counts,
            "predicted_attack_counts": self.predicted_attack_counts,
            "average_risk_score": self.average_risk_score,
            "max_risk_score": self.max_risk_score,
            "selected_binary_variant": self.selected_binary_variant,
            "selected_multiclass_variant": self.selected_multiclass_variant,
            "validation_passed": self.validation_passed,
            "notes": self.notes,
            "config": self.config,
        }


def build_phase12_logger(config: AlertingConfig) -> logging.Logger:
    """Create the dedicated Phase 12 logger."""
    return configure_logging(config.log_path, config.log_level, logger_name="sentinelnet.phase12")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _build_recommended_action(level: str, predicted_label_name: str) -> str:
    """Generate a concise recommended action for an alert row."""
    if level == "Attack":
        if predicted_label_name == "DDoS":
            return "Trigger containment workflow and rate-limit suspected sources."
        if predicted_label_name == "PortScan":
            return "Escalate to perimeter monitoring and investigate scanning hosts."
        return "Escalate to incident response and isolate the affected flow or host."
    if level == "Suspicious":
        return "Increase monitoring, capture packet context, and validate against baseline."
    return "Continue monitoring."


def _assign_level(risk_score: float, config: AlertingConfig) -> str:
    """Convert a numeric risk score into an alert level."""
    if risk_score >= config.attack_threshold:
        return "Attack"
    if risk_score >= config.suspicious_threshold:
        return "Suspicious"
    return "Normal"


def _severity_for_label(predicted_label_name: str) -> float:
    """Look up the configured attack severity for a predicted label."""
    if predicted_label_name == "BENIGN":
        return 0.0
    return float(ATTACK_SEVERITY_MAP.get(predicted_label_name, 0.75))


def _validate_prediction_columns(chunk: pd.DataFrame) -> None:
    """Fail fast when the Phase 11 stream output is missing required columns."""
    missing_columns = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in chunk.columns]
    if missing_columns:
        raise ValueError(
            "Phase 12 requires Phase 11 prediction columns that were not found: "
            + ", ".join(sorted(missing_columns))
        )


def _enrich_chunk(
    chunk: pd.DataFrame,
    config: AlertingConfig,
) -> pd.DataFrame:
    """Compute risk scores and alert metadata for a prediction chunk."""
    enriched = chunk.copy()
    binary_probability = enriched["binary_attack_probability"].astype(float)
    predicted_label_name = enriched["predicted_multiclass_label_name"].astype(str)
    attack_class_probability = enriched["multiclass_confidence"].astype(float).where(predicted_label_name != "BENIGN", 0.0)
    predicted_attack_severity = predicted_label_name.map(_severity_for_label).astype(float)
    disagreement_score = (binary_probability - attack_class_probability).abs()

    risk_score = 100.0 * (
        config.binary_probability_weight * binary_probability
        + config.attack_severity_weight * predicted_attack_severity
        + config.class_confidence_weight * attack_class_probability
        + config.disagreement_weight * disagreement_score
    )
    risk_score = risk_score.clip(lower=0.0, upper=100.0)
    alert_levels = risk_score.map(lambda value: _assign_level(float(value), config))
    is_alert = alert_levels.isin(["Suspicious", "Attack"])
    alert_generated_at = datetime.now(tz=timezone.utc).isoformat()

    alert_ids: list[str] = []
    alert_messages: list[str] = []
    actions: list[str] = []
    for position, (level, label_name, score, probability) in enumerate(
        zip(alert_levels, predicted_label_name, risk_score, binary_probability, strict=True)
    ):
        stream_order = int(enriched.iloc[position]["stream_order"])
        if level == "Normal":
            alert_ids.append("")
            alert_messages.append("Normal traffic confidence within acceptable range.")
        else:
            alert_ids.append(f"SNT-{level.upper()}-{stream_order:08d}")
            alert_messages.append(
                f"{level} alert for {label_name} | risk={float(score):.2f} | binary_attack_probability={float(probability):.4f}"
            )
        actions.append(_build_recommended_action(level, label_name))

    enriched["predicted_attack_severity"] = predicted_attack_severity.astype(float)
    enriched["attack_class_probability"] = attack_class_probability.astype(float)
    enriched["disagreement_score"] = disagreement_score.astype(float)
    enriched["risk_score"] = risk_score.astype(float)
    enriched["alert_level"] = alert_levels.astype(str)
    enriched["is_alert"] = is_alert.astype(bool)
    enriched["alert_id"] = alert_ids
    enriched["alert_timestamp_utc"] = enriched["event_time_utc"].astype(str)
    enriched["alert_generated_at_utc"] = alert_generated_at
    enriched["recommended_action"] = actions
    enriched["alert_message"] = alert_messages
    enriched["alert_rule_version"] = "phase12_v1"
    return enriched


def score_prediction_frame(
    frame: pd.DataFrame,
    config: AlertingConfig,
) -> pd.DataFrame:
    """Apply Phase 12 risk scoring and alert enrichment to an in-memory prediction frame."""
    _validate_prediction_columns(frame)
    return _enrich_chunk(frame, config)


def _write_chunk(path: Path, frame: pd.DataFrame, *, include_header: bool) -> None:
    """Append a chunk to a CSV output with an optional header."""
    frame.to_csv(path, mode="a", index=False, header=include_header)


def _build_report(
    config: AlertingConfig,
    *,
    total_rows_processed: int,
    alert_rows_written: int,
    level_counts: Counter[str],
    predicted_attack_counts: Counter[str],
    risk_score_sum: float,
    max_risk_score: float,
    streaming_report: dict[str, Any],
    notes: list[str],
) -> AlertingReport:
    """Build the final Phase 12 report."""
    average_risk_score = float(risk_score_sum / total_rows_processed) if total_rows_processed > 0 else 0.0
    return AlertingReport(
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
        input_predictions_path=str(config.input_predictions_path),
        output_dir=str(config.output_dir),
        enriched_predictions_path=str(config.enriched_predictions_path),
        alerts_path=str(config.alerts_path),
        report_path=str(config.report_path),
        total_rows_processed=total_rows_processed,
        alert_rows_written=alert_rows_written,
        level_counts={
            "Normal": int(level_counts.get("Normal", 0)),
            "Suspicious": int(level_counts.get("Suspicious", 0)),
            "Attack": int(level_counts.get("Attack", 0)),
        },
        predicted_attack_counts={label: int(count) for label, count in predicted_attack_counts.most_common()},
        average_risk_score=average_risk_score,
        max_risk_score=float(max_risk_score),
        selected_binary_variant=str(streaming_report.get("selected_binary_variant", "")),
        selected_multiclass_variant=str(streaming_report.get("selected_multiclass_variant", "")),
        validation_passed=total_rows_processed > 0 and config.enriched_predictions_path.exists() and config.alerts_path.exists(),
        notes=notes,
        config=config.to_dict(),
    )


def run_alerting_pipeline(
    config: AlertingConfig,
    logger: logging.Logger | None = None,
) -> AlertingReport:
    """Execute the complete SentinelNet Phase 12 alerting workflow."""
    active_logger = logger or LOGGER
    config.ensure_directories()

    required_paths = (config.input_predictions_path, config.streaming_report_path)
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Phase 12 dependency not found at {required_path}")

    streaming_report = _read_json(config.streaming_report_path)
    notes = [
        "Risk scores combine binary attack probability, predicted attack severity, multiclass confidence, and model disagreement.",
    ]
    if config.max_rows is not None:
        notes.append(f"Alerting run was capped at {config.max_rows} rows by configuration.")

    active_logger.info(
        "Starting alerting pipeline | selected_binary=%s | selected_multiclass=%s | suspicious_threshold=%.2f | attack_threshold=%.2f",
        streaming_report.get("selected_binary_variant", ""),
        streaming_report.get("selected_multiclass_variant", ""),
        config.suspicious_threshold,
        config.attack_threshold,
    )

    config.enriched_predictions_path.write_text("", encoding="utf-8")
    config.alerts_path.write_text("", encoding="utf-8")

    include_enriched_header = True
    include_alerts_header = True
    total_rows_processed = 0
    alert_rows_written = 0
    risk_score_sum = 0.0
    max_risk_score = 0.0
    level_counts: Counter[str] = Counter()
    predicted_attack_counts: Counter[str] = Counter()

    chunk_iterator = pd.read_csv(config.input_predictions_path, chunksize=config.chunk_size, low_memory=False)
    for chunk in chunk_iterator:
        if config.max_rows is not None:
            remaining = config.max_rows - total_rows_processed
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()

        if chunk.empty:
            continue

        enriched = score_prediction_frame(chunk, config)
        _write_chunk(config.enriched_predictions_path, enriched, include_header=include_enriched_header)
        include_enriched_header = False

        alerts_only = enriched.loc[enriched["is_alert"]].copy()
        if config.emit_normal_events:
            alerts_to_write = enriched.copy()
        else:
            alerts_to_write = alerts_only
        _write_chunk(config.alerts_path, alerts_to_write, include_header=include_alerts_header)
        include_alerts_header = False

        total_rows_processed += len(enriched)
        alert_rows_written += len(alerts_to_write)
        level_counts.update(enriched["alert_level"].astype(str).tolist())
        predicted_attack_counts.update(alerts_only["predicted_multiclass_label_name"].astype(str).tolist())
        risk_score_sum += float(enriched["risk_score"].sum())
        chunk_max = float(enriched["risk_score"].max())
        if chunk_max > max_risk_score:
            max_risk_score = chunk_max

        if total_rows_processed % max(config.chunk_size, 1) == 0:
            active_logger.info(
                "Processed %d rows | alerts_written=%d | current_max_risk=%.2f",
                total_rows_processed,
                alert_rows_written,
                max_risk_score,
            )

    report = _build_report(
        config,
        total_rows_processed=total_rows_processed,
        alert_rows_written=alert_rows_written,
        level_counts=level_counts,
        predicted_attack_counts=predicted_attack_counts,
        risk_score_sum=risk_score_sum,
        max_risk_score=max_risk_score,
        streaming_report=streaming_report,
        notes=notes,
    )
    config.report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    gc.collect()

    if not report.validation_passed:
        raise ValueError("Phase 12 validation failed because no alerting outputs were written.")

    active_logger.info(
        "Completed Phase 12 alerting | rows=%d | alerts_written=%d | attack_alerts=%d",
        report.total_rows_processed,
        report.alert_rows_written,
        report.level_counts.get("Attack", 0),
    )
    return report
