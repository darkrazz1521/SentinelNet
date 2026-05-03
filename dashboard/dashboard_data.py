"""Artifact-backed data access for the SentinelNet Phase 13 dashboard."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

from .config import DashboardConfig

ALERT_LEVEL_ORDER: tuple[str, ...] = ("Normal", "Suspicious", "Attack")
RECENT_PREDICTION_COLUMNS: tuple[str, ...] = (
    "stream_order",
    "event_time_utc",
    "source_file",
    "true_multiclass_label_name",
    "predicted_multiclass_label_name",
    "binary_attack_probability",
    "multiclass_confidence",
    "risk_score",
    "alert_level",
    "recommended_action",
)
ROC_INPUT_COLUMNS: tuple[str, ...] = ("true_binary_label", "binary_attack_probability")
CONFUSION_BINARY_COLUMNS: tuple[str, ...] = ("true_binary_label_name", "predicted_binary_label_name")
CONFUSION_MULTICLASS_COLUMNS: tuple[str, ...] = ("true_multiclass_label_name", "predicted_multiclass_label_name")


@dataclass(slots=True)
class DashboardSnapshot:
    """Aggregated data rendered by the Streamlit dashboard."""

    overview_metrics: dict[str, Any]
    alert_level_counts: pd.DataFrame
    attack_distribution: pd.DataFrame
    recent_predictions: pd.DataFrame
    recent_alerts: pd.DataFrame
    binary_confusion_matrix: pd.DataFrame
    multiclass_confusion_matrix: pd.DataFrame
    binary_roc_curve: pd.DataFrame
    binary_roc_auc: float
    alert_timeline: pd.DataFrame
    phase9_metrics: pd.DataFrame
    binary_shap_summary: pd.DataFrame
    multiclass_shap_summary: pd.DataFrame
    binary_ensemble_summary: pd.DataFrame
    multiclass_ensemble_summary: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_exists(path: Path) -> None:
    """Raise a descriptive error if a required artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required Phase 13 artifact not found at {path}")


def _tail_csv(path: Path, rows: int, *, usecols: list[str], chunk_size: int) -> pd.DataFrame:
    """Read the tail of a CSV file without materializing the whole file in memory."""
    if rows <= 0:
        return pd.DataFrame(columns=usecols)

    tail_frame: pd.DataFrame | None = None
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        if tail_frame is None:
            tail_frame = chunk.reset_index(drop=True)
        else:
            tail_frame = pd.concat([tail_frame, chunk], ignore_index=True)
        if len(tail_frame) > rows:
            tail_frame = tail_frame.iloc[-rows:].reset_index(drop=True)
    if tail_frame is None:
        return pd.DataFrame(columns=usecols)
    return tail_frame


def _load_recent_predictions(config: DashboardConfig, rows: int) -> pd.DataFrame:
    """Load the most recent enriched prediction events."""
    _ensure_exists(config.enriched_predictions_path)
    recent = _tail_csv(
        config.enriched_predictions_path,
        rows,
        usecols=list(RECENT_PREDICTION_COLUMNS),
        chunk_size=config.chunk_size,
    )
    if recent.empty:
        return recent
    recent = recent.sort_values("stream_order", ascending=False).reset_index(drop=True)
    return recent


def _load_recent_alerts(config: DashboardConfig, rows: int) -> pd.DataFrame:
    """Load the most recent emitted alert rows."""
    _ensure_exists(config.alerts_path)
    alert_columns = list(RECENT_PREDICTION_COLUMNS) + ["alert_id", "alert_message"]
    recent = _tail_csv(
        config.alerts_path,
        rows,
        usecols=alert_columns,
        chunk_size=config.chunk_size,
    )
    if recent.empty:
        return recent
    recent = recent.sort_values("stream_order", ascending=False).reset_index(drop=True)
    return recent


def _build_alert_level_counts(alerting_report: dict[str, Any]) -> pd.DataFrame:
    """Convert the alert-level summary from the Phase 12 report into a chartable frame."""
    counts = alerting_report.get("level_counts", {})
    return pd.DataFrame(
        {
            "alert_level": list(ALERT_LEVEL_ORDER),
            "count": [int(counts.get(level, 0)) for level in ALERT_LEVEL_ORDER],
        }
    )


def _build_attack_distribution(alerting_report: dict[str, Any]) -> pd.DataFrame:
    """Build the predicted attack-family distribution from the Phase 12 report."""
    counts = alerting_report.get("predicted_attack_counts", {})
    distribution = pd.DataFrame(
        {
            "attack_type": list(counts.keys()),
            "count": [int(value) for value in counts.values()],
        }
    )
    if distribution.empty:
        return pd.DataFrame(columns=["attack_type", "count"])
    distribution = distribution.sort_values(["count", "attack_type"], ascending=[False, True]).reset_index(drop=True)
    return distribution


def _load_binary_arrays(config: DashboardConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the binary labels, predictions, and probabilities needed for metrics."""
    _ensure_exists(config.predictions_path)
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    usecols = ["true_binary_label", "predicted_binary_label", "binary_attack_probability"]

    for chunk in pd.read_csv(config.predictions_path, usecols=usecols, chunksize=config.chunk_size, low_memory=False):
        y_true.append(chunk["true_binary_label"].to_numpy(dtype=int))
        y_pred.append(chunk["predicted_binary_label"].to_numpy(dtype=int))
        y_score.append(chunk["binary_attack_probability"].to_numpy(dtype=float))

    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_score)


def _build_binary_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Compute the binary confusion matrix from the Phase 11 prediction log."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["Actual BENIGN", "Actual ATTACK"],
        columns=["Predicted BENIGN", "Predicted ATTACK"],
    )


def _build_binary_roc_curve(y_true: np.ndarray, y_score: np.ndarray, *, max_points: int = 400) -> tuple[pd.DataFrame, float]:
    """Compute a downsampled binary ROC curve for the live dashboard."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))

    if len(fpr) > max_points:
        indices = np.linspace(0, len(fpr) - 1, num=max_points, dtype=int)
        fpr = fpr[indices]
        tpr = tpr[indices]
        thresholds = thresholds[indices]

    curve = pd.DataFrame(
        {
            "false_positive_rate": fpr.astype(float),
            "true_positive_rate": tpr.astype(float),
            "threshold": thresholds.astype(float),
        }
    )
    return curve, roc_auc


def _build_multiclass_confusion_matrix(config: DashboardConfig, top_k_labels: int) -> pd.DataFrame:
    """Compute a compact multiclass confusion matrix focused on the most common labels."""
    _ensure_exists(config.predictions_path)
    actual_counter: Counter[str] = Counter()
    pair_counter: Counter[tuple[str, str]] = Counter()

    for chunk in pd.read_csv(
        config.predictions_path,
        usecols=list(CONFUSION_MULTICLASS_COLUMNS),
        chunksize=config.chunk_size,
        low_memory=False,
    ):
        actual = chunk["true_multiclass_label_name"].astype(str)
        predicted = chunk["predicted_multiclass_label_name"].astype(str)
        actual_counter.update(actual.tolist())
        pair_counter.update(zip(actual.tolist(), predicted.tolist(), strict=True))

    sorted_actual = [
        label
        for label, _ in actual_counter.most_common()
        if label != "BENIGN"
    ]
    display_labels = ["BENIGN"] + sorted_actual[: max(top_k_labels - 1, 0)]
    display_set = set(display_labels)

    include_other = any(label not in display_set for label in actual_counter)
    matrix_labels = display_labels + (["OTHER"] if include_other else [])
    matrix = pd.DataFrame(0, index=matrix_labels, columns=matrix_labels, dtype=int)

    for (actual_label, predicted_label), count in pair_counter.items():
        actual_bucket = actual_label if actual_label in display_set else "OTHER"
        predicted_bucket = predicted_label if predicted_label in display_set else "OTHER"
        if actual_bucket in matrix.index and predicted_bucket in matrix.columns:
            matrix.loc[actual_bucket, predicted_bucket] += int(count)

    matrix.index = [f"Actual {label}" for label in matrix.index]
    matrix.columns = [f"Predicted {label}" for label in matrix.columns]
    return matrix


def _build_alert_timeline(config: DashboardConfig) -> pd.DataFrame:
    """Aggregate alert counts over time for a live-operations chart."""
    _ensure_exists(config.alerts_path)
    timeline_counter: Counter[tuple[str, str]] = Counter()

    for chunk in pd.read_csv(config.alerts_path, chunksize=config.chunk_size, low_memory=False):
        timestamp_column = "alert_timestamp_utc" if "alert_timestamp_utc" in chunk.columns else "event_time_utc"
        if timestamp_column not in chunk.columns or "alert_level" not in chunk.columns:
            raise ValueError("Phase 13 timeline generation requires alert timestamp and alert_level columns.")
        timestamps = pd.to_datetime(chunk[timestamp_column], utc=True, errors="coerce").dt.floor("min")
        levels = chunk["alert_level"].astype(str)
        valid = pd.DataFrame({"timestamp": timestamps, "level": levels}).dropna()
        timeline_counter.update(
            (timestamp.isoformat(), level)
            for timestamp, level in zip(valid["timestamp"], valid["level"], strict=True)
        )

    if not timeline_counter:
        return pd.DataFrame(columns=["Suspicious", "Attack"])

    timeline_rows: list[dict[str, Any]] = []
    for (timestamp, level), count in timeline_counter.items():
        timeline_rows.append({"timestamp": timestamp, "alert_level": level, "count": int(count)})
    timeline = pd.DataFrame(timeline_rows)
    pivot = (
        timeline.pivot_table(index="timestamp", columns="alert_level", values="count", aggfunc="sum", fill_value=0)
        .sort_index()
        .reset_index()
    )
    return pivot


def _load_phase9_metrics(config: DashboardConfig) -> pd.DataFrame:
    """Load the ensemble metrics summary used by the dashboard performance view."""
    _ensure_exists(config.phase9_metrics_path)
    metrics = pd.read_csv(config.phase9_metrics_path)
    metrics = metrics.sort_values(["task", "f1_score", "roc_auc"], ascending=[True, False, False]).reset_index(drop=True)
    return metrics


def _load_summary_table(path: Path, top_n: int) -> pd.DataFrame:
    """Load and trim a summary CSV for dashboard display."""
    _ensure_exists(path)
    table = pd.read_csv(path)
    if "mean_abs_contribution" in table.columns:
        table = table.sort_values("mean_abs_contribution", ascending=False)
    return table.head(top_n).reset_index(drop=True)


def build_dashboard_snapshot(
    config: DashboardConfig,
    *,
    recent_rows: int = 50,
    explanation_top_n: int = 10,
    multiclass_top_k: int = 8,
) -> DashboardSnapshot:
    """Load all dashboard views from the persisted project artifacts."""
    streaming_report = _read_json(config.streaming_report_path)
    alerting_report = _read_json(config.alerting_report_path)

    overview_metrics = {
        "rows_streamed": int(streaming_report.get("rows_streamed", 0)),
        "throughput_rows_per_second": float(streaming_report.get("throughput_rows_per_second", 0.0)),
        "average_batch_latency_ms": float(streaming_report.get("average_batch_latency_ms", 0.0)),
        "selected_binary_variant": str(streaming_report.get("selected_binary_variant", "")),
        "selected_multiclass_variant": str(streaming_report.get("selected_multiclass_variant", "")),
        "alert_rows_written": int(alerting_report.get("alert_rows_written", 0)),
        "attack_alerts": int(alerting_report.get("level_counts", {}).get("Attack", 0)),
        "average_risk_score": float(alerting_report.get("average_risk_score", 0.0)),
        "max_risk_score": float(alerting_report.get("max_risk_score", 0.0)),
    }

    y_true_binary, y_pred_binary, y_score_binary = _load_binary_arrays(config)
    binary_roc_curve, binary_roc_auc = _build_binary_roc_curve(y_true_binary, y_score_binary)
    snapshot = DashboardSnapshot(
        overview_metrics=overview_metrics,
        alert_level_counts=_build_alert_level_counts(alerting_report),
        attack_distribution=_build_attack_distribution(alerting_report),
        recent_predictions=_load_recent_predictions(config, recent_rows),
        recent_alerts=_load_recent_alerts(config, recent_rows),
        binary_confusion_matrix=_build_binary_confusion_matrix(y_true_binary, y_pred_binary),
        multiclass_confusion_matrix=_build_multiclass_confusion_matrix(config, multiclass_top_k),
        binary_roc_curve=binary_roc_curve,
        binary_roc_auc=binary_roc_auc,
        alert_timeline=_build_alert_timeline(config),
        phase9_metrics=_load_phase9_metrics(config),
        binary_shap_summary=_load_summary_table(config.binary_shap_summary_path, explanation_top_n),
        multiclass_shap_summary=_load_summary_table(config.multiclass_shap_summary_path, explanation_top_n),
        binary_ensemble_summary=_load_summary_table(config.binary_ensemble_summary_path, explanation_top_n),
        multiclass_ensemble_summary=_load_summary_table(config.multiclass_ensemble_summary_path, explanation_top_n),
    )
    return snapshot
