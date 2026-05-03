"""Artifact-backed service layer for the SentinelNet Phase 14 API."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dashboard.config import DashboardConfig
from dashboard.dashboard_data import DashboardSnapshot, build_dashboard_snapshot
from src.data_pipeline.logging_utils import configure_logging
from src.deployment.alerting import score_prediction_frame
from src.deployment.alerting_config import AlertingConfig
from src.deployment.config import StreamingConfig
from src.deployment.data import StreamBatch, StreamingMetadata, load_streaming_metadata
from src.deployment.predictor import BatchPredictionResult, StreamingEnsemblePredictor

from .config import ApiConfig

LOGGER = logging.getLogger("sentinelnet.phase14")

PredictorFactory = Callable[[StreamingConfig, StreamingMetadata], Any]


@dataclass(slots=True)
class PageResult:
    """Paged CSV selection result."""

    records: list[dict[str, Any]]
    has_more: bool


def build_phase14_logger(config: ApiConfig) -> logging.Logger:
    """Create the dedicated Phase 14 logger."""
    return configure_logging(config.log_path, config.log_level, logger_name="sentinelnet.phase14")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    """Convert numeric-like values into JSON-safe floats."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _normalize_datetime(value: datetime | None) -> str:
    """Normalize an optional datetime onto an ISO-8601 UTC string."""
    if value is None:
        return datetime.now(tz=timezone.utc).isoformat()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


def _pythonize_scalar(value: Any) -> Any:
    """Convert numpy and pandas scalars into JSON-serializable Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-ready records."""
    if frame.empty:
        return []
    cleaned = frame.replace({np.nan: None})
    records = cleaned.to_dict(orient="records")
    return [{key: _pythonize_scalar(value) for key, value in record.items()} for record in records]


def _matrix_to_payload(frame: pd.DataFrame) -> dict[str, Any]:
    """Serialize a confusion matrix for JSON responses."""
    return {
        "index": [str(value) for value in frame.index.tolist()],
        "columns": [str(value) for value in frame.columns.tolist()],
        "data": [[int(cell) for cell in row] for row in frame.to_numpy(dtype=int).tolist()],
    }


class SentinelNetApiService:
    """Operational API service that reuses the persisted SentinelNet artifacts."""

    def __init__(
        self,
        api_config: ApiConfig,
        *,
        predictor_factory: PredictorFactory | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.api_config = api_config
        self.logger = logger or LOGGER
        self.streaming_config = StreamingConfig.from_json(api_config.streaming_config_path, project_root=api_config.project_root)
        self.alerting_config = AlertingConfig.from_json(api_config.alerting_config_path, project_root=api_config.project_root)
        self.dashboard_config = DashboardConfig.from_json(api_config.dashboard_config_path, project_root=api_config.project_root)
        self.streaming_metadata = load_streaming_metadata(self.streaming_config)
        self.inverse_multiclass_mapping = self.streaming_metadata.inverse_multiclass_mapping
        self.predictor_factory = predictor_factory or StreamingEnsemblePredictor
        self._predictor: Any | None = None
        self._predictor_lock = threading.Lock()
        self._metrics_cache: tuple[tuple[Any, ...], dict[str, Any]] | None = None

    @property
    def feature_names(self) -> list[str]:
        """Return the ordered feature names expected by the model-serving path."""
        return self.streaming_metadata.feature_names

    def close(self) -> None:
        """Release any lazily loaded predictor state."""
        predictor = self._predictor
        if predictor is not None and hasattr(predictor, "close"):
            predictor.close()
        self._predictor = None

    def _get_predictor(self) -> Any:
        """Lazily load the shared streaming predictor."""
        if self._predictor is None:
            with self._predictor_lock:
                if self._predictor is None:
                    self.logger.info("Loading Phase 14 predictor artifacts for /predict requests")
                    self._predictor = self.predictor_factory(self.streaming_config, self.streaming_metadata)
        return self._predictor

    def warmup_predictor(self) -> float:
        """Eagerly load the predictor artifacts and return the load time in seconds."""
        started = time.perf_counter()
        predictor = self._get_predictor()
        if hasattr(predictor, "reset_state"):
            predictor.reset_state()
        return float(time.perf_counter() - started)

    def reset_predictor_state(self) -> None:
        """Reset mutable predictor state between independent API requests or benchmarks."""
        predictor = self._predictor
        if predictor is not None and hasattr(predictor, "reset_state"):
            predictor.reset_state()

    def health(self) -> dict[str, Any]:
        """Return operational readiness information."""
        required_paths = [
            self.streaming_config.feature_manifest_path,
            self.streaming_config.label_mapping_path,
            self.alerting_config.streaming_report_path,
            self.alerting_config.report_path,
            self.dashboard_config.phase9_metrics_path,
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        return {
            "status": "ready" if not missing else "degraded",
            "missing_artifacts": missing,
            "feature_count": len(self.feature_names),
            "predictor_loaded": self._predictor is not None,
            "selected_binary_variant": _read_json(self.alerting_config.streaming_report_path).get("selected_binary_variant", "")
            if self.alerting_config.streaming_report_path.exists()
            else "",
            "selected_multiclass_variant": _read_json(self.alerting_config.streaming_report_path).get("selected_multiclass_variant", "")
            if self.alerting_config.streaming_report_path.exists()
            else "",
        }

    def _validate_predict_request(self, records: list[dict[str, float]]) -> None:
        """Ensure all request records contain the expected selected features."""
        missing_by_record: list[tuple[int, list[str]]] = []
        for index, feature_map in enumerate(records):
            missing = [feature_name for feature_name in self.feature_names if feature_name not in feature_map]
            if missing:
                missing_by_record.append((index, missing[:10]))
        if missing_by_record:
            sample_index, sample_missing = missing_by_record[0]
            raise ValueError(
                f"Record {sample_index} is missing {len(sample_missing)} required selected features, "
                f"including: {', '.join(sample_missing)}"
            )

    def predict(self, payload: list[dict[str, Any]]) -> dict[str, Any]:
        """Run artifact-backed inference and Phase 12 alert scoring for API-submitted records."""
        feature_maps = [dict(record["features"]) for record in payload]
        self._validate_predict_request(feature_maps)

        feature_frame = pd.DataFrame(
            [{feature_name: float(record[feature_name]) for feature_name in self.feature_names} for record in feature_maps],
            columns=self.feature_names,
        ).astype(np.float32)
        source_files = np.asarray([str(record.get("source_file", "api")) for record in payload], dtype=object)
        event_times = [_normalize_datetime(record.get("event_time_utc")) for record in payload]
        batch = StreamBatch(
            feature_frame=feature_frame,
            original_indices=np.arange(len(payload), dtype=np.int64),
            source_files=source_files,
            true_binary_labels=np.zeros(len(payload), dtype=np.int32),
            true_multiclass_labels=np.zeros(len(payload), dtype=np.int32),
        )

        predictor = self._get_predictor()
        self.reset_predictor_state()
        result: BatchPredictionResult = predictor.predict_batch(batch)
        binary_attack_probability = result.binary_probabilities[:, 1].astype(float, copy=False)
        multiclass_confidence = result.multiclass_probabilities.max(axis=1).astype(float, copy=False)

        prediction_frame = pd.DataFrame(
            {
                "stream_order": np.arange(len(payload), dtype=np.int64),
                "original_index": batch.original_indices,
                "event_time_utc": event_times,
                "source_file": source_files,
                "predicted_binary_label": result.binary_predicted_labels.astype(int, copy=False),
                "predicted_binary_label_name": np.where(result.binary_predicted_labels == 1, "ATTACK", "BENIGN"),
                "binary_attack_probability": binary_attack_probability,
                "predicted_multiclass_label": result.multiclass_predicted_labels.astype(int, copy=False),
                "predicted_multiclass_label_name": [
                    self.inverse_multiclass_mapping[int(label)] for label in result.multiclass_predicted_labels
                ],
                "multiclass_confidence": multiclass_confidence,
                "selected_binary_variant": result.binary_variant,
                "selected_multiclass_variant": result.multiclass_variant,
            }
        )
        scored = score_prediction_frame(prediction_frame, self.alerting_config)
        response_columns = [
            "stream_order",
            "event_time_utc",
            "source_file",
            "predicted_binary_label",
            "predicted_binary_label_name",
            "binary_attack_probability",
            "predicted_multiclass_label",
            "predicted_multiclass_label_name",
            "multiclass_confidence",
            "selected_binary_variant",
            "selected_multiclass_variant",
            "risk_score",
            "alert_level",
            "is_alert",
            "alert_id",
            "recommended_action",
            "alert_message",
        ]
        results = _frame_to_records(scored.loc[:, response_columns])
        return {
            "record_count": len(results),
            "feature_count": len(self.feature_names),
            "selected_binary_variant": result.binary_variant,
            "selected_multiclass_variant": result.multiclass_variant,
            "results": results,
        }

    def _page_csv_records(
        self,
        path: Path,
        *,
        limit: int,
        offset: int,
        alert_level: str | None = None,
        attack_type: str | None = None,
        min_risk_score: float | None = None,
    ) -> PageResult:
        """Read filtered CSV rows page-by-page without loading the whole file."""
        if not path.exists():
            raise FileNotFoundError(f"Required API artifact not found at {path}")

        bounded_limit = min(max(limit, 1), self.api_config.max_page_size)
        target_rows = bounded_limit + 1
        matched_rows = 0
        collected: list[pd.DataFrame] = []
        collected_count = 0

        for chunk in pd.read_csv(path, chunksize=self.dashboard_config.chunk_size, low_memory=False):
            filtered = chunk
            if alert_level is not None and "alert_level" in filtered.columns:
                filtered = filtered.loc[filtered["alert_level"].astype(str) == alert_level]
            if attack_type is not None and "predicted_multiclass_label_name" in filtered.columns:
                filtered = filtered.loc[filtered["predicted_multiclass_label_name"].astype(str) == attack_type]
            if min_risk_score is not None and "risk_score" in filtered.columns:
                filtered = filtered.loc[pd.to_numeric(filtered["risk_score"], errors="coerce") >= min_risk_score]

            filtered_count = len(filtered)
            if filtered_count == 0:
                continue
            if matched_rows + filtered_count <= offset:
                matched_rows += filtered_count
                continue

            start_index = max(0, offset - matched_rows)
            matched_rows += filtered_count
            filtered = filtered.iloc[start_index:].copy()
            if filtered.empty:
                continue

            remaining = target_rows - collected_count
            collected.append(filtered.iloc[:remaining].copy())
            collected_count += min(len(filtered), remaining)
            if collected_count >= target_rows:
                break

        if not collected:
            return PageResult(records=[], has_more=False)

        page = pd.concat(collected, ignore_index=True)
        has_more = len(page) > bounded_limit
        if has_more:
            page = page.iloc[:bounded_limit].copy()
        return PageResult(records=_frame_to_records(page), has_more=has_more)

    def get_stream_page(
        self,
        *,
        limit: int,
        offset: int,
        alerts_only: bool,
        alert_level: str | None,
    ) -> dict[str, Any]:
        """Return a paged slice of the enriched Phase 11/12 stream."""
        path = self.alerting_config.alerts_path if alerts_only else self.alerting_config.enriched_predictions_path
        page = self._page_csv_records(path, limit=limit, offset=offset, alert_level=alert_level)
        return {
            "count": len(page.records),
            "has_more": page.has_more,
            "records": page.records,
        }

    def iter_stream_ndjson(
        self,
        *,
        limit: int,
        offset: int,
        alerts_only: bool,
        alert_level: str | None,
        delay_ms: int,
    ) -> Generator[bytes, None, None]:
        """Yield enriched stream rows as newline-delimited JSON."""
        page = self.get_stream_page(limit=limit, offset=offset, alerts_only=alerts_only, alert_level=alert_level)
        for record in page["records"]:
            yield (json.dumps(record, default=str) + "\n").encode("utf-8")
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

    def get_alerts_page(
        self,
        *,
        limit: int,
        offset: int,
        alert_level: str | None,
        attack_type: str | None,
        min_risk_score: float | None,
    ) -> dict[str, Any]:
        """Return filtered alert rows plus top-level counts from the Phase 12 report."""
        page = self._page_csv_records(
            self.alerting_config.alerts_path,
            limit=limit,
            offset=offset,
            alert_level=alert_level,
            attack_type=attack_type,
            min_risk_score=min_risk_score,
        )
        report = _read_json(self.alerting_config.report_path)
        return {
            "count": len(page.records),
            "has_more": page.has_more,
            "level_counts": {str(key): int(value) for key, value in report.get("level_counts", {}).items()},
            "records": page.records,
        }

    def _metrics_cache_key(
        self,
        *,
        recent_rows: int,
        explanation_top_n: int,
        multiclass_top_k: int,
    ) -> tuple[Any, ...]:
        """Build a file-state cache key for expensive metrics views."""
        paths = [
            self.dashboard_config.streaming_report_path,
            self.dashboard_config.alerting_report_path,
            self.dashboard_config.predictions_path,
            self.dashboard_config.enriched_predictions_path,
            self.dashboard_config.alerts_path,
            self.dashboard_config.phase9_metrics_path,
            self.dashboard_config.binary_shap_summary_path,
            self.dashboard_config.multiclass_shap_summary_path,
            self.dashboard_config.binary_ensemble_summary_path,
            self.dashboard_config.multiclass_ensemble_summary_path,
        ]
        return (
            recent_rows,
            explanation_top_n,
            multiclass_top_k,
            *[(str(path), path.stat().st_mtime_ns if path.exists() else -1) for path in paths],
        )

    def get_metrics(
        self,
        *,
        recent_rows: int | None = None,
        explanation_top_n: int | None = None,
        multiclass_top_k: int | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Return operational metrics, confusion matrices, ROC data, and explainability views."""
        recent_rows = self.api_config.metrics_recent_rows if recent_rows is None else int(recent_rows)
        explanation_top_n = self.api_config.metrics_explanation_top_n if explanation_top_n is None else int(explanation_top_n)
        multiclass_top_k = self.api_config.metrics_multiclass_top_k if multiclass_top_k is None else int(multiclass_top_k)

        cache_key = self._metrics_cache_key(
            recent_rows=recent_rows,
            explanation_top_n=explanation_top_n,
            multiclass_top_k=multiclass_top_k,
        )
        if not refresh and self._metrics_cache is not None and self._metrics_cache[0] == cache_key:
            return self._metrics_cache[1]

        snapshot: DashboardSnapshot = build_dashboard_snapshot(
            self.dashboard_config,
            recent_rows=recent_rows,
            explanation_top_n=explanation_top_n,
            multiclass_top_k=multiclass_top_k,
        )
        payload = {
            "overview": snapshot.overview_metrics,
            "alert_level_counts": _frame_to_records(snapshot.alert_level_counts),
            "attack_distribution": _frame_to_records(snapshot.attack_distribution),
            "binary_confusion_matrix": _matrix_to_payload(snapshot.binary_confusion_matrix),
            "multiclass_confusion_matrix": _matrix_to_payload(snapshot.multiclass_confusion_matrix),
            "binary_roc_auc": float(snapshot.binary_roc_auc),
            "binary_roc_curve": _frame_to_records(snapshot.binary_roc_curve),
            "phase9_metrics": _frame_to_records(snapshot.phase9_metrics),
            "recent_predictions": _frame_to_records(snapshot.recent_predictions),
            "recent_alerts": _frame_to_records(snapshot.recent_alerts),
            "explainability": {
                "binary_shap_summary": _frame_to_records(snapshot.binary_shap_summary),
                "multiclass_shap_summary": _frame_to_records(snapshot.multiclass_shap_summary),
                "binary_ensemble_summary": _frame_to_records(snapshot.binary_ensemble_summary),
                "multiclass_ensemble_summary": _frame_to_records(snapshot.multiclass_ensemble_summary),
            },
        }
        self._metrics_cache = (cache_key, payload)
        return payload
