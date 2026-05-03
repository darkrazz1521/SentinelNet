"""Pydantic schemas for the SentinelNet Phase 14 API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

AlertLevel = Literal["Normal", "Suspicious", "Attack"]
ResponseFormat = Literal["json", "ndjson"]


class PredictRecord(BaseModel):
    """Single model-ready record submitted to the prediction endpoint."""

    source_file: str = Field(default="api")
    event_time_utc: datetime | None = None
    features: dict[str, float] = Field(min_length=1)

    model_config = {"extra": "forbid"}


class PredictRequest(BaseModel):
    """Batch prediction request."""

    records: list[PredictRecord] = Field(min_length=1, max_length=2048)

    model_config = {"extra": "forbid"}


class PredictionRecordResponse(BaseModel):
    """Single API prediction response record."""

    stream_order: int
    event_time_utc: str
    source_file: str
    predicted_binary_label: int
    predicted_binary_label_name: str
    binary_attack_probability: float
    predicted_multiclass_label: int
    predicted_multiclass_label_name: str
    multiclass_confidence: float
    selected_binary_variant: str
    selected_multiclass_variant: str
    risk_score: float
    alert_level: AlertLevel
    is_alert: bool
    alert_id: str
    recommended_action: str
    alert_message: str


class PredictResponse(BaseModel):
    """Batch prediction response."""

    record_count: int
    feature_count: int
    selected_binary_variant: str
    selected_multiclass_variant: str
    results: list[PredictionRecordResponse]


class AlertsResponse(BaseModel):
    """Filtered alert listing response."""

    count: int
    has_more: bool
    level_counts: dict[str, int]
    records: list[dict[str, object]]


class StreamResponse(BaseModel):
    """Paged stream replay response."""

    count: int
    has_more: bool
    records: list[dict[str, object]]
