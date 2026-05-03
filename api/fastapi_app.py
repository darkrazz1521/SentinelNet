"""FastAPI application for SentinelNet Phase 14."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

try:
    from fastapi import Depends, FastAPI, HTTPException, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
except ImportError as exc:  # pragma: no cover - exercised only when the app is launched without deps
    raise RuntimeError("Install FastAPI dependencies before running the API application.") from exc

from .config import ApiConfig
from .schemas import AlertsResponse, PredictRequest, PredictResponse, ResponseFormat
from .service import SentinelNetApiService, build_phase14_logger


def _resolve_api_config_path() -> str | None:
    """Resolve an optional Phase 14 config override from the environment."""
    return os.getenv("SENTINELNET_API_CONFIG")


def get_service(request: Request) -> SentinelNetApiService:
    """Return the shared API service instance."""
    return request.app.state.service


def create_app(
    config: ApiConfig | None = None,
    service: SentinelNetApiService | None = None,
) -> FastAPI:
    """Create the Phase 14 FastAPI application."""
    active_config = config or ApiConfig.from_json(_resolve_api_config_path())
    logger = build_phase14_logger(active_config)
    active_service = service or SentinelNetApiService(active_config, logger=logger)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = active_service
        app.state.api_config = active_config
        logger.info("Starting SentinelNet Phase 14 API")
        if active_config.preload_predictor_on_startup:
            warmup_seconds = active_service.warmup_predictor()
            logger.info("Preloaded Phase 14 predictor on startup | load_seconds=%.3f", warmup_seconds)
        try:
            yield
        finally:
            active_service.close()
            logger.info("Stopping SentinelNet Phase 14 API")

    app = FastAPI(
        title=active_config.title,
        version=active_config.version,
        description="Artifact-backed operational API for SentinelNet v2.",
        lifespan=lifespan,
    )
    allowed_origins = [
        origin.strip()
        for origin in os.getenv(
            "SENTINELNET_CORS_ORIGINS",
            "http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173,http://127.0.0.1:5174",
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.service = active_service
    app.state.api_config = active_config

    @app.get("/health", tags=["system"])
    def health(service: SentinelNetApiService = Depends(get_service)) -> dict[str, object]:
        """Return API readiness and artifact status."""
        return service.health()

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    def predict(
        payload: PredictRequest,
        service: SentinelNetApiService = Depends(get_service),
    ) -> dict[str, object]:
        """Run real ensemble inference and Phase 12 alert scoring for submitted records."""
        try:
            records = [
                record.model_dump(mode="python") if hasattr(record, "model_dump") else record.dict()
                for record in payload.records
            ]
            return service.predict(records)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/stream", tags=["streaming"])
    def stream(
        limit: int = Query(default=100, ge=1, le=5000),
        offset: int = Query(default=0, ge=0),
        alerts_only: bool = Query(default=False),
        alert_level: Literal["Normal", "Suspicious", "Attack"] | None = Query(default=None),
        response_format: ResponseFormat = Query(default="json"),
        delay_ms: int = Query(default=0, ge=0, le=5000),
        service: SentinelNetApiService = Depends(get_service),
    ):
        """Return the enriched stream replay as JSON or NDJSON."""
        try:
            if response_format == "ndjson":
                return StreamingResponse(
                    service.iter_stream_ndjson(
                        limit=limit,
                        offset=offset,
                        alerts_only=alerts_only,
                        alert_level=alert_level,
                        delay_ms=delay_ms,
                    ),
                    media_type="application/x-ndjson",
                )
            return service.get_stream_page(
                limit=limit,
                offset=offset,
                alerts_only=alerts_only,
                alert_level=alert_level,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/alerts", response_model=AlertsResponse, tags=["operations"])
    def alerts(
        limit: int = Query(default=100, ge=1, le=5000),
        offset: int = Query(default=0, ge=0),
        alert_level: Literal["Suspicious", "Attack"] | None = Query(default=None),
        attack_type: str | None = Query(default=None),
        min_risk_score: float | None = Query(default=None, ge=0.0, le=100.0),
        service: SentinelNetApiService = Depends(get_service),
    ) -> dict[str, object]:
        """Return persisted Phase 12 alerts with filters."""
        try:
            return service.get_alerts_page(
                limit=limit,
                offset=offset,
                alert_level=alert_level,
                attack_type=attack_type,
                min_risk_score=min_risk_score,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/metrics", tags=["analytics"])
    def metrics(
        refresh: bool = Query(default=False),
        recent_rows: int | None = Query(default=None, ge=0, le=250),
        explanation_top_n: int | None = Query(default=None, ge=1, le=25),
        multiclass_top_k: int | None = Query(default=None, ge=2, le=20),
        service: SentinelNetApiService = Depends(get_service),
    ) -> dict[str, object]:
        """Return operational metrics, confusion matrices, ROC data, and explainability summaries."""
        try:
            return service.get_metrics(
                recent_rows=recent_rows,
                explanation_top_n=explanation_top_n,
                multiclass_top_k=multiclass_top_k,
                refresh=refresh,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
