"""FastAPI application factory: middleware, error handling, routers.

Lean Core surface: auth (login), predict, models, admin (promote/retire), health, version.
Errors are converted to the standard envelope; every request gets a correlation id (NFR-6).
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from emotionsense.backend.routers import admin, auth, health, models, predict
from emotionsense.common.config import get_settings
from emotionsense.common.errors import AppError
from emotionsense.common.logging import bind_correlation_id, configure_logging, get_logger

log = get_logger("api")


def create_app() -> FastAPI:
    s = get_settings()
    configure_logging(s.log_level)
    app = FastAPI(
        title="EmotionSense AI API",
        version="0.1.0",
        description="Production-ready Speech Emotion Recognition platform API.",
    )

    @app.middleware("http")
    async def correlation_and_timing(request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        bind_correlation_id(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        cid = request.headers.get("X-Correlation-ID")
        log.warning("api.error", code=exc.code, message=exc.message)
        return JSONResponse(status_code=exc.http_status, content=exc.to_envelope(cid))

    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix)
    app.include_router(auth.router, prefix=prefix)
    app.include_router(predict.router, prefix=prefix)
    app.include_router(models.router, prefix=prefix)
    app.include_router(admin.router, prefix=prefix)
    return app


app = create_app()
