"""Health, readiness, version endpoints (FR-S4)."""

from __future__ import annotations

from fastapi import APIRouter

from emotionsense.backend.state import get_registry
from emotionsense.common.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/health/ready")
def ready() -> dict:
    checks = {"registry": "ok", "default_model": "ok"}
    status = "ready"
    try:
        get_registry().default()
    except Exception:
        checks["default_model"] = "missing"
        status = "degraded"
    return {"status": status, "checks": checks}


@router.get("/version")
def version() -> dict:
    s = get_settings()
    try:
        default = get_registry().default().ref
    except Exception:
        default = None
    return {"api_version": "v1", "default_model": default, "env": s.env}
