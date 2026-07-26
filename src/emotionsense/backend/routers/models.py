"""Model registry read endpoints (FR-S2)."""

from __future__ import annotations

from fastapi import APIRouter

from emotionsense.backend.state import get_registry

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models(status: str | None = None) -> list[dict]:
    recs = get_registry().list(status=status)
    return [
        {
            "ref": r.ref,
            "name": r.name,
            "version": r.version,
            "family": r.family,
            "status": r.status,
            "is_default": r.is_default,
            "headline_metric": r.headline_metric,
            "label_classes": r.label_classes,
        }
        for r in recs
    ]


@router.get("/{name}/{version}")
def get_model(name: str, version: str) -> dict:
    rec = get_registry().get(f"{name}:{version}")
    return {
        "ref": rec.ref,
        "family": rec.family,
        "status": rec.status,
        "is_default": rec.is_default,
        "feature_spec": rec.feature_spec,
        "label_classes": rec.label_classes,
        "metrics": rec.metrics,
    }
