"""Admin endpoints: model lifecycle (promote/retire) — admin role only (FR-P5)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from emotionsense.backend.deps import CurrentUser, require_admin
from emotionsense.backend.state import get_registry
from emotionsense.common.logging import get_logger

router = APIRouter(prefix="/admin", tags=["admin"])
log = get_logger("admin")


class PromoteRequest(BaseModel):
    make_default: bool = True


@router.post("/models/{name}/{version}/promote")
def promote(
    name: str, version: str, body: PromoteRequest, user: CurrentUser = Depends(require_admin)
) -> dict:
    rec = get_registry().promote(f"{name}:{version}", make_default=body.make_default)
    log.info("admin.promote", ref=rec.ref, actor=user.subject, action="promote")
    return {"ref": rec.ref, "status": rec.status, "is_default": rec.is_default}


@router.post("/models/{name}/{version}/retire")
def retire(name: str, version: str, user: CurrentUser = Depends(require_admin)) -> dict:
    rec = get_registry().retire(f"{name}:{version}")
    log.info("admin.retire", ref=rec.ref, actor=user.subject, action="retire")
    return {"ref": rec.ref, "status": rec.status}
