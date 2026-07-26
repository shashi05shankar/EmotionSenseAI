"""FastAPI dependencies: auth + role guards."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header

from emotionsense.backend.security import decode_token
from emotionsense.common.errors import ForbiddenError, UnauthenticatedError


@dataclass(slots=True)
class CurrentUser:
    subject: str
    role: str


def get_current_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise UnauthenticatedError("Missing bearer token")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    return CurrentUser(subject=payload["sub"], role=payload.get("role", "user"))


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != "admin":
        raise ForbiddenError("Admin role required")
    return user
