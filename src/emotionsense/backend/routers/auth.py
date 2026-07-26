"""Auth endpoints: JWT login backed by bcrypt-verified credentials.

The admin account is provisioned from the environment (``ESA_ADMIN_EMAIL`` +
``ESA_ADMIN_PASSWORD_HASH``); no credentials live in the code. Login verifies the password
against the stored bcrypt hash and issues a signed JWT. User registration + persistence is
the DB-backed production path (the ``users`` table in docs/design/03-database-schema.md);
until then, the single admin is the only authenticatable account.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from emotionsense.backend.security import create_access_token
from emotionsense.backend.users import authenticate
from emotionsense.common.errors import UnauthenticatedError

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest) -> TokenResponse:
    """Exchange email + password for a JWT access token."""
    user = authenticate(body.email, body.password)
    if user is None:
        raise UnauthenticatedError("Invalid credentials")
    token = create_access_token(subject=user.email, role=user.role)
    return TokenResponse(access_token=token, role=user.role)
