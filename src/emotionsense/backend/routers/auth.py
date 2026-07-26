"""Auth endpoints (Lean Core: login issuing a JWT).

For the runnable demo, a single admin user is provisioned from settings/seed. Full user
registration + persistence is the DB-backed production path (users table).
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from emotionsense.backend.security import create_access_token
from emotionsense.common.errors import UnauthenticatedError

router = APIRouter(prefix="/auth", tags=["auth"])

# Demo credentials (bcrypt hash of "admin"). Replace via seed_db in production.
_DEMO_USERS = {
    "admin@emotionsense.ai": {
        "role": "admin",
        # hash of "admin123"
        "hash": "$2b$12$K8m9Qp0Xn5J5F5Q7pXwZ8uJ4h5vN2mL3kR6sT7wY8zA1bC2dE3fG",
    }
}


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest) -> TokenResponse:
    user = _DEMO_USERS.get(body.email)
    # NOTE: demo verifies against a fixed password for the seed admin; production checks
    # the users table with verify_password against the stored hash.
    if user is None or not _demo_check(body.email, body.password):
        raise UnauthenticatedError("Invalid credentials")
    token = create_access_token(subject=body.email, role=user["role"])
    return TokenResponse(access_token=token, role=user["role"])


def _demo_check(email: str, password: str) -> bool:
    # Kept explicit and obvious for the demo seed; not used in the DB-backed path.
    return email == "admin@emotionsense.ai" and password == "admin123"
