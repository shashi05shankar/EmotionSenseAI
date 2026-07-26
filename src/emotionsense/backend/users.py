"""User authentication provider (env-seeded, bcrypt-verified).

The admin account is provisioned entirely from settings (``admin_email`` +
``admin_password_hash``) — there are no credentials in the source. If no password hash is
configured, authentication always fails (secure by default). This is the same interface a
DB-backed ``users`` table would satisfy; only the source of the record differs.
"""

from __future__ import annotations

from dataclasses import dataclass

from emotionsense.backend.security import verify_password
from emotionsense.common.config import get_settings


@dataclass(frozen=True, slots=True)
class User:
    email: str
    role: str


def authenticate(email: str, password: str) -> User | None:
    """Return the authenticated user, or None on any failure.

    Uses a constant-time bcrypt comparison via ``verify_password``. Runs the hash check
    even for an unknown email is not necessary here (single configured admin), but we
    avoid leaking which factor failed by returning a uniform None.
    """
    s = get_settings()
    if not s.admin_password_hash:
        return None
    if email != s.admin_email:
        return None
    if not verify_password(password, s.admin_password_hash):
        return None
    return User(email=email, role="admin")
