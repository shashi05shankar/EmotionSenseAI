"""Auth primitives: password hashing + JWT (Lean Core: JWT + 2 roles).

Deliberately thin per design review — refresh tokens and API-key scopes are deferred as
low-signal plumbing. Passwords hashed with bcrypt; tokens signed with the app secret.
"""

from __future__ import annotations

import time
from typing import Any

import bcrypt
from jose import JWTError, jwt

from emotionsense.common.config import get_settings
from emotionsense.common.errors import UnauthenticatedError

# bcrypt operates on the first 72 bytes of the password (algorithm limit); we truncate
# explicitly so long inputs hash deterministically instead of raising.
_BCRYPT_MAX_BYTES = 72


def hash_password(password: str) -> str:
    """Return a bcrypt hash for ``password``."""
    digest = bcrypt.hashpw(password.encode("utf-8")[:_BCRYPT_MAX_BYTES], bcrypt.gensalt())
    return digest.decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Constant-time verify of ``password`` against a stored bcrypt ``hashed``."""
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:_BCRYPT_MAX_BYTES], hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str, role: str) -> str:
    s = get_settings()
    now = int(time.time())
    payload = {
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": now + s.jwt_expire_minutes * 60,
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any]:
    s = get_settings()
    try:
        return jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    except JWTError as exc:
        raise UnauthenticatedError("Invalid or expired token") from exc
