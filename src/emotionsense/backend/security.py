"""Auth primitives: password hashing + JWT (Lean Core: JWT + 2 roles).

Deliberately thin per design review — refresh tokens and API-key scopes are deferred as
low-signal plumbing. Passwords hashed with bcrypt; tokens signed with the app secret.
"""

from __future__ import annotations

import time
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from emotionsense.common.config import get_settings
from emotionsense.common.errors import UnauthenticatedError

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return _pwd.verify(password, hashed)


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
