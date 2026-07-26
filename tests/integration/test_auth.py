"""Auth flow tests: bcrypt-verified JWT login + admin route protection.

No credentials are hardcoded anywhere — the test provisions the admin hash via the same
environment settings the app reads in production.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from emotionsense.backend.security import hash_password

ADMIN_EMAIL = "admin@emotionsense.ai"
ADMIN_PASSWORD = "s3cret-test-pw"


@pytest.fixture
def client(monkeypatch) -> TestClient:
    from emotionsense.common import config

    monkeypatch.setenv("ESA_ADMIN_EMAIL", ADMIN_EMAIL)
    monkeypatch.setenv("ESA_ADMIN_PASSWORD_HASH", hash_password(ADMIN_PASSWORD))
    monkeypatch.setenv("ESA_JWT_SECRET", "test-secret")
    config.get_settings.cache_clear()  # settings are lru_cached

    from emotionsense.backend.main import create_app

    c = TestClient(create_app())
    yield c
    config.get_settings.cache_clear()


@pytest.mark.integration
def test_login_success_returns_jwt(client: TestClient):
    r = client.post("/api/v1/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["role"] == "admin"
    assert body["access_token"]
    assert "refresh_token" not in body  # refresh deferred by design


@pytest.mark.integration
def test_login_wrong_password_401(client: TestClient):
    r = client.post("/api/v1/auth/login", json={"email": ADMIN_EMAIL, "password": "wrong"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "UNAUTHENTICATED"


@pytest.mark.integration
def test_login_unknown_email_401(client: TestClient):
    r = client.post(
        "/api/v1/auth/login", json={"email": "nobody@x.com", "password": ADMIN_PASSWORD}
    )
    assert r.status_code == 401


@pytest.mark.integration
def test_admin_route_requires_token(client: TestClient):
    # No token -> 401
    r = client.post("/api/v1/admin/models/foo/v1/retire")
    assert r.status_code == 401


@pytest.mark.integration
def test_admin_route_accepts_valid_token(client: TestClient):
    token = client.post(
        "/api/v1/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    ).json()["access_token"]
    # Valid token, but model does not exist -> 404 (proves auth passed, not 401/403)
    r = client.post(
        "/api/v1/admin/models/does-not-exist/v1/retire",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 404
