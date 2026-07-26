"""Integration tests for the FastAPI backend against the local registry."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    from emotionsense.backend.main import create_app

    return TestClient(create_app())


@pytest.mark.integration
def test_health(client: TestClient):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.integration
def test_version(client: TestClient):
    r = client.get("/api/v1/version")
    assert r.status_code == 200
    assert r.json()["api_version"] == "v1"


@pytest.mark.integration
def test_predict_requires_valid_audio(client: TestClient):
    r = client.post("/api/v1/predict", files={"file": ("x.txt", b"not audio", "text/plain")})
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


@pytest.mark.integration
def test_predict_returns_distribution(client: TestClient, tmp_path: Path):
    # Requires a registered default model (train_and_register). Skip if none.
    ready = client.get("/api/v1/health/ready").json()
    if ready["checks"]["default_model"] != "ok":
        pytest.skip("no default model registered in this environment")
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 240 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    r = client.post("/api/v1/predict", files={"file": ("a.wav", buf.read(), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_label"] in body["probabilities"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.05
