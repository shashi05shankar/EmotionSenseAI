"""Typed-ish client for the backend API."""

from __future__ import annotations

import os

import requests

BASE = os.environ.get("ESA_API_BASE_URL", "http://localhost:8000/api/v1")


def predict(file_bytes: bytes, filename: str, model: str | None = None) -> dict:
    files = {"file": (filename, file_bytes, "audio/wav")}
    data = {"model": model} if model else {}
    r = requests.post(f"{BASE}/predict", files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()


def list_models() -> list[dict]:
    r = requests.get(f"{BASE}/models", timeout=15)
    r.raise_for_status()
    return r.json()


def version() -> dict:
    r = requests.get(f"{BASE}/version", timeout=15)
    r.raise_for_status()
    return r.json()
