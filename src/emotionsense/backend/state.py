"""Process-wide serving singletons (registry + inference service).

Lean Core runnable path uses the filesystem-backed LocalRegistry so the API serves without
Postgres. The production path swaps this for a DB-backed registry with the same interface.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from emotionsense.common.config import get_settings
from emotionsense.inference.service import InferenceService
from emotionsense.registry.local_registry import LocalRegistry

REGISTRY_ROOT = Path("experiments/artifacts/registry")


@lru_cache
def get_registry() -> LocalRegistry:
    return LocalRegistry(REGISTRY_ROOT)


@lru_cache
def get_inference_service() -> InferenceService:
    s = get_settings()
    return InferenceService(
        get_registry(),
        cache_size=s.model_cache_size,
        max_duration_sec=s.max_duration_sec,
    )
