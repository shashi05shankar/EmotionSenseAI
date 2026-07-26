"""Model registry: the serving source of truth (ADR-2)."""

from emotionsense.registry.artifact import load_model_verified, save_model_checksummed
from emotionsense.registry.local_registry import LocalRegistry, ModelRecord

__all__ = ["LocalRegistry", "ModelRecord", "load_model_verified", "save_model_checksummed"]
