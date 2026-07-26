"""Serving-time inference plane (CPU): validate -> preprocess -> features -> predict."""

from emotionsense.inference.service import InferenceService

__all__ = ["InferenceService"]
