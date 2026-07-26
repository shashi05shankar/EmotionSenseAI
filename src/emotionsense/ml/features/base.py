"""Feature extractor interface + factory.

A ``FeatureExtractor`` maps preprocessed audio to a fixed-length vector (for classical/
deep heads) driven entirely by a :class:`FeatureSpec`, so the same spec reproduces the
same features at train and serve time (ADR-3).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from emotionsense.common.schemas import FeatureSpec


@runtime_checkable
class FeatureExtractor(Protocol):
    """Turns a 1-D waveform into a feature vector."""

    spec: FeatureSpec

    def extract(self, y: np.ndarray) -> np.ndarray:
        """Return a 1-D feature vector for one clip."""
        ...


def build_extractor(spec: FeatureSpec) -> FeatureExtractor:
    """Instantiate the extractor named by ``spec.extractor``."""
    if spec.extractor in {"mfcc", "mel", "chroma"}:
        from emotionsense.ml.features.spectral import SpectralExtractor

        return SpectralExtractor(spec)
    if spec.extractor == "ssl":
        from emotionsense.ml.features.ssl import SSLExtractor

        return SSLExtractor(spec)
    raise ValueError(f"Unknown feature extractor: {spec.extractor}")
