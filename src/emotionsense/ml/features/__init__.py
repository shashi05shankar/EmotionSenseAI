"""Feature extractors: spectral (MFCC/mel/chroma) and SSL embeddings."""

from emotionsense.ml.features.base import FeatureExtractor, build_extractor
from emotionsense.ml.features.spectral import SpectralExtractor

__all__ = ["FeatureExtractor", "SpectralExtractor", "build_extractor"]
