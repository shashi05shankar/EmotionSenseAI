"""Spectral features (MFCC / mel / chroma) with mean+std temporal pooling.

Pooling frame-level features to (mean, std) yields a fixed-length vector independent of
clip duration — the standard, robust representation for classical SER models.
"""

from __future__ import annotations

import librosa
import numpy as np

from emotionsense.common.schemas import FeatureSpec


class SpectralExtractor:
    def __init__(self, spec: FeatureSpec) -> None:
        self.spec = spec

    def _frames(self, y: np.ndarray) -> np.ndarray:
        s = self.spec
        if s.extractor == "mfcc":
            return librosa.feature.mfcc(
                y=y, sr=s.sample_rate, n_mfcc=s.n_mfcc, n_fft=s.n_fft, hop_length=s.hop_length
            )
        if s.extractor == "mel":
            mel = librosa.feature.melspectrogram(
                y=y, sr=s.sample_rate, n_mels=s.n_mels, n_fft=s.n_fft, hop_length=s.hop_length
            )
            return librosa.power_to_db(mel)
        if s.extractor == "chroma":
            return librosa.feature.chroma_stft(
                y=y, sr=s.sample_rate, n_fft=s.n_fft, hop_length=s.hop_length
            )
        raise ValueError(f"SpectralExtractor cannot handle: {s.extractor}")

    def extract(self, y: np.ndarray) -> np.ndarray:
        frames = self._frames(y)  # (n_features, n_frames)
        if frames.shape[1] == 0:
            frames = np.zeros((frames.shape[0], 1), dtype=np.float32)
        if self.spec.aggregate == "mean":
            vec = frames.mean(axis=1)
        else:  # mean_std (default)
            vec = np.concatenate([frames.mean(axis=1), frames.std(axis=1)])
        return vec.astype(np.float32)
