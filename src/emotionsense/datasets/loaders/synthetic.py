"""Synthetic corpus generator — makes the whole platform runnable without downloads.

Each (emotion, speaker) produces a short tone whose spectral profile is emotion-
dependent (distinct base frequency + harmonics + noise), so classical/deep models can
actually learn a separable signal. This lets CI and local dev exercise the *entire*
pipeline (features -> models -> benchmark -> serving) deterministically, with no GPU and
no multi-GB corpora. It is NOT a scientific dataset — only an execution fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from emotionsense.common.constants import CANONICAL_LABELS, DEFAULT_SAMPLE_RATE
from emotionsense.datasets.base import Sample

# Emotion -> base frequency (Hz). Distinct enough to be separable, close enough to be
# non-trivial.
_BASE_FREQ = {
    "angry": 240.0,
    "disgust": 180.0,
    "fear": 300.0,
    "happy": 330.0,
    "neutral": 200.0,
    "sad": 160.0,
    "surprise": 360.0,
}


def _synth_clip(emotion: str, speaker_seed: int, duration: float, sr: int) -> np.ndarray:
    rng = np.random.default_rng(speaker_seed)
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    f0 = _BASE_FREQ[emotion] * (1.0 + 0.02 * rng.standard_normal())  # slight speaker jitter
    signal = np.sin(2 * np.pi * f0 * t)
    # Emotion-dependent harmonic energy + speaker timbre.
    signal += 0.4 * np.sin(2 * np.pi * 2 * f0 * t)
    signal += 0.2 * np.sin(2 * np.pi * 3 * f0 * t + speaker_seed % 7)
    signal += 0.05 * rng.standard_normal(t.shape[0])  # noise
    peak = np.max(np.abs(signal)) or 1.0
    return (0.9 * signal / peak).astype(np.float32)


def generate(
    root: Path,
    speakers: int = 6,
    clips_per_class: int = 4,
    duration: float = 1.0,
    sr: int = DEFAULT_SAMPLE_RATE,
    labels: tuple[str, ...] = CANONICAL_LABELS,
) -> Path:
    """Write a synthetic corpus under ``root`` and return the path.

    Filenames encode speaker + label so :class:`SyntheticLoader` can parse them.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for spk in range(speakers):
        for label in labels:
            for c in range(clips_per_class):
                seed = spk * 1000 + hash(label) % 1000 + c
                clip = _synth_clip(label, seed, duration, sr)
                fname = f"spk{spk:02d}__{label}__{c:02d}.wav"
                sf.write(root / fname, clip, sr)
    return root


class SyntheticLoader:
    name = "synthetic"

    def load(self, root: Path) -> list[Sample]:
        samples: list[Sample] = []
        for wav in sorted(Path(root).rglob("*.wav")):
            try:
                spk, label, _idx = wav.stem.split("__")
            except ValueError:
                continue
            samples.append(
                Sample(path=wav, label=label, speaker_id=f"syn_{spk}", dataset="synthetic")
            )
        return samples
