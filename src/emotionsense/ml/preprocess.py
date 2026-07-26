"""Audio preprocessing shared by train and serve (ADR-3).

Canonical form: 16 kHz, mono, float32, silence-trimmed, peak-normalized. Handles
variable-length audio by windowing (design review R-A7): long clips are split into
fixed windows whose per-window predictions are aggregated downstream.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from emotionsense.common.constants import DEFAULT_SAMPLE_RATE


def load_audio(path: str | Path, sr: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Load audio as mono float32 at the target sample rate."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def preprocess(
    y: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    trim_db: float = 30.0,
    normalize: bool = True,
) -> np.ndarray:
    """Trim leading/trailing silence and peak-normalize."""
    if y.size == 0:
        return y
    y, _ = librosa.effects.trim(y, top_db=trim_db)
    if normalize:
        peak = float(np.max(np.abs(y))) or 1.0
        y = y / peak
    return y.astype(np.float32)


def fix_length(y: np.ndarray, sr: int, duration_sec: float) -> np.ndarray:
    """Pad or center-crop to exactly ``duration_sec`` (for fixed-input models)."""
    target = int(sr * duration_sec)
    if len(y) == target:
        return y
    if len(y) < target:
        return np.pad(y, (0, target - len(y))).astype(np.float32)
    start = (len(y) - target) // 2
    return y[start : start + target].astype(np.float32)


def window(
    y: np.ndarray, sr: int, window_sec: float, hop_sec: float | None = None
) -> list[np.ndarray]:
    """Split variable-length audio into fixed windows (R-A7).

    Returns at least one window (padded) even for short clips, so inference always has
    something to score.
    """
    win = int(sr * window_sec)
    hop = int(sr * (hop_sec if hop_sec is not None else window_sec))
    if len(y) <= win:
        return [fix_length(y, sr, window_sec)]
    windows: list[np.ndarray] = []
    for start in range(0, len(y) - win + 1, hop):
        windows.append(y[start : start + win].astype(np.float32))
    return windows
