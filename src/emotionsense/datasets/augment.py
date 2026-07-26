"""Audio augmentation — TRAIN ONLY (design review: never augment val/test).

The harness passes ``training=True`` only for the train split. :func:`augment` refuses to
run otherwise, making the classic "augmented the test set" leak structurally impossible.
"""

from __future__ import annotations

import numpy as np


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_power = np.mean(x**2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), x.shape).astype(x.dtype)
    return x + noise


def pitch_shift_cheap(x: np.ndarray, steps: float) -> np.ndarray:
    """Resample-based cheap pitch shift (no librosa dependency at augment time)."""
    if steps == 0:
        return x
    factor = 2 ** (steps / 12.0)
    idx = np.round(np.arange(0, len(x), factor)).astype(int)
    idx = idx[idx < len(x)]
    shifted = x[idx]
    # pad/truncate back to original length
    if len(shifted) < len(x):
        shifted = np.pad(shifted, (0, len(x) - len(shifted)))
    return shifted[: len(x)].astype(x.dtype)


def time_shift(x: np.ndarray, frac: float) -> np.ndarray:
    n = int(len(x) * frac)
    return np.roll(x, n)


def augment(
    x: np.ndarray,
    *,
    training: bool,
    seed: int,
    noise_snr_db: float | None = 20.0,
    pitch_steps: float = 0.0,
    shift_frac: float = 0.0,
) -> np.ndarray:
    """Apply augmentation. Raises if ``training`` is False (guardrail)."""
    if not training:
        raise ValueError("augment() called on non-training data — augmentation leak blocked")
    rng = np.random.default_rng(seed)
    out = x
    if noise_snr_db is not None:
        out = add_noise(out, noise_snr_db, rng)
    if pitch_steps:
        out = pitch_shift_cheap(out, pitch_steps)
    if shift_frac:
        out = time_shift(out, shift_frac)
    return out
