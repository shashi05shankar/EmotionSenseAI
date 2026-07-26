"""Canonical constants shared by every plane (training and serving).

The label taxonomy lives here so training and inference can never disagree on class
ordering or membership — a common source of silent Speech-Emotion-Recognition bugs.

Two taxonomies are defined deliberately (design review R-ML4):

* ``CANONICAL_LABELS`` — the 7-class in-corpus taxonomy used for training/eval on a
  single corpus.
* ``CROSS_CORPUS_LABELS`` — the 6-class intersection shared by RAVDESS, TESS and
  CREMA-D. Cross-corpus evaluation MUST project predictions and references to this set,
  because the corpora do not share the same label space (RAVDESS has ``calm`` and
  ``surprise``; CREMA-D has neither).
"""

from __future__ import annotations

from typing import Final

# Audio canonical form used everywhere (16 kHz mono float32).
DEFAULT_SAMPLE_RATE: Final[int] = 16_000
DEFAULT_CHANNELS: Final[int] = 1

# In-corpus 7-class taxonomy (sorted, canonical order).
CANONICAL_LABELS: Final[tuple[str, ...]] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

# 6-class intersection for cross-corpus evaluation (drops ``surprise``).
CROSS_CORPUS_LABELS: Final[tuple[str, ...]] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
)

# Accepted upload formats, validated by magic bytes at the serving edge.
ALLOWED_AUDIO_MIME: Final[frozenset[str]] = frozenset(
    {"audio/wav", "audio/x-wav", "audio/flac", "audio/mpeg", "audio/ogg"}
)

# Model version naming: "<family>-<dataset>:<version>".
MODEL_VERSION_SEP: Final[str] = ":"


def label_index(labels: tuple[str, ...]) -> dict[str, int]:
    """Return a stable label->index map for one-hot / class-id encoding."""
    return {label: i for i, label in enumerate(labels)}
