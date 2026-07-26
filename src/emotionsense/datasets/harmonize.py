"""Label harmonization: native corpus labels -> canonical taxonomy.

Design review R-ML4: corpora do NOT share a label space. This module maps each corpus's
native labels to ``CANONICAL_LABELS`` (7-class), dropping labels a corpus doesn't share
(e.g. RAVDESS ``calm``). For cross-corpus evaluation, callers additionally project to
``CROSS_CORPUS_LABELS`` (6-class) via :func:`project_to_cross_corpus`.

The maps mirror ``configs/datasets/*.yaml`` (single source of truth is the config; these
defaults exist so the pipeline works without external files in tests).
"""

from __future__ import annotations

from emotionsense.common.constants import CANONICAL_LABELS, CROSS_CORPUS_LABELS
from emotionsense.datasets.base import Sample

# Native -> canonical. A value of None means "drop this sample" (label not in taxonomy).
DEFAULT_LABEL_MAPS: dict[str, dict[str, str | None]] = {
    "ravdess": {
        "neutral": "neutral",
        "calm": None,  # not in canonical taxonomy -> dropped
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fearful": "fear",
        "disgust": "disgust",
        "surprised": "surprise",
    },
    "tess": {
        "neutral": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fear",
        "disgust": "disgust",
        "ps": "surprise",  # "pleasant surprise"
    },
    "crema_d": {
        "NEU": "neutral",
        "HAP": "happy",
        "SAD": "sad",
        "ANG": "angry",
        "FEA": "fear",
        "DIS": "disgust",
        # CREMA-D has no surprise/calm -> 6-class corpus
    },
    # Synthetic fixture: labels are already canonical (identity map).
    "synthetic": {lab: lab for lab in CANONICAL_LABELS},
}


def harmonize(
    samples: list[Sample], label_maps: dict[str, dict[str, str | None]] | None = None
) -> list[tuple[Sample, str]]:
    """Map each sample's native label to canonical; drop unmapped/None labels.

    Returns (sample, canonical_label) pairs. Samples whose native label maps to None or
    is missing from the map are excluded (with the count discoverable by the caller via
    len difference).
    """
    maps = label_maps or DEFAULT_LABEL_MAPS
    out: list[tuple[Sample, str]] = []
    canonical = set(CANONICAL_LABELS)
    for s in samples:
        mapping = maps.get(s.dataset, {})
        canon = mapping.get(s.label)
        if canon is None or canon not in canonical:
            continue
        out.append((s, canon))
    return out


def project_to_cross_corpus(
    pairs: list[tuple[Sample, str]],
) -> list[tuple[Sample, str]]:
    """Keep only samples whose canonical label is in the 6-class intersection (R-ML4).

    Used before cross-corpus train/eval so both corpora share an identical label space.
    """
    keep = set(CROSS_CORPUS_LABELS)
    return [(s, lab) for (s, lab) in pairs if lab in keep]
