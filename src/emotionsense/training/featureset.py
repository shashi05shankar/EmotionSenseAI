"""Build (X, y) feature matrices from (Sample, label) pairs, with on-disk caching.

Features are cached by a hash of (audio path, mtime, feature spec) so re-runs are fast and
reproducible. Training and inference use the SAME extractor built from the SAME spec,
guaranteeing parity (ADR-3).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from emotionsense.common.schemas import FeatureSpec
from emotionsense.datasets.base import Sample
from emotionsense.ml.features import build_extractor
from emotionsense.ml.preprocess import fix_length, load_audio, preprocess


def _cache_key(sample: Sample, spec: FeatureSpec) -> str:
    try:
        mtime = sample.path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    raw = json.dumps(
        {"path": str(sample.path), "mtime": mtime, "spec": spec.model_dump()}, sort_keys=True
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_matrix(
    pairs: list[tuple[Sample, str]],
    spec: FeatureSpec,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return (X, y) where X is (n_samples, n_features) and y is canonical labels."""
    extractor = build_extractor(spec)
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    feats: list[np.ndarray] = []
    labels: list[str] = []
    for sample, label in pairs:
        vec: np.ndarray | None = None
        cache_path = cache_dir / f"{_cache_key(sample, spec)}.npy" if cache_dir else None
        if cache_path and cache_path.exists():
            vec = np.load(cache_path)
        if vec is None:
            y = load_audio(sample.path, sr=spec.sample_rate)
            y = preprocess(y, sr=spec.sample_rate)
            y = fix_length(y, spec.sample_rate, spec.max_duration_sec)
            vec = extractor.extract(y)
            if cache_path is not None:
                np.save(cache_path, vec)
        feats.append(vec)
        labels.append(label)
    return np.vstack(feats), labels
