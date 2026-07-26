"""Turn a dataset config into harmonized (Sample, canonical_label) pairs.

Bridges loaders + harmonization. For the synthetic fixture it generates audio on demand,
so the whole platform runs with zero downloads.
"""

from __future__ import annotations

from pathlib import Path

from emotionsense.datasets.base import Sample
from emotionsense.datasets.harmonize import DEFAULT_LABEL_MAPS, harmonize
from emotionsense.datasets.loaders import LOADERS


def build_pairs(dataset_cfg: dict, ensure_synthetic: bool = True) -> list[tuple[Sample, str]]:
    """Load + harmonize one corpus into canonical pairs."""
    name = dataset_cfg["name"]
    loader = LOADERS[dataset_cfg.get("loader", name)]
    root = Path(dataset_cfg["root"])

    if name == "synthetic" and ensure_synthetic and not any(root.rglob("*.wav")):
        from emotionsense.datasets.loaders.synthetic import generate

        generate(root)

    samples = loader.load(root)
    label_maps = (
        {name: dataset_cfg["label_map"]} if "label_map" in dataset_cfg else DEFAULT_LABEL_MAPS
    )
    return harmonize(samples, label_maps)
