"""Per-corpus loaders. Each enumerates samples + native labels only."""

from emotionsense.datasets.loaders.crema_d import CremaDLoader
from emotionsense.datasets.loaders.ravdess import RavdessLoader
from emotionsense.datasets.loaders.synthetic import SyntheticLoader
from emotionsense.datasets.loaders.tess import TessLoader

LOADERS = {
    "ravdess": RavdessLoader(),
    "tess": TessLoader(),
    "crema_d": CremaDLoader(),
    "synthetic": SyntheticLoader(),
}

__all__ = ["LOADERS", "CremaDLoader", "RavdessLoader", "SyntheticLoader", "TessLoader"]
