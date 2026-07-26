"""LRU cache of loaded models keyed by registry ref, so hot models stay in memory."""

from __future__ import annotations

from collections import OrderedDict

from emotionsense.ml.models.base import EmotionModel


class ModelCache:
    def __init__(self, max_size: int = 3) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, EmotionModel] = OrderedDict()

    def get(self, ref: str) -> EmotionModel | None:
        if ref in self._cache:
            self._cache.move_to_end(ref)
            return self._cache[ref]
        return None

    def put(self, ref: str, model: EmotionModel) -> None:
        self._cache[ref] = model
        self._cache.move_to_end(ref)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)
