"""Filesystem-backed model registry for local/dev serving.

Mirrors the ``model_versions`` table (docs/design/03-database-schema.md) as a JSON index +
artifacts directory, so the inference service and API can run WITHOUT Postgres during local
dev. In production the same interface is backed by the DB (see backend/repositories). Keeps
the serving path identical whether registry state lives in a file or a database.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from emotionsense.common.constants import MODEL_VERSION_SEP
from emotionsense.common.errors import NotFoundError


@dataclass(slots=True)
class ModelRecord:
    name: str
    version: str
    family: str
    artifact_path: str
    sha256: str
    feature_spec: dict
    label_classes: list[str]
    status: str = "staging"  # staging | production | archived
    is_default: bool = False
    headline_metric: float = 0.0
    metrics: dict = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.name}{MODEL_VERSION_SEP}{self.version}"


class LocalRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        self._records: dict[str, ModelRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.index_path.exists():
            data = json.loads(self.index_path.read_text())
            self._records = {k: ModelRecord(**v) for k, v in data.items()}

    def _save(self) -> None:
        self.index_path.write_text(
            json.dumps({k: asdict(v) for k, v in self._records.items()}, indent=2)
        )

    def register(self, record: ModelRecord) -> ModelRecord:
        self._records[record.ref] = record
        self._save()
        return record

    def get(self, ref: str) -> ModelRecord:
        if ref not in self._records:
            raise NotFoundError(f"Model not in registry: {ref}")
        return self._records[ref]

    def list(self, status: str | None = None) -> list[ModelRecord]:
        recs = list(self._records.values())
        if status:
            recs = [r for r in recs if r.status == status]
        return sorted(recs, key=lambda r: r.headline_metric, reverse=True)

    def default(self) -> ModelRecord:
        for r in self._records.values():
            if r.is_default and r.status == "production":
                return r
        raise NotFoundError("No production default model registered")

    def promote(self, ref: str, make_default: bool = True) -> ModelRecord:
        rec = self.get(ref)
        rec.status = "production"
        if make_default:
            for other in self._records.values():
                if other.name == rec.name:
                    other.is_default = False
            rec.is_default = True
        self._save()
        return rec

    def retire(self, ref: str) -> ModelRecord:
        rec = self.get(ref)
        rec.status = "archived"
        rec.is_default = False
        self._save()
        return rec
