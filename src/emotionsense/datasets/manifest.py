"""Read/write split manifests (JSONL).

Splits are persisted as manifest files (path + label + speaker + split) rather than DB
rows — the DB stores only a pointer + seed (see docs/design/03-database-schema.md §3.4).
This keeps the DB light while preserving reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path

from emotionsense.datasets.base import Sample
from emotionsense.datasets.split import Fold


def write_fold(fold: Fold, out_dir: Path, dataset: str, seed: int) -> Path:
    """Write one fold to ``<out_dir>/<dataset>_fold<k>.jsonl`` and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}_fold{fold.index}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for split_name in ("train", "val", "test"):
            for sample, label in getattr(fold, split_name):
                fh.write(
                    json.dumps(
                        {
                            "path": str(sample.path),
                            "label": label,
                            "native_label": sample.label,
                            "speaker_id": sample.speaker_id,
                            "dataset": sample.dataset,
                            "statement_id": sample.statement_id,
                            "split": split_name,
                            "fold": fold.index,
                            "seed": seed,
                        }
                    )
                    + "\n"
                )
    return path


def read_manifest(path: Path) -> list[dict]:
    """Read a manifest JSONL into a list of row dicts."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rows_to_pairs(rows: list[dict], split: str) -> list[tuple[Sample, str]]:
    """Reconstruct (Sample, canonical_label) pairs for a given split from manifest rows."""
    out: list[tuple[Sample, str]] = []
    for r in rows:
        if r["split"] != split:
            continue
        out.append(
            (
                Sample(
                    path=Path(r["path"]),
                    label=r["native_label"],
                    speaker_id=r["speaker_id"],
                    dataset=r["dataset"],
                    statement_id=r.get("statement_id"),
                ),
                r["label"],
            )
        )
    return out
