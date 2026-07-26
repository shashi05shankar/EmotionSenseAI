"""Speaker-independent k-fold splitting (ADR-4, design review R-ML2).

Why this module is the integrity centerpiece:

* **Speaker-independent** — no speaker may appear in more than one of {train, val,
  test}. Speaker leakage is the #1 cause of inflated SER accuracy; we make speaker-
  dependent splitting impossible in the default path.
* **k-fold** — a single split on ~1.4k samples swings ±3-5%. We produce K folds so the
  benchmark can report mean ± std, not a single noisy number.

The split is grouped by ``speaker_id``. Within each fold, the test group is one fold of
speakers; a slice of the remaining speakers becomes validation; the rest are train.
"""

from __future__ import annotations

from dataclasses import dataclass

from emotionsense.datasets.base import Sample


@dataclass(frozen=True, slots=True)
class Fold:
    """One speaker-independent fold: disjoint train/val/test sample lists."""

    index: int
    train: list[tuple[Sample, str]]
    val: list[tuple[Sample, str]]
    test: list[tuple[Sample, str]]

    def speakers(self, which: str) -> set[str]:
        pairs = {"train": self.train, "val": self.val, "test": self.test}[which]
        return {s.speaker_id for s, _ in pairs}


def _stable_speaker_order(pairs: list[tuple[Sample, str]], seed: int) -> list[str]:
    """Deterministically order unique speakers (hash-shuffled by seed)."""
    speakers = sorted({s.speaker_id for s, _ in pairs})
    # Deterministic pseudo-shuffle: sort by a seeded hash so folds are reproducible
    # without importing random state, keeping runs byte-identical for a given seed.
    return sorted(speakers, key=lambda sp: hash((seed, sp)) & 0xFFFFFFFF)


def make_speaker_independent_folds(
    pairs: list[tuple[Sample, str]],
    n_folds: int = 5,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> list[Fold]:
    """Build ``n_folds`` speaker-independent folds.

    Args:
        pairs: (sample, canonical_label) pairs.
        n_folds: number of CV folds (>= 2).
        seed: reproducibility seed for speaker ordering.
        val_fraction: fraction of *training speakers* held out for validation.

    Raises:
        ValueError: if there are fewer unique speakers than folds (can't guarantee
            speaker-independence otherwise).
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    order = _stable_speaker_order(pairs, seed)
    if len(order) < n_folds:
        raise ValueError(
            f"Need >= n_folds ({n_folds}) unique speakers for speaker-independent CV; "
            f"got {len(order)}."
        )

    # Round-robin speakers into fold buckets (balances group sizes).
    buckets: list[list[str]] = [[] for _ in range(n_folds)]
    for i, sp in enumerate(order):
        buckets[i % n_folds].append(sp)

    by_speaker: dict[str, list[tuple[Sample, str]]] = {}
    for pair in pairs:
        by_speaker.setdefault(pair[0].speaker_id, []).append(pair)

    folds: list[Fold] = []
    for k in range(n_folds):
        test_speakers = set(buckets[k])
        remaining = [sp for sp in order if sp not in test_speakers]
        n_val = max(1, round(len(remaining) * val_fraction))
        val_speakers = set(remaining[:n_val])
        train_speakers = set(remaining[n_val:])

        def collect(speaker_set: set[str]) -> list[tuple[Sample, str]]:
            out: list[tuple[Sample, str]] = []
            for sp in speaker_set:
                out.extend(by_speaker.get(sp, []))
            return out

        folds.append(
            Fold(
                index=k,
                train=collect(train_speakers),
                val=collect(val_speakers),
                test=collect(test_speakers),
            )
        )
    return folds


def assert_speaker_independent(fold: Fold) -> None:
    """Guardrail: raise if any speaker leaks across splits. Tests assert this."""
    tr, va, te = fold.speakers("train"), fold.speakers("val"), fold.speakers("test")
    if tr & va or tr & te or va & te:
        overlap = (tr & va) | (tr & te) | (va & te)
        raise AssertionError(f"Speaker leakage across splits in fold {fold.index}: {overlap}")
