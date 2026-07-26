"""Core dataset value objects and the loader Protocol.

A ``Sample`` is the atomic unit flowing through the pipeline. Crucially it carries
``speaker_id`` (for speaker-independent splitting, ADR-4) and ``statement_id`` (to flag
lexical leakage, R-ML4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Sample:
    """One labelled audio utterance."""

    path: Path
    label: str  # native corpus label (pre-harmonization)
    speaker_id: str
    dataset: str
    statement_id: str | None = None  # spoken sentence id, for leakage analysis
    extra: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class DatasetLoader(Protocol):
    """Interface every corpus loader implements.

    Implementations only *enumerate* samples + native labels; harmonization and
    splitting are separate stages so loaders stay trivial and testable.
    """

    name: str

    def load(self, root: Path) -> list[Sample]:
        """Enumerate all samples under ``root`` for this corpus."""
        ...
