"""TESS loader.

TESS files are organized as ``OAF_back_angry.wav`` / ``YAF_...`` where the prefix
(OAF/YAF) identifies the speaker (Older/Younger Adult Female) and the trailing token is
the emotion. Only two speakers exist — noted as a leakage/diversity caveat in model cards.
"""

from __future__ import annotations

from pathlib import Path

from emotionsense.datasets.base import Sample

_EMOTIONS = {"neutral", "happy", "sad", "angry", "fear", "disgust", "ps"}


class TessLoader:
    name = "tess"

    def load(self, root: Path) -> list[Sample]:
        samples: list[Sample] = []
        for wav in sorted(Path(root).rglob("*.wav")):
            stem = wav.stem.lower()
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            speaker = parts[0]  # oaf | yaf
            emotion = parts[-1]
            word = parts[1]
            if emotion not in _EMOTIONS:
                continue
            samples.append(
                Sample(
                    path=wav,
                    label=emotion,
                    speaker_id=f"tess_{speaker}",
                    dataset="tess",
                    statement_id=word,  # single-word lexical content
                )
            )
        return samples
