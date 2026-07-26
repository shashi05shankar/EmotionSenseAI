"""CREMA-D loader.

Filenames like ``1001_DFA_ANG_XX.wav``: field 0 is the actor id (speaker), field 1 is
the sentence code, field 2 is the emotion. CREMA-D has 6 emotions (no surprise/calm),
91 diverse speakers — which is exactly why it's our held-out cross-corpus test set.
"""

from __future__ import annotations

from pathlib import Path

from emotionsense.datasets.base import Sample

_EMOTIONS = {"ANG", "DIS", "FEA", "HAP", "NEU", "SAD"}


class CremaDLoader:
    name = "crema_d"

    def load(self, root: Path) -> list[Sample]:
        samples: list[Sample] = []
        for wav in sorted(Path(root).rglob("*.wav")):
            parts = wav.stem.split("_")
            if len(parts) < 4:
                continue
            actor, sentence, emotion = parts[0], parts[1], parts[2]
            if emotion not in _EMOTIONS:
                continue
            samples.append(
                Sample(
                    path=wav,
                    label=emotion,
                    speaker_id=f"crema_{actor}",
                    dataset="crema_d",
                    statement_id=sentence,
                )
            )
        return samples
