"""RAVDESS loader.

Filename convention (speech files): ``03-01-06-01-02-01-12.wav`` where field 3 is the
emotion and field 7 is the actor (speaker). We use SPEECH only (modality 03, channel 01)
and exclude song, per design review (avoids mixing distributions).
"""

from __future__ import annotations

from pathlib import Path

from emotionsense.datasets.base import Sample

_EMOTION = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


class RavdessLoader:
    name = "ravdess"

    def load(self, root: Path) -> list[Sample]:
        samples: list[Sample] = []
        for wav in sorted(Path(root).rglob("*.wav")):
            parts = wav.stem.split("-")
            if len(parts) != 7:
                continue
            modality, _vocal, emotion, _intensity, statement, _rep, actor = parts
            if modality != "03":  # speech only, exclude song (02)
                continue
            label = _EMOTION.get(emotion)
            if label is None:
                continue
            samples.append(
                Sample(
                    path=wav,
                    label=label,
                    speaker_id=f"ravdess_{actor}",
                    dataset="ravdess",
                    statement_id=statement,  # only 2 statements -> lexical leakage flag
                )
            )
        return samples
