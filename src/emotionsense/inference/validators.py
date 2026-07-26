"""Upload validation guards — reject bad input early at the serving edge (NFR-7, US-10)."""

from __future__ import annotations

from emotionsense.common.errors import (
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    ValidationError,
)

# Magic-byte signatures for accepted audio containers.
_MAGIC = {
    b"RIFF": "wav",
    b"fLaC": "flac",
    b"OggS": "ogg",
    b"ID3": "mp3",
}


def validate_bytes(data: bytes, max_bytes: int) -> str:
    """Validate size + format by magic bytes. Returns the detected format."""
    if len(data) == 0:
        raise ValidationError("Empty upload")
    if len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"Upload {len(data)} bytes exceeds limit {max_bytes}",
            details={"limit_bytes": max_bytes},
        )
    head = data[:4]
    for sig, fmt in _MAGIC.items():
        if data.startswith(sig) or head.startswith(sig):
            return fmt
    # MP3 without ID3 often starts with frame sync 0xFFFB/0xFFF3/0xFFF2.
    if head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    raise UnsupportedMediaTypeError("Unrecognized audio format (magic-byte check failed)")


def validate_duration(duration_sec: float, max_sec: float) -> None:
    if duration_sec > max_sec:
        raise ValidationError(
            f"Audio duration {duration_sec:.1f}s exceeds limit of {max_sec:.0f}s",
            details={"limit_sec": max_sec},
        )
