"""Prediction endpoints (FR-S1)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from emotionsense.backend.state import get_inference_service
from emotionsense.common.config import get_settings
from emotionsense.common.schemas import Prediction
from emotionsense.inference.validators import validate_bytes

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=Prediction)
async def predict(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
) -> Prediction:
    s = get_settings()
    data = await file.read()
    validate_bytes(data, max_bytes=s.max_upload_mb * 1024 * 1024)

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return get_inference_service().predict_path(tmp_path, model_ref=model)
    finally:
        tmp_path.unlink(missing_ok=True)
