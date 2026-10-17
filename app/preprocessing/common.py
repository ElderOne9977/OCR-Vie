"""Shared image decode / validation."""

from __future__ import annotations

import cv2
import io
import numpy as np
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings


def validate_image_size(data: bytes) -> None:
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > get_settings().max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image too large")


def decode_bytes_to_bgr(data: bytes) -> np.ndarray:
    validate_image_size(data)
    settings = get_settings()
    try:
        # Read dimensions from the header before allocating decoded pixel buffers.
        with Image.open(io.BytesIO(data)) as header:
            width, height = header.size
            if width * height > settings.max_image_pixels or max(width, height) > settings.max_image_side:
                raise HTTPException(status_code=413, detail="Image dimensions exceed configured limits")
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail="Image dimensions exceed safe limits") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image") from exc
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image")
    height, width = image.shape[:2]
    if width * height > settings.max_image_pixels or max(width, height) > settings.max_image_side:
        raise HTTPException(status_code=413, detail="Image dimensions exceed configured limits")
    return image
