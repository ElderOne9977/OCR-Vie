"""Image preprocessing for Tesseract OCR.

Legacy compat: forwards to preprocessing/common.py + scene/document split.
ponytail: keep shim for /ocr legacy; new code uses preprocessing/scene.py|document.py."""

from __future__ import annotations

import cv2
import numpy as np
from app.preprocessing.common import decode_bytes_to_bgr


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes into a BGR OpenCV array."""
    return decode_bytes_to_bgr(image_bytes)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode, grayscale, denoise lightly, then Otsu-threshold.

    Returns a binary image suitable for Tesseract.
    Keep processing mild so small Vietnamese diacritics stay intact.
    """
    image = decode_image(image_bytes)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Light denoise: preserves edges better than a strong blur.
    denoised = cv2.medianBlur(gray, 3)

    _, binary = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary
