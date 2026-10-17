"""Document preprocessing: page-level normalization."""

from __future__ import annotations

import cv2
import numpy as np
from app.core.config import get_settings


def preprocess(image: np.ndarray) -> np.ndarray:
    """Preserve diacritics by default; binary cleanup is an explicit profile."""
    mode = get_settings().document_preprocess
    if mode == "original":
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == "gray":
        return gray
    denoised = cv2.medianBlur(gray, 3)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
