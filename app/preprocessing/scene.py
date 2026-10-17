"""Scene preprocessing: minimal, preserve details."""

from __future__ import annotations

import cv2
import numpy as np


def preprocess(image: np.ndarray) -> np.ndarray:
    """Light gray -> denoise; no hard binary step. Tesseract nhan grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    return denoised