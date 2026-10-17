"""Tesseract OCR service.

pytesseract is only a Python wrapper that shells out to the Tesseract binary.
The actual OCR engine is Tesseract (installed inside the Docker image).
"""

from __future__ import annotations

from typing import Any
import os

import numpy as np
import pytesseract
from app.core.config import get_settings
from app.services.errors import OCR_TIMEOUT, PipelineError
from app.services.deadline import remaining_seconds

# ponytail: Windows installer không thêm PATH. Hỗ trợ TESSERACT_CMD, nâng cấp khi cần auto-detect registry.
if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
elif os.name == "nt":
    _win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(_win):
        pytesseract.pytesseract.tesseract_cmd = _win

# Page Segmentation Modes supported by Tesseract (0-13, minus reserved 2).
VALID_PSM_MODES: frozenset[int] = frozenset(
    {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
)

DEFAULT_LANG = "vie+eng"
DEFAULT_PSM = 6


def validate_psm(psm: int) -> int:
    """Return psm if valid, otherwise raise ValueError."""
    if psm not in VALID_PSM_MODES:
        allowed = ", ".join(str(mode) for mode in sorted(VALID_PSM_MODES))
        raise ValueError(f"Invalid PSM '{psm}'. Allowed values: {allowed}")
    return psm


def _build_config(psm: int) -> str:
    """Build the Tesseract config string."""
    return f"--oem 3 --psm {validate_psm(psm)}"


def run_tesseract(operation, source, **kwargs):
    """Bound every OCR invocation, including legacy and native image-list input."""
    try:
        if "lang" in kwargs:
            kwargs["lang"] = normalize_tesseract_lang(kwargs["lang"])
        timeout = remaining_seconds(limit=get_settings().ocr_timeout_seconds)
        result = operation(source, timeout=timeout, **kwargs)
        remaining_seconds()
        return result
    except RuntimeError as exc:
        if "timeout" in str(exc).lower():
            raise PipelineError(OCR_TIMEOUT, "Tesseract recognition timed out", component="tesseract") from exc
        raise


def normalize_tesseract_lang(lang: str) -> str:
    aliases = {"vi": "vie", "vietnamese": "vie", "en": "eng", "english": "eng"}
    return "+".join(aliases.get(part.strip().lower(), part.strip())
                    for part in lang.replace(",", "+").split("+"))


def ocr_image(
    image: np.ndarray,
    lang: str = DEFAULT_LANG,
    psm: int = DEFAULT_PSM,
) -> str:
    """Run OCR on an image and return stripped text."""
    config = _build_config(psm)
    text = run_tesseract(pytesseract.image_to_string, image, lang=lang, config=config)
    return text.strip()


def ocr_image_data(
    image: np.ndarray,
    lang: str = DEFAULT_LANG,
    psm: int = DEFAULT_PSM,
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """Run OCR and return per-word data (text, confidence, bounding box)."""
    config = _build_config(psm)
    data = run_tesseract(pytesseract.image_to_data,
        image,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    items: list[dict[str, Any]] = []
    for i, raw_text in enumerate(data.get("text", [])):
        text = raw_text.strip()
        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, KeyError, IndexError):
            continue

        # Skip invalid/low-confidence detections.
        if confidence < 0 or confidence < min_confidence:
            continue

        items.append(
            {
                "text": text,
                "confidence": round(confidence, 2),
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            }
        )
    return items
