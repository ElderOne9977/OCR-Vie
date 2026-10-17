"""Tesseract recognizer adapter. Wraps pytesseract (binary engine)."""

from __future__ import annotations

import os
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import cv2
from PIL import Image

import numpy as np
import pytesseract

from app.recognizers.base import TextRecognizer
from app.schemas.ocr import ComponentStatus, RecognitionResult
from app.services.normalization import normalize_text
from app.core.config import get_settings
from app.services.errors import RECOGNITION_FAILED, PipelineError
from app.ocr_service import run_tesseract, normalize_tesseract_lang

# ponytail: Windows installer không thêm PATH. Hỗ trợ TESSERACT_CMD, nâng cấp khi cần auto-detect registry.
if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
elif os.name == "nt":
    _win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(_win):
        pytesseract.pytesseract.tesseract_cmd = _win

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
    return f"--oem 3 --psm {validate_psm(psm)}"


def _recognition_image(image: np.ndarray, psm: int) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Tight detector crops touch glyph edges. A tiny blank border prevents
    # Tesseract line segmentation dropping Vietnamese dots; no glyph rescale.
    if psm == 7:
        image = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return image


class TesseractRecognizer(TextRecognizer):
    name = "tesseract"

    def status(self, lang: str | None = None) -> ComponentStatus:
        try:
            pytesseract.get_tesseract_version(cached=True)
            if lang and not set(normalize_tesseract_lang(lang).split('+')) <= set(pytesseract.get_languages(cached=True)):
                return 'unavailable'
        except Exception:  # noqa: BLE001
            return "unavailable"
        return "ready"

    def recognize(
        self,
        image: np.ndarray,
        *,
        lang: str = DEFAULT_LANG,
        psm: int = DEFAULT_PSM,
    ) -> RecognitionResult:
        lang = normalize_tesseract_lang(lang)
        image = _recognition_image(image, psm)
        return self._result(self._data(image, lang=lang, psm=psm), lang)

    def _data(self, source, *, lang: str, psm: int) -> dict:
        return run_tesseract(pytesseract.image_to_data,
            source, lang=lang, config=_build_config(psm), output_type=pytesseract.Output.DICT,
        )

    def recognize_many(self, images: list[np.ndarray], *, lang: str = DEFAULT_LANG,
                       psm: int = 7) -> list[RecognitionResult]:
        """Use native image-list input; TSV page numbers preserve blank crops."""
        lang = normalize_tesseract_lang(lang)
        results = []
        size = get_settings().tesseract_batch_size
        for start in range(0, len(images), size):
            batch = images[start:start + size]
            with TemporaryDirectory(prefix="ocr_batch_") as directory:
                paths = []
                for index, image in enumerate(batch):
                    image = _recognition_image(image, psm)
                    path = Path(directory) / f"{index}.png"
                    Image.fromarray(image).save(path)
                    paths.append(str(path))
                source = Path(directory) / "images.txt"
                source.write_text("\n".join(paths) + "\n", encoding="utf-8")
                data = self._data(str(source), lang=lang, psm=psm)
                pages = data.get("page_num", [])
                if len(pages) != len(data.get("text", [])) or set(pages) != set(range(1, len(batch) + 1)):
                    raise PipelineError(RECOGNITION_FAILED, "Tesseract batch page count mismatch", component=self.name)
                for page in range(1, len(batch) + 1):
                    indices = [i for i, number in enumerate(pages) if number == page]
                    selected = {key: [values[i] for i in indices] for key, values in data.items()}
                    results.append(self._result(selected, lang))
        return results

    def _result(self, data: dict, lang: str) -> RecognitionResult:
        lines: dict[tuple, list[str]] = {}
        scores: list[float] = []
        zeros = [0] * len(data.get("text", []))
        columns = [data.get(k, zeros) for k in ("page_num", "block_num", "par_num", "line_num")]
        for i, word in enumerate(data.get("text", [])):
            word = str(word).strip()
            if not word:
                continue
            key = tuple(column[i] for column in columns)
            lines.setdefault(key, []).append(word)
            try:
                score = float(data["conf"][i])
                if math.isfinite(score) and 0 <= score <= 100:
                    scores.append(score)
            except (ValueError, TypeError, KeyError, IndexError):
                pass
        raw_text = "\n".join(" ".join(words) for words in lines.values())
        normalized, _info = normalize_text(raw_text)
        confidence = sum(scores) / len(scores) if scores else None
        return RecognitionResult(
            raw_text=raw_text,
            normalized_text=normalized,
            ocr_confidence=round(confidence / 100, 4) if confidence is not None else None,
            engine="tesseract",
            model=lang,
            raw_confidence=confidence,
        )
