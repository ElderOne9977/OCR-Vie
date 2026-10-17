"""VietOCR recognizer adapter (skeleton).

Status: UNAVAILABLE / NOT_CONFIGURED until vietocr + checkpoint verified.
No fake predictions.
"""

from __future__ import annotations

from app.recognizers.base import TextRecognizer
from app.schemas.ocr import ComponentStatus, RecognitionResult
from app.services.errors import MODEL_UNAVAILABLE, PipelineError


class VietOCRRecognizer(TextRecognizer):
    name = "vietocr"

    def __init__(self, model_path: str = "") -> None:
        self._model_path = model_path
        try:
            import vietocr  # noqa: F401
        except ImportError:
            self._status: ComponentStatus = "unavailable"
            return
        self._status = "not_configured" if not model_path else "not_implemented"

    def status(self) -> ComponentStatus:
        return self._status

    def recognize(self, image, *, lang: str = "vi", psm: int = 6) -> RecognitionResult:
        raise PipelineError(
            MODEL_UNAVAILABLE,
            "VietOCR not configured. Provide checkpoint via VIETOCR_MODEL_PATH.",
            component=self.name,
        )