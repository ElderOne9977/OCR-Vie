"""PP-OCR-vl / Multi-modal VL parser (skeleton).

Status: NOT_IMPLEMENTED until verified API installed.
No fake structure predictions.
"""

from __future__ import annotations

from app.parsers.base import DocumentParser
from app.schemas.document import DocumentParseResult
from app.services.errors import MODEL_UNAVAILABLE, PipelineError


class PaddleOcrVLParser(DocumentParser):
    name = "paddleocr_vl"

    def parse(self, text: str, *, filename: str | None = None) -> DocumentParseResult:
        raise PipelineError(
            MODEL_UNAVAILABLE,
            "Paddle OCR VL not implemented. Install verified paddleocr and wire vl parse.",
            component=self.name,
        )