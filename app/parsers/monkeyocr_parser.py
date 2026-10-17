"""MonkeyOCR-style parser (skeleton).

Status: UNAVAILABLE / NOT_CONFIGURED.
"""

from __future__ import annotations

from app.parsers.base import DocumentParser
from app.schemas.document import DocumentParseResult
from app.services.errors import MODEL_UNAVAILABLE, PipelineError


class MonkeyOCROParser(DocumentParser):
    name = "monkeyocr"

    def parse(self, text: str, *, filename: str | None = None) -> DocumentParseResult:
        raise PipelineError(
            MODEL_UNAVAILABLE,
            "MonkeyOCR not configured.",
            component=self.name,
        )