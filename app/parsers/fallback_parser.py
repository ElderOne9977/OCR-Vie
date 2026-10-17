"""Fallback parser: plain line list. No structure claimed."""

from __future__ import annotations

from app.parsers.base import DocumentParser
from app.schemas.document import (
    DocumentBlock,
    DocumentPage,
    DocumentParseResult,
)


class FallbackDocumentParser(DocumentParser):
    name = "fallback-parser"

    def parse(self, text: str, *, filename: str | None = None) -> DocumentParseResult:
        pages: list[DocumentPage] = [
            DocumentPage(
                page_number=1,
                width=0,
                height=0,
                blocks=[
                    DocumentBlock(
                        id=f"line-{i}",
                        type="line",
                        bbox=None,
                        raw_text=line,
                        normalized_text=line,
                        ocr_confidence=None,
                        source_engine="tesseract",
                    )
                    for i, line in enumerate(text.split("\n"))
                    if line.strip()
                ],
            ),
        ]
        fields: list[dict] = []
        return DocumentParseResult(
            document_type="unknown",
            pages=pages,
            tables=[],
            fields=fields,
            reading_order=[block.id for block in pages[0].blocks],
            parser_engine=self.name,
        )
