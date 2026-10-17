"""Internal document schema. Parser vendor map vào đây."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.schemas.ocr import BBox, FieldResult


@dataclass
class DocumentBlock:
    id: str
    type: str
    bbox: BBox | None
    raw_text: str
    normalized_text: str
    ocr_confidence: float | None
    source_engine: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "ocr_confidence": self.ocr_confidence,
            "source_engine": self.source_engine,
        }


@dataclass
class DocumentPage:
    page_number: int
    width: int
    height: int
    blocks: list[DocumentBlock] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "blocks": [b.to_dict() for b in self.blocks],
        }


@dataclass
class DocumentParseResult:
    document_type: str | None
    pages: list[DocumentPage]
    tables: list[dict[str, Any]]
    fields: list[FieldResult]
    reading_order: list[str]
    parser_engine: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_type": self.document_type,
            "pages": [p.to_dict() for p in self.pages],
            "tables": self.tables,
            "fields": [f.to_dict() for f in self.fields],
            "reading_order": self.reading_order,
            "parser_engine": self.parser_engine,
        }