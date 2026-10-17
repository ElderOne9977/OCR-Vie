"""DocumentParser interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.document import DocumentParseResult


class DocumentParser(ABC):
    name: str

    @abstractmethod
    def parse(self, text: str, *, filename: str | None = None) -> DocumentParseResult:
        raise NotImplementedError