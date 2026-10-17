"""TextRecognizer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.ocr import ComponentStatus, RecognitionResult


class TextRecognizer(ABC):
    name: str

    @abstractmethod
    def status(self) -> ComponentStatus:
        raise NotImplementedError

    @abstractmethod
    def recognize(
        self,
        image,
        *,
        lang: str = "vie+eng",
        psm: int = 6,
    ) -> RecognitionResult:
        raise NotImplementedError