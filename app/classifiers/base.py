"""InputClassifier interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.ocr import ClassificationResult


class InputClassifier(ABC):
    name: str

    @abstractmethod
    def classify(
        self,
        image,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> ClassificationResult:
        raise NotImplementedError