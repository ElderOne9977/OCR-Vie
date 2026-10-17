"""TextDetector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.ocr import ComponentStatus, DetectedRegion


class TextDetector(ABC):
    name: str

    @abstractmethod
    def status(self) -> ComponentStatus:
        raise NotImplementedError

    @abstractmethod
    def detect(self, image) -> list[DetectedRegion]:
        raise NotImplementedError