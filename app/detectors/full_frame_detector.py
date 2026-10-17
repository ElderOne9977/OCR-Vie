"""Fallback detector: treat whole image as one region. Used when Paddle unavailable."""

from __future__ import annotations

from app.detectors.base import TextDetector
from app.schemas.ocr import BBox, ComponentStatus, DetectedRegion


class FullFrameDetector(TextDetector):
    name = "full_frame"

    def status(self) -> ComponentStatus:
        return "ready"

    def detect(self, image) -> list[DetectedRegion]:
        height, width = image.shape[:2]
        polygon = [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]
        return [
            DetectedRegion(
                polygon=polygon,
                bbox=BBox(x=0, y=0, width=int(width), height=int(height)),
                detection_confidence=None,
            )
        ]