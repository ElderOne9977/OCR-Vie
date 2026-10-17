"""Heuristic document/scene classifier. ML slot later via same interface."""

from __future__ import annotations

import numpy as np

from app.classifiers.base import InputClassifier
from app.core.config import get_settings
from app.schemas.ocr import ClassificationResult


class HeuristicDocumentSceneClassifier(InputClassifier):
    name = "heuristic-v1"

    def classify(
        self,
        image,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> ClassificationResult:
        settings = get_settings()
        name = (filename or "").lower()
        ctype = (content_type or "").lower()

        if name.endswith(".pdf") or ctype == "application/pdf":
            return ClassificationResult(
                route="document",
                confidence=0.95,
                reason="pdf_file",
                classifier=self.name,
            )

        height, width = image.shape[:2]
        aspect = width / max(height, 1)
        pixels = width * height

        # Scan-like: large, near A4/letter aspect, mostly upright.
        a4ish = 0.65 <= aspect <= 0.85 or 1.2 <= aspect <= 1.55
        large = pixels >= 1_000_000

        if large and a4ish:
            confidence = 0.72
            route = "document"
            reason = "large_page_aspect"
        elif aspect > 2.4 or aspect < 0.4:
            confidence = 0.7
            route = "scene"
            reason = "extreme_aspect"
        elif pixels < 400_000:
            confidence = 0.62
            route = "scene"
            reason = "small_camera_like"
        else:
            confidence = 0.5
            route = "unknown"
            reason = "ambiguous_layout"

        if confidence < settings.classifier_min_confidence:
            return ClassificationResult(
                route="unknown",
                confidence=confidence,
                reason=reason + "_low_confidence",
                classifier=self.name,
            )
        return ClassificationResult(
            route=route,
            confidence=confidence,
            reason=reason,
            classifier=self.name,
        )