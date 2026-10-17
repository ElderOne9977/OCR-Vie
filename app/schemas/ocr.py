"""Internal OCR schemas. Detector/recognizer/parser map vào đây, không leak schema vendor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ComponentStatus = Literal[
    "ready",
    "unavailable",
    "not_configured",
    "not_implemented",
    "not_loaded",
    "loading",
    "load_failed",
]
RouteName = Literal["scene", "document", "unknown"]


@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class DetectedRegion:
    polygon: list[list[float]]
    bbox: BBox
    detection_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "polygon": self.polygon,
            "bbox": self.bbox.to_dict(),
            "detection_confidence": self.detection_confidence,
        }


@dataclass
class RecognitionResult:
    raw_text: str
    normalized_text: str
    ocr_confidence: float | None
    engine: str
    model: str | None = None
    raw_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "ocr_confidence": self.ocr_confidence,
            "engine": self.engine,
            "model": self.model,
            "raw_confidence": self.raw_confidence,
        }


@dataclass
class RegionResult:
    id: int
    polygon: list[list[float]]
    bbox: BBox
    raw_text: str
    normalized_text: str
    detection_confidence: float | None
    ocr_confidence: float | None
    recognizer: str
    raw_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "polygon": self.polygon,
            "bbox": self.bbox.to_dict(),
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "detection_confidence": self.detection_confidence,
            "ocr_confidence": self.ocr_confidence,
            "recognizer": self.recognizer,
            "raw_confidence": self.raw_confidence,
        }


@dataclass
class ClassificationResult:
    route: RouteName
    confidence: float | None
    reason: str
    classifier: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "confidence": self.confidence,
            "reason": self.reason,
            "classifier": self.classifier,
        }


@dataclass
class NormalizationInfo:
    applied: bool
    method: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "method": self.method,
            "confidence": self.confidence,
        }


@dataclass
class FieldResult:
    field_name: str
    raw_text: str
    normalized_text: str
    bbox: BBox | None = None
    ocr_confidence: float | None = None
    field_confidence: float | None = None
    detection_confidence: float | None = None
    source_engine: str | None = None
    parser_engine: str | None = None
    normalization: NormalizationInfo = field(default_factory=lambda: NormalizationInfo(applied=False))
    requires_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "ocr_confidence": self.ocr_confidence,
            "field_confidence": self.field_confidence,
            "detection_confidence": self.detection_confidence,
            "source_engine": self.source_engine,
            "parser_engine": self.parser_engine,
            "normalization": self.normalization.to_dict(),
            "requires_review": self.requires_review,
        }


@dataclass
class Provenance:
    route: str
    parser: str | None
    detector: str | None
    recognizer: str
    normalizer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "parser": self.parser,
            "detector": self.detector,
            "recognizer": self.recognizer,
            "normalizer": self.normalizer,
        }


@dataclass
class TimingMs:
    classification: float | None = None
    detection: float | None = None
    recognition: float | None = None
    parsing: float | None = None
    total: float | None = None
    decode: float | None = None
    initialization: float | None = None
    preprocessing: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "classification": self.classification,
            "detection": self.detection,
            "recognition": self.recognition,
            "parsing": self.parsing,
            "total": self.total,
            "decode": self.decode,
            "initialization": self.initialization,
            "preprocessing": self.preprocessing,
        }


@dataclass
class PipelineResult:
    success: bool
    route: dict[str, Any]
    engines: dict[str, Any]
    text: str
    regions: list[RegionResult]
    document: dict[str, Any] | None
    warnings: list[str]
    provenance: Provenance
    timing_ms: TimingMs | None = None
    filename: str | None = None
    fallback_used: bool = False

    def to_dict(self, include_debug: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "success": self.success,
            "route": self.route,
            "engines": self.engines,
            "text": self.text,
            "regions": [r.to_dict() for r in self.regions],
            "document": self.document,
            "warnings": self.warnings,
            "provenance": self.provenance.to_dict(),
            "fallback_used": self.fallback_used,
        }
        if self.filename is not None:
            payload["filename"] = self.filename
        if include_debug and self.timing_ms is not None:
            payload["timing_ms"] = self.timing_ms.to_dict()
        return payload
