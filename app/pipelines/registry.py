"""Recognizer / detector / parser factories. Single place to swap engines."""

from __future__ import annotations

from app.core.config import get_settings
from app.detectors.base import TextDetector
from app.detectors.full_frame_detector import FullFrameDetector
from app.detectors.paddle_detector import PaddleDetector
from app.parsers.base import DocumentParser
from app.parsers.fallback_parser import FallbackDocumentParser
from app.parsers.monkeyocr_parser import MonkeyOCROParser
from app.parsers.paddleocr_vl_parser import PaddleOcrVLParser
from app.recognizers.base import TextRecognizer
from app.recognizers.paddle_recognizer import PaddleRecognizer
from app.recognizers.svtr_recognizer import SVTRRecognizer
from app.recognizers.tesseract_recognizer import TesseractRecognizer
from app.recognizers.vietocr_recognizer import VietOCRRecognizer


def get_recognizer(name: str | None) -> TextRecognizer:
    settings = get_settings()
    key = (name or "auto").lower()
    # Tesseract is legacy / benchmark / optional-fallback only — never the implicit default.
    if key in ("tesseract", "tesseract_vie_eng"):
        return TesseractRecognizer()
    if key in ("auto", "", "paddle", "paddle_v5_latin", "paddle_v5", "pp-ocrv5", "latin_pp-ocrv5_mobile_rec"):
        return PaddleRecognizer()
    if key == "vietocr":
        return VietOCRRecognizer(settings.vietocr_model_path)
    if key == "svtr":
        return SVTRRecognizer(settings.svtr_model_path)
    # Unknown engine is a caller error: do NOT silently degrade to legacy Tesseract.
    raise ValueError(
        f"Unknown recognizer '{name}'. Use auto|paddle|paddle_v5_latin|tesseract|vietocr|svtr."
    )


def get_detector(name: str | None) -> TextDetector:
    key = (name or "auto").lower()
    if key in ("auto", "", "paddle", "paddle_detector"):
        return PaddleDetector()
    return FullFrameDetector()


def get_parser(name: str | None) -> DocumentParser:
    settings = get_settings()
    key = (name or settings.document_parser or "fallback").lower()
    if key in ("none", "", "fallback", "fallback-parser"):
        return FallbackDocumentParser()
    if key == "monkeyocr":
        return MonkeyOCROParser()
    if key in ("paddleocr_vl", "paddleocr-vl", "vl"):
        return PaddleOcrVLParser()
    return FallbackDocumentParser()