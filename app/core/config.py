"""Runtime config from env. Route code đọc config, không hardcode engine."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import get_args
from app.constants import RecognizerChoice, ParserChoice


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    # Auto Vietnamese Scene uses Paddle detection + Tesseract line recognition.
    scene_recognizer: str = "auto"
    scene_detector: str = "auto"
    document_recognizer: str = "tesseract"
    document_parser: str = "none"
    device: str = "cpu"
    max_concurrent_ocr: int = 2
    cpu_threads: int = 10
    detector_limit_side_len: int = 960
    max_image_pixels: int = 12_000_000
    max_image_side: int = 8192
    max_regions: int = 256
    ocr_timeout_seconds: float = 30.0
    ocr_request_timeout_seconds: float = 60.0
    tesseract_batch_size: int = 32
    document_preprocess: str = "gray"
    fallback_route: str = "scene"
    classifier_min_confidence: float = 0.45
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pdf_pages: int = 20
    max_pdf_bytes: int = 20 * 1024 * 1024
    vietocr_model_path: str = ""
    svtr_model_path: str = ""
    models_dir: str = "models"
    include_debug: bool = False
    allow_model_download: bool = True
    enable_tesseract_fallback: bool = False

    def __post_init__(self) -> None:
        for name in ("max_concurrent_ocr", "cpu_threads", "max_image_pixels", "max_image_side",
                     "max_regions", "max_upload_bytes", "tesseract_batch_size", "ocr_timeout_seconds",
                     "ocr_request_timeout_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.document_preprocess not in {"gray", "binary", "original"}:
            raise ValueError("DOCUMENT_PREPROCESS must be gray|binary|original")
        if not isinstance(self.detector_limit_side_len, int) or not 32 <= self.detector_limit_side_len <= 4000:
            raise ValueError("detector_limit_side_len must be an integer in [32, 4000]")
        if not math.isfinite(self.classifier_min_confidence) or not 0 <= self.classifier_min_confidence <= 1:
            raise ValueError('classifier_min_confidence must be finite in [0, 1]')
        if self.fallback_route not in {'scene', 'document'}:
            raise ValueError('FALLBACK_ROUTE must be scene|document')
        for name in ('scene_recognizer', 'document_recognizer'):
            if getattr(self, name).lower() not in get_args(RecognizerChoice):
                raise ValueError(f'Invalid {name}')
        if self.document_parser.lower() not in get_args(ParserChoice):
            raise ValueError('Invalid DOCUMENT_PARSER')
        if self.scene_detector.lower() not in {'auto', 'paddle', 'paddle_detector', 'full_frame'}:
            raise ValueError('Invalid OCR_SCENE_DETECTOR')

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            scene_recognizer=_env("OCR_SCENE_RECOGNIZER", "auto"),
            scene_detector=_env("OCR_SCENE_DETECTOR", "auto"),
            document_recognizer=_env("OCR_DOCUMENT_RECOGNIZER", "tesseract"),
            document_parser=_env("DOCUMENT_PARSER", "none"),
            device=_env("DEVICE", "cpu"),
            max_concurrent_ocr=_env_int("MAX_CONCURRENT_OCR", 2),
            cpu_threads=_env_int("CPU_THREADS", 10),
            detector_limit_side_len=_env_int("DETECTOR_LIMIT_SIDE_LEN", 960),
            max_image_pixels=_env_int("MAX_IMAGE_PIXELS", 12_000_000),
            max_image_side=_env_int("MAX_IMAGE_SIDE", 8192),
            max_regions=_env_int("MAX_REGIONS", 256),
            ocr_timeout_seconds=_env_float("OCR_TIMEOUT_SECONDS", 30.0),
            ocr_request_timeout_seconds=_env_float("OCR_REQUEST_TIMEOUT_SECONDS", 60.0),
            tesseract_batch_size=_env_int("TESSERACT_BATCH_SIZE", 32),
            document_preprocess=_env("DOCUMENT_PREPROCESS", "gray"),
            fallback_route=_env("FALLBACK_ROUTE", "scene"),
            classifier_min_confidence=_env_float("CLASSIFIER_MIN_CONFIDENCE", 0.45),
            max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
            max_pdf_pages=_env_int("MAX_PDF_PAGES", 20),
            max_pdf_bytes=_env_int("MAX_PDF_BYTES", 20 * 1024 * 1024),
            vietocr_model_path=_env("VIETOCR_MODEL_PATH", ""),
            svtr_model_path=_env("SVTR_MODEL_PATH", ""),
            models_dir=_env("MODELS_DIR", "models"),
            include_debug=_env("OCR_DEBUG", "0") in {"1", "true", "TRUE"},
            allow_model_download=_env("ALLOW_MODEL_DOWNLOAD", "true")
            in {"1", "true", "TRUE"},
            enable_tesseract_fallback=_env("ENABLE_TESSERACT_FALLBACK", "false")
            in {"1", "true", "TRUE"},
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reset_settings() -> None:
    """Test helper: reload env on next get_settings()."""
    global _settings
    _settings = None
