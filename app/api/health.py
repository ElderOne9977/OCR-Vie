"""Health endpoints. /health/ready never force-loads heavy models."""
from __future__ import annotations
import pytesseract
from fastapi import APIRouter, Response
from pytesseract import TesseractNotFoundError
from app.core.config import get_settings
from app.detectors.paddle_detector import PaddleDetector
from app.recognizers.paddle_recognizer import PaddleRecognizer
from app.recognizers.tesseract_recognizer import TesseractRecognizer
from app.services import paddle_runtime
from app.services.paddle_models import resolve_det_model, resolve_rec_model

# Canonical language for default-pipeline readiness.
PRIMARY_REC_LANG = "vie+eng"
# Scene recognizer values that name an engine, not a language.
_ENGINE_KEYS = {"", "auto", "paddle", "paddle_v5_latin", "paddle_v5", "paddle_tesseract", "tesseract", "vietocr", "svtr"}


def _primary_rec_lang(configured: str | None) -> str:
    """Map a configured recognizer value to a PP-OCRv5 language for model lookup."""
    key = (configured or "").strip().lower()
    return PRIMARY_REC_LANG if key in _ENGINE_KEYS else (configured or PRIMARY_REC_LANG)


router = APIRouter(tags=["health"])
@router.get("/health")
def health() -> dict[str, object]:
    """Backward-compatible health. Tesseract availability + languages."""
    try:
        version = str(pytesseract.get_tesseract_version())
        languages: list[str] = pytesseract.get_languages()
        return {"status": "ok" if {'vie', 'eng'} <= set(languages) else 'degraded', "tesseract_available": True, "version": version, "languages": languages, "required_languages": ["vie", "eng"]}
    except TesseractNotFoundError:
        return {"status": "degraded", "tesseract_available": False}
@router.get("/health/live")
def health_live() -> dict[str, object]:
    """Liveness: process is up. Does not touch models."""
    return {"status": "ok"}
def health_ready() -> dict[str, object]:
    """Readiness: report component status without loading heavy models.

    Default Vietnamese Scene needs Paddle detection and Tesseract recognition.
    Document readiness is independent. Dependency presence does not prove
    successful initialization; dependency imports happen only on actual use.

    Status vocabulary per Paddle component (never equate not_loaded with unavailable):
      ``not_loaded``  -> dependency + weights present, not in memory yet.
      ``unavailable`` -> dependency missing or weights absent and download forbidden.
      ``load_failed`` -> a real load attempt raised.
      ``ready``       -> real weights loaded in this process.
    """
    settings = get_settings()
    tesseract = TesseractRecognizer()
    t_status = tesseract.status(lang=PRIMARY_REC_LANG)
    det_model = resolve_det_model(None)
    rec_model = resolve_rec_model(_primary_rec_lang(settings.scene_recognizer)) or ""
    paddle_available = paddle_runtime.is_available()
    det_present = paddle_runtime.is_model_present(det_model) if paddle_available else False
    rec_present = paddle_runtime.is_model_present(rec_model) if (paddle_available and rec_model) else False
    det_component = PaddleDetector()
    rec_component = PaddleRecognizer()
    rec_status = rec_component.status(lang=PRIMARY_REC_LANG)
    def _paddle_component(model: str, present: bool, loaded: bool, status: str) -> dict:
        return {"status": status, "dependency_available": paddle_available, "model_present": present, "model_loaded": loaded, "model": model}
    components = {
        "tesseract": t_status,
        "classifier": "ready",
        "paddle_detector": det_component.status(),
        "paddle_recognizer": rec_status,
        "vietocr": "unavailable",
        "svtr": "unavailable",
        "document_parser": settings.document_parser,
    }
    def pipeline_status(engine, default):
        engine = engine if isinstance(engine, str) else default
        engine = engine.lower()
        if engine in ("auto", ""):
            engine = default
        if engine in ("tesseract", "tesseract_vie_eng"):
            return {"detector": "full_frame", "recognizer": "tesseract", "status": t_status}
        if engine == "paddle_tesseract":
            status = t_status if det_component.status() == "ready" else det_component.status()
            return {"detector": "paddle", "recognizer": "tesseract", "status": status}
        if engine not in ("paddle", "paddle_v5_latin", "paddle_v5", "pp-ocrv5", "latin_pp-ocrv5_mobile_rec"):
            return {"detector": None, "recognizer": engine, "status": "unavailable"}
        status = rec_status if det_component.status() == "ready" else det_component.status()
        metadata = paddle_runtime.get_model_meta("TextRecognition", rec_model)
        if metadata and metadata.get("vietnamese_supported") is False:
            status = "unavailable"
        return {"detector": "paddle", "recognizer": "paddle_v5_latin", "status": status}
    scene_pipeline = pipeline_status(settings.scene_recognizer, "paddle_tesseract")
    document_pipeline = pipeline_status(settings.document_recognizer, "tesseract")
    # Compatibility: overall readiness describes the configured Scene pipeline.
    # Document readiness is reported independently for route-aware clients.
    overall_ready = scene_pipeline["status"] == "ready" or (
        settings.scene_recognizer in ("auto", "") and settings.enable_tesseract_fallback and t_status == "ready"
    )
    return {
        "status": "ok" if overall_ready else "degraded",
        "ready": overall_ready,
        "primary_scene_pipeline": scene_pipeline,
        "primary_document_pipeline": document_pipeline,
        "readiness_scope": "scene (default language vie+eng); document reported independently",
        "components": components,
        "paddle": {
            "dependency_available": paddle_available,
            "allow_model_download": settings.allow_model_download,
            "cache_home": str(paddle_runtime._cache_home()),
            "detector_model": det_model,
            "recognizer_model": rec_model or None,
            "detector_present": det_present,
            "recognizer_present": rec_present,
            "loaded": paddle_runtime.is_model_loaded("TextDetection", det_model),
            "detector": _paddle_component(
                det_model, det_present,
                paddle_runtime.is_model_loaded("TextDetection", det_model),
                det_component.status(),
            ),
            "recognizer": _paddle_component(
                rec_model or "", rec_present,
                paddle_runtime.is_model_loaded("TextRecognition", rec_model)
                if rec_model else False,
                rec_status,
            ),
        },
        "tesseract_role": "document_and_vietnamese_scene",
        "tesseract_roles": ["document_default", "hybrid_line_recognizer", "legacy", "optional_scene_fallback"],
        "enable_tesseract_fallback": settings.enable_tesseract_fallback,
        "fallback_route": settings.fallback_route,
        "device": settings.device,
        "cpu_threads": settings.cpu_threads,
        "detector_limit_side_len": settings.detector_limit_side_len,
        "limits": {"max_concurrent_ocr": settings.max_concurrent_ocr,
                   "max_image_pixels": settings.max_image_pixels,
                   "max_regions": settings.max_regions,
                   "ocr_timeout_seconds": settings.ocr_timeout_seconds,
                   "ocr_request_timeout_seconds": settings.ocr_request_timeout_seconds},
    }


@router.get('/health/ready')
def readiness_response(response: Response):
    body = health_ready()
    response.status_code = 200 if body['ready'] else 503
    return body
