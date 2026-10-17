"""OCR API: legacy /ocr, /ocr/data + hybrid /v1/ocr."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pytesseract import TesseractError, TesseractNotFoundError

import sys

from app.image_processing import preprocess_image
from app.ocr_service import (
    DEFAULT_LANG,
    DEFAULT_PSM,
    ocr_image as _orig_ocr_image,
    ocr_image_data as _orig_ocr_image_data,
    validate_psm as _orig_validate_psm,
)
from app.pipelines.router import get_router
from app.services.errors import PipelineError
from app.services.normalization import clean_display_text
from app.services.limits import ocr_slot
from app.core.config import get_settings
from app.services.request_boundary import request_id

from app.constants import RouteChoice, RecognizerChoice, ParserChoice

logger = logging.getLogger("vietnamese-ocr")

router = APIRouter(tags=["ocr"])

from app.constants import ACCEPTED_CONTENT_TYPES


def _ocr_image(*args, **kwargs):
    mod = sys.modules.get("app.main")
    if mod is not None and hasattr(mod, "ocr_image"):
        fn = getattr(mod, "ocr_image")
        if fn is not _orig_ocr_image and fn is not _ocr_image:
            return fn(*args, **kwargs)
    return _orig_ocr_image(*args, **kwargs)


def _ocr_image_data(*args, **kwargs):
    mod = sys.modules.get("app.main")
    if mod is not None and hasattr(mod, "ocr_image_data"):
        fn = getattr(mod, "ocr_image_data")
        if fn is not _orig_ocr_image_data and fn is not _ocr_image_data:
            return fn(*args, **kwargs)
    return _orig_ocr_image_data(*args, **kwargs)


def _validate_psm(*args, **kwargs):
    mod = sys.modules.get("app.main")
    if mod is not None and hasattr(mod, "validate_psm"):
        fn = getattr(mod, "validate_psm")
        if fn is not _orig_validate_psm and fn is not _validate_psm:
            return fn(*args, **kwargs)
    return _orig_validate_psm(*args, **kwargs)


def _read_upload(file: UploadFile) -> bytes:
    content_type = (file.content_type or "").lower()
    if content_type not in ACCEPTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. "
            f"Accepted: {sorted(ACCEPTED_CONTENT_TYPES)}",
        )
    limit = get_settings().max_upload_bytes
    data = file.file.read(limit + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > limit:
        raise HTTPException(status_code=413, detail="File too large")
    return data


def _preprocess(image_bytes: bytes, psm: int):
    try:
        _validate_psm(psm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return preprocess_image(image_bytes)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Image processing failed: {exc}",
        ) from exc


@router.post("/ocr")
@ocr_slot()
def ocr(
    file: UploadFile = File(...),
    lang: str = Query(DEFAULT_LANG, description="Tesseract language(s), e.g. vie+eng"),
    psm: int = Query(DEFAULT_PSM, description="Page Segmentation Mode (0-13)"),
) -> dict[str, object]:
    """Legacy endpoint: OCR an image, return text. Unchanged contract."""
    logger.info("OCR request: filename=%s lang=%s psm=%s", file.filename, lang, psm)
    image_bytes = _read_upload(file)
    image = _preprocess(image_bytes, psm)
    try:
        text = _ocr_image(image, lang=lang, psm=psm)
    except PipelineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()["error"]) from exc
    except TesseractNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Tesseract binary not found") from exc
    except TesseractError as exc:
        logger.exception('legacy OCR failed request_id=%s', request_id.get())
        raise HTTPException(status_code=500, detail='OCR failed') from exc
    return {"success": True, "filename": file.filename, "text": text}


@router.post("/ocr/data")
@ocr_slot()
def ocr_data(
    file: UploadFile = File(...),
    lang: str = Query(DEFAULT_LANG, description="Tesseract language(s), e.g. vie+eng"),
    psm: int = Query(DEFAULT_PSM, description="Page Segmentation Mode (0-13)"),
    min_confidence: float = Query(0.0, ge=0.0, le=100.0, description="Minimum confidence"),
) -> dict[str, object]:
    """Legacy endpoint: per-word data. Unchanged contract."""
    logger.info(
        "OCR/data request: filename=%s lang=%s psm=%s min_conf=%s",
        file.filename, lang, psm, min_confidence,
    )
    image_bytes = _read_upload(file)
    image = _preprocess(image_bytes, psm)
    try:
        items = _ocr_image_data(image, lang=lang, psm=psm, min_confidence=min_confidence)
    except PipelineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()["error"]) from exc
    except TesseractNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Tesseract binary not found") from exc
    except TesseractError as exc:
        logger.exception('legacy OCR data failed request_id=%s', request_id.get())
        raise HTTPException(status_code=500, detail='OCR failed') from exc
    return {"success": True, "filename": file.filename, "items": items}


@router.post("/v1/ocr")
def ocr_v1(
    file: UploadFile = File(...),
    route: RouteChoice = Query("auto", description="auto|scene|document"),
    recognizer: RecognizerChoice = Query("auto", description="auto|tesseract|paddle|paddle_tesseract"),
    parser: ParserChoice | None = Query(None, description="fallback|monkeyocr|paddleocr_vl"),
    lang: str = Query(DEFAULT_LANG),
    psm: int = Query(DEFAULT_PSM),
    return_structure: bool = Query(False),
    response_format: Literal['json', 'text'] = Query("json", description="json|text"),
) -> dict[str, object]:
    """Hybrid endpoint: classifier -> scene/document pipeline.

    response_format=text returns text/plain with only OCR text (no JSON).
    Default json preserves full provenance for debugging/benchmarks.
    """
    logger.info(
        "OCR/v1 request: filename=%s route=%s recognizer=%s parser=%s structure=%s",
        file.filename, route, recognizer, parser, return_structure,
    )
    image_bytes = _read_upload(file)
    try:
        _validate_psm(psm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = get_router().process(
            image_bytes,
            route=route,
            recognizer=recognizer,
            parser=parser,
            lang=lang,
            psm=psm,
            filename=file.filename,
            content_type=file.content_type,
            return_structure=return_structure,
        )
    except PipelineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()["error"]) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("v1 pipeline failed request_id=%s", request_id.get())
        raise HTTPException(status_code=500, detail={'code': 'INTERNAL_ERROR', 'message': 'OCR processing failed'}) from exc

    if response_format == "text":
        return PlainTextResponse(
            clean_display_text(result.text),
            media_type="text/plain; charset=utf-8",
        )  # type: ignore[return-value]
    payload = result.to_dict(include_debug=True)
    payload["success"] = result.success
    return payload


@router.post("/v1/ocr/text")
def ocr_v1_text(
    file: UploadFile = File(...),
    route: RouteChoice = Query("auto", description="auto|scene|document"),
    recognizer: RecognizerChoice = Query("auto", description="auto|tesseract|paddle|paddle_tesseract"),
    parser: ParserChoice | None = Query(None, description="fallback|monkeyocr|paddleocr_vl"),
    lang: str = Query(DEFAULT_LANG),
    psm: int = Query(DEFAULT_PSM),
    return_structure: bool = Query(False),
):
    """Text-only endpoint: returns text/plain OCR text, same pipeline as /v1/ocr."""
    image_bytes = _read_upload(file)
    try:
        _validate_psm(psm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        result = get_router().process(
            image_bytes,
            route=route,
            recognizer=recognizer,
            parser=parser,
            lang=lang,
            psm=psm,
            filename=file.filename,
            content_type=file.content_type,
            return_structure=return_structure,
        )
    except PipelineError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()["error"]) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("v1 pipeline failed request_id=%s", request_id.get())
        raise HTTPException(status_code=500, detail={'code': 'INTERNAL_ERROR', 'message': 'OCR processing failed'}) from exc
    return PlainTextResponse(
        clean_display_text(result.text),
        media_type="text/plain; charset=utf-8",
    )
