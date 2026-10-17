"""FastAPI app: browser workspace, hybrid OCR, and legacy API routers."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.health import router as health_router
from app.api.ocr import router as ocr_router
from app.core.config import get_settings
from app.services.errors import PipelineError
from app.services.request_boundary import RequestBoundary
# Re-exports for backward compatibility: tests mock app.main.ocr_image etc.
from app.constants import ACCEPTED_CONTENT_TYPES, MAX_UPLOAD_BYTES  # noqa: F401
from app.ocr_service import (  # noqa: F401
    DEFAULT_LANG, DEFAULT_PSM, VALID_PSM_MODES,
    ocr_image, ocr_image_data, validate_psm,
)

logger = logging.getLogger("vietnamese-ocr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

@asynccontextmanager
async def lifespan(application):
    get_settings()  # Fail invalid configuration before serving requests.
    if os.getenv('OCR_WARMUP', '').lower() in {'1', 'true'}:
        def warmup():
            import cv2
            import numpy as np
            from app.pipelines.router import get_router
            from app.api.health import health_ready
            ok, image = cv2.imencode('.png', np.full((80, 240, 3), 255, np.uint8))
            if not ok:
                raise RuntimeError('Cannot create warmup image')
            get_router().process(image.tobytes(), route='scene', recognizer='auto', lang='vie+eng')
            if not health_ready()['ready']:
                raise RuntimeError('Configured OCR pipeline not ready after warmup')
        await run_in_threadpool(warmup)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Vietnamese OCR API",
    description=(
        "OCR for Vietnamese and English text. Primary Scene engine: "
        "Paddle text detection + Tesseract recognition for Vietnamese Scene images. "
        "Tesseract handles documents, optional hybrid line recognition, legacy compatibility, "
        "and opt-in Scene fallback."
    ),
    version="1.0.0",
)
app.add_middleware(RequestBoundary)


@app.exception_handler(PipelineError)
async def pipeline_error_handler(request, exc: PipelineError):
    # Admission/deadline decorators may raise outside the endpoint's try block.
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.to_dict()["error"]})


INDEX_HTML = Path(__file__).with_name("ui.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Browser UI; upload limit follows the active server configuration."""
    return INDEX_HTML.replace("__MAX_UPLOAD_BYTES__", str(get_settings().max_upload_bytes)).replace(
        "__REQUEST_TIMEOUT_MS__", str(int((get_settings().ocr_request_timeout_seconds + 60) * 1000)))


@app.get("/api")
def api_root() -> dict[str, str]:
    """JSON ping for scripts/tests."""
    return {
        "message": "Vietnamese OCR API is running",
        "primary_scene_engine": "paddle_tesseract",
        "legacy_engine": "tesseract",
    }


app.include_router(health_router)
app.include_router(ocr_router)
