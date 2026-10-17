"""Real PP-OCRv5 integration: detector + recognizer + /v1/ocr scene route.

Requires Python 3.11 env (.venv311) with paddlepaddle + paddleocr installed and
models under MODELS_DIR / ~/.paddlex. Run:
    .venv311\\Scripts\\python -m pytest -m "integration and paddle" -v
"""
from __future__ import annotations

import io
import os

import numpy as np
import pytest

def _have_paddle() -> bool:
    from importlib.util import find_spec
    return find_spec('paddleocr') is not None


pytestmark = [pytest.mark.integration, pytest.mark.paddle] + [
    pytest.mark.skipif(not _have_paddle() and os.getenv('OCR_REQUIRE_MODELS') != '1', reason="paddleocr not installed")
]


def test_vietnamese_hybrid_strict():
    """Release gate: real model, no 503-as-pass and no blank fixture."""
    from fastapi.testclient import TestClient
    from app.main import app
    from scripts.smoke_test import _simple_png
    response = TestClient(app).post('/v1/ocr?route=scene&lang=vie%2Beng',
                                   files={'file': ('viet.png', _simple_png(), 'image/png')})
    assert response.status_code == 200, response.text
    result = response.json()
    assert ' '.join(result['text'].split()) == 'Xin chào Việt Nam'
    assert result['engines']['recognizer'] == 'tesseract'
    assert result['engines']['detector'] == 'paddle_detector'


def _sample_scene() -> np.ndarray:
    """Use the same text fixture on local machines and fresh checkouts."""
    from app.preprocessing.common import decode_bytes_to_bgr
    from scripts.smoke_test import _simple_png

    return decode_bytes_to_bgr(_simple_png())


def test_detector_returns_polys():
    from app.detectors.paddle_detector import PaddleDetector

    d = PaddleDetector()
    if not d.ensure_ready():
        pytest.skip(f"detector not ready: {d.status()}")
    regions = d.detect(_sample_scene())
    assert isinstance(regions, list)
    assert len(regions) >= 1
    assert all(len(r.polygon) >= 3 for r in regions)


def test_recognizer_returns_text_on_crop():
    from app.detectors.paddle_detector import PaddleDetector
    from app.recognizers.paddle_recognizer import PaddleRecognizer
    from app.services.polygon_utils import crop_polygon

    d = PaddleDetector()
    if not d.ensure_ready():
        pytest.skip(f"detector not ready: {d.status()}")
    r = PaddleRecognizer()
    from app.services.errors import MODEL_LANGUAGE_UNSUPPORTED, PipelineError
    try:
        if not r.ensure_ready("vie+eng"):
            pytest.skip(f"recognizer not ready: {r.status()}")
    except PipelineError as error:
        assert error.code == MODEL_LANGUAGE_UNSUPPORTED
        assert "Vietnamese" in error.message
        return  # Current artifact cannot represent Vietnamese; rejecting it is correct.

    image = _sample_scene()
    regions = d.detect(image)
    crop = crop_polygon(image, regions[0].polygon)
    assert crop is not None and crop.size > 0

    result = r.recognize(crop, lang="vie+eng", psm=6)
    assert result.engine == "paddle_v5_latin"
    assert result.raw_text is not None
    assert result.normalized_text is not None
    assert result.model


def test_api_v1_scene_paddle_no_fallback():
    from fastapi.testclient import TestClient
    from PIL import Image

    from app.main import app

    c = TestClient(app)
    im = Image.new("RGB", (400, 200), color="white")
    buf = io.BytesIO()
    im.save(buf, format="PNG")

    r = c.post(
        "/v1/ocr?route=scene&recognizer=paddle_v5_latin",
        files={"file": ("x.png", buf.getvalue(), "image/png")},
    )
    # If paddle is unavailable, must be explicit 503 MODEL_UNAVAILABLE (no silent fallback).
    if r.status_code == 503:
        j = r.json()
        assert j["detail"]["code"] in ("MODEL_UNAVAILABLE", "MODEL_LANGUAGE_UNSUPPORTED")
        return
    assert r.status_code == 200
    j = r.json()
    assert j["provenance"]["recognizer"] in ("paddle_v5_latin", "paddle")
    assert j.get("fallback_used") in (False, None)
