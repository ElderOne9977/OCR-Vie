"""Non-Vietnamese Paddle policy and Vietnamese hybrid readiness contracts.

All Paddle/Tesseract components are stubbed: these tests never load real models
and never require paddleocr or the Tesseract binary.

Covered cases A-J:
  A. default scene (recognizer=auto) resolves to Paddle
  B. auto + Paddle READY -> Paddle
  C. auto + Paddle NOT_LOADED -> ensure_ready() -> Paddle
  D. auto + Paddle unavailable + fallback disabled -> MODEL_UNAVAILABLE
  E. auto + Paddle unavailable + fallback enabled -> Tesseract, fallback_used=true
  F. explicit paddle unavailable -> MODEL_UNAVAILABLE (never fallback)
  G. explicit tesseract -> Tesseract (never tries Paddle)
  H. UI default is Paddle, Tesseract is labelled legacy
  I. /health/ready readiness is driven by Paddle
  J. Tesseract alone does not make the service ready
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.pipelines.router import OCRRouter
from app.schemas.ocr import (
    BBox,
    ClassificationResult,
    DetectedRegion,
    RecognitionResult,
    TimingMs,
)
from app.services.errors import MODEL_UNAVAILABLE, PipelineError
from app.core.config import Settings


class _Detector:
    name = "paddle_detector"

    def __init__(self, status: str = "ready", ready: bool = True) -> None:
        self._status = status
        self._ready = ready
        self.ensure_ready_calls = 0

    def status(self) -> str:
        return self._status

    def ensure_ready(self) -> bool:
        self.ensure_ready_calls += 1
        if self._ready:
            self._status = "ready"
        return self._ready

    def detect(self, image) -> list[DetectedRegion]:
        return [
            DetectedRegion(
                polygon=[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                bbox=BBox(0, 0, 10, 10),
                detection_confidence=0.9,
            )
        ]


class _PaddleRec:
    name = "paddle"

    def __init__(self, status: str = "ready", ready: bool = True) -> None:
        self._status = status
        self._ready = ready
        self.ensure_ready_calls = 0

    def status(self) -> str:
        return self._status

    def ensure_ready(self, lang: str = "vie+eng") -> bool:
        self.ensure_ready_calls += 1
        if self._ready:
            self._status = "ready"
        return self._ready

    def recognize(self, image, *, lang="vie+eng", psm=6) -> RecognitionResult:
        return RecognitionResult(
            raw_text="Xin chào",
            normalized_text="Xin chào",
            ocr_confidence=0.93,
            engine="paddle_v5_latin",
            model="latin_PP-OCRv5_mobile_rec",
        )


class _TessRec:
    name = "tesseract"

    def status(self) -> str:
        return "ready"

    def recognize(self, image, *, lang="vie+eng", psm=6) -> RecognitionResult:
        return RecognitionResult(
            raw_text="Xin chào",
            normalized_text="Xin chào",
            ocr_confidence=88.0,
            engine="tesseract",
            model=lang,
        )


def _classification() -> ClassificationResult:
    return ClassificationResult(
        route="scene", confidence=1.0, reason="forced_by_request", classifier="manual"
    )


def _run_scene(*, recognizer: str, detector: _Detector, paddle: _PaddleRec | None,
               tess: _TessRec | None = None, fallback_enabled: bool = False,
               scene_recognizer: str = "auto"):
    router = OCRRouter.__new__(OCRRouter)
    warnings: list[str] = []

    def fake_get_recognizer(name):
        if name == "tesseract":
            assert tess is not None, "Tesseract path taken unexpectedly"
            return tess
        # Router always constructs a Paddle recognizer before checking status.
        return paddle if paddle is not None else _PaddleRec("unavailable", ready=False)

    settings = Settings(scene_recognizer=scene_recognizer, enable_tesseract_fallback=fallback_enabled)

    with patch("app.pipelines.router.get_detector", return_value=detector), patch(
        "app.pipelines.router.get_recognizer", side_effect=fake_get_recognizer
    ), patch("app.pipelines.router.get_settings", return_value=settings):
        result = router._run_scene(
            np.zeros((40, 40, 3), dtype=np.uint8),
            _classification(),
            recognizer=recognizer,
            lang="fr",
            psm=6,
            filename=None,
            warnings=warnings,
            timing=TimingMs(),
            started=0.0,
        )
    return result, warnings


# ---- A / B ---------------------------------------------------------------


def test_a_default_scene_uses_paddle():
    det = _Detector("not_loaded", ready=True)
    rec = _PaddleRec("not_loaded", ready=True)
    result, warnings = _run_scene(recognizer="auto", detector=det, paddle=rec)
    assert result.engines["detector"] == "paddle_detector"
    assert result.engines["recognizer"] in ("paddle", "paddle_v5_latin")
    assert result.provenance.recognizer == "paddle_v5_latin"
    assert result.fallback_used is False
    assert not any("TESSERACT" in w for w in warnings)


def test_b_auto_ready_uses_paddle():
    det = _Detector("ready", ready=True)
    rec = _PaddleRec("ready", ready=True)
    result, _ = _run_scene(recognizer="auto", detector=det, paddle=rec)
    assert result.provenance.recognizer == "paddle_v5_latin"
    assert result.fallback_used is False
    # Region-level provenance kept per polygon.
    assert result.regions[0].recognizer == "paddle_v5_latin"
    assert result.regions[0].polygon
    assert result.regions[0].bbox.width == 10


def test_c_auto_not_loaded_triggers_ensure_ready():
    det = _Detector("not_loaded", ready=True)
    rec = _PaddleRec("not_loaded", ready=True)
    result, _ = _run_scene(recognizer="auto", detector=det, paddle=rec)
    assert det.ensure_ready_calls == 1
    assert rec.ensure_ready_calls == 1
    assert result.provenance.recognizer == "paddle_v5_latin"


# ---- D / E / F -----------------------------------------------------------


def test_d_auto_unavailable_fallback_disabled_raises_model_unavailable():
    det = _Detector("unavailable", ready=False)
    with pytest.raises(PipelineError) as exc:
        _run_scene(recognizer="auto", detector=det, paddle=None, fallback_enabled=False)
    assert exc.value.code == MODEL_UNAVAILABLE
    assert exc.value.status_code == 503


def test_e_auto_unavailable_fallback_enabled_uses_legacy_tesseract():
    det = _Detector("unavailable", ready=False)
    result, warnings = _run_scene(
        recognizer="auto",
        detector=det,
        paddle=None,
        tess=_TessRec(),
        fallback_enabled=True,
    )
    assert result.provenance.recognizer == "tesseract"
    assert result.fallback_used is True
    assert "PADDLE_UNAVAILABLE_USING_LEGACY_TESSERACT" in warnings


def test_f_explicit_paddle_unavailable_never_falls_back():
    det = _Detector("unavailable", ready=False)
    with pytest.raises(PipelineError) as exc:
        _run_scene(
            recognizer="paddle_v5_latin",
            detector=det,
            paddle=None,
            tess=_TessRec(),  # present but MUST NOT be used
            fallback_enabled=True,
        )
    assert exc.value.code == MODEL_UNAVAILABLE


def test_g_explicit_tesseract_never_tries_paddle():
    det = _Detector("ready", ready=True)
    result, _ = _run_scene(
        recognizer="tesseract",
        detector=det,
        paddle=None,  # paddle must not be requested
        tess=_TessRec(),
    )
    assert result.provenance.recognizer == "tesseract"
    assert result.fallback_used is False
    assert det.ensure_ready_calls == 0


# ---- H -------------------------------------------------------------------


def test_h_ui_defaults_to_paddle_not_tesseract():
    c = TestClient(__import__("app.main", fromlist=["app"]).app)
    html = c.get("/").text
    assert "Paddle" in html
    assert 'value="auto" selected' in html
    assert "Scene tiếng Việt: Paddle + Tesseract" in html
    assert 'id="lang"' in html
    assert 'value="paddle_tesseract"' in html
    # Legacy Tesseract-only controls are gone from the primary form.
    assert 'name="mode"' not in html
    assert 'id="psm"' not in html
    api = c.get("/api").json()
    assert api["primary_scene_engine"] == "paddle_tesseract"
    assert api["legacy_engine"] == "tesseract"


# ---- I / J ---------------------------------------------------------------


def _patch_health(monkeypatch, *, paddle_ready: bool, tesseract_ready: bool):
    import app.api.health as health_mod
    from app.services import paddle_runtime

    det_status = "ready" if paddle_ready else "unavailable"
    rec_status = "ready" if paddle_ready else "unavailable"

    def _comp(status):
        obj = MagicMock()
        obj.status.return_value = status
        return obj

    # health.py imports these at module level: patch the module attributes it reads.
    monkeypatch.setattr(health_mod, "PaddleDetector", lambda: _comp(det_status))
    monkeypatch.setattr(health_mod, "PaddleRecognizer", lambda: _comp(rec_status))
    tess = MagicMock()
    tess.status.return_value = "ready" if tesseract_ready else "unavailable"
    monkeypatch.setattr(health_mod, "TesseractRecognizer", lambda: tess)
    monkeypatch.setattr(paddle_runtime, "is_available", lambda: True)
    monkeypatch.setattr(paddle_runtime, "is_model_present", lambda name: paddle_ready)
    monkeypatch.setattr(paddle_runtime, "is_model_loaded", lambda pred, name: paddle_ready)
    monkeypatch.setattr(paddle_runtime, "_cache_home", lambda: "C:/tmp/paddlex")
    settings = Settings(allow_model_download=False)
    monkeypatch.setattr(health_mod, "get_settings", lambda: settings)


def test_i_health_ready_driven_by_paddle(monkeypatch):
    from app.api.health import health_ready

    _patch_health(monkeypatch, paddle_ready=True, tesseract_ready=False)
    body = health_ready()
    assert body["ready"] is False
    assert body["status"] == "degraded"
    assert body["primary_scene_pipeline"]["detector"] == "paddle"
    assert body["primary_scene_pipeline"]["recognizer"] == "tesseract"
    assert body["primary_scene_pipeline"]["status"] == "unavailable"
    assert body["tesseract_role"] == "document_and_vietnamese_scene"
    assert body["enable_tesseract_fallback"] is False


def test_j_tesseract_alone_is_not_ready(monkeypatch):
    from app.api.health import health_ready

    _patch_health(monkeypatch, paddle_ready=False, tesseract_ready=True)
    body = health_ready()
    assert body["ready"] is False
    assert body["status"] == "degraded"
    assert body["primary_scene_pipeline"]["status"] == "unavailable"
