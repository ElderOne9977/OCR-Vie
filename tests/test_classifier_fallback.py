"""Classifier fallback when confidence low."""
from __future__ import annotations
import io
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app
from app.pipelines.router import OCRRouter
from app.schemas.ocr import ClassificationResult

def _png(size=(600, 600)) -> bytes:
    im = Image.new("RGB", size, color="white")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def test_classifier_fallback_warning():
    c = TestClient(app)
    class Fake:
        name = "fake"
        def classify(self, img, **kw):
            return ClassificationResult(route="unknown", confidence=0.1, reason="ambiguous_layout", classifier="fake")
    router = OCRRouter(classifier=Fake())
    with patch("app.pipelines.router.get_settings") as gs:
        m = gs.return_value
        m.fallback_route = "scene"
        m.classifier_min_confidence = 0.45
        m.scene_recognizer = "tesseract"
        with patch("app.api.ocr.get_router", return_value=router):
            url = "/v1/ocr?route=auto"
            r = c.post(url, files={"file": ("x.png", _png(), "image/png")})
            assert r.status_code == 200
            assert any("fallback" in w for w in r.json()["warnings"])
