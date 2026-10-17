"""Router + hybrid endpoint checks."""
from __future__ import annotations
import io
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

def _png(size=(400, 200)) -> bytes:
    im = Image.new("RGB", size, color="white")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def test_v1_scene_forced():
    c = TestClient(app)
    r = c.post("/v1/ocr?route=scene&recognizer=tesseract", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200
    j = r.json()
    assert j["route"]["route"] == "scene"
    assert j["provenance"]["recognizer"] == "tesseract"
    assert "raw_text" in j["regions"][0]
    assert "normalized_text" in j["regions"][0]

def test_v1_document_with_structure():
    c = TestClient(app)
    r = c.post("/v1/ocr?route=document&return_structure=true", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200
    assert "document" in r.json()