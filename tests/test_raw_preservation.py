"""raw_text immutable; normalized_text deterministic."""
from __future__ import annotations
import io
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont
from app.main import app
from app.services.normalization import normalize_text

def _png_with_text(text=" Cao   Xin chao ") -> bytes:
    im = Image.new("RGB", (400, 200), color="white")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.load_default()
    except Exception:
        f = None
    d.text((8, 8), text, fill="black", font=f)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def test_normalization_does_not_overwrite_raw():
    raw = " Cao   Xin chao "
    norm, info = normalize_text(raw)
    assert info.method is None or info.method == "deterministic-v1"

def test_v1_preserves_raw_and_normalized():
    c = TestClient(app)
    url = "/v1/ocr?route=scene&recognizer=tesseract"
    r = c.post(url, files={"file": ("x.png", _png_with_text(), "image/png")})
    assert r.status_code == 200
    reg = r.json()["regions"][0]
    assert "raw_text" in reg and "normalized_text" in reg
    recon, _ = normalize_text(reg["raw_text"])
    assert reg["normalized_text"] == recon
