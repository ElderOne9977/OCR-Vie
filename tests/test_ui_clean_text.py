"""UI clean-text + API contract tests (spec 14 A-H). No real models needed."""
from __future__ import annotations

import io
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, INDEX_HTML
from app.services.normalization import clean_display_text, normalize_text


def _png() -> bytes:
    im = Image.new("RGB", (120, 40), color="white")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _mock_pipeline(text="HELLO\\nWORLD", raw="HELLO\\nWORLD"):
    from app.schemas.ocr import BBox, PipelineResult, Provenance, RegionResult, TimingMs
    return PipelineResult(
        success=True,
        route={"route": "scene", "confidence": 0.9, "reason": "forced", "classifier": "test"},
        engines={"classifier": "test", "detector": "paddle_detector", "recognizer": "paddle_v5_latin", "parser": None, "normalizer": "deterministic-v1"},
        text=text,
        regions=[RegionResult(id=0, polygon=[[0,0],[10,0],[10,10],[0,10]], bbox=BBox(0,0,10,10), raw_text=raw, normalized_text=text, detection_confidence=0.9, ocr_confidence=0.95, recognizer="paddle_v5_latin")],
        document=None,
        warnings=[],
        provenance=Provenance(route="scene", parser=None, detector="paddle_detector", recognizer="paddle_v5_latin", normalizer="deterministic-v1"),
        timing_ms=TimingMs(classification=1, detection=10, recognition=5, total=16),
        fallback_used=False,
    )


# A: UI renders only text (INDEX_HTML must not dump JSON into main result box)
def test_ui_has_clean_text_container():
    assert 'id="ocr-text"' in INDEX_HTML
    assert 'white-space: pre-wrap' in INDEX_HTML
    assert 'word-break: break-word' in INDEX_HTML


# B: JSON metadata not in normal output — main container is #ocr-text, JSON only in collapsed <details>
def test_ui_json_only_in_details():
    assert 'id="debug-json"' in INDEX_HTML
    assert '<details' in INDEX_HTML
    # main text container is separate from debug pre
    assert INDEX_HTML.count('id="ocr-text"') == 1
    assert INDEX_HTML.count('id="debug-json"') == 1
    # collapsed by default: no open attribute
    assert '<details id="tech-details">' in INDEX_HTML or '<details' in INDEX_HTML
    assert 'Chi tiết kỹ thuật' in INDEX_HTML


# C: line breaks preserved via pre-wrap
def test_ui_preserves_line_breaks_style():
    assert 'white-space: pre-wrap' in INDEX_HTML


# D: empty text shows message
def test_ui_empty_message_present():
    assert 'Không phát hiện văn bản.' in INDEX_HTML


# E: copy copies only text (JS uses lastCleanText, not debugJson)
def test_ui_copy_copies_text_only():
    assert 'Sao chép văn bản' in INDEX_HTML
    assert 'navigator.clipboard.writeText(lastCleanText)' in INDEX_HTML
    assert 'Đã sao chép' in INDEX_HTML
    # download also uses lastCleanText
    assert 'new Blob([lastCleanText]' in INDEX_HTML


# F: backend JSON schema unchanged (still returns success, route, engines, text, regions, provenance, timing_ms, fallback_used)
def test_backend_json_schema_preserved():
    c = TestClient(app)
    with patch("app.api.ocr.get_router") as m:
        m.return_value.process.return_value = _mock_pipeline()
        r = c.post("/v1/ocr?route=scene", files={"file": ("x.png", _png(), "image/png")})
        assert r.status_code == 200
        j = r.json()
        for k in ["success", "route", "engines", "text", "regions", "provenance", "fallback_used", "timing_ms"]:
            assert k in j, f"missing {k}"
        assert j["success"] is True
        assert "raw_text" in j["regions"][0]


# G: raw_text immutable
def test_raw_text_preserved_and_normalized_deterministic():
    c = TestClient(app)
    raw = "  Cao   Xin chao  "
    with patch("app.api.ocr.get_router") as m:
        m.return_value.process.return_value = _mock_pipeline(text="Cao Xin chao", raw=raw)
        r = c.post("/v1/ocr?route=scene", files={"file": ("x.png", _png(), "image/png")})
        reg = r.json()["regions"][0]
        assert reg["raw_text"] == raw
        recon, _ = normalize_text(raw)
        assert reg["normalized_text"] == recon


# H: whitespace cleanup does not alter Vietnamese
def test_clean_display_preserves_vietnamese():
    src = "GIẤY CHỨNG NHẬN\nBẢO HIỂM TỰ NGUYỆN"
    assert clean_display_text(src) == src
    src2 = "Tiếng Việt có dấu: ơ ư á à ả ã ạ\nDòng 2"
    assert "ơ" in clean_display_text(src2)
    assert "Tiếng" in clean_display_text(src2)
    # deterministic rules
    assert clean_display_text("a\r\n b") == "a\n b"
    assert clean_display_text("hello   \nworld   ") == "hello\nworld"
    assert clean_display_text("a\n\n\n\nb") == "a\n\n\nb"  # 4 newlines -> 3
    assert clean_display_text("\n\n hello \n\n") == "hello"
    # does not transliterate
    assert "GIẤY" in clean_display_text("GIẤY")


def test_error_message_present():
    assert 'Không thể nhận dạng văn bản.' in INDEX_HTML


def test_download_txt_uses_text_only():
    assert 'Tải .txt' in INDEX_HTML
    assert 'ocr.txt' in INDEX_HTML


def test_text_mode_endpoints():
    c = TestClient(app)
    with patch("app.api.ocr.get_router") as m:
        m.return_value.process.return_value = _mock_pipeline(text="Xin chào")
        r = c.post("/v1/ocr?response_format=text", files={"file": ("x.png", _png(), "image/png")})
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]
        assert r.text == "Xin chào"
        r2 = c.post("/v1/ocr/text", files={"file": ("x.png", _png(), "image/png")})
        assert r2.status_code == 200
        assert "text/plain" in r2.headers["content-type"]
        assert r2.text == "Xin chào"


def test_empty_text_via_api_still_json_with_empty_string():
    c = TestClient(app)
    with patch("app.api.ocr.get_router") as m:
        m.return_value.process.return_value = _mock_pipeline(text="", raw="")
        r = c.post("/v1/ocr", files={"file": ("x.png", _png(), "image/png")})
        assert r.json()["text"] == ""
        # UI would show Không phát hiện... but API keeps empty string
        assert clean_display_text(r.json()["text"]) == ""
