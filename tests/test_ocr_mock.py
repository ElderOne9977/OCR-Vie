"""Mocked OCR tests that do not require a real Tesseract binary."""

from __future__ import annotations

import io
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.ocr_service import DEFAULT_LANG, DEFAULT_PSM


def _simple_png(
    size: tuple[int, int] = (240, 80),
    text: str = "Hello",
) -> bytes:
    """Generate a simple in-memory PNG (white background, black text)."""
    from PIL import ImageDraw, ImageFont

    im = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None
    draw.text((8, 8), text, fill="black", font=font)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_ocr_mocked_success(client: TestClient) -> None:
    with mock.patch("app.main.ocr_image") as mock_ocr:
        mock_ocr.return_value = "Xin chao Viet Nam"
        resp = client.post(
            "/ocr",
            files={"file": ("hello.png", _simple_png(), "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["text"] == "Xin chao Viet Nam"
        # Verify language/psm forwarded.
        mock_ocr.assert_called_once()
        _arr, kwargs = mock_ocr.call_args
        assert kwargs.get("lang") == DEFAULT_LANG
        assert kwargs.get("psm") == DEFAULT_PSM


def test_ocr_mocked_psm_forwarded(client: TestClient) -> None:
    with mock.patch("app.main.ocr_image") as mock_ocr:
        mock_ocr.return_value = "single line"
        resp = client.post(
            "/ocr?psm=7",
            files={"file": ("line.png", _simple_png(text="line"), "image/png")},
        )
        assert resp.status_code == 200
        assert mock_ocr.call_args[1]["psm"] == 7


def test_ocr_data_mocked(client: TestClient) -> None:
    fake_items = [
        {"text": "Xin", "confidence": 98.5, "left": 8, "top": 8, "width": 40, "height": 16},
    ]
    with mock.patch("app.main.ocr_image_data") as mock_data:
        mock_data.return_value = fake_items
        resp = client.post(
            "/ocr/data",
            files={"file": ("data.png", _simple_png(text="Xin chao"), "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["items"] == fake_items


def test_tesseract_not_found_maps_to_503(client: TestClient) -> None:
    import pytesseract

    with mock.patch("app.main.ocr_image") as mock_ocr:
        mock_ocr.side_effect = pytesseract.TesseractNotFoundError()
        resp = client.post(
            "/ocr",
            files={"file": ("x.png", _simple_png(), "image/png")},
        )
        assert resp.status_code == 503
        assert "Tesseract" in resp.json()["detail"]


def test_tesseract_error_maps_to_500(client: TestClient) -> None:
    import pytesseract

    with mock.patch("app.main.ocr_image") as mock_ocr:
        mock_ocr.side_effect = pytesseract.TesseractError(status=1, message="broken")
        resp = client.post(
            "/ocr",
            files={"file": ("x.png", _simple_png(), "image/png")},
        )
        assert resp.status_code == 500
        assert "OCR failed" in resp.json()["detail"]


def test_health_with_mock(client: TestClient) -> None:
    with mock.patch("pytesseract.get_tesseract_version") as m_ver, \
         mock.patch("pytesseract.get_languages") as m_langs:
        m_ver.return_value = "5.0.0"
        m_langs.return_value = ["eng", "vie", "osd"]
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["tesseract_available"] is True
        assert "vie" in data["languages"]


def test_health_degraded_when_missing(client: TestClient) -> None:
    import pytesseract

    with mock.patch("pytesseract.get_tesseract_version") as m_ver:
        m_ver.side_effect = pytesseract.TesseractNotFoundError()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["tesseract_available"] is False