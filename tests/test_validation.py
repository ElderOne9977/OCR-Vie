"""Runnable checks that do not require the Tesseract binary.

Run: python -m tests.test_validation / pytest tests/test_validation.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Allow `python -m tests.test_validation` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from PIL import Image

from app.main import ACCEPTED_CONTENT_TYPES, app
from app.ocr_service import VALID_PSM_MODES, validate_psm


def _png_bytes(size: tuple[int, int] = (40, 20), color: str = "white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_validate_psm() -> None:
    assert validate_psm(6) == 6
    assert 3 in VALID_PSM_MODES
    assert 2 not in VALID_PSM_MODES
    try:
        validate_psm(2)
        raise AssertionError("PSM 2 should be rejected")
    except ValueError as exc:
        assert "Invalid PSM" in str(exc)


def test_root_and_rejects() -> None:
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert "text/html" in root.headers.get("content-type", "")
    assert "Vietnamese OCR" in root.text

    api = client.get("/api")
    assert api.status_code == 200
    assert "running" in api.json()["message"]

    # Non-image content type.
    bad_type = client.post(
        "/ocr",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert bad_type.status_code == 400

    # Empty image payload (with accepted content type).
    empty = client.post(
        "/ocr",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert empty.status_code == 400

    # Corrupted image bytes.
    corrupted = client.post(
        "/ocr",
        files={"file": ("bad.png", b"not-an-image", "image/png")},
    )
    assert corrupted.status_code == 400
    assert "Invalid" in corrupted.json()["detail"] or "corrupted" in corrupted.json()["detail"]

    # Invalid PSM.
    invalid_psm = client.post(
        "/ocr?psm=99",
        files={"file": ("ok.png", _png_bytes(), "image/png")},
    )
    assert invalid_psm.status_code == 400
    assert "PSM" in invalid_psm.json()["detail"]

    # File too large should be 413.
    big = b"x" * (10 * 1024 * 1024 + 1)
    too_big = client.post(
        "/ocr",
        files={"file": ("big.png", big, "image/png")},
    )
    assert too_big.status_code == 413


if __name__ == "__main__":
    test_validate_psm()
    test_root_and_rejects()
    print("ok: validation checks passed")