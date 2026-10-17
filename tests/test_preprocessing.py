"""Tests for image preprocessing."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.image_processing import decode_image, preprocess_image


def _png_bytes(size: tuple[int, int] = (24, 24)) -> bytes:
    im = Image.new("RGB", size, color="white")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_returns_binary_image() -> None:
    out = preprocess_image(_png_bytes())
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2  # grayscale/binary
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})


def test_decode_empty_raises() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        decode_image(b"")
    assert exc.value.status_code == 400


def test_decode_corrupted_raises() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        decode_image(b"not-an-image")
    assert exc.value.status_code == 400


def test_preprocess_empty_raises() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        preprocess_image(b"")
    assert exc.value.status_code == 400