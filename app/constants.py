"""Shared constants for API validation."""

from __future__ import annotations
from typing import Literal

RouteChoice = Literal['auto', 'scene', 'document']
RecognizerChoice = Literal['auto', 'tesseract', 'tesseract_vie_eng', 'paddle', 'paddle_v5_latin',
                           'paddle_v5', 'pp-ocrv5', 'latin_pp-ocrv5_mobile_rec', 'paddle_tesseract', 'vietocr', 'svtr']
ParserChoice = Literal['none', 'fallback', 'fallback-parser', 'monkeyocr', 'paddleocr_vl', 'paddleocr-vl', 'vl']

ACCEPTED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
