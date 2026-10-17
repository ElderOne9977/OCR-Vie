"""Deterministic text normalization. Never overwrite raw_text. No LLM."""

from __future__ import annotations

import re
import unicodedata

from app.schemas.ocr import NormalizationInfo

NORMALIZER_ID = "deterministic-v1"
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"[ \t]+")


def normalize_text(raw: str) -> tuple[str, NormalizationInfo]:
    """Unicode NFKC + whitespace/control cleanup. Semantic content unchanged."""
    if raw is None:
        return "", NormalizationInfo(applied=False, method=None)
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = text.strip()
    applied = text != raw
    return text, NormalizationInfo(
        applied=applied,
        method=NORMALIZER_ID if applied else None,
    )


def clean_display_text(text: str | None) -> str:
    """UI-only deterministic whitespace cleanup. No semantic correction. Preserves Vietnamese."""
    # ponytail: whitespace-only; extend here if richer display rules needed.
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # remove trailing spaces per line, preserve leading spaces and Vietnamese chars
    lines = [ln.rstrip(" \t") for ln in t.split("\n")]
    t = "\n".join(lines)
    # collapse 3+ consecutive blank lines to at most 2 (i.e. 4+ newlines -> 3 newlines)
    t = re.sub(r"\n{4,}", "\n\n\n", t)
    # trim leading/trailing whitespace and empty beginning/end lines
    t = t.strip()
    return t