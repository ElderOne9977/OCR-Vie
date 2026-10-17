"""Confidence helpers. Never invent scores."""

from __future__ import annotations


def as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def tesseract_conf_to_unit(conf: float | None) -> float | None:
    """Tesseract reports 0-100. Keep original scale; callers document units."""
    return as_optional_float(conf)