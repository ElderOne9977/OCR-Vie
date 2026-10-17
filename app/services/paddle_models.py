"""Pure model-name resolution for PaddleOCR PP-OCRv5. No paddle import; unit-testable.

Facts pinned from paddleocr 3.7.0 ``_pipelines/ocr.py::_get_ocr_model_names``:

- PP-OCRv5 detection: ``PP-OCRv5_server_det``
- PP-OCRv5 rec: latin langs (incl. ``vi``) -> ``latin_PP-OCRv5_mobile_rec``
- PP-OCRv5 rec: ``en`` -> ``en_PP-OCRv5_mobile_rec``
- PP-OCRv5 rec: ch/chinese_cht/japan -> ``PP-OCRv5_server_rec``

Language mapping does not guarantee complete charset coverage; runtime validates
the loaded decoder before accepting Vietnamese recognition.
"""

from __future__ import annotations
import unicodedata

VIETNAMESE_CHARACTERS = frozenset(
    char for base in "aăâeêioôơuưyAĂÂEÊIOÔƠUƯYđĐ"
    for tone in ("", "\u0300", "\u0301", "\u0303", "\u0309", "\u0323")
    if len(char := unicodedata.normalize("NFC", base + tone)) == 1
)


def is_vietnamese_lang(lang: str) -> bool:
    return any(part.strip().lower() in {"vie", "vi", "vietnamese"}
               for part in lang.replace("+", ",").split(","))

OCR_VERSION = "PP-OCRv5"
DET_MODEL = "PP-OCRv5_server_det"
LATIN_REC_MODEL = "latin_PP-OCRv5_mobile_rec"
EN_REC_MODEL = "en_PP-OCRv5_mobile_rec"
CJK_REC_MODEL = "PP-OCRv5_server_rec"

# Subset of paddleocr._utils.langs.LATIN_LANGS; 'vi' is Latin.
LATIN_LANGS = frozenset(
    {
        "af", "az", "bs", "cs", "cy", "da", "de", "es", "et", "fr", "ga", "hr",
        "hu", "id", "is", "it", "ku", "la", "lt", "lv", "mi", "ms", "mt", "nl",
        "no", "oc", "pi", "pl", "pt", "ro", "rs_latin", "sk", "sl", "sq", "sv",
        "sw", "tl", "tr", "uz", "vi", "french", "german", "fi", "eu", "gl",
        "lb", "rm", "ca", "qu",
    }
)
CJK_LANGS = frozenset({"ch", "chinese_cht", "japan"})

_ALIASES = {"vie": "vi", "vietnamese": "vi", "eng": "en", "english": "en"}


def normalize_lang(lang: str) -> str:
    """Lowercase, strip, map tesseract-style 'vie+eng' to a single paddle code."""
    if not lang or not isinstance(lang, str):
        return "vi"
    parts = [p.strip().lower() for p in lang.replace("+", ",").split(",") if p.strip()]
    parts = [_ALIASES.get(p, p) for p in parts]
    for p in parts:
        if p in LATIN_LANGS:
            return p
    return parts[0] if parts else "vi"


def is_latin_lang(lang: str) -> bool:
    return normalize_lang(lang) in LATIN_LANGS


def resolve_rec_model(lang: str) -> str | None:
    """Return the PP-OCRv5 recognizer model name for a language, or None if unsupported."""
    code = normalize_lang(lang)
    if code in CJK_LANGS:
        return CJK_REC_MODEL
    if code == "en":
        return EN_REC_MODEL
    if code in LATIN_LANGS:
        return LATIN_REC_MODEL
    return None


def resolve_det_model(lang: str | None = None) -> str:
    """PP-OCRv5 uses one shared server detector for all languages."""
    return DET_MODEL


def resolve_all_models(lang: str) -> tuple[str, str | None]:
    """Return ``(detector_name, recognizer_name_or_None)``."""
    return resolve_det_model(lang), resolve_rec_model(lang)
