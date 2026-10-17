"""Lazy PaddleOCR runtime + predictor cache. Import-safe without paddle installed.

Model location / download policy
--------------------------------
PaddleX stores official models under ``PADDLE_PDX_CACHE_HOME`` (default
``~/.paddlex``) at ``official_models/<model_name>``. Resolution order here:
  1) Explicit ``PADDLE_PDX_CACHE_HOME`` env var (framework knob, wins).
  2) ``MODELS_DIR`` env var when user set it (Docker: ``MODELS_DIR=/models``).
  3) Framework default ``~/.paddlex`` (already holds downloaded weights).

``ALLOW_MODEL_DOWNLOAD=false`` refuses to construct a predictor whose
weights are absent locally — no silent download, health never triggers I/O.
"""

from __future__ import annotations

import logging
import os
import threading
import importlib.util
import unicodedata
from pathlib import Path
from typing import Any
from app.services.deadline import deadline_lock, remaining_seconds

logger = logging.getLogger("vietnamese-ocr.paddle_runtime")

_lock = threading.Lock()
# ponytail: serialize shared native predictors; per-worker locks when load tests justify a worker pool.
_predict_lock = threading.Lock()
_CACHE: dict[tuple[str, str, tuple], Any] = {}
_LOADED: set[tuple[str, str]] = set()
_LOAD_ERRORS: dict[tuple[str, str], str] = {}
_MODEL_META: dict[tuple[str, str], dict] = {}
_ENV_READY = False


def _settings():
    from app.core.config import get_settings

    return get_settings()


def is_available() -> bool:
    """Dependency presence only; actual import/load failures are reported on use."""
    try:
        return importlib.util.find_spec("paddleocr") is not None
    except (ImportError, ValueError):
        return False


def _ensure_paddle_env() -> None:
    """Point PaddleX cache at MODELS_DIR when user set it.

    Must run before ``import paddleocr`` (which freezes CACHE_DIR at import).
    """
    global _ENV_READY
    if _ENV_READY:
        return
    if "PADDLE_PDX_CACHE_HOME" not in os.environ:
        env_models_dir = os.environ.get("MODELS_DIR")
        if env_models_dir:
            os.environ["PADDLE_PDX_CACHE_HOME"] = str(Path(env_models_dir))
    _ENV_READY = True


def _cache_home() -> Path:
    env = os.environ.get("PADDLE_PDX_CACHE_HOME") or os.environ.get("MODELS_DIR")
    if env:
        return Path(env)
    return Path.home() / ".paddlex"


def model_dir(model_name: str) -> Path:
    """Local directory where a PaddleX official model would be stored."""
    return _cache_home() / "official_models" / model_name


def is_model_present(model_name: str) -> bool:
    """True when model weights already exist locally (no download needed)."""
    d = model_dir(model_name)
    try:
        return (d.is_dir() and (d / 'inference.pdiparams').is_file()
                and (d / 'inference.pdiparams').stat().st_size > 0
                and any((d / name).is_file() and (d / name).stat().st_size > 0
                        for name in ('inference.json', 'inference.pdmodel'))
                and (d / 'inference.yml').is_file())
    except OSError:
        return False


def is_model_loaded(pred_name: str, model_name: str) -> bool:
    return (pred_name, model_name) in _LOADED


def is_any_model_loaded(pred_name: str) -> bool:
    """True when any model for this predictor is loaded in this process.

    Load state is process-global (the predictor cache), so a freshly
    constructed adapter can report ``ready`` after a sibling instance loaded it.
    """
    return any(pred == pred_name for pred, _ in _LOADED)


def get_load_error(pred_name: str, model_name: str) -> str | None:
    """Sanitized load-error if a previous load failed (no stack trace)."""
    return _LOAD_ERRORS.get((pred_name, model_name))


def component_status(pred_name: str, model_name: str) -> str:
    if not is_available():
        return 'unavailable'
    if is_model_loaded(pred_name, model_name):
        return 'ready'
    if get_load_error(pred_name, model_name):
        return 'load_failed'
    if not _settings().allow_model_download and not is_model_present(model_name):
        return 'unavailable'
    return 'not_loaded'


def get_model_meta(pred_name: str, model_name: str) -> dict | None:
    """Cached model metadata (name + local path) if loaded."""
    return _MODEL_META.get((pred_name, model_name))


def _freeze_dict(d: dict) -> tuple:
    return tuple(sorted((k, v) for k, v in d.items()))


def _make_key(name: str, model_name: str, kwargs: dict) -> tuple[str, str, tuple]:
    kw = _freeze_dict(kwargs) if isinstance(kwargs, dict) else ()
    return (name, model_name, kw)


def _get_or_create_predictor(pred_name: str, model_name: str, init_kwargs: dict) -> Any:
    remaining_seconds()
    _ensure_paddle_env()
    key = _make_key(pred_name, model_name, init_kwargs)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    with deadline_lock(_lock):
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        meta_key = (pred_name, model_name)
        if meta_key in _LOAD_ERRORS:
            from app.services.errors import MODEL_UNAVAILABLE, PipelineError

            raise PipelineError(MODEL_UNAVAILABLE, _LOAD_ERRORS[meta_key], component=pred_name)
        if not _settings().allow_model_download and not is_model_present(model_name):
            from app.services.errors import MODEL_UNAVAILABLE, PipelineError

            msg = (
                f"Model '{model_name}' not found in {_cache_home()} and "
                "ALLOW_MODEL_DOWNLOAD=false. Set ALLOW_MODEL_DOWNLOAD=true or pre-populate MODELS_DIR."
            )
            _LOAD_ERRORS[meta_key] = msg
            raise PipelineError(MODEL_UNAVAILABLE, msg, component=pred_name)
        try:
            import paddleocr
            remaining_seconds()
            cls = getattr(paddleocr, pred_name)
            inst = cls(model_name=model_name, **init_kwargs)
        except Exception as exc:  # noqa: BLE001
            from app.services.errors import PipelineError, OCR_TIMEOUT
            if isinstance(exc, PipelineError) and exc.code == OCR_TIMEOUT:
                raise
            _LOAD_ERRORS[meta_key] = f"{type(exc).__name__}: {exc}"
            logger.warning("Paddle %s load failed for %s: %s", pred_name, model_name, exc)
            raise
        _CACHE[key] = inst
        _LOADED.add(meta_key)
        _MODEL_META[meta_key] = {"model_name": model_name, "predictor": pred_name, "path": str(model_dir(model_name))}
        if pred_name == "TextRecognition":
            from app.services.errors import PipelineError
            try:
                validate_recognition_language(inst, "vie")
                _MODEL_META[meta_key]["vietnamese_supported"] = True
            except PipelineError:
                _MODEL_META[meta_key]["vietnamese_supported"] = False
        remaining_seconds()
        return inst


def get_detector_model(**kwargs: float | int | str | None) -> Any:
    """Get or create a TextDetection predictor (cached)."""
    from app.services.paddle_models import resolve_det_model

    settings = _settings()
    options = {"device": settings.device, "cpu_threads": settings.cpu_threads,
               "limit_side_len": settings.detector_limit_side_len, "limit_type": "max", **kwargs}
    return _get_or_create_predictor("TextDetection", resolve_det_model(None), options)


def get_recognition_model(lang: str, **kwargs: float | int | str | None) -> Any:
    from app.services.paddle_models import resolve_rec_model

    rec_name = resolve_rec_model(lang)
    if rec_name is None:
        raise ValueError(f"Unsupported language for PP-OCRv5 recognition: {lang!r}")
    options = {"device": _settings().device, "cpu_threads": _settings().cpu_threads, **kwargs}
    predictor = _get_or_create_predictor("TextRecognition", rec_name, options)
    validate_recognition_language(predictor, lang)
    return predictor


def validate_recognition_language(predictor: Any, lang: str) -> None:
    from app.services.paddle_models import VIETNAMESE_CHARACTERS, is_vietnamese_lang
    from app.services.errors import MODEL_LANGUAGE_UNSUPPORTED, PipelineError
    if not is_vietnamese_lang(lang):
        return
    decoder = getattr(getattr(predictor, "paddlex_predictor", None), "post_op", None)
    vocabulary = set(getattr(decoder, "character", ()))
    missing = sorted(char for char in VIETNAMESE_CHARACTERS
                     if char not in vocabulary and not all(
                         part in vocabulary for part in unicodedata.normalize("NFD", char)))
    if missing:
        raise PipelineError(
            MODEL_LANGUAGE_UNSUPPORTED,
            f"Paddle model decoder lacks {len(missing)} Vietnamese characters "
            f"({''.join(missing[:12])}). Use paddle_tesseract or tesseract.",
            component="paddle_recognizer",
        )


def predict(predictor: Any, image: Any) -> list:
    # Paddle returns lazy generators: consume inside the lock as well.
    with deadline_lock(_predict_lock):
        # ponytail: native inference cannot be interrupted safely; process workers if hard cancellation is required.
        return list(predictor.predict(image))


def reset_cache() -> None:
    """Clear predictor cache (for testing)."""
    with _lock:
        _CACHE.clear()
        _LOADED.clear()
        _LOAD_ERRORS.clear()
        _MODEL_META.clear()
