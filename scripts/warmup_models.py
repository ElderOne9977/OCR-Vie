#!/usr/bin/env python3
"""Verify/warm the configured Scene pipeline in this diagnostic process.

Default Vietnamese auto loads only PaddleDetector, verifies Tesseract, and
prints environment/model metadata. This separate process does not warm a
running Uvicorn server. Explicit incompatible recognizers exit non-zero.

Run inside the Python 3.11 environment:
    .venv311\\Scripts\\python scripts\\warmup_models.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _version(mod_name: str) -> str:
    try:
        mod = __import__(mod_name)
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__})"
    return str(getattr(mod, "__version__", "unknown"))


def _print_env() -> None:
    from app.core.config import get_settings
    from app.services import paddle_runtime

    settings = get_settings()
    paddle_runtime._ensure_paddle_env()
    print("=== Environment ===")
    print(f"python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"paddlepaddle: {_version('paddle')}")
    print(f"paddleocr   : {_version('paddleocr')}")
    print(f"device      : {settings.device}")
    print(f"MODELS_DIR  : {settings.models_dir!r} (env={os.environ.get('MODELS_DIR')!r})")
    print(f"cache_home  : {paddle_runtime._cache_home()}")
    print(f"ALLOW_MODEL_DOWNLOAD : {settings.allow_model_download}")
    print(f"paddle dependency present : {paddle_runtime.is_available()}")
    print()


def _report_model(label: str, pred_name: str, model_name: str) -> bool:
    from app.services import paddle_runtime

    present = paddle_runtime.is_model_present(model_name)
    loaded = paddle_runtime.is_model_loaded(pred_name, model_name)
    meta = paddle_runtime.get_model_meta(pred_name, model_name)
    err = paddle_runtime.get_load_error(pred_name, model_name)
    print(f"--- {label} ---")
    print(f"model      : {model_name}")
    print(f"present    : {present} ({paddle_runtime.model_dir(model_name)})")
    print(f"loaded     : {loaded}")
    if meta:
        print(f"meta       : {meta}")
    if err:
        print(f"load_error : {err}")
    print()
    return loaded


def main() -> int:
    _print_env()

    from app.detectors.paddle_detector import PaddleDetector
    from app.pipelines.registry import get_recognizer
    from app.core.config import get_settings
    from app.services.errors import PipelineError
    from app.services.paddle_models import resolve_det_model, resolve_rec_model

    det_model = resolve_det_model(None)
    rec_model = resolve_rec_model("vie+eng") or ""

    engine = get_settings().scene_recognizer.lower()
    if engine in ("auto", ""):
        engine = "paddle_tesseract"
    detector = PaddleDetector()
    recognizer = get_recognizer("tesseract" if engine == "paddle_tesseract" else engine)
    print(f"before: detector={detector.status()} recognizer={recognizer.status()}")

    t0 = time.perf_counter()
    det_ready = detector.ensure_ready() if engine not in ("tesseract", "tesseract_vie_eng") else True
    det_ms = round((time.perf_counter() - t0) * 1000, 2)
    print(f"after detector : status={detector.status()} ready={det_ready} load_ms={det_ms}")

    t0 = time.perf_counter()
    try:
        rec_ready = recognizer.ensure_ready("vie+eng") if recognizer.name == "paddle" else recognizer.status() == "ready"
    except PipelineError as error:
        print(f"Recognizer rejected: {error.code}: {error.message}", file=sys.stderr)
        rec_ready = False
    rec_ms = round((time.perf_counter() - t0) * 1000, 2)
    print(f"after recognizer: status={recognizer.status()} ready={rec_ready} load_ms={rec_ms}")
    print()

    ok_det = _report_model("Detector", "TextDetection", det_model) if engine not in ("tesseract", "tesseract_vie_eng") else True
    ok_rec = _report_model("Recognizer", "TextRecognition", rec_model) if recognizer.name == "paddle" else rec_ready

    if det_ready and rec_ready and ok_det and ok_rec:
        print(f"WARMUP OK: {engine} READY in this process.")
        return 0
    print("WARMUP FAILED: see load_error above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
