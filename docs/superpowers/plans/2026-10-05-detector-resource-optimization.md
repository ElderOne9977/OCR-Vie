# Detector/resource optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the measured detector tradeoff and bound crop allocation/request work without changing default OCR quality.

**Architecture:** Use Paddle's native maximum-side option, a crop pixel budget checked before allocation, and a ContextVar deadline shared by the existing admission context. Native Paddle remains serialized; timeouts are cooperative outside its running call.

**Tech Stack:** Existing Python, FastAPI, OpenCV, PaddleOCR, pytest; no new dependencies.

## Global Constraints

- Default detector maximum side remains 960 px; preserve MKL-DNN and Vietnamese charset gate.
- Production timeout per Tesseract call remains 30 seconds; new cooperative request budget defaults to 60 seconds and begins after admission, before decode.
- No new model downloads, auto quality switching, persistent OCR cache, worker pool, or Git commits (workspace is not a Git checkout).
- Execute inline as requested; optional executing-plans skill is not available in this session.

### Task 1: Native detector configuration

**Files:** `app/core/config.py`, `app/services/paddle_runtime.py`, `tests/test_detector_optimization.py`, `app/api/health.py`, `README.md`.
**Interface:** `Settings.detector_limit_side_len: int = 960`, env `DETECTOR_LIMIT_SIDE_LEN`; range 32..4000. Runtime forwards `limit_side_len` and `limit_type="max"` to the existing predictor constructor.

- [x] Red: assert env 768 reaches the native constructor and reject 0/31/4001 with `ValueError`; run `.venv311\Scripts\python -m pytest tests/test_detector_optimization.py -q`.
- [x] Green: add validated field/env binding and constructor kwargs; same command must pass.
- [x] Expose effective side in readiness and document 768/960/1280 choices; do not resize recognition image.

### Task 2: Preallocation crop budget

**Files:** `app/services/polygon_utils.py`, `app/pipelines/router.py`, `tests/test_detector_optimization.py`.
**Interface:** `crop_polygon(image, polygon, *, max_pixels: int | None = None)` raises HTTP 413 before copying/warping a crop exceeding budget.

- [x] Red: call `crop_polygon(..., max_pixels=99)` for a 10×10 quad; assert 413 and no `cv2.warpPerspective` invocation. Cover triangle bbox, exact-budget success, and router cumulative budget.
- [x] Green: compare exact calculated output area before `.copy()`/warp; pass remaining budget from router. Preserve interpolation, coordinates and degenerate handling. Run focused tests and polygon tests.

### Task 3: Cooperative request deadline

**Files:** `app/services/deadline.py`, `app/services/limits.py`, `app/core/config.py`, `app/ocr_service.py`, `app/services/paddle_runtime.py`, `app/main.py`, Paddle adapters, `tests/test_detector_optimization.py`.
**Interface:** `remaining_seconds(limit=None)` returns active deadline budget capped by optional limit, raises `OCR_TIMEOUT`/504 when expired; without a deadline returns the limit. `ocr_deadline()` initializes/reset context. Existing `ocr_slot()` encloses work in it.

- [x] Red: fake monotonic clock, run two Tesseract batch calls inside `ocr_slot()`, verify remaining budget shrinks rather than resets; expired budget must not start another native operation and contexts/slots must be released.
- [x] Red: predictor lock acquisition obeys deadline, and a native generator finishing after deadline raises 504 rather than returning success. Verify legacy endpoint maps deadline errors to 504.
- [x] Green: implement ContextVar deadline with monotonic time, cap Tesseract timeout, bound lock acquisition and check before/after native work. Map legacy decorator boundary errors via a PipelineError exception handler. No thread-based cancellation.

### Task 4: Verify and handoff

- [x] Run `.venv311\Scripts\python -m pytest -q` and `.venv311\Scripts\python -m compileall -q app scripts tests` — 189 passed.
- [x] Run `scripts/verify_ocr_quality.py` and performance verifier in fresh processes with detector side 960 and 768; report synthetic CER, latency/RAM and timeout limitations.
- [x] Update README, review report and project log with effective options, completed tests and server restart instructions.
