# OCR Performance Implementation Plan

**Goal:** Cải thiện OCR tiếng Việt, giảm chi phí hybrid và giới hạn tài nguyên bằng các dependency hiện có.

**Architecture:** Auto scene tiếng Việt dùng Paddle detector + Tesseract; document dùng Tesseract. Batch crop dùng danh sách ảnh native của Tesseract, giữ ánh xạ page/region. Giới hạn request và kích thước ảnh ở các entry point dùng chung.

**Tech Stack:** Python 3.11, FastAPI, Pillow, OpenCV, pytesseract, PaddleOCR; threading/tempfile/stdlib.

## Constraints

- Không thêm dependency/model mới, không sửa dictionary của model.
- Explicit Paddle không fallback ngầm; model thiếu charset trả lỗi rõ ràng.
- Giữ raw text, confidence, bbox, thứ tự và vùng trắng trong batch.
- Giữ vie+eng mặc định để không làm giảm hỗ trợ song ngữ; cho phép profile vie và grayscale.
- Thực hiện inline; thư mục hiện tại không phải Git repository nên không tạo commit.

## Task 1: Resource bounds and runtime settings

Files: `app/core/config.py`, `app/preprocessing/common.py`, `app/image_processing.py`, `app/api/ocr.py`, `app/pipelines/router.py`, `app/services/paddle_runtime.py`, `app/detectors/paddle_detector.py`, `app/recognizers/paddle_recognizer.py`, `app/services/limits.py`; tests in `tests/test_performance_regressions.py`.

- [x] Write failing tests: MAX_UPLOAD_BYTES limits read to limit+1; MAX_IMAGE_PIXELS rejects image headers before imdecode; concurrent OCR returns 503 and releases slot after exceptions; DEVICE/CPU_THREADS reach model construction; shared predictor calls serialize including generator consumption.
- [x] Run `.venv311/Scripts/python -m pytest tests/test_performance_regressions.py -q`; confirm expected failures.
- [x] Implement Settings limits (positive validated), bounded upload read and Pillow header check before OpenCV decode. Delegate legacy decode to shared helper. Add shared BoundedSemaphore context with locked initialization around router processing and legacy endpoints; reject excess requests immediately. Pass runtime settings into predictor cache construction; serialize prediction with lock.
- [x] Repeat tests and existing validation/mock checks.

## Task 2: Vietnamese model selection and document detail preservation

Files: router/runtime/model helpers, `app/preprocessing/document.py`, `app/api/health.py`, UI/API in `app/main.py`; policy/quality/integration tests.

- [x] Add failures asserting `scene+auto+vie` chooses Tesseract crops; non-Vietnamese auto retains Paddle; incomplete decoder rejects Vietnamese before recognition. Assert grayscale retains original gray values and binary remains opt-in.
- [x] Implement language-aware auto, validate loaded decoder against NFC Vietnamese characters and decomposed alternatives. Explicit incompatible model reports MODEL_LANGUAGE_UNSUPPORTED (503). Add DOCUMENT_PREPROCESS=gray|binary|original (gray default), preserve language default.
- [x] Update readiness/UI/docs to describe the actual default and dependencies, test health does not label hybrid ready without Tesseract.
- [x] Run policy/quality tests and real incompatible-model integration.

## Task 3: Tesseract crop batching

Files: `app/recognizers/tesseract_recognizer.py`, router, legacy service timeout handling, tests and `scripts/verify_ocr_quality.py`.

- [x] Add failing batch checks for blank middle page, per-page confidence, RGB conversion, one invocation, cleanup, timeout and mismatched page count. Add router check for preserving region order and failing visibly on batch errors.
- [x] Implement `recognize_many(images, *, lang, psm)` in bounded chunks: TemporaryDirectory PNGs + native image-list input; group TSV by page_num, including header-only pages; reuse existing TSV-to-result parser. Pass OCR_TIMEOUT_SECONDS to all OCR subprocess invocations; propagate timeout as 504.
- [x] Router uses batch only for hybrid; cap region count, never silently truncate. Existing non-batch adapters retain per-region route.
- [x] Run complete pytest, real single-vs-batch equivalence including blank crops, original A4 regression, and 12-line warm latency/RAM comparison. Do not claim synthetic CER generalizes.

## Completion

- [x] Save measured before/after evidence and reproduction command beside `OCR_PERFORMANCE_ANALYSIS.md`.
- [x] Document limits per process, timeout scope, cold-start cost and remaining data needs.
