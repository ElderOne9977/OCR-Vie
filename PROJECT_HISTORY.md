# OCR-Vie — Vietnamese OCR Project — Full History

> Cập nhật 2026-10-05: Scene auto tiếng Việt dùng detector Paddle + Tesseract
> batch; Document grayscale/Tesseract. Model Latin thiếu charset bị từ chối.
> Đã nối cấu hình tài nguyên, timeout, giới hạn ảnh/vùng, khóa native predictors
> và health nhẹ. Xem OCR_PERFORMANCE_FIX.md; các mục frozen/Paddle-first bên
> dưới là snapshot lịch sử, không phải chính sách runtime hiện hành.

> Cập nhật chất lượng 2026-10-04: document auto hiện dùng Tesseract cố định;
> explicit Paddle phải detect/crop từng vùng. Có thêm hybrid `paddle_tesseract`.
> Benchmark và synthetic geometry đã sửa, bộ mới `benchmark_data_quality_v2/`.
> Xem `OCR_QUALITY_FIX.md`; các số liệu và chính sách legacy-only bên dưới là snapshot trước sửa.

Chi tiết từng phase nằm trong `PROJECT_LOG.md`. File này giữ các snapshot;
trạng thái runtime mới nhất nằm trong README và OCR_PERFORMANCE_FIX.md.

## 1. Mục tiêu

API OCR tiếng Việt + tiếng Anh, local/offline, Docker-packable. Scene primary = Paddle text detection + PP-OCRv5 Latin. Tesseract = legacy / benchmark baseline / explicit opt-in fallback.

## 2. Kiến trúc hiện tại

- Thin FastAPI `app/main.py`: `/` HTML, `/api` ping, mounts `app/api/health.py` + `app/api/ocr.py`.
- Hybrid `/v1/ocr` (`route=auto|scene|document`, `recognizer=auto|paddle_v5_latin|tesseract`).
- Legacy `/ocr`, `/ocr/data`, `/health` giữ nguyên contract Tesseract.
- Scene pipeline (frozen): `PaddleDetector` (`PP-OCRv5_server_det`) → `crop_polygon` → `PaddleRecognizer` (`latin_PP-OCRv5_mobile_rec`) → NFKC deterministic-v1.
- Tesseract không chạy trừ khi `recognizer=tesseract` hoặc `ENABLE_TESSERACT_FALLBACK=true`.
- Health: `/health/live` process-up; `/health/ready` dựa Paddle (Tesseract một mình không làm `ready`).
- Runtime canonical: Python 3.11 (`.venv311`) paddlepaddle 3.2.0 + paddleocr 3.7.0. Python 3.14 = fast unit tests, không có wheel Paddle.

## 3. Các phase (tóm tắt)

1. Tesseract-only FastAPI + preprocessing + Docker skeleton + UI.
2. Hybrid router (classifier → scene/document). Paddle/VietOCR/SVTR skeletons `MODEL_UNAVAILABLE`.
3. Real PP-OCRv5 Scene OCR trên `.venv311`. Weights `~/.paddlex/official_models/{PP-OCRv5_server_det,latin_PP-OCRv5_mobile_rec}`.
4. Engine policy: Paddle primary, Tesseract legacy. Provenance + `fallback_used` nhất quán.
5. Phase 3 freeze production Scene. Benchmark framework (CER/WER/exact/accent_stripped_cer/diacritic_loss_rate, rec-only vs scene, never-overwrite outputs).
6. Phase 3.1: dataset schema + privacy gate + annotator + filter CLI `--source/--split/--reviewed-only`.
7. Phase 3.2: coverage flags SEVERE n<5 / LOW n<10 (advisory) + freeze gate ≥100 real reviewed test (`freeze_testset.py`).

## 4. Benchmark (Phase 3)

- Dataset: `benchmark_data/` v0.3.1 — 12 synthetic smoke, real=0, reviewed_real=0.
- Gate: `MIN_REAL_TEST_REVIEWED=100`. Freeze `benchmark_data/frozen/testset_v<version>.json` (ids+sha256 only).
- Official run chỉ khi freeze + `--split test --source real --reviewed-only`. Dưới 100 = `PRELIMINARY NOT REPRESENTATIVE`.
- Aggregation per-category/tag/error-bucket/review_queue **chưa** mở rộng (chờ real data >100).
- Không cài VietOCR / SVTR / MonkeyOCR / PaddleOCR-VL.

## 5. Vận hành

```
.venv311\Scripts\python -m pytest -m "not integration" -q
.venv311\Scripts\python -m pytest -m "integration and paddle" -v
.venv311\Scripts\python benchmarks/validate_dataset.py --data benchmark_data
.venv311\Scripts\python benchmarks/dataset_summary.py --data benchmark_data
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 6. Trạng thái hiện tại (2026-10-04)

- Tests: **105 passed** (102 non-integration + 3 paddle integration). `pip check` clean.
- Production OCR: frozen. API smoke `/v1/ocr?route=scene&recognizer=auto` → detector=`paddle_detector` recognizer=`paddle_v5_latin` fallback_used=false. Explicit tesseract vẫn 200.
- Dataset: total=12 synthetic=12 real=0. `Real reviewed test: 0 / 100`. Smoke CER rec paddle 0.134 / tesseract 0.176 / scene F1 0.917 — **không đại diện**.
- Workspace: không phải git repo (standalone copy, `git_hash=null`).
- Blocked: thu thập ≥100 real reviewed `split=test` qua `benchmark_incoming/` → import `--privacy safe` → annotate → validate/audit → freeze v0.1.

<!-- ponytail: chi tiết commit log từng patch đã gộp; tách lại khi cần audit từng diff -->
