# PROJECT_LOG — Vietnamese OCR API

> Tài liệu lưu trữ toàn bộ chi tiết dự án và quá trình phát triển từ đầu.

## Cập nhật 2026-10-07 — Thực hiện kế hoạch audit

Sửa crop suy biến/45°, reading_order, TSV cấp phát O(n²), annotator và manifest
v2; scene benchmark gọi pipeline sản phẩm, field mapping/IoU/hash rõ ràng.
Thêm cap body trước multipart, lỗi an toàn/request ID, readiness đúng runtime và
traineddata, chuẩn `.venv311`/OpenCV, warmup trong server, strict smoke. Smoke thật
phát hiện mất dấu ở crop sát mép, sửa border 2 px riêng Tesseract PSM7 chung single/batch.
UI busy/timeout giữ ảnh; browser đã kiểm tra copy/TXT/mobile. README/DEMO hiện hành
tách lịch sử; bỏ ModelRegistry và patch_benchmarks không caller, snapshot có thể phục hồi.

253 tests passed; base Python 3.11 mới không cài Paddle đạt 249 unit/mock tests.
Pip check/environment/dataset validation đạt. TSV 8k từ
median 433,335→17,776 ms, output/confidence tương đương. Soak scene 300,65 s:
456 request đúng fixture, không lỗi; không dùng synthetic làm bằng chứng chất
lượng thực tế. Không đổi threads/side/batch/PSM defaults; không thêm queue/model mới.

Chưa có 100 real reviewed test, Docker daemon chưa chạy, CI và một
phần browser E2E chưa nghiệm thu. Server kiểm thử 8017 đã dừng, không triển khai public.
Bằng chứng và trạng thái chi tiết: [IMPLEMENTATION_RESULTS_2026-10-07.md](docs/IMPLEMENTATION_RESULTS_2026-10-07.md).

## Cập nhật 2026-10-07 — UX/UI workspace OCR

Áp dụng `redesign-existing-projects` và `minimalist-ui`: hai vùng ảnh nguồn / kết quả,
nền trắng ngà, màu trạng thái nhẹ, mobile một cột, focus bàn phím và reduced motion.
Tách trang sang `app/ui.html`, giữ `app.main.INDEX_HTML`, re-export và toàn bộ API/OCR.
Đưa engine vào mục nâng cao; khóa form khi chạy, chặn gửi trùng, tắt xuất khi
văn bản rỗng/lỗi, xóa kết quả khi đổi ảnh và thu hồi URL ảnh cũ. Giới hạn upload
trên giao diện lấy từ cấu hình server; lỗi/cảnh báo tiếng Việt, JSON chỉ trong details.

Kiểm chứng: 191 tests passed (có regression JavaScript chạy bằng Node nếu được cài),
compileall đạt; trình duyệt kiểm tra 1280/390 px không tràn ngang. OCR scene thật
trả 3 vùng; mẫu tài liệu trả đúng “Xin chào Việt Nam”; explicit Latin trả thông báo
model không đủ ký tự và khóa xuất. File tải xuống chỉ chứa văn bản. Clipboard có
kiểm thử thành công/thất bại; browser báo đã sao chép nhưng API clipboard tích hợp
không đọc lại được nội dung, nên chưa xác minh clipboard hệ điều hành end-to-end.
Không cài dependency, tải model hoặc thay đổi engine. Server kiểm thử đã dừng.
Ảnh xác minh trong `artifacts/ui-result.jpg` và `artifacts/ui-mobile.jpg`.

## Cập nhật 2026-10-05 — tối ưu detector và ngân sách request

Thêm `DETECTOR_LIMIT_SIDE_LEN` (mặc định 960; profile 768/1280 tùy chọn), giữ
ảnh gốc khi crop và MKL-DNN. Tổng crop neural được chặn trước cấp phát copy/warp.
Thêm `OCR_REQUEST_TIMEOUT_SECONDS=60` dạng cooperative deadline; Tesseract batch
và chờ khóa Paddle dùng thời gian còn lại. Hết hạn trả 504, không ghi timeout
thành lỗi tải model. Native Paddle đang chạy không bị ngắt cưỡng bức.
Đo cả request synthetic 12 dòng: median nóng 960/768 là 1,43/1,26 giây,
RSS đỉnh median 875/765 MiB, CER/WER 0; không tự bật profile nhanh.
Chi tiết đo, giới hạn và cách bật trong OCR_OPTIMIZATION_REVIEW.md.
Kiểm chứng: 189 tests passed, compileall và A4 cold/warm/hybrid đạt; không tải
model hay cài dependency mới. Cần khởi động lại server để áp dụng code/env mới.

## Cập nhật 2026-10-05 — charset, batch và giới hạn tài nguyên

Scene auto tiếng Việt chuyển sang Paddle detector + Tesseract PSM 7 theo lô.
Model Latin cục bộ thiếu 90/146 ký tự Việt; explicit model không tương thích
trả 503 MODEL_LANGUAGE_UNSUPPORTED kể cả sau warmup. Document giữ grayscale
mặc định, cho phép binary/original. Truyền DEVICE/CPU_THREADS vào SDK, giới hạn
request/pixel/region/upload, timeout Tesseract, serialize shared predictors.
Health không import Paddle; UI có chọn vie/vie+eng/eng. Chi tiết và số đo trong
OCR_PERFORMANCE_FIX.md; snapshot Paddle-first phía dưới đã được thay thế.
Kiểm chứng cuối: 160 tests passed; compileall và A4 cold/warm đạt. Hybrid
synthetic 12 dòng warm ~4,64 s → ~1,6–1,8 s, CER 0; RAM peak vẫn ~900 MiB.

## Cập nhật 2026-10-04 — sửa routing và benchmark chất lượng

Đã loại lỗi Document phụ thuộc trạng thái cache: auto dùng Tesseract cố định qua
`OCR_DOCUMENT_RECOGNIZER`; explicit Paddle dùng detector/crop thay vì nhận nguyên
trang. Thêm `paddle_tesseract` (Paddle detection + Tesseract PSM 7) vào UI/API.
Tesseract adapter chạy một lần, confidence chuẩn 0–1 và giữ raw score. Benchmark
sửa GT geometry/seed, missing text, page WER và recognition confidence; dataset
`benchmark_data_quality_v2` sinh riêng, dữ liệu cũ được giữ nguyên.

Kiểm chứng A4 cold/warm cùng text, CER 0 trên mẫu tái hiện; hybrid 4 vùng, CER 0.
133 tests passed (gồm 3 Paddle integration) tại lần kiểm tra đầu sau sửa.
Xem `OCR_QUALITY_FIX.md` và phần đầu README cho chính sách hiện hành; các mục
Paddle-first/legacy-only bên dưới là lịch sử trước bản sửa này.

## 1. Thông tin tổng quan

| Thuộc tính | Giá trị |
|---|---|
| Tên dự án | Vietnamese OCR API |
| Mục tiêu | Nhận dạng chữ tiếng Việt + tiếng Anh, chạy local/offline, đóng gói được bằng Docker |
| Stack | FastAPI + Uvicorn + Tesseract OCR + pytesseract + OpenCV + Pillow |
| Scene primary | PaddleDetector + PP-OCRv5 Latin (frozen production pipeline) |\n| Tesseract role | legacy/benchmark/opt-in fallback only |\n| Language default | `vie+eng` (Tesseract & Paddle) |\n| Upload limit | 10 MB |\n| Image formats | JPEG,PNG,WebP,BMP,TIFF |\n-| Runtime Python | 3.11 trong Docker; 3.14.6 trên máy phát triển Windows |
| Runtime Python | 3.11 trong Docker; 3.14.6 dev (unit fast) |\n| Runtime canonical | `.venv311`: Python 3.11.7 + paddlepaddle==3.2.0 + paddleocr==3.7.0 |
| Ngôn ngữ OCR mặc định | `vie+eng` |
| Port | `8000` |
| Giới hạn upload | 10 MB |
| Định dạng ảnh chấp nhận | JPEG, PNG, WebP, BMP, TIFF |
| Runtime Python | 3.11 trong Docker; 3.14.6 trên máy phát triển Windows |
| Runtime canonical | Python 3.11.7 (`.venv311`) paddlepaddle 3.2.0 + paddleocr 3.7.0 |

## 2. Mục tiêu thiết kế

1. Đơn giản, dễ debug, ít file, logic rõ ràng, không auth/db/queue.
2. Chạy offline hoàn toàn: Docker image chứa Tesseract + vie + eng.
3. Dễ test: unit tests mock toàn bộ Tesseract, không cần binary.
4. Bảo toàn dấu tiếng Việt: preprocessing nhẹ, không morphology mạnh.
5. Trải nghiệm người dùng tốt: có UI upload trực tiếp tại `/`, không bắt buộc dùng curl.

## 3. Kiến trúc hệ thống

```
Client (Browser / curl / mobile)
  ↓
FastAPI (app/main.py)
  ↓
Validation (MIME, size, PSM, corrupted image)
  ↓
Image preprocessing (OpenCV)
  ↓
pytesseract (wrapper)
  ↓
Tesseract OCR binary
  ↓
Trained data: vie + eng
  ↓
JSON response (text hoặc word-level bbox)
```

Lưu ý: `pytesseract` chỉ là wrapper gọi binary Tesseract. Engine OCR thật sự là Tesseract, phải có sẵn trong môi trường chạy.

## 4. Cấu trúc project

```
OCR-Vie/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI endpoints + UI HTML
│   ├── ocr_service.py       # Tesseract wrapper, PSM validation
│   └── image_processing.py  # OpenCV decode/preprocess
├── tests/
│   ├── __init__.py
│   ├── test_validation.py   # Validation tests (MIME, size, PSM, HTML)
│   ├── test_ocr_mock.py     # Mock OCR tests
│   └── test_preprocessing.py # Preprocessing tests
├── scripts/
│   ├── check_environment.py # Kiểm tra Python libs + Tesseract
│   ├── smoke_test.py        # Test API đang chạy
│   ├── run_local.ps1        # PowerShell: cài deps + chạy server
│   └── test.ps1             # PowerShell: cài test deps + chạy pytest
├── requirements.txt         # Dependencies runtime (pinned)
├── requirements-dev.txt     # + pytest
├── Dockerfile               # python:3.11-slim + tesseract + vie + eng
├── .dockerignore            # Loại venv, cache, IDE
├── README.md                # Hướng dẫn tiếng Việt
└── PROJECT_LOG.md           # File này
```

## 5. Chi tiết từng file

### 5.1 `app/image_processing.py`

Pipeline:
```
bytes → numpy array → cv2.imdecode (BGR) → grayscale (COLOR_BGR2GRAY) → medianBlur(3) → Otsu threshold → binary uint8
```

Lý do:
- `medianBlur(3)`: lọc nhiễu nhẹ nhưng giữ biên, tránh làm mờ dấu tiếng Việt.
- `Otsu threshold`: tự động chọn ngưỡng.
- Không deskew, không morphology mạnh.

### 5.2 `app/ocr_service.py`

Cấu hình:
```
DEFAULT_LANG = 'vie+eng'
DEFAULT_PSM = 6
VALID_PSM_MODES = {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
tesseract config = '--oem 3 --psm 6'
```

Hai hàm chính:
- `ocr_image(image, lang, psm)` → trả text đã strip.
- `ocr_image_data(image, lang, psm, min_confidence)` → trả list word với text, confidence, left, top, width, height.

Fix Tesseract PATH trên Windows:
```python
if os.getenv('TESSERACT_CMD'):
    pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD')
elif os.name == 'nt':
    _win = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.isfile(_win):
        pytesseract.pytesseract.tesseract_cmd = _win
```

### 5.3 `app/main.py`

| Method | Path | Mô tả |
|---|---|---|
| GET | `/` | Trang HTML upload ảnh (UI) |
| GET | `/api` | JSON ping |
| GET | `/health` | Tình trạng Tesseract: available, version, languages |
| POST | `/ocr` | OCR trả text |
| POST | `/ocr/data` | OCR trả word-level bbox |

UI HTML có: upload file + preview, chọn mode (text/data), chọn PSM (3/6/7/8), fetch `/health` hiển thị trạng thái, JS gọi API và hiển thị kết quả.

Validation:
- content_type thuộc `{image/jpeg, image/png, image/webp, image/bmp, image/tiff}`.
- File rỗng → 400.
- File > 10 MB → 413.
- Decode lỗi → 400.
- PSM không hợp lệ → 400.
- TesseractNotFoundError → 503.
- TesseractError khác → 500.

### 5.4 `Dockerfile`

```
FROM python:3.11-slim
RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng
COPY requirements.txt . && pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8000
HEALTHCHECK ... /health
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5.5 `requirements.txt`

```
fastapi==0.142.2
uvicorn[standard]==0.54.0
pytesseract==0.3.13
pillow==12.3.0
python-multipart==0.0.32
opencv-python-headless==5.0.0.93
numpy==2.3.5
httpx==0.28.1
```

Lý do các phiên bản:
- numpy==2.3.5: numpy 2.1.3 gây OverflowError trên Python 3.14 Windows.
- uvicorn[standard]==0.54.0 tương thích FastAPI 0.142.2.
- Pin để build reproducible, tránh source build chậm.

## 6. Quá trình phát triển theo giai đoạn

### Giai đoạn 1 — Khởi tạo
- Tạo app/, tests/, scripts/.
- app/main.py, ocr_service.py, image_processing.py.
- Endpoints: `/`, `/health`, `/ocr`, `/ocr/data`.
- OCR lang vie+eng, PSM 6.
- Preprocessing: grayscale → medianBlur(3) → Otsu.

### Giai đoạn 2 — Validation và logging
- non-image/empty/corrupted/PSM → 400; OCR fail → 500; Tesseract missing → 503.
- MAX_UPLOAD_BYTES = 10 MB → 413.
- ACCEPTED_CONTENT_TYPES.
- /health trả tesseract_available, version, languages.
- Logging.

### Giai đoạn 3 — Docker và dependencies
- Dockerfile python:3.11-slim + tesseract + vie + eng.
- .dockerignore.
- README.md tiếng Việt.
- Pin deps trong requirements.txt.
- requirements-dev.txt + pytest.
- HEALTHCHECK.

### Giai đoạn 4 — Tests
- test_validation.py.
- test_ocr_mock.py mock pytesseract.
- test_preprocessing.py.
- Sửa lỗi TesseractNotFoundError() không nhận argument.

### Giai đoạn 5 — Scripts hỗ trợ
- check_environment.py.
- smoke_test.py.
- run_local.ps1.
- test.ps1.

### Giai đoạn 6 — Dependency fix trên Windows
- pip install treo → cài từng gói + --prefer-binary.
- numpy 2.1.3 lỗi OverflowError → nâng 2.3.5.
- 13/13 pytest pass.

### Giai đoạn 7 — UI tại `/`
- Người dùng báo không hiển thị UI khi mở IP local.
- GET / trả JSON, không có HTML.
- Fix: thêm INDEX_HTML, JSON ping sang /api.
- Cập nhật tests, smoke_test, README.
- Bind 0.0.0.0 cho LAN.

### Giai đoạn 8 — Fix Tesseract binary not found
- Vấn đề: /health tesseract_available false, /ocr 503.
- Nguyên nhân: Tesseract chưa cài; winget install xong không có PATH và thiếu vie.
- Bước fix:
  1. winget install --id UB-Mannheim.TesseractOCR -e → 5.4.0.
  2. Download vie.traineddata (tessdata_fast, 531275 bytes) bằng Python urllib về %TEMP%.
  3. Copy bằng Start-Process powershell -Verb RunAs (Program Files cần admin).
  4. Thêm fallback Windows path trong app/ocr_service.py.
  5. Thêm C:\Program Files\Tesseract-OCR vào User PATH.
- /health ok với languages eng, osd, vie.

### Giai đoạn 9 — Demo end-to-end
- Tạo ảnh test bằng Pillow: Xin chào Việt Nam.
- POST /ocr?lang=vie → chính xác 100%.
- POST /ocr/data → 4 items với confidence ~95-96%.
- Smoke test pass.

## 7. Các lỗi đã gặp và cách fix

| Lỗi | Nguyên nhân | Cách fix |
|---|---|---|
| pip install treo/chậm | Mạng yếu, thiếu wheel | --prefer-binary, cài từng gói |
| OverflowError khi import numpy | numpy 2.1.3 không tương thích Python 3.14 | Nâng numpy lên 2.3.5 |
| TesseractNotFoundError() nhận argument | Constructor không có tham số | Gọi TesseractNotFoundError() không đối số |
| Không hiển thị UI tại IP local | GET / trả JSON | Thêm HTML UI tại /, JSON sang /api |
| Tesseract binary not found / 503 | Tesseract chưa cài | Cài Tesseract + language pack vie |
| Vẫn 503 sau khi cài | Installer không thêm PATH | Fallback path trong ocr_service + User PATH |
| --list-langs thiếu vie | Installer không có Vietnamese | Download vie.traineddata vào tessdata |
| Permission denied khi copy vie.traineddata | Program Files cần admin | Copy bằng PowerShell -Verb RunAs |
| Smoke test /api 404 | Server cũ chạy | Kill hết uvicorn, khởi động lại |

## 8. Kết quả hiện tại

### Unit tests
```
13 passed, 1 warning
```

### Health check
```json
{
  "status": "ok",
  "tesseract_available": true,
  "version": "5.4.0.20240606",
  "languages": ["eng", "osd", "vie"],
  "required_languages": ["vie", "eng"]
}
```

### OCR demo
```bash
curl -X POST http://127.0.0.1:8000/ocr -F file=@test.png
# {"success":true,"filename":"test.png","text":"Xin chào Việt Nam"}

curl -X POST http://127.0.0.1:8000/ocr/data -F file=@test.png
# {"success":true,"filename":"test.png","items":[{"text":"Xin","confidence":95.0,...}]}
```

### UI
Mở trình duyệt tại `http://localhost:8000` hoặc `http://<IP>:8000` để thấy form upload.

## 9. Hướng dẫn chạy demo

### 9.1 Local (Windows)

```powershell
cd 'C:\Users\PC\OneDrive\Desktop\My Project\OCR-Vie'
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install --prefer-binary -r requirements.txt
python -m pip install --prefer-binary -r requirements-dev.txt
winget install --id UB-Mannheim.TesseractOCR -e
# Tải vie.traineddata vào C:\Program Files\Tesseract-OCR\tessdata\
uvicorn app.main:app --host 0.0.0.0 --port 8000
start http://localhost:8000
```

### 9.2 Docker

```bash
docker build -t vietnamese-ocr .
docker run -p 8000:8000 --name vietnamese-ocr-test vietnamese-ocr
```

### 9.3 Kiểm tra nhanh

```bash
python scripts/check_environment.py
python -m pytest -v tests/
python scripts/smoke_test.py --base-url http://localhost:8000
```

## 10. Tính năng đã hoàn thành

- [x] FastAPI endpoints: /, /api, /health, /ocr, /ocr/data
- [x] UI HTML upload ảnh tại /
- [x] Validation MIME, size, PSM, corrupted image
- [x] Preprocessing OpenCV nhẹ
- [x] OCR vie+eng
- [x] Word-level OCR với bbox + confidence
- [x] Dockerfile + HEALTHCHECK
- [x] Unit tests (mock Tesseract)
- [x] Scripts hỗ trợ
- [x] README tiếng Việt
- [x] Fix Tesseract PATH trên Windows
- [x] Demo end-to-end thành công

## 11. Giới hạn hiện tại

- Docker daemon chưa kiểm tra trong môi trường này.
- Chưa có deskew / orientation detection.
- Chưa hỗ trợ PDF / batch upload.
- Preprocessing cố định, chưa adaptive.

## 12. Hướng mở rộng

- Deskew + orientation detection.
- Model OCR khác (VietOCR, PaddleOCR).
- PDF support.
- Batch OCR với queue.
- Frontend React/Vue/Streamlit.
- Export TXT/JSON/CSV.
- CI/CD với GitHub Actions.

## 13. Ghi chú cho người maintain

- KHÔNG xóa fallback Windows path trong app/ocr_service.py.
- LUÔN giữ preprocessing nhẹ để bảo toàn dấu tiếng Việt.
- GIỮ tests mock Tesseract để CI nhanh.
- CẬP NHẬT requirements.txt khi nâng Python (numpy đặc biệt dễ lỗi).

---


## 14. Hybrid Phase 1 (refactor scaffolding)

- Thin `app/main.py` mounts `health_router` + `ocr_router`; keeps `GET /` HTML + `GET /api`.
- Shared `app/constants.py` (`ACCEPTED_CONTENT_TYPES`, `MAX_UPLOAD_BYTES`) re-exported via `app.main` for legacy tests mocking `app.main.ocr_image`.
- Schemas: `app/schemas/ocr.py` (BBox/DetectedRegion/RecognitionResult/RegionResult/Provenance/TimingMs/PipelineResult), `app/schemas/document.py`.
- Config `app/core/config.py` Settings.from_env, registry `app/core/model_registry.py` lazy singleton.
- Services: `normalization.py` (NFKC deterministic-v1), `confidence.py`, `errors.py` (PipelineError codes).
- Classifiers: `HeuristicDocumentSceneClassifier` heuristic-v1 (pdf/aspect/pixels).
- Detectors: `FullFrameDetector` ready, `PaddleDetector` skeleton unavailable/not_implemented.
- Recognizers: `TesseractRecognizer` ready (wraps pytesseract, VALID_PSM_MODES), `Paddle/VietOCR/SVTR` skeletons MODEL_UNAVAILABLE.
- Parsers: `FallbackDocumentParser` ready, `PaddleOCR-VL`/`MonkeyOCR` skeletons.
- Preprocessing split: `preprocessing/scene.py` gray+median, `preprocessing/document.py` gray+median+Otsu; legacy `app/image_processing.py` kept for compat.
- Pipelines `registry.py` + `router.py` OCRRouter.process(route auto|scene|document, fallback settings.fallback_route, timing_ms, Provenance, warnings).
- API `app/api/health.py` (/health compat, /health/live, /health/ready degraded-ok), `app/api/ocr.py` (/ocr,/ocr/data unchanged, /v1/ocr hybrid) with shim for `app.main.ocr_image` mocks.
- Benchmarks: `benchmarks/metrics.py` (CER/WER word-level lev, exact), `dataset.py` (data/ layout), `run_recognition_benchmark.py`; `data/README.md`; `tests/test_router.py`, `test_raw_preservation.py`, `test_classifier_fallback.py` — 18 passed.
- READY vs skeleton: ready = Tesseract, heuristic classifier, full_frame, fallback parser, deterministic normalizer; skeleton = Paddle/VietOCR/SVTR/MonkeyOCR/PaddleOCR-VL (log MODEL_UNAVAILABLE, not fake READY).

## 15. Phase 2 — Real Paddle PP-OCRv5 Scene OCR (2026-10-02)

*Scope:* Real neural scene `detector -> crop_polygon -> recognizer` alongside unchanged Tesseract path. No DB/auth, no hardcoded Windows `/models`, no fake `READY`.

*Runtime:* Canonical `Python 3.11` (`.venv311`) — `paddlepaddle==3.2.0 + paddleocr==3.7.0 + paddlex==3.7.2`; `requirements-paddle.txt` pins the trio. Python 3.14 has no `paddlepaddle` cp314 wheel (kept for fast pre-validation). All Paddle imports lazy; absent dep => `status="unavailable"`, never fake READY.

*Verified API (3.7.0):* `TextDetection(model_name="PP-OCRv5_server_det").predict(bgr) -> [{dt_polys:Nx4x2, dt_scores:N}]`; `TextRecognition(model_name="latin_PP-OCRv5_mobile_rec").predict(crop) -> [{rec_text:[str], rec_score:[float]}]` — confirmed on `data/sample_scene.png` (detected polylines + non-empty `rec_text`).

*Model location & policy:* PaddleX caches under `PADDLE_PDX_CACHE_HOME` (default `~/.paddlex`) at `official_models/<model_name>`; presence check gates construction — `ALLOW_MODEL_DOWNLOAD=false` + absent weights => `PipelineError(MODEL_UNAVAILABLE,503)` with no download, health never downloads. Resolution order: 1) `PADDLE_PDX_CACHE_HOME` (explicit, wins) 2) `MODELS_DIR` (when set) 3) framework `~/.paddlex` (holds real weights on dev). Real weights verified at `C:\Users\PC\.paddlex\official_models\{PP-OCRv5_server_det, latin_PP-OCRv5_mobile_rec}`. `MODELS_DIR=/models` only in Docker; no Windows hardcode.

*Changes:* `app/schemas/ocr.py` ComponentStatus += `loading|load_failed`; `app/core/config.py` adds `models_dir` + `allow_model_download` from `ALLOW_MODEL_DOWNLOAD`; `app/services/paddle_runtime.py` rewritten (no `disable_download` kwarg — was invalid): `_ensure_paddle_env()` before `import paddleocr`, `is_model_present()`, `_LOADED/_LOAD_ERRORS/_MODEL_META`, `get_load_error/get_model_meta`; `app/detectors/paddle_detector.py` + `app/recognizers/paddle_recognizer.py` init `not_loaded`, `ensure_ready()` lazy `LOADING->READY|LOAD_FAILED`; `app/pipelines/registry.py` recognizer aliases `paddle_v5_latin|paddle_v5|pp-ocrv5|latin_*`; `app/pipelines/router.py` neural-path: when `recognizer in (paddle, paddle_v5_latin, paddle_v5)` OR `scene_detector==paddle`, eagerly `ensure_ready()` both components, per-region `crop_polygon` + `rec.recognize(crop)` (BGR, no Tesseract preprocessing), explicit paddle unavailable => raise `MODEL_UNAVAILABLE` no fallback, `auto` may fall back to Tesseract; `app/api/health.py` `GET /health/ready` adds `paddle.{dependency_available, cache_home, detector/recognizer models, present flags}` (no load); `scripts/warmup_models.py` new (versions, device, MODELS_DIR, ALLOW_MODEL_DOWNLOAD, before/after status, meta, load_ms); `scripts/check_environment.py` extended (paddle versions, cache_home, present vs READY); `pytest.ini` `integration,paddle` markers; `tests/integration/test_paddle_scene_ocr.py` (real detector/recognizer/API, `skipif` no paddle); `.gitignore`, `.dockerignore` (`.venv311`, `.paddlex`, `models/`), `Dockerfile` `libgl1 libglib2.0-0 + ALLOW_MODEL_DOWNLOAD=false`.

*Verification (GREEN 2026-10-02 on .venv311):* `python -m py_compile` all touched files; `pytest -m "not integration"` 31 passed on Py3.14 + 31 passed on Py3.11; `pytest -m "integration and paddle"` 3 passed on Py3.11 (detector polys, crop+rec text `engine=paddle_v5_latin, model=latin_PP-OCRv5_mobile_rec`, API `route=scene recognizer=paddle_v5_latin` => `provenance.recognizer paddle_v5_latin` with `fallback_used=false`, not `503`); `scripts/warmup_models.py` READY detector 3361ms + recognizer 560ms both `loaded=True`; `scripts/check_environment.py` reports `cache_home C:\Users\PC\.paddlex` both models present.

*Log tạo dựa trên toàn bộ quá trình phát triển dự án Vietnamese OCR.*

## 16. Primary OCR Engine Migration — Paddle-first Scene OCR

*Scope:* Chuyển engine Scene chính từ **Tesseract-first → Paddle-first** (`PaddleDetector` + `PP-OCRv5 Latin`). Tesseract trở thành **legacy / benchmark / optional fallback**, không bao giờ fallback ngầm. Document route giữ nguyên khung parser (sẽ nâng lên MonkeyOCR / PaddleOCR-VL sau).

*Policy matrix (enforce trong `OCRRouter._run_scene`):*

| recognizer | Paddle state | ENABLE_TESSERACT_FALLBACK | Kết quả |
|---|---|---|---|
| `auto` (default) | ready / not_loaded → `ensure_ready()` → ready | bất kỳ | Paddle, `fallback_used=false` |
| `auto` | unavailable / load_failed | `false` | `503 MODEL_UNAVAILABLE` (không fallback) |
| `auto` | unavailable / load_failed | `true` | Tesseract legacy, `fallback_used=true`, warning `PADDLE_UNAVAILABLE_USING_LEGACY_TESSERACT` |
| `paddle` / `paddle_v5_latin` | unavailable / load_failed | bất kỳ | `503 MODEL_UNAVAILABLE` (không bao giờ fallback) |
| `tesseract` | (không thử Paddle) | bất kỳ | Tesseract legacy, `fallback_used=false` |

*Changes:*
- `app/core/config.py`: thêm `enable_tesseract_fallback: bool = False`, đọc env `ENABLE_TESSERACT_FALLBACK` (`1`/`true`/`TRUE`). Comment nêu Paddle là primary Scene engine.
- `app/pipelines/registry.py`: `get_recognizer` gom alias về Paddle hoặc Tesseract; engine lạ → `ValueError` (không fallback ngầm sang Tesseract); comment ghi rõ Tesseract chỉ legacy/benchmark/optional-fallback.
- `app/pipelines/router.py`: `want_neural = not explicit_tesseract`; `explicit_paddle` gate; 3 return path cho scene (neural success / explicit paddle fail → 503 / auto fail → fallback gate); `fallback_used` flag; fix imports `polygon_utils.crop_polygon` + `MODEL_UNAVAILABLE`.
- `app/api/health.py`: viết lại `/health/ready` — readiness dựa trên `paddle_detector` + `paddle_recognizer`; `overall_ready = paddle_ready or (enable_tesseract_fallback and tesseract ready)`; thêm `primary_scene_pipeline{detector, recognizer, status}`, `tesseract_role="legacy"`, `enable_tesseract_fallback`; không bao giờ load model nặng.
- `app/main.py`: FastAPI description đổi sang "Paddle text detection + PP-OCRv5 Latin"; UI HTML mô tả Paddle primary, form dùng `route` (auto/scene/document) + `recognizer` (auto/paddle_v5_latin/tesseract legacy); `/api` trả `primary_scene_engine="paddle_v5_latin"` + `legacy_engine="tesseract"`; health check UI gọi `/health/ready`.
- `README.md`: cập nhật sơ đồ kiến trúc (Scene = Paddle Detector + PP-OCRv5 Latin + text regions; Document = parser layout/table/reading order + re-OCR; Legacy/Benchmark block Tesseract tách riêng), bảng công nghệ (Paddle primary, Tesseract legacy), mục API endpoints (thêm `/health/ready`, `/v1/ocr` policy matrix, PSM chỉ áp dụng Tesseract legacy).

*Verification (GREEN):*
- `python -m py_compile` toàn bộ file chạm → exit 0, không lỗi cú pháp.
- `python -m pytest -q -k "not integration and not paddle"` → 23 passed (Py3.14 venv).
- `tests/test_engine_policy.py` mới (A–J, mock hoàn toàn, không cần paddle/Tesseract binary): default scene → Paddle; auto READY → Paddle; auto NOT_LOADED → `ensure_ready()` → Paddle; auto unavailable + fallback disabled → `MODEL_UNAVAILABLE` 503; auto unavailable + fallback enabled → Tesseract `fallback_used=true` + warning; explicit paddle unavailable → `MODEL_UNAVAILABLE`; explicit tesseract → Tesseract (không thử Paddle); UI default != Tesseract; `/health/ready` dựa Paddle; Tesseract một mình không làm service `ready`.
- Router integrity: 582 dòng, tail `return _router`, `fallback_used` xuất hiện ở cả 3 return path scene, imports có `polygon_utils` + `MODEL_UNAVAILABLE`.

*Kết quả:* Engine Scene chính giờ là **Paddle PP-OCRv5**; Tesseract chỉ chạy khi opt-in (`ENABLE_TESSERACT_FALLBACK=true`) hoặc gọi tường minh `recognizer=tesseract`. Provenance (`detector=paddle_detector`, `recognizer=paddle_v5_latin`), `fallback_used`, và health readiness đều nhất quán giữa router, schema, API và UI.

## 17. Primary OCR Engine Migration — REAL VERIFIED (2026-10-03)

*Env:* Python 3.11.7 (`.venv311`) trên Windows, `paddlepaddle==3.2.0`, `paddleocr==3.7.0`, `pip check` → `No broken requirements found.`

*Models (warmup_models.py WARMUP OK):* `PP-OCRv5_server_det` (TextDetection, `C:\Users\PC\.paddlex\official_models\PP-OCRv5_server_det`, load ~4.3 s) + `latin_PP-OCRv5_mobile_rec` (TextRecognition, `C:\Users\PC\.paddlex\official_models\latin_PP-OCRv5_mobile_rec`, load ~0.5 s), device `cpu`, cache_home `C:\Users\PC\.paddlex`.

*Test counts:* non-integration `41 passed, 3 deselected` (1 lần fail trung gian `test_v1_document_with_structure` do adapter Paddle trạng thái process-global chạm ảnh trắng — fix `paddle_recognizer.recognize()` gray→BGR, sau đó GREEN); integration `3 passed, 41 deselected` (`-m "integration and paddle"`).

*Live cold uvicorn :8001 (fresh process, health không load model):* trước OCR `degraded ready=false detector=not_loaded recognizer=not_loaded model_present=true dependency_available=true` — sau OCR `ok ready=true detector=ready recognizer=ready model_loaded=true`. Fix `status()` cả hai adapter consult `paddle_runtime.is_any_model_loaded()` khi `not_loaded`.

*Kết quả 4 request trên `data/sample_scene.png` (`Xin chao Viet Nam / OCR tieng Viet 2026 / Ma: AB123456`):*

| Request | HTTP | engines.detector / recognizer | fallback_used | regions |
|---|---|---|---|---|
| `route=scene&recognizer=auto` (cold, ~5.9 s) | 200 | `paddle_detector` / `paddle_v5_latin` | `false` | 3, `Xin chao Viet Nam` |
| `route=scene` (default, ~1.1 s) | 200 | `paddle_detector` / `paddle_v5_latin` | `false` | 3, giống auto |
| `route=scene&recognizer=paddle_v5_latin` (~1.0 s) | 200 | `paddle_detector` / `paddle_v5_latin` | `false` | 3, giống auto |
| `route=scene&recognizer=tesseract` (~1.1 s) | 200 | `None` / `tesseract` | `false` (explicit legacy, không phải fallback) | 1 |

*Provenance nhất quán:* 3 kết quả Paddle `engines.recognizer == provenance.recognizer == paddle_v5_latin`, `region.recognizer == paddle_v5_latin`; Tesseract `None / tesseract`.

*Policy checks (mock, isolation):* empty detection `0 regions` → success `text=""` `fallback_used=false` provenance Paddle; `auto` + Paddle unavailable + fallback disabled → `503 MODEL_UNAVAILABLE`; + fallback enabled → Tesseract `fallback_used=true` warning `PADDLE_UNAVAILABLE_USING_LEGACY_TESSERACT`.

*Cleanup:* `scripts/_verify_live.py`, `uvicorn_8001.out/.err`, `scripts/__pycache__` đã xóa; port 8001 rảnh.

## 18. Hybrid Phase 3 — Benchmark & Error Analysis (2026-10-03)

*Scope:* Phase 3 **freeze production Scene pipeline** — `PaddleDetector` + `PP-OCRv5 Latin` + `crop_polygon` + NFKC deterministic-v1, Tesseract legacy-only. Không đổi `app/pipelines/router.py` architecture; preprocessing experiments giữ trong benchmark code only. Không cài VietOCR/SVTR/MonkeyOCR/PaddleOCR-VL.

*Audit:* Reuse `benchmarks/metrics.py` CER/WER/exact (extend không rewrite), `data/` legacy sidecar `*.txt` giữ nguyên, `tests/` reuse conventions, `README.md` + `PROJECT_LOG.md` read trước. `benchmarks/dataset.py` extend giữ `EXPECTED_LAYOUT`, thêm `benchmark_data/` + `load_annotations()` + `sha256_file()`.

*Dataset (synthetic safe, no PII):*

- `benchmark_data/README.md` (system fonts only, deterministic seed, tag `synthetic`, variants)
- `benchmark_data/annotations.jsonl` — JSONL spec: `id,image,split,category,language,text,tags,regions{polygon,text},fields{amount,tax_id,document_id}`
- `benchmark_data/images/` 12 synthetic PNGs, `benchmark_data/crops/` (derived, empty placeholder), `benchmark_data/splits/train.txt/val.txt/test.txt` (ids per split, SHA256 no duplicate across splits)
- Generator: `scripts/generate_ocr_samples.py --seed 42 --variants clean blur jpeg rotation perspective low_contrast small_text` (Pillow + OpenCV + NumPy only, system font `C:/Windows/Fonts/arial.ttf` fallback)
- Validator: `benchmarks/validate_dataset.py` — JSONL syntax, unique id, image exists, valid split, text exists, polygon ≥3 pts + numeric [x,y], field structure, SHA256 duplicate across splits, splits consistency, exit non-zero on fail. `python benchmarks/validate_dataset.py --data benchmark_data` → `Validation OK: 12 records, 12 unique images`

*Benchmark code:*

- `benchmarks/run_recognition_benchmark.py --data benchmark_data --recognizer paddle_v5_latin|tesseract --preprocessing original|gray|otsu` — GT-crop → recognizer isolation (fair paddle vs tesseract)
- `benchmarks/run_scene_benchmark.py --data benchmark_data --recognizer paddle_v5_latin --preprocessing original` — full image → `PaddleDetector` (baseline original image) → `crop_polygon` → recognizer → regions, measured separately (IoU threshold 0.5 documented)
- Preprocessing ablation A: original perspective crop, B: gray→BGR, C: Otsu binary→BGR (không đổi production default)

*Metrics (stdlib-first, `ponytail: upgrade to shapely` for exact polygon IoU):*

- CER `lev(pred,ref)/len(ref)`, WER word-level Levenshtein (token list, không phải char distance trên joined string), exact_match `strip(pred)==strip(ref)`
- accent_stripped_cer via NFD Mn filter + `đ/Đ` map, diacritic_loss_rate = `cer - accent_stripped_cer` clamped ≥0 (documented; ví dụ `Xin chào Việt Nam` vs `Xin chao Viet Nam`: CER>0, accent_stripped==0 ⇒ loss==CER, raw_text primary, không spell-correct/LLM/strip trước CER/WER)
- Vietnamese breakdown `ă â ê ô ơ ư đ` + tone marks via NFD, critical field exact_field_match strip case-sensitive (`012345678901` vs `0123456789O1` false; `1.250.000` true)
- Detection precision/recall/F1/IoU threshold 0.5 documented, polygon IoU via bbox approx (shapely if installed), greedy 1-1 region matching by IoU + counts GT/pred/matched/missed/fp
- Latency split `model_load_ms/detection_ms/recognition_ms/total_ms` cold (first call) vs warm avg; confidence bins `0.0-0.5/0.5-0.7/0.7-0.8/0.8-0.9/0.9-1.0` with count/exact-rate/avg CER; raw_text primary

*Output (timestamped never overwrite):*

- `benchmark_results/<run-id>/` with `recognition_results.csv` (id,image,split,category,region_idx,gt_text,pred_text,cer,wer,exact_match,accent_stripped_cer,diacritic_loss_rate,confidence,recognition_ms,preprocessing), `scene_results.csv` (id,image,gt_regions,pred_regions,matched,missed,fp,precision,recall,f1,mean_iou,cer,wer,detection_ms,recognition_ms,total_ms), `errors.csv` (top CER>0.3), `summary.json` + `summary.md` with model_meta (Python/PaddlePaddle/PaddleOCR/detector/recognizer/Tesseract/device/preprocessing/CLI/dataset/sample_count/git_hash) + warning if `sample_count<100`
- `.gitignore` + `benchmark_results/` (không commit outputs)

*Tests:*

- `tests/test_benchmark_metrics.py` 17 tests: CER/WER/exact/accent strip/diacritic/IoU/region matching + required regressions (`Xin chào Việt Nam` vs `Xin chao Viet Nam` normal CER>0 accent_stripped==0, `012345678901` vs `0123456789O1` false, `1.250.000` true, WER word-level vs char-distance check)
- `tests/test_benchmark_validation.py` 7 tests: valid minimal, duplicate id, missing image, invalid split, duplicate SHA256 across splits, empty text, serialization roundtrip
- `pytest -m "not integration" -q` → **65 passed, 3 deselected** (tăng từ 41 do 24 tests benchmark mới)

*Smoke runs (SMOKE ONLY NOT REPRESENTATIVE — n=12 <100):*

| run | preprocessing | CER | WER | exact | detection P/R/F1 | n | meta |
|---|---|---|---|---|---|---|---|
| rec paddle | original | 0.134 | 0.426 | 0.167 | — | 12 | Python 3.11.7 Paddle 3.2.0 PaddleOCR 3.7.0 Tesseract 5.4.0 device cpu load ~0.5s |
| rec paddle | gray | 0.134 | 0.426 | 0.167 | — | 12 | — |
| rec paddle | otsu | 0.150 | 0.507 | 0.083 | — | 12 | — |
| rec tesseract | original | 0.176 | 0.369 | 0.500 | — | 12 | — |
| scene paddle | original | 0.092 | — | — | 0.917/0.917/0.917 (IoU 0.5, matched 11/12) | 12 | det PP-OCRv5_server_det rec latin_PP-OCRv5_mobile_rec avg_total ~539ms cold det 310ms cold rec 413ms |

Validation + 5 benchmark_results runs (2×rec limit 4 pilot + 3×rec full 12 ablation + 1×scene full 12) đều `SMOKE ONLY` warn. Diacritic loss contributes ~0.02–0.06 CER; gray không cải thiện, Otsu tệ hơn; Tesseract exact cao hơn nhưng CER tệ hơn trên synthetic này — không kết luận chất lượng thực khi n nhỏ.

*Verification (GREEN):*

- `python benchmarks/validate_dataset.py --data benchmark_data` → OK
- `.venv311\Scripts\python benchmarks/run_recognition_benchmark.py --data benchmark_data --recognizer paddle_v5_latin` → Done CER 0.134
- `.venv311\Scripts\python benchmarks/run_recognition_benchmark.py --data benchmark_data --recognizer tesseract` → Done CER 0.176
- `.venv311\Scripts\python benchmarks/run_scene_benchmark.py --data benchmark_data --recognizer paddle_v5_latin` → Done P 0.917 R 0.917
- `python -m pytest -m "not integration" -q` → 65 passed

*Status:* **FRAMEWORK READY** — benchmark framework hoàn chỉnh, diacritic loss isolated, detection/recognition tách bạch, latency/confidence bins, never-overwrite outputs. **REAL QUALITY BENCHMARK PENDING DATA** — cần ≥100 real Vietnamese images (scene + document + fields) để quyết định training/fine-tuning recognizer mới; hiện tại chỉ synthetic smoke, không đại diện. `README.md` §14 Benchmarking đã thêm, `benchmark_results/` gitignored, production router không đổi.

### 18.1 Phase 3.1 supplement (tooling + filters + privacy, 2026-10-03)

*Tooling hoàn thiện:* `benchmarks/dataset.py` thêm vocab chuẩn `SOURCE_TYPES/REVIEW_STATUSES/ANNOTATION_LEVELS/CATEGORY_PLAN/VIET_SPECIAL_CHARS/VIET_TONES/CRITICAL_FIELD_KINDS` + helpers `record_source/record_review_status/record_annotation_level/filter_records/count_records` (duy nhất dùng cho mọi runner). `benchmarks/import_images.py` (privacy gate `real→safe` required, SHA256 dedup, stubs unreviewed), `benchmarks/annotate.py` (stdlib HTTP + canvas rect/polygon, prev/next, image/region level, ignore, reviewed/rejected, atomic save), `benchmarks/dataset_summary.py` (+vi diacritics/tone/field/category_plan coverage, `Real reviewed test: x/100`), `benchmarks/audit_annotations.py` (+all QA checks), `benchmark_incoming/` gitignored intake, `benchmark_data/README.md` §Privacy Gate + §Building dataset + §Coverage + §CLI flags.

*Benchmark runners:* cả hai đều `sys.path` ROOT fix (tránh ModuleNotFoundError khi run direct), import `count_records/filter_records`, CLI thêm `--source/--split/--reviewed-only/--include-unreviewed/--include-rejected` (canonical, `--reviewed-only` recommended cho official). Summary ghi `selection` + `dataset_counts`. `--help` đã verify.

*Version metadata:* `benchmark_data/dataset.json` `v0.3.1` Phase 3.1 thêm `privacy_gate/build_tools/coverage_targets/benchmark_cli_flags/provenance`; `git_hash=null` khi standalone copy (không phải git checkout) + note hướng set `git rev-parse HEAD`. Patch ban đầu tạo malformed (hai JSON object nối nhau) đã xóa + ghi sạch lại, validate JSON ok. `validate_dataset/dataset_summary/audit` đều green trên 12 synthetic hiện tại.

*Tests 23 mới (`tests/test_benchmark_dataset_filters.py`):* record_source/review_status/annotation_level fallbacks; filter split/source/reviewed_only/include_rejected + official selection (`test+real+reviewed`); count by_source/review/split/category/tag; metadata structure + counts khớp `annotations.jsonl` live; privacy_gate tồn tại; `benchmark_data/README.md` có Privacy Gate + flags; import `existing_hashes` dedup + `next_id` increment. Đã fix harness sinh file `Add File` bị thừa leading space (helper `scripts/_fix_leading_space.py` tạm rồi xóa). `git_hash` populated provenance do không có git repo.

*Docs:* `README.md` §14 thêm §Building Real Benchmark Dataset + §Privacy gate + canonical filtering table + `selection/dataset_counts` note; count test cập nhật 88. `benchmark_data/dataset.json` gitignored lỗi đã revert (metadata tracked). `scripts/patch_benchmarks.py` giữ làm idempotent patcher cho legacy runners.

*Verification (Phase 3.1):* `validate OK`/`audit PASSED`/`summary total=12 real=0 reviewed=0` + category LOW, `pip-audit 0`, `pytest -m "not integration"` **88 passed, 3 deselected** (từ 65), `--help` cả hai runner show 5 flags mới.

*Còn lại:* per-category/diacritic/critical-field/error-bucket/confidence/review-queue reporting trong runner summary (mở rộng metrics aggregation, không đổi core formula), và thu thập ≥100 real reviewed test để benchmark đại diện.

### 18.2 Phase 3.2 coverage gate + freeze manifest (2026-10-04)

*Dataset summary + coverage flags:* `benchmarks/dataset.py` adds `MIN_REAL_TEST_REVIEWED=100`, `COVERAGE_SEVERE_THRESHOLD=5`, `COVERAGE_LOW_THRESHOLD=10`, helper `coverage_report()` counts per-category-plan tags+category and tags SEVERE/LOW/ok (advisory only, never suppresses data), `real_reviewed_test_count()`, `record_tags()`, `build_manifest()` builds manifest ids/splits/sha256 + `manifest_hash` via hashlib (no image bytes, safe to commit). `dataset_summary.py` fixes `UnicodeEncodeError` on Windows consoles with `sys.stdout.reconfigure(encoding='utf-8')`; JSON output now includes `real_reviewed_test`, `min_real_test_reviewed`, `category_plan_coverage`.

*Freeze tool:* `benchmarks/freeze_testset.py` gates: refuses if real reviewed test <100, unless `--allow-preliminary` labels the manifest `PRELIMINARY NOT REPRESENTATIVE`. Output `benchmark_data/frozen/testset_v<version>.json` contains only id/split/sha256/source/review_status/manifest_hash; no image bytes. `--help` verified.

*New tests (`tests/test_benchmark_coverage.py`):* 14 tests covering thresholds sanity, `record_tags`, `coverage_report` flags, `real_reviewed_test_count`, `build_manifest` determinism/hash stability/no-bytes, freeze gate refusal + allow-preliminary labelling. All pass.

*Docs:* `README.md` §14 updated with coverage flags + freeze; `benchmark_data/README.md` adds §Freeze the Real Test Set + per-category thresholds; `benchmark_data/dataset.json` updates `coverage_targets` with `severe_below/low_below/flags_are_advisory/freeze_output` and `build_tools.freeze`.

*Cleanup:* removed stray temp `_tmp_*.txt`, `_tmp_rec_full.py`, `_tmp_scene_full.py`, `_cur_*.py`.

*Verification:* `pytest -m 'not integration' -q` -> 102 passed, 3 deselected (88 prior + 14 new); `validate_dataset` OK; `audit_annotations` PASSED; `dataset_summary --data benchmark_data` works on Windows; `freeze_testset --help` OK. Status still blocked on real data: `Real reviewed test: 0 / 100` (no real photos in `benchmark_incoming`). Did not extend runner aggregation (item 4) because real data <100; did not install VietOCR/SVTR/MonkeyOCR/PaddleOCR-VL. Production pipeline untouched.

*Verification pass (2026-10-04, integrity check, no code change):* `compileall` app/benchmarks/scripts/tests EXIT=0. JSON/JSONL parse OK (12 lines, 0 bad). Full `.venv311` pytest **105 passed** (102 non-integration + 3 paddle integration). `pip check` clean. Python 3.11.7 / Paddle 3.2.0 / PaddleOCR 3.7.0. API: `/health/live` 200, `/health/ready` 200 degraded-before-load then models cache, POST `/v1/ocr?route=scene&recognizer=auto` 200 detector=`paddle_detector` recognizer=`paddle_v5_latin` fallback_used=false, explicit tesseract 200. Benchmark smoke `--limit` writes valid timestamped outputs. Workspace not a git repo. No corruption, no stray temp files. `PROJECT_HISTORY.md` rewritten to match current architecture (was stale Tesseract-only 13 tests).
