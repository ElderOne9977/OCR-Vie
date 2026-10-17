# Vietnamese OCR

## Cập nhật chất lượng và hiệu năng OCR — 2026-10-05

Chính sách hiện hành (thay thế mô tả Tesseract chỉ là legacy ở các phần lịch sử bên dưới):

| Route / lựa chọn | Pipeline |
|---|---|
| `scene + auto`, có `vie`/`vi` | Paddle detector → crop màu → Tesseract PSM 7 theo lô |
| `scene + auto`, ngôn ngữ khác | Paddle detector → crop màu → Paddle recognizer |
| `document + auto` | Tesseract toàn trang; engine cố định, không phụ thuộc Paddle đã load |
| `scene/document + paddle_v5_latin` | Paddle detector → crop từng dòng → Paddle recognizer; thiếu model/charset trả 503 |
| `scene/document + paddle_tesseract` | Paddle detector → crop từng dòng → Tesseract PSM 7 theo lô; thiếu component trả 503 |
| `scene/document + tesseract` | Tesseract toàn ảnh, PSM theo query |

`OCR_DOCUMENT_RECOGNIZER=tesseract` là mặc định mới, độc lập với
`OCR_SCENE_RECOGNIZER=auto`. Có thể đặt một trong hai thành `paddle_tesseract`.
Document không đưa cả trang vào Paddle line recognizer. Parser tài liệu vẫn là
khung; đọc nhiều cột/bảng chưa được bảo đảm bởi cách sắp xếp bbox hiện tại.

Trong UI chọn **Paddle phát hiện + Tesseract tiếng Việt** để dùng hybrid.
Auto tiếng Việt hiện cũng chọn hybrid cho Scene. Chọn ngôn ngữ `vie` nếu ảnh chỉ
có tiếng Việt; mặc định vẫn `vie+eng` để hỗ trợ song ngữ. Alias `vi+en` được đổi
thành `vie+eng` khi gọi Tesseract.
Ví dụ API: `POST /v1/ocr?route=auto&recognizer=paddle_tesseract` (multipart `file`).
Khởi động lại server đang chạy để nạp code mới.

Model Latin đang cài thiếu 90/146 ký tự tiếng Việt trong decoder. Explicit Paddle
với ngôn ngữ Việt bị từ chối bằng `503 MODEL_LANGUAGE_UNSUPPORTED`, cả trước/sau
warmup; không tự sửa dictionary hay chuyển engine ngầm. Model thay thế phải có
trọng số và charset tương thích.

Document mặc định dùng grayscale, giữ nét/dấu nhỏ. Có thể chọn
`DOCUMENT_PREPROCESS=gray|binary|original`; `binary` giữ thuật toán median+Otsu cũ.
Profile này áp dụng `/v1/ocr`; tiền xử lý legacy `/ocr` giữ contract cũ.

| Cấu hình tài nguyên | Mặc định | Ý nghĩa |
|---|---:|---|
| `MAX_CONCURRENT_OCR` | 2 | Request OCR đang chạy **mỗi process**; quá tải trả 503 OCR_BUSY + Retry-After |
| `MAX_UPLOAD_BYTES` | 10485760 | Đọc tối đa giới hạn+1 byte rồi từ chối 413 |
| `MAX_IMAGE_PIXELS` | 12000000 | Kiểm tra header trước/sau decode; kiểm tra tổng crop neural trước copy/warp |
| `MAX_IMAGE_SIDE` | 8192 | Chiều dài tối đa mỗi cạnh ảnh |
| `MAX_REGIONS` | 256 | Vượt giới hạn trả 413; không âm thầm bỏ bớt vùng |
| `OCR_TIMEOUT_SECONDS` | 30 | Timeout mỗi invocation Tesseract; timeout trả 504 OCR_TIMEOUT |
| `OCR_REQUEST_TIMEOUT_SECONDS` | 60 | Deadline hợp tác từ lúc nhận slot OCR; batch/đợi predictor dùng thời gian còn lại, hết hạn trả 504 |
| `TESSERACT_BATCH_SIZE` | 32 | Crop tối đa mỗi invocation; TSV page_num giữ ánh xạ vùng trắng |
| `DEVICE` / `CPU_THREADS` | cpu / 10 | Truyền vào constructor Paddle; không đồng nghĩa giới hạn mọi native thread |
| `DETECTOR_LIMIT_SIDE_LEN` | 960 | Giới hạn cạnh dài detector, 32–4000; 768 ưu tiên tốc độ, 1280 ưu tiên chữ nhỏ |

Các giá trị số phải dương/hữu hạn. Paddle predictor dùng chung được khóa cả lúc
consume generator. Admission không tạo hàng đợi; chạy nhiều Uvicorn worker sẽ
tăng tổng số request và nhân bản cache/RAM. Deadline không ngắt tức thì native
Paddle/import/load model đang chạy: kiểm tra trước/sau thao tác, giữ slot và
khóa cho đến khi native trả về. Cold start vẫn cần nạp framework/model.
Upload/network trước slot và serialization phản hồi sau OCR không thuộc budget.

Chọn cấu hình detector bằng biến môi trường rồi khởi động lại server:
`DETECTOR_LIMIT_SIDE_LEN=768` cho ảnh chữ đủ lớn hoặc `1280` cho chữ nhỏ;
mặc định vẫn `960`. Detector resize nội bộ, crop nhận dạng giữ ảnh gốc. Không
tự chuyển profile theo confidence. Số đo và đánh đổi trong
[OCR_OPTIMIZATION_REVIEW.md](OCR_OPTIMIZATION_REVIEW.md).

Đo lại bằng `.venv311\Scripts\python scripts/verify_ocr_performance.py --engine hybrid`
hoặc `--engine document --lang vie`. Trên mẫu synthetic 12 dòng: hybrid warm
~4,64 s → ~1,63 s, CER 0; RAM peak vẫn ~900 MiB. Document default ~0,55 s →
~0,47 s, CER 3,57% → 2,68%. Xem [OCR_PERFORMANCE_FIX.md](OCR_PERFORMANCE_FIX.md)
và [OCR_PERFORMANCE_ANALYSIS.md](OCR_PERFORMANCE_ANALYSIS.md) cho phương pháp/giới hạn.

Hybrid API trả `ocr_confidence` trong khoảng 0–1 và thêm `raw_confidence`
(Tesseract 0–100; Paddle 0–1). Điểm này chưa được hiệu chỉnh thành xác suất đúng.
Legacy `/ocr` và `/ocr/data` giữ contract cũ. Tesseract adapter lấy text và
confidence từ một lần `image_to_data`; nối từ theo line/page/block/paragraph.
`/health/ready` giữ overall readiness cho Scene, thêm `primary_document_pipeline`
và `readiness_scope` để kiểm tra Document riêng. Health không import/load Paddle;
dependency presence khác successful initialization. `timing_ms` thêm `decode`,
`initialization` (kiểm tra dependency/khởi tạo model), `preprocessing`.

Kiểm chứng lỗi A4 lạnh/nóng bằng `.venv311\Scripts\python scripts/verify_ocr_quality.py`.
Chi tiết kết quả và giới hạn: [OCR_QUALITY_FIX.md](OCR_QUALITY_FIX.md).

Benchmark đã sửa polygon (kể cả offset font, rotation và perspective), seed ổn
định giữa process, Tesseract crop dùng `--psm 7`, confidence lấy từ recognizer,
WER từng trang và điểm end-to-end tính cả chữ thiếu/thừa. `recognition` trong
summary Scene hiện tính unmatched GT là chuỗi rỗng; `end_to_end` là macro CER/WER
trên toàn trang. GT regions phải có thứ tự đọc, vùng `ignore` được loại.

Dataset cũ được giữ nguyên để truy vết. Bộ sửa mới là `benchmark_data_quality_v2/`,
chỉ 12 synthetic smoke, **không đại diện ảnh thật**. Generator từ chối ghi đè
annotations đã có: dùng `--output` mới. Các số benchmark lịch sử bên dưới không
so sánh trực tiếp được với phiên bản đánh giá mới.

OCR tiếng Việt chạy local/offline, dùng FastAPI + Paddle detector + Tesseract.

Scene tiếng Việt: **Paddle detection + Tesseract theo lô**. `ENABLE_TESSERACT_FALLBACK=false` mặc định; nhận dạng Tesseract trong hybrid là đường chính, không phải fallback.
Document parse là nhánh riêng (parser khung), sẽ nâng lên MonkeyOCR / PaddleOCR-VL sau.

## 1. Giới thiệu

API OCR đơn giản, dễ đọc, dễ debug. Không có database, authentication, queue.
Chạy offline khi đã có Paddle detector weights và Tesseract `vie`/`eng` traineddata.

## 2. Kiến trúc

```
Client
  → FastAPI
    → InputClassifier (heuristic-v1)
      ├─ scene tiếng Việt → PaddleDetector → crop_polygon → Tesseract batch → regions[{bbox, polygon, raw_text, normalized_text}]
      ├─ document → grayscale → Tesseract toàn trang → parser tùy chọn (khung, chưa có layout/table model)
      └─ explicit/legacy/benchmark → engine người dùng chọn; Paddle recognition có charset gate
        → PipelineResult { route, engines{classifier, detector, recognizer, parser, normalizer}, text, regions, document?, warnings, provenance, timing_ms, fallback_used }
```

### Hybrid (Phase 2) — snapshot lịch sử trước 2026-10-05

```
Client -> FastAPI /v1/ocr?route=auto&recognizer=auto
  -> InputClassifier (heuristic-v1: pdf/aspect/pixels)
     +- scene (primary)    -> paddle_detector(PP-OCRv5_server_det) -> crop_polygon -> paddle_recognizer(latin_PP-OCRv5_mobile_rec) -> raw/normalized per-region -> NFKC  [fallback_used=false]
     |                      └─ khi Paddle unavailable + ENABLE_TESSERACT_FALLBACK=false → 503 MODEL_UNAVAILABLE (không fallback)
     |                      └─ khi Paddle unavailable + ENABLE_TESSERACT_FALLBACK=true  + recognizer=auto → Tesseract legacy, fallback_used=true, warning PADDLE_UNAVAILABLE_USING_LEGACY_TESSERACT
     |                      └─ explicit recognizer=paddle_v5_latin unavailable → 503 MODEL_UNAVAILABLE (không fallback)
     |                      └─ explicit recognizer=tesseract → Tesseract legacy ngay, không thử Paddle
     +- document            -> doc_preprocess (gray+median+Otsu) -> full_frame detector -> recognizer -> parser (fallback)  [FallbackParser != pipeline cuối; sẽ thay MonkeyOCRv2-B-Parsing / PaddleOCR-VL]
  -> PipelineResult { route, engines, text, regions[{raw_text, normalized_text, bbox, polygon, recognizer, provenance}], document?, warnings, fallback_used, timing_ms }
```
Phase 2 pins PaddleOCR 3.7.0: `TextDetection.predict(bgr) -> {dt_polys, dt_scores}` + `TextRecognition.predict(crop) -> {rec_text, rec_score}`. Scene neural `raw_text` verbatim recognizer output (một entry mỗi `dt_poly` qua `crop_polygon`), `normalized_text` = NFKC deterministic-v1, không LLM overwrite. Tesseract chỉ là legacy (`provenance.recognizer == \"tesseract\"`); explicit `recognizer=paddle_v5_latin` không bao giờ fallback ngầm — thiếu weight => `503 MODEL_UNAVAILABLE`.

Config: `OCR_SCENE_DETECTOR=paddle|full_frame`, `OCR_SCENE_RECOGNIZER=auto|paddle|paddle_v5_latin|tesseract`, `MODELS_DIR`, `ALLOW_MODEL_DOWNLOAD=false|true`, `ENABLE_TESSERACT_FALLBACK=false|true` (mặc định false), `PADDLE_PDX_CACHE_HOME` (framework, wins), `DEVICE`. Health: `GET /health/live` không load model; `GET /health/ready` báo `components.paddle_*` + `paddle.{cache_home, detector_present, recognizer_present}` không I/O, `primary_scene_pipeline` drives overall `ready`/`degraded`, Tesseract `ready` alone != service `ready` khi fallback disabled. `MODELS_DIR` priority: 1) `PADDLE_PDX_CACHE_HOME` 2) `MODELS_DIR` 3) `~/.paddlex` (chứa weight thật trên dev; `models/` là mount hint rỗng — không hardcode `/models` trên Windows).

Quick verify (Python 3.11 only — 3.14 has no paddlepaddle wheel):
```bash
.venv311/Scripts/python -m pip install -r requirements-paddle.txt
.venv311/Scripts/python scripts/warmup_models.py   # Detector + Tesseract cho default VI; kiểm chứng trong process này
.venv311/Scripts/python -m pytest -m "integration and paddle" -v
curl.exe -X POST "http://localhost:8000/v1/ocr?route=scene&recognizer=auto" -F "file=@data/sample_scene.png"
```

`pytesseract` là wrapper gọi binary Tesseract. Script warmup chạy ở process riêng;
không làm nóng cache model của server Uvicorn đang chạy.

## 3. Công nghệ sử dụng

| Thành phần | Vai trò |
|---|---|
| Python 3.11 | Runtime (3.14 chưa có paddlepaddle wheel) |
| FastAPI + Uvicorn | REST API |
| PaddleOCR 3.7.0 + PP-OCRv5 | Scene detector; recognizer cho ngôn ngữ tương thích |
| Tesseract OCR + tesseract-ocr-vie/eng + pytesseract | Document và Scene tiếng Việt mặc định; legacy/benchmark |
| OpenCV (headless) + NumPy + polygon_utils.crop_polygon | Preprocessing + perspective rectify |
| Pillow | Đọc ảnh |
| python-multipart | Upload file |
| Docker | Đóng gói, chạy offline |

## 4. Cấu trúc project

```
OCR-Vie/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI endpoints
│   ├── ocr_service.py       # Tesseract wrapper
│   └── image_processing.py  # OpenCV preprocessing
├── tests/
│   ├── test_validation.py
│   ├── test_ocr_mock.py
│   └── test_preprocessing.py
├── scripts/
│   ├── check_environment.py
│   ├── smoke_test.py
│   ├── run_local.ps1
│   └── test.ps1
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## 5. Chạy local

Local — Paddle phát hiện Scene, Tesseract nhận dạng tiếng Việt và tài liệu.

* Scene OCR tiếng Việt mặc định cần PaddleDetector — **yêu cầu Python 3.11** và detector weights trong `~/.paddlex` hoặc `MODELS_DIR`.
* Tesseract kèm `vie` + `eng` cần cho cả Document và Scene tiếng Việt mặc định.
* Nếu muốn chạy legacy Tesseract local, cài Tesseract kèm language pack `vie` + `eng` như mục 6 bên dưới.

Tạo virtualenv:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Windows CMD:

```bat
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

Cài dependency (ưu tiên wheel, tránh build từ source):

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --prefer-binary -r requirements.txt
```

PowerShell shortcut:

```powershell
.\scripts\run_local.ps1
```

Chạy API:

```bash
uvicorn app.main:app --reload
```

Swagger: <http://localhost:8000/docs>

Giao diện upload: <http://localhost:8000> (trang HTML tại `GET /`).

Truy cập từ máy khác trong LAN: chạy `uvicorn app.main:app --host 0.0.0.0 --port 8000`
rồi mở `http://<IP-máy-host>:8000`. Cần mở firewall cho port 8000.

Kiểm tra Tesseract trên máy host:

```bash
tesseract --version
tesseract --list-langs
```

Kết quả `--list-langs` phải có ít nhất `eng` và `vie`.

### Cài Tesseract trên host (chỉ khi chạy local, không dùng Docker)

- **Windows:** cài nhanh bằng winget (không cần mở trình duyệt):
  ```powershell
  winget install --id UB-Mannheim.TesseractOCR -e
  ```
  Hoặc tải installer từ <https://github.com/UB-Mannheim/tesseract/wiki> và
  chọn language pack Vietnamese + English.

  Installer **không** tự thêm `tesseract.exe` vào `PATH`, nên project tự
  dò `C:\Program Files\Tesseract-OCR\tesseract.exe`. Nếu cài chỗ khác,
  đặt biến môi trường:
  ```powershell
  $env:TESSERACT_CMD = "D:\Tools\Tesseract-OCR\tesseract.exe"
  ```

  Thêm `vie` nếu `--list-langs` thiếu (tải `vie.traineddata` từ
  <https://github.com/tesseract-ocr/tessdata_fast> vào thư mục `tessdata`):
  ```powershell
  # chạy trong PowerShell admin
  Invoke-WebRequest "https://github.com/tesseract-ocr/tessdata_fast/raw/main/vie.traineddata" `
    -OutFile "C:\Program Files\Tesseract-OCR\tessdata\vie.traineddata"
  ```
- **Debian/Ubuntu:**
  ```bash
  sudo apt-get install tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng
  ```

## 6. Chạy bằng Docker

Docker image tự chứa Tesseract + `vie`/`eng`. Host không cần cài Tesseract.

Build:

```bash
docker build -t vietnamese-ocr .
```

Run (host port 8000 → container port 8000):

```bash
docker run -p 8000:8000 vietnamese-ocr
```

Sau đó:

- API: <http://localhost:8000>
- Swagger: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

Kiểm tra language pack trong container:

```bash
docker run --rm vietnamese-ocr tesseract --list-langs
```

Kết quả phải có `eng` và `vie`.

## 7. API endpoints

| Method | Path | Mô tả |
|---|---|---|
| GET | `/` | UI upload (HTML) |
| GET | `/api` | JSON ping — `primary_scene_engine: paddle_tesseract`, `legacy_engine: tesseract` |
| GET | `/health` | Backward-compat: Tesseract availability |
| GET | `/health/live` | Liveness — không load model |
| GET | `/health/ready` | Readiness — primary `paddle` quyết định `ready`/`degraded`; Tesseract `ready` alone != service ready khi `ENABLE_TESSERACT_FALLBACK=false` |
| POST | `/ocr` | Legacy Tesseract OCR — trả text |
| POST | `/ocr/data` | Legacy Tesseract — word bbox + confidence |
| POST | `/v1/ocr?route=auto&recognizer=auto` | Classifier → Scene detector/crop/Tesseract cho tiếng Việt hoặc Document Tesseract toàn trang |

### GET `/`

```json
{ "message": "Vietnamese OCR API is running", "primary_scene_engine": "paddle_tesseract", "legacy_engine": "tesseract" }
```

### GET `/health`

```json
{ "status": "ok", "tesseract_available": true, "version": "5.x.x", "languages": ["eng","vie"] }
```

### GET `/health/ready`

```json
{
  "status": "ok|degraded", "ready": true,
  "primary_scene_pipeline": { "detector": "paddle", "recognizer": "tesseract", "status": "ready|not_loaded|unavailable|load_failed" },
  "components": { "paddle_detector": "ready", "paddle_recognizer": "ready", "tesseract": "ready" },
  "paddle": { "dependency_available": true, "detector_present": true, "recognizer_present": true },
  "tesseract_role": "document_and_vietnamese_scene", "enable_tesseract_fallback": false
}
```

### POST `/v1/ocr`

Multipart `file`, query `route=auto|scene|document`, `recognizer=auto|paddle_tesseract|paddle_v5_latin|tesseract`, `lang`, `psm`, `return_structure`. Auto tiếng Việt dùng hybrid cho Scene, toàn trang Tesseract cho Document.
Response `PipelineResult` gồm `route`, `engines{classifier, detector, recognizer, parser, normalizer}`, `text`, `regions[{raw_text, normalized_text, bbox, polygon, recognizer, provenance}]`, `warnings`, `fallback_used`, `timing_ms`.

Policy:
- `recognizer=auto` + tiếng Việt → Paddle detection/Tesseract batch, `fallback_used=false`.
- `recognizer=auto` + Paddle unavailable + `ENABLE_TESSERACT_FALLBACK=false` → `503 MODEL_UNAVAILABLE` (không fallback).
- `recognizer=auto` + Paddle unavailable + `ENABLE_TESSERACT_FALLBACK=true` → Tesseract legacy, `fallback_used=true`, `warnings: ["PADDLE_UNAVAILABLE_USING_LEGACY_TESSERACT"]`.
- `recognizer=paddle_v5_latin` unavailable → `503 MODEL_UNAVAILABLE`; thiếu charset Việt → `503 MODEL_LANGUAGE_UNSUPPORTED`, không fallback.
- `recognizer=tesseract` → Tesseract legacy ngay, không thử Paddle.

### POST `/ocr` (legacy)

Multipart form-data, field `file`. Query params tùy chọn:

- `lang` (default `vie+eng`)
- `psm` (default `6`)

Response:

```json
{
  "success": true,
  "filename": "document.jpg",
  "text": "Nội dung OCR ở đây"
}
```

### POST `/ocr/data` (legacy)

Cùng input như `/ocr`, thêm `min_confidence` (default `0`).

Response:

```json
{
  "success": true,
  "filename": "image.jpg",
  "items": [
    {
      "text": "Xin",
      "confidence": 96.2,
      "left": 120,
      "top": 80,
      "width": 45,
      "height": 20
    }
  ]
}
```

Item có text rỗng hoặc confidence `< 0` bị bỏ.

PSM chỉ áp dụng cho Tesseract legacy; Paddle bỏ qua `psm`.

## 8. Test bằng Swagger

Mở <http://localhost:8000/docs>, chọn `POST /ocr`, bấm **Try it out**,
upload ảnh, Execute.

## 9. Test bằng curl

Linux / macOS / Git Bash:

```bash
curl -X POST "http://localhost:8000/ocr" -F "file=@test.png"
```

Với PSM 7 (single text line):

```bash
curl -X POST "http://localhost:8000/ocr?psm=7" -F "file=@test.png"
```

Word-level data:

```bash
curl -X POST "http://localhost:8000/ocr/data" -F "file=@test.png"
```

**Windows PowerShell:** `curl` là alias của `Invoke-WebRequest`. Dùng
`curl.exe` để gọi binary thật:

```powershell
curl.exe -X POST "http://localhost:8000/ocr" -F "file=@test.png"
```

## 10. PSM là gì

PSM = **Page Segmentation Mode** — cách Tesseract cắt trang thành vùng text.

| PSM | Ý nghĩa |
|---|---|
| 3 | Automatic page segmentation (default Tesseract, không OSD) |
| 6 | Single uniform block of text (default API này) |
| 7 | Single text line |
| 8 | Single word |
| 11 | Sparse text |
| 13 | Raw line, không dùng hack đặc biệt |

Giá trị hợp lệ: `0, 1, 3–13` (PSM 2 bị Tesseract reserve). PSM không hợp lệ
trả HTTP 400.

Chọn PSM:

- Ảnh tài liệu / đoạn văn: `6` hoặc `3`
- Một dòng (biển số, tiêu đề): `7`
- Một từ: `8`

## 11. OCR tiếng Việt + tiếng Anh

Language mặc định: `vie+eng`.

Tesseract sẽ dùng **cả hai** traineddata. Hữu ích khi ảnh lẫn chữ Việt và
tiếng Anh (tên riêng, số, từ viết tắt).

Đổi ngôn ngữ:

```
POST /ocr?lang=vie
POST /ocr?lang=eng
```

## 12. Development / Tests

Cài test extras:

```bash
python -m pip install --prefer-binary -r requirements-dev.txt
```

Chạy unit tests (không cần Tesseract binary):

```bash
pytest -v tests/
```

PowerShell:

```powershell
.\scripts\test.ps1
```

Kiểm tra môi trường:

```bash
python scripts/check_environment.py
```

Smoke test API đang chạy:

```bash
python scripts/smoke_test.py
# hoặc
python scripts/smoke_test.py --base-url http://localhost:8000
```

## 13. Troubleshooting

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `tesseract: command not found` khi chạy local | Cài Tesseract + thêm vào PATH |
| `/health` trả `tesseract: false` | Binary không trong PATH của process |
| OCR ra rỗng / sai | Thử PSM khác (`3`, `7`); ảnh quá nhỏ / mờ / nghiêng |
| `Invalid or corrupted image` | File không phải ảnh, hoặc file hỏng |
| `Uploaded file must be an image` | Content-Type không bắt đầu bằng `image/` |
| `Invalid PSM` | Truyền PSM ngoài tập hợp hợp lệ |
| Docker build fail ở `apt-get` | Kiểm tra mạng; retry |
| Docker daemon not running | Mở Docker Desktop rồi `docker version` |
| pip treo / chậm | `python -m pip install --upgrade pip setuptools wheel` rồi `python -m pip install --prefer-binary -r requirements.txt` |
| `File too large` / 413 | Ảnh > 10 MB |
| `Unsupported file type` | Chỉ chấp nhận jpeg/png/webp/bmp/tiff |
| Chữ dấu tiếng Việt bị mất | Preprocessing đã giữ nhẹ (median 3 + Otsu). Nếu vẫn mất, thử PSM 3, hoặc ảnh độ phân giải cao hơn |
| PowerShell `curl` lỗi lạ | Dùng `curl.exe` thay vì `curl` |

## 14. Benchmarking (Phase 3 — Scene OCR)

Phase 3 **freeze production Scene pipeline** (`PaddleDetector` + `PP-OCRv5 Latin` + `crop_polygon` + NFKC deterministic-v1, Tesseract legacy-only). Preprocessing experiments giữ trong benchmark code, không đổi router.

### Dataset format

```
benchmark_data/
  README.md
  annotations.jsonl   # JSONL: id,image,split,category,language,text,tags,regions{polygon,text},fields{amount,tax_id,document_id}
  images/             # synthetic/public/anonymized safe samples only
  crops/              # GT crops (derived)
  splits/train.txt,val.txt,test.txt  # ids per split, SHA256 no duplicate across splits
data/                 # legacy sidecar *.txt (Phase 2) giữ nguyên, reuse không rewrite
```

Validator: `python benchmarks/validate_dataset.py --data benchmark_data` — check JSONL syntax, unique id, image exists, valid split, text exists, polygon ≥3 pts, field structure, SHA256 duplicate, exit non-zero on fail.

Sinh synthetic safe samples (chỉ system fonts, deterministic seed, tag `synthetic`):

```bash
.venv311\Scripts\python scripts/generate_ocr_samples.py --count 12 --seed 42 --variants clean blur jpeg rotation
python benchmarks/validate_dataset.py --data benchmark_data
```

Biến thể hỗ trợ: `clean blur jpeg rotation perspective low_contrast small_text`. Không commit dữ liệu cá nhân/nhạy cảm.

### Building Real Benchmark Dataset (Phase 3.1 workflow)

```bash
# ① Import real images (requires privacy=safe; dedup by SHA256; creates unreviewed stubs)
.venv311\Scripts\python benchmarks/import_images.py --source my_real_images --data benchmark_data \
  --category camera --split test --source-type real --privacy safe --tags signage diacritics_heavy

# ② Annotate in browser (polygon edit, text entry, reviewed/rejected marking, image/region level)
.venv311\Scripts\python benchmarks/annotate.py --data benchmark_data --port 8765
#  → open http://localhost:8765/

# ③ Validate + audit + summary
.venv311\Scripts\python benchmarks/validate_dataset.py --data benchmark_data
.venv311\Scripts\python benchmarks/audit_annotations.py --data benchmark_data
.venv311\Scripts\python benchmarks/dataset_summary.py --data benchmark_data
```

* Intake private `benchmark_incoming/` (gitignored). Import never overwrites images. Requires `--privacy safe` for real data.
* Annotator writes atomically to `annotations.jsonl`; preserves `sha256`/`source_type`. Supports `ignore` regions.
* Coverage target: **≥100 real reviewed test** samples across `CATEGORY_PLAN` (`clean_print`, `camera`, `signage`, `diacritics_heavy`, `mixed_vi_en`, `low_contrast`, …). Check `Real reviewed test: x / 100` in summary/audit.
* Per-category flags (advisory, never suppresses): **SEVERE n<5**, **LOW n<10**, ok. Freeze at ≥100: `python benchmarks/freeze_testset.py --data benchmark_data --version 0.1` writes `benchmark_data/frozen/testset_v0.1.json` (ids+sha256 only, no image bytes). Refuses <100 unless `--allow-preliminary` (labels `PRELIMINARY NOT REPRESENTATIVE`). Do not run official benchmark on an unfrozen/<100 set.
* `benchmark_data/dataset.json` (`v0.3.1` Phase 3.1) records `counts`/`schema_version`/`privacy_gate`/`build_tools`/`coverage_targets`/`benchmark_cli_flags` + `provenance`; `git_hash=null` when not a git checkout (set via `git rev-parse HEAD` when versioned).

### Privacy gate

`benchmark_incoming/` và các file private (`dataset_summary.json`, `audit_report.txt`, `coverage_report.json`) bị `.gitignore`. Không commit `benchmark_data` khi chưa có `privacy=safe` trên mọi record (xem `benchmark_data/README.md` §Privacy Gate). Validate trước commit: `python benchmarks/validate_dataset.py --data benchmark_data`.

### Recognition-only vs End-to-end

```bash
# GT-crop → recognizer isolation (công bằng paddle vs tesseract, filtering flags documented)
.venv311\Scripts\python benchmarks/run_recognition_benchmark.py --data benchmark_data --recognizer paddle_v5_latin --preprocessing original
.venv311\Scripts\python benchmarks/run_recognition_benchmark.py --data benchmark_data --recognizer tesseract --preprocessing original
# full image → Detector → crop → recognizer → regions (đo riêng)
.venv311\Scripts\python benchmarks/run_scene_benchmark.py --data benchmark_data --recognizer paddle_v5_latin --preprocessing original
```

Preprocessing ablation (không đổi production default): **A: original perspective crop**, **B: grayscale→BGR**, **C: Otsu binary→BGR**. PaddleDetector baseline luôn trên ảnh gốc.

Canonical filtering (cả hai runner, via `benchmarks/dataset.py:filter_records`):

```bash
# Official quality reporting (đề xuất cho báo cáo chính thức)
.venv311\Scripts\python benchmarks/run_recognition_benchmark.py --data benchmark_data --recognizer paddle_v5_latin --split test --source real --reviewed-only
.venv311\Scripts\python benchmarks/run_scene_benchmark.py --data benchmark_data --recognizer paddle_v5_latin --split test --source real --reviewed-only
```

| flag | mô tả | mặc định |
|---|---|---|
| `--source real\|synthetic` | lọc nguồn | all |
| `--split train\|val\|test` | lọc split | all |
| `--reviewed-only` | chỉ `reviewed` (khuyến nghị cho báo cáo chính) | false |
| `--include-unreviewed` | giữ unreviewed (no-op khi `--reviewed-only`) | false |
| `--include-rejected` | giữ rejected (mặc định drop) | false |

`selection` + `dataset_counts` (by_source/review/split/category/tag) được ghi vào `summary.json` để tái tạo run.

### Metrics

- **CER** = `lev(pred,ref)/len(ref)`, **WER** = word-level Levenshtein / len(ref_words) (không phải char distance trên chuỗi nối), **exact_match** = `strip(pred)==strip(ref)` (strip trước so).
- **accent_stripped_cer** = CER sau `strip_accents()` (NFD Mn filter + đ/Đ map), **diacritic_loss_rate** = `cer - accent_stripped_cer` clamped ≥0. Ví dụ `Xin chào Việt Nam` vs `Xin chao Viet Nam`: normal CER>0, accent-stripped==0 ⇒ loss==CER. Không thay thế CER chuẩn, `raw_text` primary trước CER/WER, không spell-correct/LLM/strip.
- Vietnamese breakdown `ă â ê ô ơ ư đ` + tone marks qua NFD, **critical field exact_field_match** (`amount`/`tax_id`/`document_id`) strip case-sensitive. Ví dụ `012345678901` vs `0123456789O1` ⇒ false; `1.250.000` ⇒ true.
- **Detection**: IoU threshold 0.5 (ghi rõ), polygon IoU = shapely nếu có else bbox approximation (`ponytail: upgrade to shapely`). Greedy 1-1 matching → `precision/recall/F1/IoU` + counts `GT/pred/matched/missed/false_positive`. **Region matching by IoU**.

Công thức diacritic: `diacritic_loss_rate = max(0, cer(pred,ref) - cer(strip(pred),strip(ref)))`.

### Latency & confidence

`model_load_ms`, `detection_ms`, `recognition_ms`, `total_ms` riêng biệt, cold (first call) vs warm avg. Confidence bins `0.0-0.5/0.5-0.7/0.7-0.8/0.8-0.9/0.9-1.0` với `count/exact-rate/avg CER`.

### Output

`benchmark_results/<run-id>/` (timestamped, không overwrite):

```
recognition_results.csv  # id,image,split,category,region_idx,gt_text,pred_text,cer,wer,exact_match,accent_stripped_cer,diacritic_loss_rate,confidence,recognition_ms,preprocessing
scene_results.csv        # id,image,gt_regions,pred_regions,matched,missed,fp,precision,recall,f1,mean_iou,cer,wer,detection_ms,recognition_ms,total_ms
errors.csv               # id,region,gt,pred,cer,error,type (top CER>0.3)
summary.json / summary.md # model_meta: Python/PaddlePaddle/PaddleOCR/detector/recognizer/Tesseract/device/preprocessing/CLI/dataset/sample_count/git_hash + warnings
```

`sample_count <100` ⇒ warn `SMOKE ONLY NOT REPRESENTATIVE`. Smoke hiện tại (synthetic 12 samples, `benchmark_data`):

| run | CER | WER | exact | detection P/R/F1 | n | note |
|---|---|---|---|---|---:|---|
| rec paddle original | 0.134 | 0.426 | 0.167 | — | 12 | ablations: gray 0.134, otsu 0.150 |
| rec tesseract original | 0.176 | 0.369 | 0.500 | — | 12 | — |
| scene paddle original | 0.092 | — | — | 0.917/0.917/0.917 | 12 | end-to-end |

> FRAMEWORK READY — REAL QUALITY BENCHMARK PENDING DATA (add ≥100 real Vietnamese images for representative numbers). Không cài VietOCR/SVTR/MonkeyOCR/PaddleOCR-VL trong Phase 3.

### Tests

```bash
python -m pytest -m "not integration" -q    # 88 passed (23 new filter/metadata tests in Phase 3.1)
python benchmarks/validate_dataset.py --data benchmark_data
```

Preprocessing pipeline:

```
bytes → decode → grayscale → medianBlur(3) → Otsu threshold → Tesseract
```

Cố ý **không** deskew / morphology mạnh để tránh nuốt dấu tiếng Việt.

## Khả năng mở rộng

| Phase | Nội dung |
|---|---|
| 1 | Python + Tesseract |
| 2 | FastAPI |
| 3 | Docker |
| 4 | OpenCV preprocessing |
| 5 | Frontend React/Vue hoặc Streamlit |
| 6 | Deploy |

Hướng mở rộng khác:

- Batch OCR nhiều ảnh
- PDF OCR
- Bounding boxes trên frontend (`/ocr/data` đã trả sẵn)
- Export JSON / TXT
- Deskew, document orientation detection
- Lọc theo confidence (query `min_confidence` đã có)

## License

Open source. Tesseract dùng Apache 2.0.
