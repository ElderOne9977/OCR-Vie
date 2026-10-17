# OCR-Vie

OCR tiếng Việt/Anh từ ảnh, FastAPI và UI upload cục bộ. Python **3.11**, môi trường chuẩn `.venv311`.

## Chạy trên Windows

Cài Python 3.11 và Tesseract có traineddata `vie`, `eng`; nếu binary nằm ngoài vị trí thông thường, đặt `TESSERACT_CMD`. Setup không tự cài binary hoặc tải model.

```powershell
.\scripts\setup.ps1
$env:ALLOW_MODEL_DOWNLOAD = 'false'
$env:MODELS_DIR = (Resolve-Path .\models).Path
.\scripts\run_local.ps1
```

UI: http://127.0.0.1:8000 — API: http://127.0.0.1:8000/docs. Ctrl+C để dừng. `run_local.ps1` không cài/upgrade packages; `-Port 8002`, `-Reload` là tùy chọn. `-Warmup` nạp model **trong process server** và từ chối khởi động nếu pipeline không ready.

Repository kèm hai model PaddleOCR tại `models/official_models`; xem [model và giấy phép](models/README.md). Lệnh trên dùng bản model trong checkout, không cần tải lại weights. Tesseract và traineddata vẫn cần cài riêng.

Hybrid cần Paddle detector đã chuẩn bị offline. Cache mặc định: `%USERPROFILE%\.paddlex\official_models\PP-OCRv5_server_det`; nếu đặt `MODELS_DIR`, dùng `<MODELS_DIR>/official_models/<model>`. `PADDLE_PDX_CACHE_HOME` ưu tiên cao hơn. Artifact tối thiểu: `inference.pdiparams`, `inference.json` hoặc `inference.pdmodel`, `inference.yml`. Có thư mục không đồng nghĩa model nạp được.

Chỉ cần document/Tesseract: cài `requirements.txt`, `requirements-dev.txt` trong Python 3.11, đặt `OCR_SCENE_RECOGNIZER=tesseract`. Setup mặc định cài cả hybrid. Không cài đồng thời nhiều wheel OpenCV: PaddleX hiện yêu cầu `opencv-contrib-python==4.10.0.84`.

## Pipeline và giới hạn

| Lựa chọn | Đường chạy |
| --- | --- |
| Scene auto, `vie`/`vi` | Paddle detector → crop phối cảnh → Tesseract PSM 7 batch |
| Scene auto, ngôn ngữ khác | Paddle detector → Paddle recognizer theo language |
| Document auto | Tesseract toàn trang, mặc định grayscale/PSM 6 |
| Explicit `paddle_tesseract` | Hybrid ở scene/document |
| Explicit `tesseract` | Tesseract toàn ảnh |
| Explicit `paddle_v5_latin` | Paddle, không tự chuyển engine khi thiếu model/charset |

Model Latin hiện có thiếu ký tự tiếng Việt; explicit Paddle tiếng Việt trả `503 MODEL_LANGUAGE_UNSUPPORTED`. Không sửa charset bằng cách thêm dictionary không tương thích trọng số. Confidence 0–1 không phải xác suất đúng đã hiệu chỉnh. Legacy `/ocr`, `/ocr/data` giữ contract cũ; raw OCR không bị suy đoán sửa dấu/số.

Nhận PNG/JPG/WebP/BMP/TIFF, multipart field `file`. **Chưa hỗ trợ PDF, layout nhiều cột/bảng đáng tin cậy hoặc chữ viết tay.** Scene gom vùng cùng dòng theo overlap rồi đọc trái→phải, không phải bộ phân tích cột.

Mặc định: file 10 MiB, tổng body ≤ file cap + 64 KiB metadata, 12 triệu pixel, cạnh 8192, 256 vùng, 2 OCR đồng thời/process. Body vượt cap bị chặn khi đang đọc, trước khi spool hết multipart. `CPU_THREADS=10`, detector side 960, batch 32 giữ nguyên. `OCR_BUSY` trả 503 + Retry-After, không tạo hàng đợi. Typo route/recognizer/parser/response_format trả 422. Lỗi nội bộ được che, response có `X-Request-ID` để truy log.

Deadline OCR 60 giây là cooperative; mỗi lần Tesseract tối đa 30 giây và dùng budget còn lại. Không ngắt tức thì native Paddle/model load. Timeout mạng/upload phải cấu hình ở gateway trước khi public. Nhiều Uvicorn worker nhân bản model/RAM và tăng admission tổng.

## Health và kiểm thử

`/health/live`: HTTP 200 nếu process sống. `/health/ready`: **200 khi primary scene ready, 503 khi chưa ready**; cold `not_loaded` khác `load_failed`/`unavailable`. Health không nạp model. Document báo độc lập trong JSON; `/health` legacy vẫn 200 nhưng có thể `degraded`.

```powershell
.\scripts\test.ps1
.\.venv311\Scripts\python.exe -m pytest -m 'not integration and not paddle' -q
$env:OCR_REQUIRE_MODELS = '1'
.\.venv311\Scripts\python.exe -m pytest tests/integration/test_paddle_scene_ocr.py::test_vietnamese_hybrid_strict -q
.\.venv311\Scripts\python.exe scripts/smoke_test.py --base-url http://127.0.0.1:8000
```

Strict integration/smoke yêu cầu model thật và chữ Việt đúng trên fixture cố định; 503/ảnh rỗng không được xem là pass. `smoke_test.py --allow-unavailable` chỉ chẩn đoán, **không là release gate**. Test không tự tải model. CI chưa cấu hình; kết quả test cục bộ không thay thế nghiệm thu triển khai.

## Benchmark đúng phạm vi

```powershell
.\.venv311\Scripts\python.exe -m benchmarks.validate_dataset --data benchmark_data_quality_v2
.\.venv311\Scripts\python.exe -m benchmarks.run_scene_benchmark --data benchmark_data_quality_v2 --include-unreviewed --mode production --recognizer auto
.\.venv311\Scripts\python.exe -m benchmarks.run_recognition_benchmark --data benchmark_data_quality_v2 --include-unreviewed --recognizer tesseract
.\.venv311\Scripts\python.exe scripts/verify_ocr_performance.py --engine hybrid --repeats 30
```

Scene production gọi router giống API; `--mode ablation` tách đường thử preprocess/engine cũ. Recognition là **GT-crop only**, không thay thế end-to-end. Polygon IoU mặc định OpenCV convex, `--iou-backend bbox` là metric khác; Shapely phải cài sẵn nếu yêu cầu polygon concave. Không fallback metric âm thầm. Field chỉ chấm khi region có `field_key` tồn tại trong `fields`; thiếu mapping báo null/coverage chưa đánh giá.

Manifest v2 hash ảnh thực và canonical GT. Freeze version không ghi đè; GT/ảnh thay đổi cần version mới. Annotator có kiểm tra revision để tránh tab cũ ghi đè, chỉ phục vụ ảnh trong dataset. Đọc [data/README.md](data/README.md) trước khi nhập dữ liệu riêng tư.

Hai bộ hiện tại chỉ có 24 ảnh synthetic, **0 real reviewed test**: kết quả là preliminary, không chứng minh độ chính xác ngoài thực tế. Gate ban đầu cần ≥100 ảnh thật được phép sử dụng, có review và holdout.

## Docker

Dockerfile dùng hybrid/Python 3.11, non-root UID 10001, tải model bị tắt, warmup trong server. Mount cache `/models` có `official_models/PP-OCRv5_server_det` và quyền đọc/ghi cho UID này; không mount dataset riêng tư vào image.

```powershell
docker build -t ocr-vie .
docker run --rm -p 127.0.0.1:8000:8000 -v C:/ocr-model-cache:/models ocr-vie
```

Thay đường dẫn mount bằng cache đã chuẩn bị. **Container chưa nghiệm thu**: Docker daemon đã hoạt động ngày 16/10; build/kiểm thử container vẫn cần hoàn tất. Không dùng hướng dẫn này như bằng chứng production-ready.

## Tài liệu

- [Kế hoạch cải thiện 16/10](docs/superpowers/plans/2026-10-16-project-improvements.md), [tiến độ thực hiện](docs/IMPLEMENTATION_RESULTS_2026-10-16.md)
- [Kế hoạch và trạng thái](docs/superpowers/plans/2026-10-07-project-improvements.md)
- [Kết quả triển khai](docs/IMPLEMENTATION_RESULTS_2026-10-07.md)
- [Audit trước sửa](docs/PROJECT_AUDIT_2026-10-07.md)
- [Demo cục bộ](DEMO.md), [lịch sử dự án](PROJECT_LOG.md)
- [README lịch sử](docs/HISTORICAL_README_2026-10-07.md), [demo lịch sử](docs/HISTORICAL_DEMO_2026-10-07.md): chỉ để tra cứu, không dùng làm hướng dẫn chạy hiện hành.
