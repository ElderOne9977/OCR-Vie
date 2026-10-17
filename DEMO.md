# Demo OCR-Vie cục bộ

Đọc yêu cầu Python 3.11/Tesseract/model offline trong [README](README.md) trước. Demo không cài lại packages mỗi lần chạy.

```powershell
$env:ALLOW_MODEL_DOWNLOAD = 'false'
.\scripts\run_local.ps1 -Port 8000 -Warmup
```

Mở http://127.0.0.1:8000, chọn ảnh → Nhận dạng → kiểm tra dấu/số → Sao chép hoặc Tải .txt. Chọn Document cho trang in, Scene cho biển hiệu/ảnh thực tế; mặc định tự động. Đổi/bỏ ảnh sẽ bỏ kết quả cũ. Nếu server bận, giữ ảnh và thử lại sau thời gian UI báo.

```powershell
curl.exe http://127.0.0.1:8000/health/live
curl.exe http://127.0.0.1:8000/health/ready
curl.exe -X POST 'http://127.0.0.1:8000/v1/ocr?route=scene&lang=vie%2Beng' -F 'file=@data/sample_scene.png'
.\.venv311\Scripts\python.exe scripts/smoke_test.py --base-url http://127.0.0.1:8000
```

Thay ảnh mẫu bằng ảnh được phép sử dụng. `/health/ready` trả 503 lúc cold nếu không bật warmup, hoặc khi thiếu component; `/health/live` không xác nhận model. Explicit Paddle tiếng Việt có thể trả `MODEL_LANGUAGE_UNSUPPORTED`, không phải lý do cài lại Tesseract.

Dừng bằng Ctrl+C tại terminal đã chạy server. Không kill mọi process Python hoặc mở listener `0.0.0.0` chỉ để demo trên máy này. Phiên kiểm thử triển khai không để demo chạy nền sau khi kết thúc.
