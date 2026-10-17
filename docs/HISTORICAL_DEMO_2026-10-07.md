# Demo nhanh — Vietnamese OCR

API OCR tiếng Việt. Scene auto = Paddle detector + Tesseract theo lô;
Document auto = Tesseract grayscale. Model Latin thiếu dấu bị từ chối 503.
Upload **multipart** field `file`. Không gửi raw octet-stream.

## 1. Chạy server

Python 3.11 bắt buộc cho Paddle (`.venv311`). Port 8000 mặc định; demo live dùng 8002 vì 8000 có thể bận.

```powershell
.venv311\Scripts\python -m pip install --prefer-binary -r requirements.txt
.venv311\Scripts\python -m pip install --prefer-binary -r requirements-paddle.txt
.venv311\Scripts\python scripts/warmup_models.py
.venv311\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

UI: http://localhost:8000  
Swagger: http://localhost:8000/docs

## 2. Health

```powershell
curl.exe http://localhost:8000/health/live
curl.exe http://localhost:8000/health/ready
```

- `/health/live` → `{"status":"ok"}` (process up, không load model).
- `/health/ready` trước OCR đầu: `status=degraded` `ready=false` `not_loaded`. Sau OCR: `status=ok` `ready=true`.

## 3. OCR ảnh mẫu

```powershell
# Scene tiếng Việt (route=scene, recognizer=auto hoặc paddle_tesseract)
curl.exe -X POST "http://localhost:8000/v1/ocr?route=scene&recognizer=auto" -F "file=@data/sample_scene.png"

# Tesseract legacy (opt-in)
curl.exe -X POST "http://localhost:8000/v1/ocr?route=scene&recognizer=tesseract" -F "file=@data/sample_scene.png"
```

PowerShell tương đương:

```powershell
$uri = "http://localhost:8000/v1/ocr?route=scene&recognizer=auto"
curl.exe -X POST $uri -F "file=@data/sample_scene.png"
```

Kỳ vọng Paddle 200:

```json
{
  "provenance": {
    "route": "scene",
    "detector": "paddle_detector",
    "recognizer": "paddle_v5_latin",
    "normalizer": "deterministic-v1"
  },
  "fallback_used": false,
  "text": "Xin chao Viet Nam\nOCR tieng Viet 2026\nMa: AB123456",
  "regions": [ "...3 regions..." ]
}
```

## 4. Kết quả demo live (2026-10-04, port 8002)

| Bước | Kết quả |
|---|---|
| `/health/live` | 200 `status=ok` |
| `/health/ready` trước OCR | 200 `degraded` `ready=false` detector/recognizer=`not_loaded` |
| POST scene + `paddle_v5_latin` `data/sample_scene.png` | 200 detector=`paddle_detector` recognizer=`paddle_v5_latin` `fallback_used=false` text=`Xin chao Viet Nam / OCR tieng Viet 2026 / Ma: AB123456` regions=3 |
| `/health/ready` sau OCR | 200 `ok` `ready=true` |
| POST scene + `tesseract` | 200 recognizer=`tesseract` `fallback_used=false` |

Ảnh trắng → 200 `text=""` `regions=[]` (đúng: không phát hiện text).

## 5. UI

Mở http://localhost:8000 — chọn `route` (auto/scene/document), `recognizer` (auto/paddle_v5_latin/tesseract), upload ảnh.

## 6. Lỗi thường gặp

| Lỗi | Sửa |
|---|---|
| `ModuleNotFoundError: paddle` | Dùng `.venv311`, `pip install -r requirements-paddle.txt` |
| `MODEL_UNAVAILABLE` 503 | Weights trong `%USERPROFILE%\.paddlex\official_models`; chạy `scripts/warmup_models.py` |
| `/health/ready` degraded | Bình thường trước OCR đầu. Sau load → `ok` |
| Port bận | `--port 8002` hoặc kill process giữ 8000 |
| PowerShell `Invoke-RestMethod` + `octet-stream` | Sai. Dùng `curl.exe -F "file=@..."` |

## 7. Benchmark smoke (n=12, không đại diện)

```powershell
.venv311\Scripts\python benchmarks/run_recognition_benchmark.py --data benchmark_data --recognizer paddle_v5_latin --limit 4
.venv311\Scripts\python benchmarks/run_scene_benchmark.py --data benchmark_data --recognizer paddle_v5_latin
```

Chi tiết: `README.md`, `PROJECT_LOG.md` §18.
