# Sửa chất lượng OCR — 2026-10-04

> Snapshot kết quả ngày 04. Bản 2026-10-05 thay default Scene tiếng Việt bằng
> hybrid, batch Tesseract và kiểm tra charset. Explicit Paddle trên artifact
> thiếu dấu hiện trả MODEL_LANGUAGE_UNSUPPORTED; số CER Paddle bên dưới là
> phép đo trước charset gate. Xem OCR_PERFORMANCE_FIX.md cho trạng thái mới.

## Thay đổi đã triển khai

- Document `auto` dùng `OCR_DOCUMENT_RECOGNIZER` (mặc định Tesseract), độc lập cache Paddle.
- Explicit Paddle ở Document phải detect/crop màu từng vùng và load model; không tự fallback.
- `paddle_tesseract` dùng Paddle detector + Tesseract PSM 7, cho cả Scene và Document.
- Scene mặc định vẫn là Paddle; dùng hybrid bằng UI hoặc cấu hình để thử trên dữ liệu thật.
- Tesseract adapter chỉ OCR một lần, dựng dòng từ TSV, chuyển BGR sang RGB, confidence 0–1 và raw score riêng.
- API, UI và health mô tả lựa chọn Document/hybrid; legacy endpoints giữ contract.
- Benchmark crop mặc định PSM 7, tính missing GT như chuỗi rỗng, tính page-level CER/WER với toàn bộ prediction, dùng confidence recognition thật.
- Generator dùng SHA-256 cho seed, polygon theo textbbox offset và cùng ma trận biến đổi ảnh.
- Không ghi đè dataset/benchmark cũ. Output benchmark dùng timestamp microsecond và từ chối thư mục trùng.

## Bằng chứng A4

Chạy trong process mới:

```powershell
.venv311\Scripts\python scripts/verify_ocr_quality.py
```

Trang dựng từ bốn ảnh synthetic có dấu trong bộ cũ, trên canvas 1240×1754:

| Request | Engine | Regions | CER | WER |
|---|---|---:|---:|---:|
| auto, lạnh | Tesseract | 1 | 0 | 0 |
| scene Paddle | Paddle Latin | 4 | 0.0946 | 0.4375 |
| auto, sau warmup | Tesseract | 1 | 0 | 0 |
| document hybrid | Tesseract trên crop Paddle | 4 | 0 | 0 |
| document explicit Paddle | Paddle Latin trên crop | 4 | 0.0946 | 0.4375 |

Script assert text/engine cold=warm, không fallback ngầm và explicit Paddle/hybrid tạo bốn vùng.
Trước sửa, auto sau warmup trả rỗng (CER 1.0) trên cùng trang.

## Benchmark synthetic đã sửa

```powershell
.venv311\Scripts\python benchmarks/validate_dataset.py --data benchmark_data_quality_v2
.venv311\Scripts\python benchmarks/run_scene_benchmark.py --data benchmark_data_quality_v2 --recognizer tesseract --psm 7
.venv311\Scripts\python benchmarks/run_scene_benchmark.py --data benchmark_data_quality_v2 --recognizer paddle_v5_latin
```

12 dòng synthetic, font hệ thống, seed 42, clean/blur/jpeg/rotation. Tesseract
dùng ngôn ngữ `vie` từ annotation; API mặc định `vie+eng`. Detection P/R/F1 = 1.0
cho cả hai. Macro CER toàn trang: hybrid Tesseract 0.0000; Paddle Latin 0.0799.
Đây là smoke test trên dữ liệu đơn giản, không phải bằng chứng đạt 100% trên ảnh thật.

Run ban đầu sau sửa:
- `benchmark_results/20261004_162846_924753_scene_tesseract_original/`
- `benchmark_results/20261004_162945_608601_scene_paddle_v5_latin_original/`

Hai run này tạo trước bản bổ sung metadata `languages`; nhãn model Tesseract trong
run đầu là `vie+eng` tĩnh, nhưng code thực sự truyền `vie` theo annotation. Các run
tiếp theo ghi ngôn ngữ thực tế.

## Giới hạn và bước tiếp theo

- Cần 30–50 ảnh thật cho pilot, tối thiểu 100 real/reviewed/test để đánh giá theo gate dự án.
- Chưa chọn engine mới toàn cục, chưa cài VietOCR hoặc fine-tune. Cần so sánh trên cùng crop và giữ test set tách khỏi tuning.
- Chưa có layout parser thật cho bảng/nhiều cột. Thứ tự bbox y/x chỉ là baseline.
- CER loại dấu chỉ là phép đo phụ; không diễn giải chênh lệch như phân rã nguyên nhân tuyệt đối.
- Confidence 0–1 chưa được calibration; không tự chọn engine bằng điểm confidence lớn nhất.
- Nếu đổi cấu hình Scene sang hybrid, rollback lựa chọn bằng `OCR_SCENE_RECOGNIZER=auto`; Document giữ Tesseract hoặc chọn explicit Paddle đã được segmentation.
