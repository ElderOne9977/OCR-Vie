# Cải thiện OCR đã triển khai — 2026-10-05

## Kết quả

Auto Scene với tiếng Việt dùng Paddle detector + Tesseract PSM 7 theo lô;
Document dùng Tesseract toàn trang và grayscale. Mặc định ngôn ngữ vẫn là
`vie+eng`, UI cho phép chọn `vie` hoặc `eng` riêng.

Model Latin đang cài thiếu 90/146 chữ/nguyên âm tiếng Việt kiểm tra theo NFC,
và không có combining mark để ghép những chữ thiếu. Runtime kiểm tra decoder
thực tế, gồm cả cách viết decomposed; explicit model không đủ charset trả
`503 MODEL_LANGUAGE_UNSUPPORTED` cả trước và sau warmup. Không sửa dictionary
hay thay trọng số của model; không fallback ngầm khi user chọn explicit.

## Số đo trên máy hiện tại

Môi trường và cách dựng ảnh giống `OCR_PERFORMANCE_ANALYSIS.md`: Windows,
Ryzen 7 H255 8 nhân/16 luồng, Python 3.11.7, PaddleOCR 3.7.0/PaddleX 3.7.2/
PaddlePaddle 3.2.0, Tesseract 5.4.0.20240606. Dataset synthetic 12 ảnh được ghép
không resize lên trang trắng BGR 1240×1754 tại x=60, y=60+130*i.

SHA-256 annotations: `C8E88ECAA2E75B10888713F361A1D3B40DA76458A012724E4BB33E4CB7010D5C`.
Đo tuần tự trong process riêng cho mỗi engine; warm 3 lần, không chạy benchmark
CPU khác đồng thời. Peak RAM cộng RSS của process và subprocess, lấy mẫu 20 ms;
CPU-seconds có thể bỏ sót subprocess ngắn, không phải điện năng. Không có p95
hay kiểm thử throughput tải thật đủ để chốt năng lực sản xuất.

| Phép đo 12 dòng | Trước | Sau |
|---|---|---|
| Hybrid end-to-end warm | 4,616 / 4,639 / 4,676 s | Lượt A: 1,436 / 1,628 / 1,639 s; lượt B: 1,637 / 1,678 / 1,789 s |
| Hybrid CER/WER | 0 / 0 | 0 / 0 ở cả hai lượt |
| Hybrid CPU-seconds ước lượng/request | 17,34–17,98 | 12,61–13,13 ở lượt B |
| Hybrid peak RAM | ~886–897 MiB | ~882–899 MiB, không giảm rõ rệt |
| Document end-to-end warm, vie+eng | 0,530 / 0,545 / 0,563 s | 0,458 / 0,467 / 0,470 s |
| Document CER | 3,57% | 2,68% |
| Document peak RAM | ~164 MiB | ~160–161 MiB |

Median hybrid end-to-end giảm khoảng 64–65% trong các lượt này. RAM vẫn bị
chi phối bởi Paddle detector; giữ một cache dùng chung giúp tránh nhân bản
model theo mỗi request. CPU giảm là ước lượng theo cùng cách lấy mẫu, không
diễn giải thành mức tiết kiệm điện hay bảo đảm cho mọi máy.

So riêng recognition trên **cùng 12 crop do detector trả về**:

| Chế độ | Lượt A (ms) | Lượt B (ms) |
|---|---|---|
| Tesseract từng crop | 2881 / 3246 / 3743 | 3091 / 3422 / 3633 |
| Tesseract batch | 525 / 551 / 556 | 445 / 535 / 671 |

Mỗi lượt assert toàn bộ `RecognitionResult.to_dict()` của batch bằng từng
invocation: raw/normalized text, confidence, model, thứ tự. Có thêm crop trắng
ở giữa để kiểm tra không dịch ánh xạ. Tesseract PSM 7 có thể tự sinh ký tự trên
ảnh trắng ở cả hai chế độ; kiểm tra này xác nhận vị trí/kết quả tương đương,
không cam kết mọi ảnh trắng sẽ trả chuỗi rỗng.

Cold hybrid vẫn ~8,81–10,00 s trên trang 12 dòng; lượt B đo `initialization`
~7,85 s. Không so trực tiếp số này với cold 4 dòng trong báo cáo trước. Script
warmup chạy ở process riêng chỉ kiểm chứng/cache trọng số trên đĩa; không làm
nóng process Uvicorn. Framework import/load vẫn là phần cần lưu ý khi khởi động.

Health ready trong process mới: ~1,73 s trước → ~0,033 s sau, không import
`paddleocr`. Tesseract status tận dụng version cache native của pytesseract;
binary/cấu hình thay đổi cần khởi động lại process, OCR thật vẫn báo lỗi nếu
binary không chạy được.

## Kiểm soát tài nguyên

- `MAX_CONCURRENT_OCR=2` áp dụng chung cho `/ocr`, `/ocr/data`, `/v1/ocr`,
  `/v1/ocr/text` trong một process. Router gọi trực tiếp cũng được giới hạn.
  Admission không có queue: quá tải trả `503 OCR_BUSY` và `Retry-After: 1`.
  Khóa khởi tạo semaphore ngăn nhiều limiter được tạo khi request đầu đến đồng thời.
- Upload chỉ đọc tối đa `MAX_UPLOAD_BYTES+1`, mặc định 10 MiB; FastAPI vẫn có
  bước parse/spool multipart trước khi endpoint chạy. Giới hạn này không phải
  giới hạn tổng lưu lượng request body ở reverse proxy.
- Pillow đọc header trước OpenCV decode: `MAX_IMAGE_PIXELS=12000000`,
  `MAX_IMAGE_SIDE=8192`; kiểm tra lại sau decode. Legacy và hybrid dùng chung
  decoder. Vượt giới hạn trả 413.
- `MAX_REGIONS=256`, giới hạn tổng pixel crop hybrid bằng `MAX_IMAGE_PIXELS`;
  không cắt bớt vùng âm thầm. Polygon invalid hoặc kết quả batch không đủ trang
  trả lỗi rõ ràng.
- `TESSERACT_BATCH_SIZE=32`: lưu crop PNG trong TemporaryDirectory và dùng native
  image-list input. TSV `page_num` ánh xạ mỗi crop về region, kể cả trang chỉ có
  header. Temp files được dọn khi thành công hoặc có exception/timeout.
- `OCR_TIMEOUT_SECONDS=30` áp dụng mỗi invocation Tesseract, cả legacy và batch;
  timeout trả `504 OCR_TIMEOUT`. Đây không phải timeout toàn request/Paddle;
  một request nhiều batch có thể kéo dài hơn 30 s.
- `DEVICE`/`CPU_THREADS` được truyền vào constructor và tham gia khóa cache model.
  Default `cpu/10` giữ nguyên thread setting của SDK trước đây; không áp dụng
  thread=1 vì phép đo trước chưa chứng minh lợi ích rõ ràng. Native thread count
  toàn process có thể lớn hơn CPU_THREADS.
- Shared Paddle predictors serialize cả lúc consume generator; Tesseract có
  thể chạy trong slot khác sau khi detector hoàn tất. Nhiều Uvicorn worker sẽ
  nhân cache model và tổng mức concurrency; giới hạn không xuyên process.

`timing_ms` giữ các trường cũ và thêm `decode`, `initialization`, `preprocessing`.
Initialization gồm status/dependency/khởi tạo model; recognition hybrid gồm
rectify crop, lưu ảnh, subprocess và parse TSV. Chưa đo riêng queue nội bộ
predictor hoặc điện năng.

## Chạy lại kiểm chứng

Kiểm chứng cuối: **160 tests passed**, gồm 3 Paddle integration; compileall
thành công. Một warning `ccache` đến từ Paddle SDK, không ảnh hưởng kết quả.

```powershell
.venv311\Scripts\python -m pytest -q
.venv311\Scripts\python -m compileall -q app scripts tests
.venv311\Scripts\python scripts/verify_ocr_quality.py
.venv311\Scripts\python scripts/verify_ocr_performance.py --engine document
.venv311\Scripts\python scripts/verify_ocr_performance.py --engine hybrid
```

Performance verifier tắt model download trong process của nó; `psutil` là tùy
chọn, không thêm dependency sản xuất. Không có psutil thì latency/quality vẫn
đo được, các trường RAM/CPU trả null.

Mẫu A4 từ dataset cũ vẫn CER/WER 0 cho Document auto lạnh/nóng và hybrid 4 vùng;
explicit Paddle hiện trả đúng lỗi charset. Mẫu đó khác trang 12 dòng ở bảng trên.

## Phạm vi áp dụng và rollback

Đợt kiểm tra tiếp theo (không đổi cấu hình sản xuất):
[OCR_OPTIMIZATION_REVIEW.md](OCR_OPTIMIZATION_REVIEW.md), gồm ablation detector
768/960/1280 px, CPU threads, MKL-DNN và stress test chữ nhỏ.

Các số trên chỉ là synthetic smoke. Cần ảnh thực tế độc lập để đánh giá scan,
hóa đơn, ảnh điện thoại, chữ nhỏ/mờ/xoay, bảng/nhiều cột và song ngữ. Thứ tự đọc
hiện vẫn theo bbox y/x; parser không có layout model thật. Không tự chuyển
engine dựa trên confidence chưa hiệu chuẩn.

Nếu grayscale kém trên nhóm tài liệu cụ thể, đổi `DOCUMENT_PREPROCESS=binary`
để dùng tiền xử lý cũ hoặc `original`; khởi động lại server. `lang=vie` là profile
tùy chọn đã cho CER tốt hơn ở ablation trước; không đổi default song ngữ chỉ
dựa trên synthetic. Charset gate không được bỏ qua bằng sửa tay dictionary.

Thay đổi giữ các dataset/report cũ, không cài dependency hay tải model mới.
