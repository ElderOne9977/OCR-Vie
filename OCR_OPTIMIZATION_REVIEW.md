# Kiểm tra tối ưu tiếp — 2026-10-05

## Kết luận

Còn dư địa, chủ yếu ở **detector**, nhưng không có một cấu hình vừa nhanh nhất,
vừa ít RAM nhất, vừa tốt nhất cho chữ nhỏ. Đợt phân tích ban đầu chỉ thêm công
cụ đo và báo cáo. Sau yêu cầu “Bắt đầu tối ưu”, đã triển khai các thay đổi ở
phần cuối tài liệu; vẫn giữ mặc định detector 960 px. Không cài thư viện hoặc tải model.

## Thí nghiệm trên máy hiện tại

Model cached `PP-OCRv5_server_det`, CPU; cùng trang 1240×1754 ghép 12 ảnh đầu
của `benchmark_data_quality_v2/annotations.jsonl`, ngôn ngữ `vie+eng`.
Mỗi cấu hình chạy trong process mới: 1 lượt detector lạnh + 4 lượt nóng.
Bảng lấy median 4 lượt nóng. CPU là user+system của process Python trong detector,
không phải điện năng. RAM là RSS đỉnh của process detector, lấy mẫu mỗi 20 ms;
không gồm subprocess Tesseract. CER đo kết quả crop + Tesseract batch sau detector.

| Cấu hình detector | Median nóng | CPU giây/lượt | RAM đỉnh | Vùng | CER |
|---|---:|---:|---:|---:|---:|
| 960 px, 10 luồng, MKL-DNN bật | 996,9 ms | 12,344 | 881,1 MiB | 12 | 0% |
| 960 px, 4 luồng, MKL-DNN bật | 1075,9 ms | 11,648 | 879,4 MiB | 12 | 0% |
| 768 px, 10 luồng, MKL-DNN bật | 709,6 ms | 8,008 | 748,0 MiB | 12 | 0% |
| 960 px, 10 luồng, MKL-DNN tắt | 11743,0 ms | 60,555 | 3523,1 MiB | 12 | 0% |

768 px giảm thời gian detector **28,8%**, CPU **35,1%**, RSS đỉnh **15,1%**
so với baseline trong thí nghiệm này. Đây **không** phải tỷ lệ giảm toàn request:
decode, crop và Tesseract vẫn mất thời gian. Cấu hình 4 luồng không có lợi ích
tốc độ/RAM rõ ràng; CPU giảm nhẹ cần đo lại trước khi kết luận.
Tắt MKL-DNN không phù hợp trên máy này: chậm khoảng 11,8 lần, RAM đỉnh gấp 4 lần.
Không suy từ RSS sau xử lý thấp rằng RAM đỉnh cũng thấp.

### Stress test chữ nhỏ

Thu nhỏ từng ảnh chữ xuống 50% bằng `INTER_AREA`, giữ nguyên kích thước trang.
Đây là ảnh synthetic mới dẫn xuất; không so CER trực tiếp với bảng chữ gốc.

| Giới hạn detector, 10 luồng, MKL-DNN bật | Median nóng | RAM đỉnh | Vùng | CER |
|---|---:|---:|---:|---:|
| 768 px | 691,8 ms | 745,2 MiB | 11 | 9,38% |
| 960 px | 1053,8 ms | 884,1 MiB | 11 | 9,82% |
| 1280 px | 1670,4 ms | 1153,4 MiB | 12 | 2,23% |

1280 px phát hiện đủ 12 dòng, CER giảm khoảng 77% tương đối so với 960 px,
đổi lại detector chậm thêm 58,5% và RAM đỉnh tăng 30,5%. 768 px chưa làm chất
lượng kém hơn 960 px trên mẫu này, nhưng cả hai cùng bỏ sót một dòng; không có
cơ sở coi 768 px an toàn cho mọi ảnh. Số vùng chỉ kiểm tra độ đầy đủ của mẫu
12 dòng, **không phải** metric detection recall/IoU tổng quát.

## Khuyến nghị theo thứ tự

1. **Cho phép chọn độ phân giải detector**, giữ 960 px làm mặc định hiện tại.
   Profile nhanh 768 px cho ảnh chữ đủ lớn; profile chữ nhỏ 1280 px khi cần độ
   đầy đủ. Dùng tùy chọn native `limit_side_len` + `limit_type="max"`, không tự
   resize ảnh gốc dùng nhận dạng. Paddle trả polygon theo tọa độ ảnh gốc nên
   crop vẫn giữ độ phân giải gốc. Chưa tự động chọn profile theo confidence:
   vùng bị bỏ sót không có confidence để cảnh báo.
2. **Siết RAM trước cấp phát crop**. Trong `OCRRouter._run_scene_neural`, tổng
   pixel hiện được kiểm tra *sau* `crop_polygon`/`warpPerspective`. Một crop
   làm vượt ngân sách vẫn được cấp phát rồi mới bị từ chối. Nên tính trước
   kích thước đầu ra bằng chính quy tắc hình học đang dùng và chặn trước warp;
   cần regression test chứng minh warp không được gọi khi vượt ngân sách.
   Đây là cải thiện an toàn tài nguyên, không phải cách tăng CER.
3. **Giới hạn thời gian toàn request**. `OCR_TIMEOUT_SECONDS` hiện là timeout
   mỗi lần chạy Tesseract; 256 vùng chia batch 32 có thể gọi 8 lần, chưa kể
   Paddle/khóa predictor. Không diễn giải 30 giây thành deadline toàn pipeline.
   Deadline còn lại có thể truyền xuống từng batch; hủy native Paddle giữa
   chừng cần worker process, không dùng thread timeout để giả vờ đã dừng.
4. **Dữ liệu thực và routing**. Classifier hiện dùng kích thước/aspect ratio;
   scan nhỏ có thể đi scene tốn detector, ảnh điện thoại giống A4 có thể đi
   document kém chất lượng. Đo routing cùng CER/độ đầy đủ trên ảnh thực rồi
   mới chỉnh heuristic. Tài liệu sạch có thể chọn rõ `route=document` hiện có;
   nội dung thuần Việt có thể chọn `lang=vie` hiện có. Không đổi mặc định song ngữ.

Giữ MKL-DNN bật. Chưa thay detector mobile/model mới vì chưa có weights cached
và chưa đo chất lượng. Warmup trong server có thể chuyển chi phí cold start
sang lúc khởi động, nhưng không giảm tổng chi phí và sẽ giữ RAM thường trực.
Tăng số worker nhân bản model/cache; cần load test trước khi tăng concurrency.

## Chạy lại

`scripts/probe_detector.py` dùng `psutil` đã có trong `.venv311`; chạy từng lệnh
tuần tự để tránh tranh CPU/RAM. Tắt MKL-DNN có thể mất hơn một phút và dùng
trên 3 GiB RAM trên máy này.

```powershell
.venv311\Scripts\python scripts/probe_detector.py --threads 10 --side 960
.venv311\Scripts\python scripts/probe_detector.py --threads 4 --side 960
.venv311\Scripts\python scripts/probe_detector.py --threads 10 --side 768
.venv311\Scripts\python scripts/probe_detector.py --threads 10 --side 960 --no-mkldnn
.venv311\Scripts\python scripts/probe_detector.py --threads 10 --side 960 --text-scale 0.5
.venv311\Scripts\python scripts/probe_detector.py --threads 10 --side 768 --text-scale 0.5
.venv311\Scripts\python scripts/probe_detector.py --threads 10 --side 1280 --text-scale 0.5
```

Các kết quả là kiểm tra synthetic, một process cho mỗi cấu hình, chưa phải
benchmark p95/tải đồng thời hoặc cam kết chất lượng trên hóa đơn, bảng nhiều
cột, tài liệu thực. Sai số đo, tải nền và mức RAM trống có thể ảnh hưởng kết quả.

## Đã triển khai sau kiểm tra

- `DETECTOR_LIMIT_SIDE_LEN`: mặc định 960, giới hạn 32–4000; 768 là lựa chọn
  ưu tiên tốc độ, 1280 cho chữ nhỏ. Truyền trực tiếp tùy chọn native Paddle,
  giữ ảnh gốc khi crop, giữ MKL-DNN. Cấu hình hiệu lực có trong `/health/ready`.
- `crop_polygon(..., max_pixels=...)`: kiểm tra diện tích đầu ra trước copy
  hoặc warp; router trừ ngân sách còn lại cho cả batch và đường từng vùng.
  Vượt ngân sách trả 413, không cấp phát crop làm vượt giới hạn trước khi từ chối.
- `OCR_REQUEST_TIMEOUT_SECONDS=60`: deadline hợp tác, bắt đầu sau admission
  trước decode; ContextVar tách biệt theo request. `OCR_TIMEOUT_SECONDS=30`
  vẫn giới hạn mỗi invocation Tesseract, được cắt theo thời gian request còn lại.
  Chờ khóa model initialization/predictor cũng dùng thời gian còn lại.
  Timeout trả 504 `OCR_TIMEOUT`, không bị đổi thành 503 model thiếu hoặc 500
  recognition lỗi; không ghi timeout thành model load failure.
- Đã kiểm thử trước/sau deadline, nhả slot/khóa/context khi lỗi, batch cleanup,
  crop đúng bằng ngân sách và không gọi warp cho crop vượt tổng ngân sách.

Deadline **không phải hard cancellation**: Paddle/import/load model đang chạy
vẫn phải trả về rồi mới nhả slot/khóa; nếu quá hạn thì không tiếp tục hoặc trả
success. Upload/network trước admission và serialization sau pipeline không
thuộc budget. Chạy OCR adapter riêng ngoài request vẫn dùng timeout Tesseract
mỗi invocation; muốn deadline chung dùng context `ocr_deadline()`.

### Đo cả pipeline sau triển khai

`verify_ocr_performance.py --engine hybrid`, hai process mới, cùng synthetic 12
dòng và `vie+eng`. Mỗi profile 1 lạnh + 3 nóng; bảng ghi 3 lượt nóng:

| Detector side | Toàn request (ms) | Median | RSS đỉnh cây process (MiB) | CER/WER |
|---|---|---:|---|---|
| 960 | 1414,3 / 1427,6 / 1461,4 | 1427,6 ms | 872,7 / 875,0 / 883,9 | 0 / 0 |
| 768 | 1175,0 / 1261,3 / 1269,2 | 1261,3 ms | 752,8 / 764,8 / 766,4 | 0 / 0 |

768 giảm median **11,6% toàn request**, RSS đỉnh median **12,6%**;
CPU ước lượng median giảm 12,016 → 8,969 giây (**25,4%**). RSS cây process có
thể đếm lặp trang chia sẻ; CPU subprocess lấy mẫu có thể thiếu process ngắn.
Không gọi đây là tiết kiệm điện hoặc p95. Cả hai nhận 12 vùng, batch bằng kết
quả/confidence từng crop, ánh xạ vùng trắng giữ nguyên. Cold request tương ứng
9753,5 / 8681,9 ms, phần lớn là khởi tạo; không suy mức cải thiện cold start
ổn định từ chỉ một mẫu/profile. Mặc định không tự bật 768.

Ví dụ bật profile nhanh trước khi khởi động server trong PowerShell:

```powershell
$env:DETECTOR_LIMIT_SIDE_LEN = '768'
.venv311\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Đổi thành `'960'` để giữ cấu hình cân bằng, `'1280'` cho chữ nhỏ. Không tăng
worker đồng thời chỉ vì detector nhanh hơn; đo chất lượng ảnh thực và load test
trước khi thay đổi mặc định hoặc mức concurrency.

Kiểm chứng cuối: **189 tests passed** (29 kiểm thử mới), gồm Paddle integration;
compileall thành công. `verify_ocr_quality.py` trên mẫu A4: Document auto
cold/warm và hybrid 4 vùng vẫn CER/WER 0; explicit Paddle tiếng Việt vẫn bị
charset gate từ chối đúng. Warning thiếu `ccache` của SDK không làm test thất bại.
Server chưa được khởi động/restart trong đợt này.
