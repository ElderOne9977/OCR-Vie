# Phân tích tốc độ, chất lượng và tài nguyên OCR — 2026-10-04

> Snapshot phân tích trước triển khai. Bản sửa và số đo sau thay đổi ngày
> 2026-10-05 nằm trong [OCR_PERFORMANCE_FIX.md](OCR_PERFORMANCE_FIX.md).

## Phạm vi

Đây là kết quả chẩn đoán và đề xuất, chưa thay đổi mã chạy OCR. Không cài thêm thư viện, tải model hay thay đổi cấu hình hệ thống. Các thử nghiệm cấu hình chỉ áp dụng trong tiến trình đo.

Mục tiêu: giảm CER tiếng Việt, độ trễ và RAM/CPU cho ứng dụng hiện tại. Chưa có ngân sách latency/chất lượng do người dùng chốt; chưa đủ dữ liệu thật để chọn cấu hình sản xuất.

## Môi trường và phương pháp

- Windows; AMD Ryzen 7 H255, 8 nhân/16 luồng; RAM 15,31 GiB, còn khoảng 2,07 GiB tại thời điểm bắt đầu.
- Python 3.11.7 trong `.venv311`; PaddleOCR 3.7.0, PaddleX 3.7.2, PaddlePaddle 3.2.0; pytesseract 0.3.13; Tesseract 5.4.0.20240606; psutil 7.2.2.
- Nguồn: 12 ảnh và ground truth trong `benchmark_data_quality_v2/annotations.jsonl`, gồm sạch/mờ/JPEG/xoay. Không phải bộ dữ liệu tài liệu thực tế.
- Ghép 4 hoặc 12 ảnh đầu theo thứ tự annotation, không resize, lên canvas trắng BGR 1240×1754. Mỗi ảnh đặt tại x=60, y=60+130*i. Ground truth nối bằng newline.
- Mỗi engine chạy trong tiến trình mới: trang 4 dòng trước, trang 12 dòng sau; mỗi trang lặp 3 lần, tuần tự, không tải đồng thời.
- Đây KHÔNG phải cùng mẫu A4 của `scripts/verify_ocr_quality.py` dùng dữ liệu cũ. Không so trực tiếp CER giữa hai phép thử.
- Wall time bao gồm lời gọi router; thời gian import ứng dụng đo riêng. Cold call bao gồm import/load model bên trong router. Thời gian nhận dạng trong thử nghiệm ablation không phải end-to-end.
- Peak RAM là tổng working set/RSS của tiến trình và tiến trình con, lấy mẫu mỗi 20 ms; không phải bộ nhớ unique, không phải VRAM. CPU-seconds là ước lượng có thể bỏ sót subprocess ngắn; không phải điện năng.
- Không có load test concurrency, p95 đáng tin cậy hay đo watt. Kết quả nhỏ này dùng tìm nguyên nhân, không bảo đảm hiệu năng sản xuất.

## 1. Lỗi bảng ký tự đã xác nhận

Artifact cục bộ `C:/Users/PC/.paddlex/official_models/latin_PP-OCRv5_mobile_rec/inference.yml` có 836 mục trong `PostProcess.character_dict`. Tập kiểm tra 146 chữ tiếng Việt được tạo bằng NFC từ `aăâeêioôơuưyAĂÂEÊIOÔƠUƯYđĐ` với các dấu thanh rỗng/0300/0301/0303/0309/0323, giữ kết quả một ký tự.

Bảng thiếu **90/146 ký tự**, gồm `ệ`, `ộ`, `ố`, `ợ`, `ử`. Không có combining mark trong các mục ký tự. Kiểm tra trực tiếp `model.paddlex_predictor.post_op.character` sau khi load cũng xác nhận thiếu ký tự: decoder có 838 mục tính cả token đặc biệt/khoảng trắng.

Kết luận giới hạn ở artifact đang cài: model không thể trực tiếp sinh nhiều chữ tiếng Việt cần thiết. Việc thư viện ánh xạ `vi` sang model Latin không đủ chứng minh hỗ trợ đầy đủ tiếng Việt. Tăng resolution, padding, confidence hay số luồng không khắc phục được lỗi này. Không tự thêm ký tự vào dictionary vì chỉ số lớp phải khớp trọng số model.

Đề xuất: kiểm tra charset khi xác nhận model hỗ trợ ngôn ngữ; không dùng artifact này làm mặc định cho tiếng Việt. Giữ detector nếu phù hợp, chọn recognizer có trọng số và decoder tương thích. Model thay thế phải vượt cả kiểm tra charset và benchmark thực tế; không mặc định model lớn hơn/mới hơn sẽ giải quyết được.

## 2. Đo end-to-end trên trang 12 dòng

| Pipeline | Ba lần chạy (ms) | CER | Peak RAM quan sát | CPU-seconds ước lượng/request |
|---|---|---|---|---|
| Document Tesseract, tiền xử lý hiện tại, vie+eng | 530 / 563 / 545 | 3,57% | ~164 MiB | 0,69–0,78 |
| Paddle detector + Tesseract từng dòng, vie+eng | 4676 / 4616 / 4639 | 0% | 886–897 MiB | 17,34–17,98 |
| Paddle detector + Latin recognizer | 1844 / 1481 / 1569 | 7,59% | 1025–1163 MiB | 20,30–25,47 |

Paddle đã được load bởi trang 4 dòng; lần đầu trang 12 dòng vẫn có thể chịu chi phí cache theo shape. Không coi ba mẫu là phân phối latency.

Cold call 4 dòng: document ~0,50 s, hybrid ~12,94 s, Paddle ~10,06 s; chưa gồm import ứng dụng ~1,4–1,8 s. CER trang 4 dòng: document 14,86%, hybrid 0%, Paddle 9,46%.

Hybrid 12 dòng dành ~0,90–0,98 s cho detector và ~3,53–3,66 s cho nhận dạng. Mã hiện tại gọi Tesseract riêng cho từng crop; đo quan sát khoảng 12 subprocess OCR. Phần nhận dạng gồm cả khởi tạo tiến trình, nạp ngôn ngữ và inference; chưa tách riêng để khẳng định phần trăm overhead.

Đề xuất thử batch nhiều crop trong một invocation Tesseract, ánh xạ `page_num` về region, giữ bbox/thứ tự. Tesseract hỗ trợ file danh sách ảnh theo [FAQ chính thức](https://github.com/tesseract-ocr/tessdoc/blob/main/FAQ.md). Đây là hướng thử nghiệm, chưa có số đo speedup. Native API worker sống lâu là lựa chọn sau, cần kiểm soát state/adaptive learning và thread safety. Không tạo số thread/subprocess không giới hạn theo số dòng.

## 3. Tiền xử lý và ngôn ngữ: ablation

Trên cùng trang 12 dòng; mỗi cấu hình dưới đây lặp 3 lần, PSM 6. Chỉ đo `recognize`, không tính preprocessing/router.

| Cấu hình | Median nhận dạng | CER cả ba lần |
|---|---|---|
| Median blur + Otsu, vie+eng (hiện tại) | 455 ms | 3,57% |
| Gray, vie+eng | 424 ms | 2,68% |
| Gray, vie | 313 ms | 0,45% |
| Gray, vie, PSM 3 | 320 ms | 0,45% |

Đổi riêng tiền xử lý từ hiện tại sang gray cải thiện mẫu này; đổi riêng ngôn ngữ từ vie+eng sang vie cải thiện tiếp. Không suy rộng sang tài liệu song ngữ. Một phép thử riêng Otsu không median cho CER 5,36% với vie+eng và 2,68% với vie.

Mẫu 4 dòng vẫn CER ~13,5–16,2% trên các biến thể, với dòng xoay bị nhận dạng thành nhiều dòng. Vì vậy tiền xử lý/ngôn ngữ không thay thế xử lý bố cục và xoay.

Đề xuất: có profile tài liệu tiếng Việt và song ngữ; giữ ảnh gray/original làm baseline, threshold/denoise là tùy chọn có chứng cứ. Với trang dễ, chạy Tesseract toàn trang; chỉ dùng detector/crop cho trường hợp cần thiết. Tiêu chí chuyển pipeline phải được hiệu chuẩn bằng dữ liệu thật, không dựa vào một ngưỡng confidence tùy ý.

## 4. Kiểm soát tài nguyên còn thiếu

- `MAX_CONCURRENT_OCR` được đọc trong `app/core/config.py` nhưng chưa có admission limiter dùng giá trị này. Lock trong `paddle_runtime.py` bảo vệ tạo/cache model, không bảo vệ các lời gọi predict dùng chung. Chưa chứng minh lỗi concurrent predict xảy ra; cần test hoặc serialize theo worker.
- `DEVICE` hiện được đọc/báo cáo nhưng không truyền tới model construction ở callers. Cần truyền và kiểm tra cấu hình runtime thực tế, không chỉ giá trị hiển thị health.
- `app/api/ocr.py::_read_upload` đọc toàn bộ dữ liệu rồi mới kiểm tra giới hạn byte. Decode ảnh chưa có giới hạn số pixel; ảnh nén nhỏ vẫn có thể tạo buffer lớn.
- Tesseract chưa có timeout ở `image_to_data`. Cần giới hạn request/queue, pixel, vùng nhận dạng và thời gian; trả lỗi rõ ràng thay vì âm thầm mất dòng.
- Health ready trong tiến trình mới mất ~1,73 s, RSS từ ~94,5 lên ~141,7 MiB. Không load trọng số nhưng có import dependency. Nên tách liveness rẻ, kiểm tra dependency và trạng thái loaded/ready, cache kết quả kiểm tra phù hợp.
- Paddle latency báo cáo det/rec chưa phản ánh rõ phần model-load trong cold request. Cần trường đo riêng import/load/warmup/queue/decode/preprocess/detect/recognize/postprocess.

Thử truyền `cpu_threads=1, device='cpu'` vào constructor trong tiến trình chẩn đoán: hybrid 12 dòng ~4,39–4,47 s so với ~4,62–4,68 s; CPU-seconds không giảm, số native thread vẫn ~49–51. Chưa đủ căn cứ khuyến nghị thread=1; phải xác minh backend nào nhận cấu hình và load test.

Thử `OMP_THREAD_LIMIT=1`: median gray+vie ~319 ms so với ~313 ms mặc định; cấu hình hiện tại ~448 ms so với ~455 ms. Không có lợi ích tốc độ rõ ràng ở phép thử tuần tự này. Dù [Tesseract FAQ](https://github.com/tesseract-ocr/tessdoc/blob/main/FAQ.md) nêu thread limiting có thể hữu ích với nhiều ảnh, cần đo workload thật trước khi đổi mặc định.

## 5. Thứ tự triển khai và tiêu chí kiểm chứng

1. **Chất lượng nền tảng:** charset gate; loại recognizer không đủ chữ khỏi lựa chọn auto tiếng Việt. Giữ lựa chọn explicit và lỗi/cảnh báo minh bạch. Lưu model ID, artifact hash, ngôn ngữ và preprocessing cùng kết quả benchmark.
2. **Giới hạn tài nguyên:** nối cấu hình concurrency/device tới runtime; bounded queue, timeout, pixel/upload/region caps; bắt đầu ít worker do RAM trống thấp. Test quá tải phải trả lỗi có kiểm soát, không deadlock hoặc tăng RAM không giới hạn.
3. **Tối ưu đường nhẹ:** đánh giá gray/original và profile vie/vie+eng trên tập độc lập. Không tự đổi mặc định song ngữ thành đơn ngữ chỉ vì mẫu hiện tại tốt hơn.
4. **Tối ưu hybrid:** batch crop trong một invocation; kiểm tra một-một giữa crop và output, dòng rỗng, thứ tự, dấu và timeout. So sánh với phiên bản từng dòng trước khi chuyển.
5. **Model mới:** chỉ sau charset gate; benchmark cùng máy, cùng tập ảnh, cấu hình được khóa. Chưa cần tải model nặng, mua GPU hay fine-tune ở bước này.

Trước khi chọn cấu hình sản xuất: bổ sung ảnh thực tế có ground truth (scan sạch, hóa đơn, ảnh điện thoại, chữ nhỏ/xoay/mờ, bảng/nhiều cột, đơn ngữ/song ngữ), tách tập điều chỉnh và tập kiểm tra theo tài liệu nguồn. Không coi các biến thể cùng một câu là mẫu độc lập.

Ghi CER/WER, lỗi dấu, tỷ lệ bỏ dòng, sai số tiền/mã, p50/p95 cold/warm, throughput, peak/idle RAM, CPU-seconds và lỗi/timeout ở concurrency 1/2/4. Ngưỡng chất lượng và latency cần chốt theo nhu cầu thực tế. Không promote nếu mất dòng, sai ánh xạ vùng, hoặc chất lượng tụt ở nhóm quan trọng. Giữ cấu hình/artifact cũ để rollback.
