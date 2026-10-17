# OCR-Vie — kiểm tra toàn dự án ngày 07/10/2026

> Đây là snapshot **trước sửa**. Kế hoạch đã được triển khai từng phần sau audit;
> xem [kết quả thực hiện](IMPLEMENTATION_RESULTS_2026-10-07.md) cho trạng thái hiện tại.
> Điểm audit bên dưới không phải điểm đánh giá lại sau sửa.

## Kết luận

Production audit: **45/100 — chưa nên triển khai công khai**. Đây là điểm ưu tiên kỹ thuật định tính, không phải độ chính xác OCR. Bản demo nội bộ có nền tảng tốt, nhưng còn lỗi crop, sai lệch công cụ đánh giá, giới hạn upload đặt quá muộn và cấu hình triển khai không khớp pipeline mặc định.

Khuyến nghị: sửa lỗi có thể tái hiện và làm benchmark đáng tin cậy trước; sau đó mới chọn cấu hình cân bằng độ chính xác, tốc độ và RAM. Chưa có cơ sở để thay toàn bộ engine, thêm GPU, hàng đợi hay viết lại frontend.

Phạm vi đã kiểm tra: API, decode/preprocess, định tuyến, detector/recognizer, crop, parser, giới hạn tài nguyên, trạng thái model, UI, bộ kiểm thử, benchmark/dataset/annotation, script vận hành, Docker, dependencies và tài liệu. Không có database, thanh toán hoặc hệ thống tài khoản cần audit trong phạm vi hiện tại.

Chỉ kiểm tra và lập kế hoạch: không sửa mã nguồn ứng dụng, không cài thư viện, không tải model, không bật demo, không gửi dữ liệu dự án ra dịch vụ bên ngoài.

## Bằng chứng kiểm tra

| Kiểm tra | Kết quả |
| --- | --- |
| `.venv311/Scripts/python.exe -m pytest -q`, `ALLOW_MODEL_DOWNLOAD=false` | **191 passed**, 1 cảnh báo ccache của Paddle, 29,25 giây |
| `.venv311/Scripts/python.exe -m pip check` | Không phát hiện requirements bị xung đột |
| `python -m benchmarks.validate_dataset --data benchmark_data` | 12 records, 12 ảnh hợp lệ |
| `python -m benchmarks.validate_dataset --data benchmark_data_quality_v2` | 12 records, 12 ảnh hợp lệ |
| Phân loại hai bộ dữ liệu bằng helper hiện có | Mỗi bộ có 12 mẫu synthetic; **0 real reviewed test** |
| Git | Thư mục hiện tại không phải Git repository; không có lịch sử/diff để đối chiếu |
| Docker | Có CLI nhưng không kết nối được Docker Desktop Linux daemon; **chưa build/run container** |
| Coverage | Chưa cài công cụ coverage; không đưa ra tỷ lệ coverage |
| Cổng demo 8000 | Không có listener khi kiểm tra cuối |

Các kiểm tra tái hiện bên dưới chạy trong process riêng, dùng TestClient hoặc mock; không cần server và không thay đổi model/dataset. Kết quả tốc độ OCR trong tài liệu cũ là bằng chứng lịch sử trên ảnh synthetic, không phải số đo mới của đợt audit này.

## Kiến trúc hiện tại và điểm nên giữ

- `scene + auto + tiếng Việt`: Paddle detector → crop phối cảnh từ ảnh màu → Tesseract theo batch. Chính sách chặn model Latin thiếu ký tự tiếng Việt là đúng hướng.
- `document + auto`: Tesseract toàn trang; lựa chọn này nhẹ và phù hợp làm baseline cho văn bản in.
- Có giới hạn pixel ảnh/crop, số vùng, semaphore OCR, timeout Tesseract và deadline hợp tác. Không được hiểu deadline này là khả năng hủy ngay native Paddle.
- Giữ văn bản raw tách khỏi normalized; lỗi engine không bị âm thầm biến thành kết quả giả.
- UI hiện có trạng thái tải/lỗi/rỗng, chống gửi lặp, giải phóng URL preview, xuất kết quả và bố cục mobile. Không cần thêm framework chỉ để tiếp tục cải thiện.

## Phát hiện ưu tiên

P1: sửa trước khi mở rộng sử dụng, tin vào benchmark hoặc triển khai công khai. P2: cải thiện ở đợt tiếp theo. Không phát hiện P0 có bằng chứng trong phạm vi kiểm tra này; đây không phải chứng nhận an toàn.

### F01 — P1: crop bốn điểm có thể tạo phép biến đổi suy biến

**Vị trí:** `app/services/polygon_utils.py:30–43`, gọi ở `crop_polygon`.

`argmin/argmax` của tổng và hiệu tọa độ có thể chọn cùng một điểm khi hòa giá trị. Với polygon hợp lệ `[[50,0],[100,50],[50,100],[0,50]]`, kết quả thực tế là:

```text
ordered = [[50,0], [50,0], [100,50], [50,100]]
unique_points = 3
perspective_matrix_rank = 1
```

Ảnh xoay/phối cảnh có thể bị crop sai trước khi tới recognizer. Đổi model nhận dạng không giải quyết nguyên nhân này.

**Hướng sửa:** sắp xếp vòng quanh tâm với quy tắc góc bắt đầu xác định, kiểm tra điểm hữu hạn/khác nhau và diện tích; giữ đúng hướng crop. Kiểm thử đủ hoán vị, hình thoi, xoay gần 45°, điểm trùng và polygon suy biến. Dùng NumPy/OpenCV đang có.

### F02 — P1: một số phép đo benchmark chưa phản ánh sản phẩm

**Vị trí:** `benchmarks/run_scene_benchmark.py:45–51,169`; `benchmarks/run_recognition_benchmark.py:232–241`; `benchmarks/metrics.py`.

- `--recognizer auto` của benchmark vẫn chọn Paddle Latin, khác chính sách auto tiếng Việt trong API. Nhánh Tesseract scene gọi từng crop, không dùng batch như sản phẩm. Vì vậy không dùng số latency này để kết luận hiệu năng API mặc định.
- `field_total` tăng cho từng cặp region × field; việc nhận diện field lại dựa vào tên khóa xuất hiện trong GT hoặc chỉ có một region. Tái hiện logic với hai region `100`, `123`, hai field amount/tax_id và prediction đúng hoàn toàn cho kết quả **0/4 = 0**, thay vì hai field đúng. Đây là tái hiện khối tính điểm, không phải chạy model.
- IoU dùng polygon nếu có Shapely, tự chuyển bbox nếu không có hoặc có exception; backend tính điểm có thể khác giữa môi trường.

**Hướng sửa:** benchmark end-to-end gọi router sản phẩm; giữ GT-crop benchmark như phép đo riêng. Chỉ chấm field có liên kết GT rõ ràng, một lần/field; không suy đoán vị trí từ tên khóa. Ghi rõ backend IoU và không đổi thuật toán ngầm khi lỗi. Ghi model/config/dataset hash cùng kết quả.

### F03 — P1: dữ liệu đánh giá chưa đủ và bản freeze không bảo vệ nhãn

**Vị trí:** `benchmarks/dataset.py:277–310`; `benchmarks/freeze_testset.py:57–65`; hai thư mục benchmark.

- Hai bộ hiện có đều synthetic, mỗi bộ 12 mẫu; chưa có ảnh thực tế đã duyệt trong test set. Chưa thể kết luận CER hoặc mức cải thiện trên tài liệu/ảnh chụp thực tế.
- Thay `text` và polygon trong một record không làm thay `manifest_hash` — đã tái hiện. Manifest không hash nội dung GT và tin `sha256` có sẵn thay vì luôn đối chiếu ảnh.
- Freeze cùng version ghi đè file hiện có. Nhãn và phiên bản kiểm thử có thể thay đổi mà không được phát hiện đầy đủ.

**Hướng sửa:** hash canonical GT gồm text, regions, fields, ignore và metadata ảnh hưởng chấm điểm; xác minh hash ảnh; từ chối ghi đè phiên bản. Thu thập tối thiểu 100 ảnh thực tế đã duyệt theo gate của dự án, phân tầng theo lỗi và nguồn. Con số 100 là mốc khởi đầu, không tự bảo đảm tính đại diện thống kê. Chỉ dùng ảnh có quyền sử dụng và chính sách riêng tư phù hợp.

### F04 — P1: công cụ gán nhãn có đường dẫn làm mất dữ liệu

**Vị trí:** `benchmarks/annotate.py:194–219`; `benchmarks/audit_annotations.py:57–70`; `benchmarks/dataset_summary.py`.

Loader của annotator bỏ qua dòng JSON hỏng không báo lỗi; thao tác save sau đó ghi lại danh sách đã bị lọc. Với một dòng hợp lệ và một dòng JSON hỏng, loader thực tế chỉ trả dòng hợp lệ. Chưa ghi thử lên dữ liệu người dùng; nguy cơ mất dòng được xác định từ đường đọc → save.

Audit/summary cũng có loader riêng bỏ qua lỗi; audit có thể in lỗi cú pháp nhưng không đưa lỗi đó vào danh sách quyết định exit code.

**Hướng sửa:** dùng loader strict hiện có, báo file/số dòng và chặn save khi đọc lỗi; kiểm tra schema trước lưu. Thêm cảnh báo dữ liệu chưa lưu khi đổi ảnh và phát hiện tab cũ trước ghi đè.

**Bảo mật bổ sung, P2 khi chỉ dùng local:** annotator nội suy tags vào `innerHTML`; fallback `/image/…` cho phép tìm file ngoài dataset trong repository. Chuyển dữ liệu nhập vào `textContent`/`value`, chỉ phục vụ ảnh nằm trong dataset. Không công khai annotator khi chưa có kiểm soát truy cập. Chưa thực hiện khai thác XSS hay đọc file nhạy cảm.

### F05 — P1 khi nhận request không tin cậy: giới hạn upload chưa chặn tài nguyên đầu vào

**Vị trí:** `app/api/ocr.py:61–75`; `app/services/limits.py`; `app/main.py`.

FastAPI xử lý multipart trước khi chạy `_read_upload`. Trong thử nghiệm đặt giới hạn **64 byte**, upload **2.097.152 byte** đã được spool xuống đĩa rồi API mới trả 413. Semaphore/deadline OCR không bảo vệ giai đoạn nhận multipart này.

Đây không phải vượt giới hạn để chạy OCR: OCR vẫn bị chặn. Vấn đề là băng thông, file tạm, thời gian nhận và số request đồng thời vẫn có thể gây quá tải trước giới hạn đó.

**Hướng sửa:** chặn tổng body thực tế trước multipart, kể cả chunked/không có Content-Length; cho phép phần overhead multipart có giới hạn riêng, giữ cap từng file. Dùng giới hạn body/connections/timeouts của reverse proxy nếu có; nếu app cần tự bảo vệ thì dùng middleware ASGI nhỏ, không tạo hàng đợi OCR mới. Local mặc định bind loopback; khi public cần auth/rate-limit phù hợp tại gateway.

### F06 — P1: đóng gói và trạng thái sẵn sàng không khớp pipeline

**Vị trí:** `Dockerfile:24–25,36–37`; `scripts/run_local.ps1`; `scripts/test.ps1`; `app/api/health.py:40`; `app/detectors/paddle_detector.py:31`; `app/recognizers/tesseract_recognizer.py:53`.

- Docker chỉ cài requirements cơ bản, không có Paddle, trong khi scene auto mặc định cần Paddle. Kết luận từ recipe; chưa xác minh image thực tế vì daemon đang tắt.
- Script local chọn `venv`, cài lại/upgrades khi chạy, không chuẩn hóa `.venv311` và bộ phụ thuộc hybrid hiện dùng.
- Health tạo adapter mới không đọc cache lỗi load toàn cục. Mock lỗi detector có sẵn, weights vắng: health vẫn trả `not_loaded`, không phải `load_failed`/`unavailable` như tài liệu mô tả.
- Tesseract status chỉ kiểm tra version. Mock chỉ có `eng`, không có `vie`: status vẫn `ready`.
- `/health/ready` trả HTTP 200 kể cả `ready=false`; Docker probe chỉ kiểm tra HTTP của `/health`, không đọc readiness.

**Hướng sửa:** một hướng dẫn/cấu hình hybrid chuẩn với Python 3.11; setup tách run. Probe liveness và readiness riêng; trạng thái model lấy từ một nguồn thật, kiểm tra traineddata theo ngôn ngữ. Không load model nặng khi gọi health. Deployment muốn ready ngay phải warm model trong chính server process trước nhận traffic, không dựa vào warmup process khác.

### F07 — P2: thứ tự đọc và cấu trúc có lỗi độc lập recognizer

**Vị trí:** `app/parsers/fallback_parser.py:43`; `app/pipelines/router.py:528`; `app/classifiers/document_scene_classifier.py`.

- Đã tái hiện `parse('A\n\nB')`: blocks `line-0,line-2`, nhưng reading_order `line-0,line-1`. Tham chiếu `line-1` không tồn tại. Sửa trực tiếp bằng danh sách ID của block thật.
- Sort `(y,x)` có thể đảo các từ cùng dòng: từ bên phải y=99 sẽ đứng trước từ bên trái y=100; có thể xen cột. Chưa đo tần suất trên ảnh thật.
- Classifier chỉ dùng kích thước/tỷ lệ ảnh; document default PSM 6 giả định một khối văn bản. Bảng, nhiều cột, trang xoay cần benchmark riêng. Đây là giới hạn thiết kế, không phải bằng chứng rằng mọi ảnh đều sai.

**Hướng sửa:** gom dòng theo giao nhau theo trục dọc/chiều cao chữ và sắp x trong dòng; chỉ mở rộng phân cột khi dataset chứng minh cần. So sánh PSM 3/6/11 và giữ tùy chọn route thủ công. Không thêm layout model lớn ngay.

### F08 — P2: bước ghép TSV có cấp phát bậc hai không cần thiết

**Vị trí:** `app/recognizers/tesseract_recognizer.py:113`.

`dict.get(k, [0] * n)` tạo danh sách mặc định trước mỗi lần gọi, ngay cả khi key tồn tại. Lặp bốn key cho mỗi từ tạo công việc O(n²).

Microbenchmark `_result` trên TSV tổng hợp, lấy min 3 lần trong cùng process:

| Số từ | Thời gian |
| ---: | ---: |
| 1.000 | 6,37 ms |
| 4.000 | 114,09 ms |
| 8.000 | 406,69 ms |

Không phải latency OCR end-to-end và không đo speedup sau sửa. Hướng sửa nhỏ: lấy các cột một lần ngoài vòng lặp, dùng chung danh sách zero khi thiếu. Giữ nguyên text, nhóm dòng, confidence và xử lý TSV thiếu cột.

### F09 — P2: API/UX xử lý sai một số trạng thái

**Vị trí:** `app/api/ocr.py:151,186,188`; `app/pipelines/registry.py`; `app/ui.html:258–259`.

Query string tự do làm format không hợp lệ rơi về JSON; parser không hợp lệ bị thay bằng fallback. Lỗi 500 trả nguyên exception ra client. UI gộp mọi 503 thành “bộ nhận dạng chưa sẵn sàng”, dù `OCR_BUSY` là quá tải tạm thời và đã có Retry-After.

**Hướng sửa:** kiểm tra giá trị query tại API, phân biệt tham số không hợp lệ với engine hợp lệ nhưng chưa sẵn sàng; trả lỗi nội bộ an toàn và log có request ID, không log văn bản OCR/ảnh. UI phân loại theo error code, giữ ảnh để thử lại; không tự động retry vô hạn.

### F10 — P2: kiểm thử vận hành và tài liệu đang lệch nhau

**Vị trí:** `scripts/smoke_test.py:53,92–94`; `scripts/check_environment.py`; `README.md`; `DEMO.md`; `.dockerignore`.

- Smoke chỉ đọc 500 ký tự HTML rồi tìm `Vietnamese OCR`. Thực tế chuỗi không nằm trong 500 ký tự đầu nhưng có trong toàn trang: smoke báo lỗi sai sau redesign.
- Smoke dùng ảnh trắng, endpoint legacy, chấp nhận 503; chưa chứng minh scene auto tiếng Việt hoạt động.
- Check environment có nhánh báo thiếu phụ thuộc nhưng không thoát với mã lỗi; chưa là gate tự động đáng tin.
- Không thấy CI trong cây dự án; test xanh hiện tại không tương đương kiểm thử engine tiếng Việt thực đầy đủ hoặc browser E2E.
- README/DEMO trộn hướng dẫn hiện tại và lịch sử; `.dockerignore` chưa loại `venv_broken`, ảnh incoming, kết quả benchmark và artifacts khỏi build context.
- `ModelRegistry` là lớp cache song song chưa có consumer runtime; có cấu hình chưa được dùng và adapter stub. Không nên phát triển tiếp những lớp này khi chưa có nhu cầu.

**Hướng sửa:** sửa smoke theo contract hiện tại, tách smoke thiếu-model khỏi release gate bắt buộc model thật; chuẩn hóa setup/run/check; phân tách tài liệu current/history, thu gọn mã chết sau khi kiểm tra callers. Không xóa dữ liệu/venv người dùng trong công việc này.

## Điều chưa được chứng minh

- Chưa có CER/WER đại diện cho ảnh thực tế, trường quan trọng hoặc nhiều cột; chưa có ngưỡng chất lượng được chấp thuận.
- Chưa đo p95/p99, tải đồng thời, RAM lâu dài, giới hạn CPU/RAM của máy triển khai; không hứa phần trăm tăng tốc.
- Chưa build Docker, quét CVE hiện hành, kiểm thử từ máy sạch hoặc kiểm tra production URL.
- Đợt này không chạy lại browser E2E; bằng chứng UI thực tế trước đó có trong PROJECT_LOG/artifacts. Clipboard ở mức hệ điều hành chưa được xác minh end-to-end.
- Không có Git history/CI evidence/rollback đã thử. Giữ snapshot trước đợt sửa; việc khởi tạo Git hay cấu hình remote là quyết định riêng.

## Bước tiếp theo

Thực hiện đợt A trong kế hoạch `docs/superpowers/plans/2026-10-07-project-improvements.md`: crop, reading_order, bảo toàn annotation và freeze/metric. Sau đó sửa giới hạn đầu vào, readiness và packaging trước khi public. Các bước tối ưu model chỉ bắt đầu sau khi benchmark đúng pipeline và có baseline phù hợp.

Skill `code-reviewer` giúp xác định lỗi theo đường gọi; `production-audit` mở rộng kiểm tra sang triển khai và toàn vẹn dữ liệu; `writing-plans` chuyển kết quả thành các bước có test và điều kiện nghiệm thu. Không dùng dịch vụ audit bên ngoài.
