# OCR-Vie — rà soát và kế hoạch cải thiện 16/10/2026

Tiến độ thực thi sau audit: xem `docs/IMPLEMENTATION_RESULTS_2026-10-16.md`. Đã sửa các lỗi ưu tiên với 266 tests pass; phần dữ liệu thật, tối ưu profile và nghiệm thu triển khai còn chưa hoàn tất. Các checkbox dưới đây giữ nội dung kế hoạch, không đại diện mọi hạng mục đã hoàn thành.

## 1. Kết luận và phạm vi

Giữ FastAPI, UI JavaScript thuần và pipeline hiện tại. Ưu tiên sửa lỗi báo kết quả, bảo toàn nhãn và độ tin cậy của benchmark trước khi đổi model hoặc tham số hiệu năng. Chưa có bằng chứng cần framework frontend, hàng đợi, Redis hay worker pool mới.

Đây là **kế hoạch, chưa thực thi sửa mã nguồn**. Rà soát luồng API → decode → classifier → detector/recognizer/parser, admission/deadline, UI, annotator, dataset/benchmark, scripts vận hành và Docker; đối chiếu README, tài liệu triển khai và kế hoạch 07/10. Không coi các sửa lỗi đã hoàn tất ở đợt trước là việc tồn đọng.

### Trạng thái xác minh ngày 16/10

| Hạng mục | Kết quả hiện tại |
| --- | --- |
| Full pytest trong `.venv311`, tắt tải model | **253 passed**, 37,28 giây; một warning thiếu ccache |
| `pip check` | Không broken requirements |
| `scripts/check_environment.py` | PASSED; Python 3.11.7 |
| Hai dataset hiện có | Mỗi bộ 12 records/12 ảnh unique trong bộ; synthetic/unreviewed; **0 real reviewed test** |
| Docker daemon | Đã phản hồi, server **29.4.2**; chưa build/chạy image dự án trong lượt audit |
| Git | Thư mục hiện tại chưa là Git repository |

Python 3.11 và Docker bị sandbox từ chối truy cập ban đầu; kiểm tra lại ngoài sandbox thành công. **Không kết luận venv hỏng hoặc Docker chưa chạy** từ lần thử đầu. Python mặc định trên PATH là 3.14.6, không phải runtime chuẩn của dự án.

Test pass không bao phủ các trường hợp biên mới dưới đây. Các tái hiện dùng mock hoặc bản sao dữ liệu trong bộ nhớ; không đổi nhãn, không tạo frozen dataset, không bật demo.

## 2. Phát hiện có bằng chứng

### F01 — P1: lưu nhãn bất đồng bộ có thể làm mất trạng thái chỉnh sửa

**Vị trí:** `benchmarks/annotate.py:194–205`, kết hợp `show()` tại dòng 99.

`save()` gửi bản chụp JSON nhưng khi response về lại gán `original` từ object đang sống và xóa `dirty` toàn cục. Người dùng vẫn sửa hoặc chuyển ảnh trong thời gian chờ. Revision/ETag bảo vệ tab cũ ở server, không bảo vệ race trong chính tab đó.

Tái hiện bằng Node VM chạy script thật:

- Gửi “sent A”, sửa tiếp thành “new unsaved A”, nhận response: ô nhập còn bản mới nhưng `dirty=false`, snapshot vẫn “sent A”. Rời ảnh không còn cảnh báo mất sửa.
- Lưu ảnh A, chuyển sang B rồi sửa B, nhận response A: `idx=1`, record hiện tại là B, `original` lại thuộc A và `dirty=false`. Sau đó thao tác discard có thể khôi phục sai record.

**Tác động:** mất công gán nhãn hoặc nhầm nhãn giữa ảnh; ảnh hưởng trực tiếp chất lượng đánh giá OCR.

### F02 — P1: mọi vùng nhận dạng lỗi nhưng API vẫn HTTP 200 / success=true

**Vị trí:** `app/pipelines/router.py:559–596`; callers `/v1/ocr`, `/v1/ocr/text`, UI và scene benchmark.

Nhánh Paddle recognizer không-hybrid bắt lỗi từng vùng, thêm warning rồi trả thành công vô điều kiện. Tái hiện qua FastAPI TestClient: detector trả một vùng hợp lệ, recognizer ném `RuntimeError` → **HTTP 200, `success=true`, `text=""`, warning `region 0 recognition error`**.

UI có thể hiểu đây là ảnh không có chữ; text-only trả chuỗi trống không có warning. Benchmark hiện phải dò nội dung warning để đoán all-failed (`benchmarks/run_scene_benchmark.py:173–175`). Đây là lỗi contract, không phải lỗi chữ Việt của nhánh hybrid mặc định.

### F03 — P1 trước public: thông báo lỗi classifier còn lộ chi tiết nội bộ

**Vị trí:** `app/pipelines/router.py:149–156`, `app/services/errors.py:25` và `to_dict()`.

Exception bất ngờ được ghép nguyên văn thành `CLASSIFICATION_FAILED`, mã này trả 422; cơ chế che message chỉ áp dụng status 500. Tái hiện exception có marker `internal C:/private/model-path`: response HTTP 422 chứa nguyên marker.

Đây là đường lộ thông tin đã tái hiện với mock; chưa có bằng chứng dữ liệu thật đã bị lộ. Lỗi vận hành của classifier cần khác lỗi input không hợp lệ và phải được log kèm request ID.

### F04 — P1 cho benchmark: validate/freeze chưa có cùng điều kiện hợp lệ

**Vị trí:** `benchmarks/validate_dataset.py:77–84,151–164`; `benchmarks/dataset.py:310`; `benchmarks/freeze_testset.py:52–63`.

- Validator chấp nhận `image: 17` và trả `Validation OK: 1 records, 0 unique images` nếu không có split files; `split: []` gây `TypeError` thay vì báo lỗi schema có kiểm soát. Đã tái hiện bằng input mock.
- Kiểm tra split files hiện phát hiện ID lạ/trùng, nhưng chưa đối chiếu đầy đủ ID thuộc đúng split và ID bị thiếu.
- Freeze đếm số records đã filter, gọi manifest nhưng không bắt buộc validation toàn dataset. Bản sao trong bộ nhớ gồm 100 ID khác nhau trỏ cùng một ảnh tạo được manifest `count=100`, một image hash duy nhất, tất cả `image_verified=true`. Theo gate hiện tại, số record này đủ nhánh gắn nhãn chính thức; **không chạy ghi freeze thật**.
- Vì freeze chỉ xét tập đã chọn, kiểm tra rò rỉ ảnh giữa train/val/test không được bảo đảm tại điểm freeze.

**Tác động:** tập đánh giá không đủ đa dạng hoặc nhiễm dữ liệu vẫn có thể được đóng dấu như tập chuẩn. Ngưỡng 100 là gate nội bộ tối thiểu, không tự chứng minh tính đại diện.

### F05 — P2: UI thiếu giới hạn chờ request phía trình duyệt

**Vị trí:** `app/ui.html:274–286` và khối `finally`.

UI khóa các control rồi chờ `fetch()` không có abort/timeout. Nếu kết nối treo hoặc native inference không trả về, `finally` chưa chạy và UI không tự mở khóa. Đây là kết luận từ mã nguồn, chưa tái hiện trên trình duyệt thật trong lượt này. Deadline backend là cooperative, không phải hard cancellation của Paddle.

### Khoảng trống chất lượng/hiệu năng, không khẳng định là bug

- Không có ảnh thật reviewed: chưa đủ cơ sở đo cải thiện dấu tiếng Việt, chọn model mới hoặc đổi default. Các số CER/RSS/p95 ngày 07/10 chỉ thuộc fixture synthetic đã ghi rõ phạm vi.
- Classifier hiện dùng kích thước/tỉ lệ ảnh, không phân tích nội dung trang; cần đo nhầm route trên dữ liệu có nhãn trước khi thay heuristic.
- Hybrid tạo danh sách toàn bộ crops trước `recognize_many()`; hàm này chia batch và ghi PNG tạm. Tổng crop pixels đã bị chặn, nên đây là ứng viên giảm peak RAM/I/O, **chưa có số đo chứng minh là bottleneck**.
- Admission OCR nằm sau multipart parsing; v1 đọc file trước khi vào router semaphore. Cap từng body không phải cap tổng kết nối/upload. Cần kiểm tra tải ingress trước public, không suy ra bộ nhớ toàn dịch vụ chỉ từ `MAX_CONCURRENT_OCR=2`.
- `OCR_SCENE_DETECTOR` được nhận trong config nhưng router chọn detector trực tiếp; `OCR_DEBUG` không quyết định `include_debug=True` trong API. Cần chốt contract thay vì thêm lớp cấu hình mới. `include_debug` hiện chủ yếu điều khiển timing, không đồng nghĩa một lỗi lộ secret.

## 3. Kế hoạch triển khai theo thứ tự

Mỗi task: thêm regression tái hiện → sửa nhỏ ở điểm chung → chạy test liên quan → full suite trước bàn giao. Không sửa thuật toán chỉ để làm một fixture đẹp hơn.

### Đợt A — tính đúng đắn và bảo toàn dữ liệu; có thể bắt đầu ngay

#### A1. Chặn race lưu nhãn — F01

**Phạm vi:** `benchmarks/annotate.py`, `tests/test_annotator_ui.py`, `tests/test_annotator_http.py`.

- [ ] Thêm regression response trễ, sửa trong lúc lưu, đổi ảnh/filter, double-save, lỗi mạng và HTTP 409.
- [ ] Giải pháp nhỏ nhất: một save đang chạy; chụp payload/id/revision cố định; khóa chỉnh sửa/navigation/filter trong thời gian lưu, bao gồm shortcut và canvas handlers. Không thêm state library.
- [ ] Thành công chỉ cập nhật snapshot của payload đã lưu. Thất bại giữ nhãn và trạng thái chưa lưu; luôn mở khóa trong `finally`.
- [ ] Xử lý kết quả list/image đến trễ nếu test chứng minh ghi đè state mới; không để filter rỗng giữ editor của ảnh cũ có thể lưu nhầm.

**Đạt khi:** không mất sửa hoặc đổi ID/snapshot khi response đảo thứ tự; không gửi save thứ hai trong lúc pending; retry sau lỗi giữ dữ liệu; test ETag hiện có vẫn đạt. Không tự autosave/ghi đè khi conflict.

#### A2. Sửa contract lỗi OCR và che lỗi nội bộ — F02/F03

**Phạm vi:** `app/pipelines/router.py`, `app/services/errors.py`, API handlers; `tests/test_router.py`, `tests/test_request_boundary.py`, `tests/test_engine_policy.py`, scene benchmark tests và UI interactions.

- [ ] Phân biệt ba trường hợp: không phát hiện chữ; recognizer chạy hợp lệ nhưng đọc rỗng; mọi lần nhận dạng đều lỗi. Chỉ trường hợp cuối bắt buộc trả lỗi.
- [ ] Dùng `PipelineError` hiện có để all-failed trả 5xx, cả JSON/text-only. Giữ phần thành công và warning khi chỉ lỗi một phần; không đổi raw text hay tự fallback engine explicit.
- [ ] Exception bất ngờ của classifier trả lỗi nội bộ an toàn; validation input vẫn 4xx. Rà các chỗ ghép `str(exc)` vào response khác, giữ chi tiết trong log có request ID.
- [ ] Benchmark dùng kết quả/exception chuẩn; bỏ việc phụ thuộc câu chữ warning nếu đã không cần. Kiểm tra cả crop không hợp lệ, timeout và lỗi batch.

**Đạt khi:** mock all-failed không còn HTTP 200; ảnh trắng hợp lệ vẫn không bị báo lỗi hệ thống; marker đường dẫn nội bộ không xuất hiện ở JSON/text response; partial result và legacy contract hợp lệ được giữ.

#### A3. Đồng nhất validation và freeze — F04

**Phạm vi:** helpers hiện có trong `benchmarks/dataset.py`, validator, annotator save, freeze; kiểm tra callers import/audit/summary/scene/recognition; tests dataset, coverage và annotator.

- [ ] Dùng validation dùng chung cho kiểu dữ liệu bắt buộc, enum, ID/image/text và geometry/field mapping; phân biệt bản nháp chưa gán xong với điều kiện được benchmark/freeze. Không ép bản nháp thành reviewed.
- [ ] Mọi JSON hợp lệ nhưng schema sai phải nhận lỗi có vị trí, không crash `TypeError`; ảnh không phải string/file hợp lệ không được pass.
- [ ] Đối chiếu split files với `annotations.jsonl` theo cả membership và tính đầy đủ; chốt JSONL là nguồn chính, split files là dữ liệu dẫn xuất.
- [ ] Trước freeze, kiểm tra toàn dataset, ID/hash trùng và leakage giữa splits. Đếm ảnh thật reviewed **duy nhất** sau validation; giữ gate 100, version không ghi đè và manifest hash hiện có.
- [ ] Kiểm thử lỗi ghi split files sau atomic replace JSONL: báo rõ tình trạng, giữ nguồn chính có thể đọc/khôi phục; không trả thông báo lưu thành công sai. Chưa đưa database/transaction framework vào annotator cục bộ.

**Đạt khi:** các input F04 bị từ chối có kiểm soát; 100 bản ghi cùng ảnh không vượt gate; duplicate train/test không freeze được; dữ liệu synthetic hiện tại vẫn validate, nhưng không được đổi thành real/reviewed.

#### A4. UI chờ có giới hạn và contract cấu hình — F05

**Phạm vi:** `app/ui.html`, `tests/test_ui_interactions.py`; config/health/API tests nếu chỉnh contract.

- [ ] Dùng AbortController native cho request treo; đặt thời gian chờ phù hợp budget backend và upload, tránh một hằng số nhỏ làm hỏng cold start.
- [ ] Khi timeout: giữ ảnh, mở khóa thao tác, thông báo rõ; không tự retry liên tục. Abort phía trình duyệt **không được mô tả là đã dừng native OCR trên server**.
- [ ] Kiểm tra route/recognizer aliases, requested language thiếu traineddata và readiness; giảm các bảng ánh xạ trùng nếu có mismatch được tái hiện.
- [ ] Chốt `OCR_SCENE_DETECTOR` và `OCR_DEBUG`: triển khai đúng contract nếu thực sự cần, hoặc deprecate rõ cấu hình không có tác dụng; không âm thầm đổi behavior đang dùng.
- [ ] Browser E2E: keyboard, mobile, upload/đổi ảnh, duplicate submit, empty result, 413/503/504, response treo, copy/export. Giữ ARIA/focus/reduced motion; chưa thiết kế lại giao diện.

**Đạt khi:** request không hoàn tất không khóa UI vô hạn; ảnh và nội dung đang làm việc được giữ; không có kết quả cũ ghi đè request mới; hướng dẫn cấu hình phản ánh đúng runtime.

### Đợt B — đo đúng chất lượng tiếng Việt; cần nguồn ảnh được phép sử dụng

#### B1. Bộ dev và holdout thật

**Phạm vi:** workflow import/annotate/audit/freeze đang có, `data/README.md`, các benchmark runners. Phụ thuộc A1/A3; người dùng xác định nguồn ảnh và quyền sử dụng.

- [ ] Thu thập dev riêng và tối thiểu 100 ảnh thật reviewed test; chia theo nguồn tài liệu trước tạo biến thể/crop để giảm leakage. Ngưỡng số lượng không thay thế coverage theo nhóm.
- [ ] Bao phủ trang in, ảnh chụp, screenshot, biển hiệu; chữ nhỏ, mờ, lóa, xoay/phối cảnh; Việt/Anh; dấu, số tiền và mã. Dữ liệu nhạy cảm chỉ dùng khi có quyền, không đưa lên dịch vụ ngoài.
- [ ] Gán raw GT, thứ tự đọc, polygon, ignore đúng phạm vi; critical fields có mapping. Review chéo những vùng dấu/số quan trọng.
- [ ] Đo cùng ảnh: route auto so với forced scene/document; end-to-end so với GT-crop. Phân loại lỗi do route, detection/crop, recognition và thứ tự đọc.
- [ ] Báo CER/WER/exact theo nhóm, miss/false positive và field exact-match; công bố cỡ mẫu. Chỉ số accent-stripped delta hiện có không được gọi là tỷ lệ sai dấu edit-aligned.
- [ ] Freeze sau validation, giữ holdout không dùng để chọn tham số. Không gán nhãn “official quality” chỉ vì đủ 100 records.

**Đạt khi:** có baseline lặp lại được, hash dữ liệu/model/config, nhóm yếu được xác định bằng lỗi cụ thể. Thiếu ảnh thật thì giữ task chưa hoàn thành, không lấy synthetic thay thế kết luận thực tế.

### Đợt C — tốc độ, tài nguyên và nghiệm thu

#### C1. Tối ưu theo số đo; không đổi default trước baseline

**Phạm vi:** tận dụng `scripts/verify_ocr_performance.py`, `scripts/verify_ocr_load.py`, scene/recognition benchmark; chỉ sửa router/Tesseract/runtime khi profiling chỉ ra điểm nghẽn.

- [ ] Đo lại trên máy mục tiêu, không chạy pytest/benchmark nặng song song. Tách cold/warm; warm ít nhất 30 mẫu để sàng lọc, tăng mẫu cho ứng viên cuối và công bố n của p95.
- [ ] Báo p50/p95, throughput thành công, busy/error/timeout riêng; RSS toàn cây process, CPU-s/request và temp I/O. Concurrency 1/2/4, soak ít nhất 10 phút cho ứng viên cuối. Không chỉ đo latency của vài request thành công trong burst bị reject.
- [ ] Sàng lọc từng biến: detector side 768/960/1280; threads 4/8/10; batch 8/16/32. Xác minh giới hạn thread thực tế ở cả Paddle và subprocess Tesseract, không giả định một biến điều khiển tất cả engine.
- [ ] Với document, so original/gray và PSM 3/6/11 theo nhóm đã gán nhãn. Chỉ sửa heuristic route khi confusion/error analysis của B1 chứng minh cần.
- [ ] Nếu crop allocation/I/O đáng kể: thử tạo và giải phóng crops theo batch, tái sử dụng `recognize_many()`; giữ tổng pixel budget toàn request, mapping vùng trắng, deadline và cleanup temp. Không tối ưu chỉ dựa vào cảm giác.
- [ ] Nếu phân nhóm TSV vẫn chiếm tỷ trọng đáng kể, thử index page một lần thay vì quét toàn TSV cho từng page; phải giữ output/confidence và đo end-to-end. Không làm nếu chi phí không đáng kể.

**Gate đề xuất để chốt trước khi đo:** không tăng CER/WER tổng và critical-field error trên dev; xem riêng từng nhóm và các lỗi mới. Chấp nhận thay đổi nếu p95 hoặc CPU/RSS cải thiện lặp lại rõ ràng (mục tiêu sàng lọc khoảng 10% cho chỉ số được chọn), không đẩy lỗi/busy lên để làm số đẹp. Xác nhận một lần trên holdout; nếu không đạt giữ cấu hình cũ. Chưa cam kết tốc độ/RAM tuyệt đối khi chưa có máy triển khai và SLO.

Hard timeout native/worker isolation chỉ mở task riêng nếu có yêu cầu cứng hoặc profiling/load test chứng minh cần; không mặc định thêm hàng đợi.

#### C2. Nghiệm thu chạy sạch và triển khai

**Phạm vi:** Dockerfile, requirements, setup/run/test/smoke scripts, README/DEMO; CI chỉ sau khi người dùng quyết định Git/runner.

- [ ] Docker đã sẵn sàng để kiểm tra tiếp: build image, xác minh pip/import, non-root, offline model mount, startup warmup và `/health/ready`. Audit này chưa build/pull image.
- [ ] Smoke HTTP thật scene/document tiếng Việt, thiếu model/traineddata, timeout và lỗi tải ảnh. Không lấy 503 làm pass.
- [ ] Kiểm tra setup Python 3.11 từ môi trường trống, base không Paddle và full hybrid; không thay runtime hệ thống hoặc xóa venv cũ tự động.
- [ ] Trước public: chốt ingress body/read/connection timeout, tổng concurrency/upload, quyền truy cập, lưu log và dữ liệu nhạy cảm. Kiểm thử slow upload/disconnect/temp cleanup; giữ loopback khi chưa nghiệm thu.
- [ ] Nếu đưa vào Git: rà `.gitignore` cho `data/`, artifacts chứa GT/prediction và dữ liệu thật trước commit; chia CI base/mock và integration có model. Không public repository/dataset tự động.
- [ ] Cập nhật trạng thái kế hoạch cũ/README theo bằng chứng mới; ghi rõ Docker daemon sẵn sàng khác với container đã nghiệm thu. Lưu báo cáo chất lượng/hiệu năng cùng cấu hình và giới hạn.

**Đạt khi:** chạy lặp lại từ môi trường sạch, smoke bắt lỗi thật, container không tải model ngoài ý muốn; release report nêu rõ dữ liệu, phạm vi chất lượng và các giới hạn còn lại.

## 4. Thứ tự bàn giao và điều kiện mở khóa

1. **Mốc 1:** A1 → A2 → A3, full suite + regression mới. Giá trị cao nhất: không mất nhãn, không báo thành công giả, không freeze tập sai.
2. **Mốc 2:** A4 và C2 phần local/container; không cần đợi ảnh thật. Chưa mở public.
3. **Mốc 3:** B1 → C1 → báo cáo nghiệm thu. Cần ảnh có quyền sử dụng, máy triển khai và ngân sách latency/RAM mong muốn; chưa có thì chỉ làm đo synthetic chẩn đoán và giữ default.

Không đặt lịch hoàn thành giả cho phần chờ dữ liệu/review. Mỗi mốc bàn giao độc lập, có test và cách quay lại cấu hình trước; không tuyên bố xong toàn kế hoạch khi B1/C1 chưa đủ bằng chứng.

## 5. Giới hạn audit

Chưa chạy lại browser E2E, benchmark tải/soak hoặc Docker build/container trong lượt này; không audit CVE/dependency online, không chứng nhận production/security. Bộ test chạy model thật có pass nhưng dữ liệu chất lượng hiện vẫn chỉ synthetic. Các phát hiện F01–F04 có tái hiện; F05 và ứng viên tối ưu đã được phân biệt rõ là đọc mã/giả thuyết cần đo.

Chủ ý chưa làm: đổi model, thêm LLM sửa dấu, PDF/bảng/chữ viết tay, thiết kế lại UI, framework mới, queue và dọn/xóa dữ liệu hay môi trường người dùng.
