# OCR-Vie Project Improvements — Implementation Plan

## Trạng thái triển khai — 07/10/2026

Đã thực thi các sửa lỗi độc lập; checkbox đánh dấu theo bằng chứng, không có
nghĩa cả đợt đã nghiệm thu production. [Báo cáo kết quả](../../IMPLEMENTATION_RESULTS_2026-10-07.md).

| Phần | Trạng thái |
| --- | --- |
| Tasks 1–3 | Đã sửa và regression kiểm chứng; không ghi dataset thật |
| Task 4 | Production benchmark/field/hash đã sửa; IoU kiểm tra thiếu Shapely bằng mock, chưa chạy hai venv độc lập |
| Task 5 | ASGI streaming/cap/cleanup/query/lỗi đạt; load upload trên gateway triển khai chưa kiểm tra |
| Task 6 | Runtime/scripts/smoke đạt; Docker/setup từ môi trường trống chưa nghiệm thu |
| Task 7 | Chờ ≥100 ảnh thật được phép dùng, 0 real reviewed test hiện tại |
| Task 8 | TSV, gom dòng, 30 warm + soak 5 phút đã đo; tuning/dev/holdout/PSM còn chờ Task 7 |
| Task 9 | UX, tài liệu, dọn code, smoke/full tests và clean base đạt; browser E2E mọi nhánh/CI chưa đủ |

Server kiểm thử đã dừng; không public/deploy và không tạo Git/remote. Lưu lại
checklist/spec gốc bên dưới; các câu mô tả “chưa thực thi” trong template là ngữ
cảnh trước triển khai, trạng thái hiện tại do bảng này và báo cáo quyết định.

> **For agentic workers:** Thực hiện từng task với một vòng regression test → sửa nhỏ nhất → kiểm tra lại. Các skill `superpowers:subagent-driven-development` và `superpowers:executing-plans` được template writing-plans nhắc tới không có trong phiên hiện tại; khi triển khai dùng các skill sẵn có phù hợp, không tự cài skill hoặc tự mở agent. Đây là kế hoạch, chưa phải các thay đổi đã thực hiện.

**Goal:** Nâng độ tin cậy OCR tiếng Việt, đo chất lượng đúng và giảm chi phí xử lý có kiểm chứng, đồng thời làm bản hybrid triển khai được một cách nhất quán.

**Architecture:** Giữ FastAPI, Paddle detector, Tesseract batch và UI HTML/CSS/JS hiện tại. Sửa nguyên nhân ở helper dùng chung, benchmark gọi pipeline sản phẩm, tái sử dụng công cụ dataset có sẵn. Chỉ thêm hạ tầng khi số đo chứng minh cần.

**Tech Stack:** Python 3.11, FastAPI, NumPy/OpenCV, Pillow, Paddle/PaddleOCR, Tesseract, pytest, PowerShell, Docker; không yêu cầu dependency mới cho các sửa nhỏ.

## Global Constraints

- Căn cứ: `docs/PROJECT_AUDIT_2026-10-07.md`, baseline 191 tests passed ngày 07/10/2026.
- Giữ contract endpoint legacy và raw text; nếu đổi HTTP readiness hoặc validation phải ghi rõ thay đổi contract và cập nhật client/tests.
- Không tải model tự động trong test; dùng `ALLOW_MODEL_DOWNLOAD=false`. Không xóa, sửa nhãn hay dùng ảnh riêng tư nếu chưa nằm trong phạm vi được cho phép.
- Mọi tối ưu cần so sánh cùng dữ liệu, cùng phần cứng, cùng model và cấu hình; không đánh đổi dấu tiếng Việt hoặc trường số quan trọng để chỉ giảm latency.
- Không khởi tạo Git/remote, không triển khai công khai và không bật lại demo trong bước lập kế hoạch này. Khi thực thi, giữ snapshot trước sửa; nếu Git được thiết lập thì commit riêng mỗi task đã đạt nghiệm thu.
- Đường dẫn trong các mục Files tính từ root `C:\Users\PC\OneDrive\Desktop\My Project\OCR-Vie`.
- Mỗi đợt là một đơn vị có thể dừng và bàn giao; không refactor toàn dự án cùng lúc.

## Thứ tự và điều kiện bắt đầu

| Đợt | Nội dung | Phụ thuộc | Quy mô tương đối |
| --- | --- | --- | --- |
| A | Lỗi crop/parser, bảo toàn dữ liệu, độ đúng benchmark | Bắt đầu ngay | Vừa |
| B | Giới hạn upload, health, setup/Docker và smoke | Trước khi public; có thể làm sau từng task A | Vừa |
| C | Bộ ảnh thật và baseline end-to-end | Task 3–4 của A; cần ảnh có quyền sử dụng | Lớn, phụ thuộc gán nhãn |
| D | Tối ưu đo được, thứ tự đọc/PSM | C cho quyết định đổi default; sửa cấp phát có thể làm sớm | Vừa |
| E | UX lỗi, kiểm thử tự động, tài liệu và thu gọn | A/B; số đo D nếu có | Nhỏ–vừa |

Không có ước tính ngày chính xác khi chưa biết lượng ảnh thực tế và cấu hình triển khai. Mốc 100 ảnh là gate ban đầu của dự án, không phải bảo đảm thống kê.

## Task 1 — Crop phối cảnh an toàn (A, F01)

**Files:** sửa `app/services/polygon_utils.py`; bổ sung `tests/test_polygons.py`.

**Interfaces:** giữ `order_quad(points: np.ndarray) -> np.ndarray` và `crop_polygon(image, polygon, *, max_pixels=None) -> np.ndarray | None`. Đầu ra hợp lệ gồm đúng bốn góc khác nhau, thứ tự vòng nhất quán; input suy biến bị từ chối trước warp.

- [x] Thêm regression test cụ thể sau:

```python
def test_order_quad_diamond_keeps_all_corners():
    import itertools
    import numpy as np
    from app.services.polygon_utils import order_quad
    points = [(50, 0), (100, 50), (50, 100), (0, 50)]
    expected = order_quad(np.array(points, dtype=np.float32))
    for permutation in itertools.permutations(points):
        actual = order_quad(np.array(permutation, dtype=np.float32))
        assert len(np.unique(actual, axis=0)) == 4
        np.testing.assert_array_equal(actual, expected)
```

- [x] Chạy `.\.venv311\Scripts\python.exe -m pytest tests/test_polygons.py -q`; test mới phải fail ở bản hiện tại.
- [x] Thay phương pháp tổng/hiệu bằng thứ tự vòng quanh tâm với tie-break xác định; kiểm tra finite, điểm trùng, convexity/diện tích trước warp bằng NumPy/OpenCV. Không tạo lớp geometry mới.
- [x] Thêm ảnh có bốn góc khác màu để kiểm tra không lật/ngược crop; thêm góc 44/45/46°, polygon collinear, NaN và budget pixel. Bốn điểm có diện tích nhưng thứ tự đầu vào lộn xộn vẫn được xử lý.
- [x] Chạy lại suite polygon + `tests/test_ocr_quality_regressions.py`; đạt mọi test, crop cũ không đổi bất hợp lý và không nới giới hạn bộ nhớ.

## Task 2 — Sửa tham chiếu reading_order (A, F07)

**Files:** sửa `app/parsers/fallback_parser.py`; test tại `tests/test_router.py`.

**Interfaces:** `FallbackDocumentParser.parse` tiếp tục trả `DocumentParseResult`; mọi ID trong reading_order phải tồn tại ở pages/blocks.

- [x] Thêm test và chạy để thấy lỗi hiện tại:

```python
def test_fallback_reading_order_references_real_blocks():
    from app.parsers.fallback_parser import FallbackDocumentParser
    for text in ("A\n\nB", "\nA\n", "", "   \n"):
        result = FallbackDocumentParser().parse(text)
        assert result.reading_order == [b.id for b in result.pages[0].blocks]
```

- [x] Sửa đúng một biểu thức của constructor:

```python
reading_order=[block.id for block in pages[0].blocks],
```

- [x] Chạy `.\.venv311\Scripts\python.exe -m pytest tests/test_router.py tests/test_raw_preservation.py -q`; text raw và ID block hiện có không bị đổi.

## Task 3 — Bảo toàn annotation và freeze thật sự (A, F03/F04)

**Files:** sửa `benchmarks/annotate.py`, `benchmarks/audit_annotations.py`, `benchmarks/dataset_summary.py`, `benchmarks/dataset.py`, `benchmarks/freeze_testset.py`; test tại `tests/test_benchmark_validation.py`, `tests/test_benchmark_coverage.py`.

**Interfaces:** tái sử dụng `load_annotations(path: Path) -> list[dict]`. `build_manifest(records, data_dir)` giữ các field hiện có, thêm schema version và fingerprint GT; caller phải phân biệt manifest cũ không đủ thông tin xác minh.

- [x] Thêm fixture JSONL có một dòng hợp lệ và một dòng sai; loader phải raise kèm số dòng, audit exit khác 0, annotator không cho save. Dùng `tmp_path`, không dùng dataset thật cho test ghi.
- [x] Tái sử dụng loader strict, loại bỏ các nhánh `except: pass` làm mất record; báo lỗi dễ sửa thay vì lọc dữ liệu. Schema sai cũng chặn ghi; file đích giữ nguyên khi lỗi.
- [x] Thêm regression cho fingerprint:

```python
def test_manifest_changes_when_ground_truth_changes(tmp_path):
    import copy
    from benchmarks.dataset import build_manifest
    image = tmp_path / "sample.bin"
    image.write_bytes(b"audit-image")
    rows = [{"id": "one", "image": "sample.bin", "split": "test",
             "text": "A", "source_type": "synthetic", "review_status": "reviewed"}]
    changed = copy.deepcopy(rows)
    changed[0]["text"] = "B"
    assert build_manifest(rows, tmp_path)["manifest_hash"] != build_manifest(changed, tmp_path)["manifest_hash"]
```

- [x] Hash canonical nội dung GT ảnh hưởng scoring: language, text, regions/polygon/text/ignore, fields, split/source/review; xác minh lại image hash thay vì chỉ tin metadata. Sort key và quy tắc serialization cố định; không đưa raw GT vào manifest công khai.
- [x] Từ chối version đã tồn tại và version chứa separator đường dẫn; thay đổi GT/ảnh buộc tạo version mới. Test file freeze cũ nguyên byte sau thao tác bị từ chối, và thứ tự records không làm đổi hash.
- [x] Chặn đổi ảnh khi form chưa lưu hoặc yêu cầu người dùng xác nhận; kiểm tra fingerprint file trước save để tránh tab cũ ghi đè tab mới. Dùng file hash đang có, chưa cần database.
- [x] Sửa annotator dùng DOM value/textContent với dữ liệu nhập; chỉ phục vụ ảnh trong dataset bằng containment check sau resolve, không fallback toàn repo.
- [x] Chạy `.\.venv311\Scripts\python.exe -m pytest tests/test_benchmark_validation.py tests/test_benchmark_coverage.py -q`, rồi hai lệnh validate_dataset của audit. Không chạy freeze trên dữ liệu thật trong test.

## Task 4 — Benchmark cùng đường chạy và chấm field đúng (A, F02)

**Files:** sửa `benchmarks/run_scene_benchmark.py`, `benchmarks/run_recognition_benchmark.py`, `benchmarks/metrics.py`; đọc/tái sử dụng `app/pipelines/router.py`; test tại `tests/test_benchmark_quality_regressions.py`, `tests/test_benchmark_metrics.py`.

**Interfaces:** scene end-to-end dùng `get_router().process` với keyword theo signature hiện tại của API; không tạo engine-selection thứ hai. Recognition-only giữ chức năng GT-crop, được gắn nhãn rõ trong summary.

- [x] Test spy router để `auto + vie+eng` của benchmark đi cùng policy API, giữ batch; test ngôn ngữ truyền từ annotation chứ không gọi ensure_ready mặc định tiếng Việt cho mọi dataset.
- [x] Scene runner ghi prediction/region/timing từ PipelineResult; các tùy chọn ablation không đại diện production phải có nhãn riêng, không gộp latency vào cùng bảng.
- [x] Chấm field chỉ khi GT có ánh xạ region rõ ràng. Với dữ liệu cũ không có ánh xạ, báo `field_exact_match=null` và số field chưa đánh giá; không tự đoán từ tên khóa hoặc chấm GT với GT. Khi cần mapping, thêm `field_key` optional trong region và kiểm tra field tồn tại/không trùng.
- [x] Test hai field `amount=100`, `tax_id=123`: hai prediction đúng cho score 1; một sai cho 0.5; thiếu mapping báo chưa đánh giá; thêm region không liên quan không đổi mẫu số.
- [ ] Chọn và ghi rõ backend IoU cho một lần chạy; nếu yêu cầu polygon backend mà thiếu dependency thì fail có thông báo, không âm thầm đổi sang bbox. Thử cùng fixture trong hai môi trường để chứng minh summary ghi đúng loại metric.
- [x] Ghi Python/package versions, model ID/hash, dataset manifest, threads/side/batch/PSM/preprocess, route, sample count, số lỗi và cold/warm scope. All-failed run phải exit khác 0.
- [x] Chạy `.\.venv311\Scripts\python.exe -m pytest tests/test_benchmark_quality_regressions.py tests/test_benchmark_metrics.py tests/test_engine_policy.py -q`. Không so số đo cũ và mới như cùng một baseline nếu policy đổi.

## Task 5 — Chặn upload trước multipart và làm rõ lỗi API (B, F05/F09)

**Files:** sửa `app/main.py`, `app/api/ocr.py`, `app/core/config.py`; test tại `tests/test_validation.py`, `tests/test_performance_regressions.py`.

**Interfaces:** giữ cap file `MAX_UPLOAD_BYTES`; có cap tổng request gồm multipart overhead hữu hạn. Error code phân biệt input sai, quá dung lượng, busy, timeout, model unavailable và lỗi nội bộ.

- [x] Test stream ASGI nhiều chunk có tổng vượt cap, thiếu Content-Length, khai báo Content-Length sai, nhiều file/field và request vừa dưới cap. Test chứng minh body không bị nhận hết/spool toàn bộ trước 413, OCR không được gọi.
- [x] Nếu chỉ phục vụ qua gateway đã có, cấu hình body/connection/read timeout ở đó và kiểm thử đường bypass. Nếu chạy app trực tiếp, thêm một lớp ASGI đếm số byte nhận trước parser; không dùng kiểm tra header làm bảo vệ duy nhất.
- [x] Giữ kiểm tra per-file trong `_read_upload`; cleanup file tạm khi lỗi/cancel. Semaphore OCR tiếp tục bảo vệ compute, không coi nó là giới hạn upload.
- [x] Validate enum/alias query hiện hỗ trợ; typo format/parser/route/recognizer trả lỗi input ổn định, không fallback ngầm. Engine có tên hợp lệ nhưng chưa triển khai vẫn trả unavailable rõ ràng.
- [x] Không đưa exception nguyên văn ra client; log stacktrace gắn request ID mà không log ảnh/raw OCR. Validate config hữu hạn và miền hợp lệ ngay lúc khởi động, không đợi request đầu.
- [ ] Chạy `.\.venv311\Scripts\python.exe -m pytest tests/test_validation.py tests/test_performance_regressions.py tests/test_engine_policy.py -q`. Kiểm tra tải upload đồng thời trong môi trường test trước public; không mở listener public để audit.

## Task 6 — Readiness, setup, Docker và smoke nhất quán (B, F06/F10)

**Files:** sửa `app/api/health.py`, `app/detectors/paddle_detector.py`, `app/recognizers/paddle_recognizer.py`, `app/recognizers/tesseract_recognizer.py`, `app/services/paddle_runtime.py`, `Dockerfile`, `.dockerignore`, `scripts/run_local.ps1`, `scripts/test.ps1`, `scripts/check_environment.py`, `scripts/smoke_test.py`; test tại `tests/test_paddle_adapters.py`, `tests/test_engine_policy.py`, `tests/test_ocr_mock.py`.

**Interfaces:** trạng thái lấy từ cache/load-error runtime hiện có; `/health/live` chỉ xác nhận process. Readiness đánh giá pipeline và ngôn ngữ cấu hình; chuyển HTTP 503 khi không ready nếu chọn làm probe triển khai, cập nhật tests/docs cùng thay đổi.

- [x] Test thiếu vie nhưng có binary → Tesseract không ready cho vie; detector từng fail → adapter mới/health thấy load_failed; weights thiếu và offline → unavailable; weights có chưa load → not_loaded. Health không khởi tạo predictor.
- [x] Đọc trạng thái chung thay vì state riêng che lỗi; kiểm tra model files tối thiểu theo artifact đang dùng, không coi một thư mục có file bất kỳ là model đầy đủ. Giữ retry có chủ đích, không tự retry load model trong health.
- [x] Chuẩn hóa `.venv311` và Python 3.11. Setup cài requirements Paddle một lần; run/test chỉ gọi interpreter xác định và kiểm tra `$LASTEXITCODE`, không upgrade pip khi bật demo. Local bind `127.0.0.1` mặc định.
- [ ] Docker hybrid cài bộ requirements tương ứng và nhận model offline đã chuẩn bị; user không đặc quyền có quyền đọc models/ghi tmp. Healthcheck kiểm tra đúng contract, không chỉ HTTP 200 của legacy health. Không dùng script warmup process riêng như bằng chứng model nằm trong RAM server.
- [x] Loại venv_broken, benchmark_incoming, artifacts, benchmark_results khỏi build context. Không xóa các thư mục này khỏi máy người dùng.
- [x] Smoke kiểm tra marker ổn định của trang hiện tại, ảnh có chữ qua `/v1/ocr` scene auto và document. Chế độ release phải fail nếu thiếu model/traineddata hoặc output rỗng; kiểm tra môi trường thiếu-model tách riêng và không báo đã đạt release gate.
- [ ] Chạy tests liên quan; khi Docker daemon sẵn sàng mới build/run, xác minh offline/no model/missing vie/valid model và graceful shutdown. Chưa có daemon thì ghi rõ chưa nghiệm thu container, không tự bật Docker Desktop.

## Task 7 — Baseline chất lượng ảnh thật (C, F03/F07)

**Files:** dùng `benchmarks/import_images.py`, `benchmarks/annotate.py`, `benchmarks/validate_dataset.py`, `benchmarks/audit_annotations.py`, `benchmarks/freeze_testset.py`; cập nhật `data/README.md` và metadata dataset trong thư mục được người dùng chọn.

**Interfaces:** schema/fingerprint từ Task 3, summary/scoring từ Task 4. Đầu ra là bộ test đã duyệt và báo cáo baseline có manifest; không phải bộ ảnh tự tải từ Internet.

- [ ] Xin/nhận ảnh đúng phạm vi sử dụng nếu chưa có: in rõ, camera, ảnh màn hình, biển hiệu, chữ nhỏ, mờ, xoay/phối cảnh, nhiều cột, dấu tiếng Việt, số tiền/mã định danh. Dùng thông tin giả hoặc che dữ liệu nhạy cảm có kiểm soát.
- [ ] Thu tối thiểu 100 ảnh real reviewed test; tránh train/test cùng ảnh hoặc cùng nguồn gần trùng. Giữ tập dev riêng để chọn tham số; không tối ưu đi tối ưu lại trên test đã freeze.
- [ ] Đánh nhãn raw text/regions/reading order cần thiết, ignore vùng thực sự không đọc được, mapping field nếu chấm field. Reviewer đối chiếu dấu, chữ số và GT của từng nhóm lỗi.
- [ ] Chạy validate + audit + coverage report trước freeze; không có lỗi cú pháp/hình học, mọi ảnh có nguồn/quyền sử dụng và review status thật. Freeze phiên bản mới, không dùng `--allow-preliminary` cho báo cáo chính thức.
- [ ] Chạy scene/document baseline theo policy sản phẩm; báo CER/WER toàn trang, detection recall, dấu tiếng Việt, exact-match field đã map, tỷ lệ lỗi/rỗng và số mẫu từng nhóm. Không chỉ báo CER trên crop đã match vì sẽ bỏ qua vùng bị detector bỏ sót.
- [x] Nếu chưa có ảnh thật được phép dùng: kết thúc ở báo cáo synthetic/preliminary, giữ Task 7 chưa hoàn thành; không tuyên bố chất lượng thực tế đã đạt.

## Task 8 — Tối ưu nhỏ trước, thử cấu hình sau (D, F07/F08)

**Files:** sửa `app/recognizers/tesseract_recognizer.py`; test tại `tests/test_performance_regressions.py`; mở rộng `scripts/verify_ocr_performance.py`; sau baseline mới cân nhắc `app/pipelines/router.py`, `app/preprocessing/document.py`.

**Interfaces:** `_result(data, lang)` giữ output tương đương; benchmark mới ghi latency/RSS/CPU với config và manifest, không thay default ngay khi chạy probe.

- [x] Test `_result` có đủ cột, thiếu cột nhóm, nhiều dòng, từ rỗng và confidence không hợp lệ. Dùng output hiện tại hợp lệ làm expected, không dùng time threshold mong manh trong unit test.
- [x] Thay cấp phát trong vòng lặp bằng các cột đã lấy một lần:

```python
# Đặt trước vòng for i, word của _result.
count = len(data.get("text", []))
zeros = [0] * count
group_columns = [data.get(k, zeros) for k in
                 ("page_num", "block_num", "par_num", "line_num")]
# Thay biểu thức tạo key bên trong vòng lặp.
key = tuple(column[i] for column in group_columns)
```

- [x] Chạy tests và microbenchmark 1k/4k/8k từ trước/sau, ít nhất 5 lần; lưu kết quả tách biệt với OCR end-to-end. Xác nhận text/confidence không đổi.
- [x] Đo cold riêng, warm tối thiểu 30 request/cấu hình cho sàng lọc; p95 từ mẫu nhỏ chỉ là sơ bộ. Đo RSS toàn cây process kể cả Tesseract, CPU-seconds/request, temp bytes, throughput và error rate ở concurrency 1/2/4; soak 5–10 phút tìm xu hướng bộ nhớ tăng.
- [ ] Dùng baseline dev để so side 768/960/1280, threads 4/8/10, batch 8/16/32 theo từng biến, không chạy tích mọi tổ hợp ngay. Giữ MKLDNN baseline đã có bằng chứng; chỉ thử tắt lại khi có regression/phần cứng khác.
- [ ] Thêm fixture cùng dòng lệch y một pixel và nhiều cột trước thay sort. Gom dòng theo overlap/chiều cao rồi x; chưa thêm layout engine. So PSM 3/6/11 và preprocessing original/gray trên tập document có nhãn, không chọn từ một ảnh đẹp.
- [x] Chỉ đổi default nếu giảm p95 hoặc CPU/RSS đo được mà không làm xấu chất lượng trong ngưỡng đã chốt trên dev; xác nhận một lần trên holdout. Ngưỡng tuyệt đối về latency/RAM phải dựa vào máy triển khai, chưa có thì chỉ báo trade-off.
- [x] Nếu vẫn cần hard timeout native hoặc throughput vượt lock hiện tại, lập task riêng cho worker isolation; không đưa multiprocessing/Redis/Celery vào đợt tối ưu nhỏ.

## Task 9 — UX lỗi, automation và tài liệu (E, F09/F10)

**Files:** sửa `app/ui.html`, `tests/test_ui_interactions.py`, `README.md`, `DEMO.md`, `PROJECT_LOG.md`; nếu dự án được quản lý bằng Git thì bổ sung CI tại `.github/workflows/tests.yml`. Chỉ dọn `app/core/model_registry.py`/script lịch sử sau xác minh không có callers.

**Interfaces:** UI dùng error code từ API, giữ Retry-After khi busy, không mất ảnh đang chọn hoặc kết quả mới hơn do request cũ. README là nguồn hướng dẫn chạy hiện tại; log/history giữ ngữ cảnh cũ.

- [x] Test `OCR_BUSY` khác `MODEL_UNAVAILABLE`, timeout khác ảnh sai định dạng, không tự retry vòng lặp. UI hiển thị server bận và cho thử lại; model thiếu có hướng kiểm tra cấu hình.
- [ ] Chạy lại UI test Node hiện có; browser E2E desktop/mobile cho chọn ảnh, drag/drop, gửi lặp, đổi ảnh, kết quả rỗng, lỗi 413/503/timeout, copy/export. Kiểm tra clipboard thực nếu môi trường cho phép, nếu không ghi chưa xác minh.
- [x] Giữ keyboard/focus/status ARIA/reduced motion. Không thêm animation/framework. Overlay vùng OCR chỉ làm ở task riêng nếu người dùng cần kiểm tra vị trí lỗi thường xuyên.
- [ ] Chạy unit/mock trong môi trường base không model; job integration có Paddle/model/traineddata thật phải kiểm tra tiếng Việt không rỗng và CER trên fixture cố định, không coi 503 là pass. Test thiếu-dependency là job riêng. Không tự tải ảnh riêng tư vào CI.
- [x] Tách README thành quickstart hiện hành ngắn và liên kết lịch sử; sửa lệnh Python/venv/Paddle, mô tả giới hạn không hỗ trợ PDF/layout, cập nhật health/smoke/setup. Pin dependencies theo môi trường đã kiểm chứng, không nâng đồng loạt chỉ vì có bản mới.
- [x] Dùng `rg` rà callers trước bỏ cache ModelRegistry không dùng, cấu hình chết hoặc script patch lịch sử. Không xóa adapter/config đang là contract nếu chưa có migration/deprecation rõ ràng.
- [x] Nghiệm thu cuối: full pytest, pip check, dataset validation, smoke model thật, container check nếu đủ điều kiện; lưu báo cáo những phần chưa kiểm tra. Chỉ ghi hoàn thành cho các bước có bằng chứng.

## Handoff ban đầu (trước thực thi)

Bắt đầu Task 1–4: lỗi có bằng chứng, diff nhỏ theo từng task, không phụ thuộc việc thu ảnh hoặc bật Docker. Task 5–6 bắt buộc trước khi public. Task 7 cần nguồn ảnh thực tế được phép sử dụng; Task 8 không được đổi default chỉ dựa vào 24 mẫu synthetic hiện có.

Không cần lựa chọn kiến trúc mới để bắt đầu. Kế hoạch chưa được thực thi; khi được yêu cầu triển khai có thể làm tuần tự trong phiên này, mỗi task có kiểm tra riêng và một mốc bàn giao.
