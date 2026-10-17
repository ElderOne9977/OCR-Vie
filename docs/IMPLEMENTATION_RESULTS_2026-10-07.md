# Kết quả thực hiện kế hoạch — 07/10/2026

## Trạng thái

Đã triển khai các sửa lỗi độc lập trong A/B/E và tối ưu nhỏ của D. **Chưa hoàn thành toàn bộ kế hoạch**: C cần ảnh thật; lựa chọn tham số D chờ baseline dev/holdout; nghiệm thu container chờ Docker daemon. Không public/deploy, không khởi tạo Git, không tải model/ảnh mới và không sửa dataset người dùng. Server loopback kiểm thử đã dừng.

| Task | Kết quả |
| --- | --- |
| 1–2 | Crop convex/finite/khác góc, thứ tự vòng xác định; parser reading_order trỏ block thật |
| 3 | Loader strict, chặn schema/geometry/mapping sai, revision/ETag, safe DOM và ảnh containment, manifest v2/immutable freeze |
| 4 | Scene gọi router sản phẩm, ablation tách riêng; GT-crop field mapping đúng, IoU explicit, metadata/weights/GT hash và exit nonzero khi all-failed |
| 5 | Cap body khi đang đọc; file spool được đóng khi 413; enum query/config và lỗi nội bộ an toàn có request ID |
| 6 | Readiness dùng cache/load-error và vie/eng traineddata; chuẩn `.venv311`, setup/run/test/smoke; Docker hybrid non-root offline đã sửa, **chưa chạy container** |
| 7 | **Chưa có dữ liệu**: 0 real reviewed test; không freeze/gán nhãn giả |
| 8 | TSV cấp phát một lần, gom dòng lệch y, đo 30 warm và soak 5 phút; **chưa chọn profile mới** |
| 9 | UX busy/timeout giữ ảnh, Node checks/browser copy/export; tài liệu hiện hành tách lịch sử; bỏ code không caller. CI chưa áp dụng vì chưa có Git/remote |

## Sửa bổ sung từ smoke thật

Fixture `Xin chào Việt Nam` được document đọc đúng nhưng scene đọc `Xin chào Viét Nam` dù HTTP 200. Crop `[11,12]..[275,42]` quá sát mép: bbox và perspective crop đều sai; thêm viền trắng 1/2/4/8 pixel trong thí nghiệm đều đọc đúng. Tesseract adapter hiện thêm **2 pixel trắng riêng PSM 7**, chung cho single/batch, không resize chữ hay suy đoán thay dấu.

Test có thể chạy lại: `tests/integration/test_paddle_scene_ocr.py::test_vietnamese_hybrid_strict`. Yêu cầu chữ đúng, detector Paddle và recognizer Tesseract thật; không chấp nhận 503/ảnh trắng. 12 dòng ghép synthetic giữ CER 0, batch/single và confidence tương đương, vùng trắng giữa batch giữ mapping. Viền này là sửa segmentation có bằng chứng fixture, **chưa chứng minh cải thiện mọi nhóm ảnh thật**.

## Bằng chứng kiểm thử

- Full pytest offline: **253 passed**, một warning ccache không ảnh hưởng runtime. Log: `artifacts/pytest-final-2026-10-07.log`.
- Môi trường **base Python 3.11 mới**, chỉ cài requirements-dev/base, xác minh không có `paddle`/`paddleocr`: **249 passed/4 deselected**, pip check đạt. Log `artifacts/pytest-base-2026-10-07.log`. Có warning deprecation Starlette TestClient/httpx, chưa là lỗi và không thêm httpx2 chỉ để tắt warning. Venv tạm giữ tại `C:\Users\PC\AppData\Local\Temp\ocr-vie-base-audit-da2d90f60cf543fbb5b20d29b1775799` vì thao tác dọn bị policy chặn; không thử vượt chặn. Package inventory: `artifacts/base-packages-2026-10-07.txt`.
- `pip check`: không broken requirements. Đã bỏ wheel `opencv-python-headless` trùng namespace và khôi phục `opencv-contrib-python==4.10.0.84` đúng yêu cầu PaddleX. Không nâng model framework; pytest dev pin khớp 9.1.1 đang kiểm thử.
- Validate hai dataset: mỗi bộ 12 records/12 ảnh; audit quality_v2 không lỗi. Mỗi bộ đều synthetic/unreviewed, tổng real reviewed test = 0.
- `check_environment.py`: PASSED; không import/nạp Paddle, kiểm tra Python/binary/vie/eng/artifact cục bộ.
- PowerShell parser: mọi `scripts/*.ps1` không lỗi syntax. Setup Windows toàn bộ từ môi trường trống **chưa chạy lại**; môi trường hiện hữu đã sửa OpenCV và kiểm tra.
- Strict smoke HTTP: root marker, API, live, document + scene tiếng Việt chính xác, readiness đều đạt. Log: `artifacts/smoke-2026-10-07.log`.
- Body regression: thiếu/khai gian Content-Length, multi-file/field, exact cap/+1 byte, chặn trước nhận hết; test đi qua file **đã rollover xuống đĩa** và xác minh đóng file.
- Nhánh Paddle partial failure cũng che exception khỏi `warnings`, log request ID; benchmark không đếm mọi vùng lỗi như ảnh thành công chỉ vì pipeline trả HTTP 200.
- Browser thật: upload ảnh synthetic, scene OCR `Cửa hàng 24h`, controls khóa lúc xử lý, copy đọc lại clipboard đúng, tải TXT đúng UTF-8, bỏ ảnh vô hiệu hóa copy/export và bỏ kết quả cũ. Viewport mobile đo DOM width=scrollWidth=375, một cột. TXT kiểm thử được giữ tại `C:\Users\PC\Downloads\ocr (1).txt`.
- Node chạy JS thực: gửi lặp, đổi ảnh, output rỗng, 413, unsupported/busy 503 và timeout 504; annotator discard phục hồi region edits/cancel giữ ảnh. **Drag/drop và các nhánh lỗi chưa chạy browser E2E toàn bộ**, chỉ các nhánh đã nêu được kiểm chứng.
- Docker CLI có nhưng daemon không kết nối được. **Chưa build/run, chưa xác minh Linux/offline/no-model/missing-vie/graceful shutdown container.** Không tự bật Docker Desktop.

## Số đo, không đánh đồng với chất lượng thực tế

Máy kiểm thử: Windows 10.0.26200, AMD64 Family 25 Model 117, 16 logical CPUs; Python 3.11.7, Paddle 3.2.0/PaddleOCR 3.7.0/PaddleX 3.7.2. CPU_THREADS=10, detector side=960, batch=32. Hash model/traineddata/GT trong summary JSON; phiên bản benchmark khác policy không được so như cùng baseline.

### TSV — trước/sau cùng input, không chạy OCR

Đọc implementation trước sửa từ snapshot, so toàn bộ output/confidence với implementation mới, năm lần cho mỗi kích thước.

| Số từ | Median trước | Median sau |
| --- | ---: | ---: |
| 1.000 | 12,027 ms | 2,755 ms |
| 4.000 | 77,975 ms | 9,961 ms |
| 8.000 | 433,335 ms | 17,776 ms |

8.000 từ nhanh hơn khoảng 24,4 lần **ở parser TSV**, không phải toàn pipeline. Log `artifacts/tsv-performance-2026-10-07.json`; script `scripts/verify_tsv_performance.py` so source snapshot thực, không dùng code baseline đoán lại.

### OCR 12 dòng ghép synthetic — cold riêng + 30 warm

| Pipeline | Cold | Warm p50 / p95 | CER | Peak tree RSS |
| --- | ---: | ---: | ---: | ---: |
| Hybrid | 15,28 s | 2,07 / 2,42 s | 0 | 903,9 MiB |
| Document | 0,76 s | 1,10 / 1,32 s | 0,0268 | 164,1 MiB |

CPU warm estimate: hybrid 14,89 CPU-s/request, document 1,38; sampling 20 ms có thể bỏ lỡ subprocess ngắn. `artifacts/performance-hybrid-isolated-2026-10-07.json` (tên tệp **không bảo đảm máy isolated**), `artifacts/performance-document-2026-10-07.json`. Có test/process khác trong một phần cửa sổ đo; p95 chỉ sơ bộ 30 mẫu, **không dùng so tốc độ tuyệt đối với các số đo lịch sử**. Mỗi script là process mới, model cache offline.

### HTTP tải đồng thời và soak — fixture một dòng

Sau viền 2 px: `artifacts/load-scene-padded-2026-10-07.json`.

| Concurrency | Requests | 200 / Busy | p50 / p95 của 200 | Throughput 200 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 30 | 30 / 0 | 636 / 965 ms | 1,50/s |
| 2 | 30 | 30 / 0 | 623 / 935 ms | 3,01/s |
| 4 | 30 | 2 / 28 | 802 / 847 ms | 2,36/s |

Mọi request 200 đọc đúng fixture. Concurrency 4 **không có 30 successful samples**: fast rejection làm phần lớn burst bị busy, nên p95 n=2 không có ý nghĩa thống kê. Giữ admission=2; không thêm queue/worker pool từ số đo này.

Soak 300,65 s: 456/456 thành công/đúng chữ, không lỗi, p50=630 ms, p95=917 ms. Peak RSS cây process 717,46 MiB; median RSS cửa sổ 30–60 s =655,1 MiB, 60 s cuối =522,8 MiB. Không thấy xu hướng tăng liên tục trong cửa sổ này; **không chứng minh không leak lâu dài**. CPU tổng estimate 1306,56 s, khoảng 2,52 CPU-s/200 tính cả cold/burst/soak. Temp batch sampled peak=12.416 bytes: đo toàn bộ prefix `ocr_batch_*`, không attribution tuyệt đối cho một process; lấy mẫu 100 ms có thể bỏ lỡ peak.

Log `artifacts/load-scene-2026-10-07.json` là **trước sửa viền**, lưu để tái hiện lỗi: mọi 200 sai fixture, script exit 1. Không gộp với kết quả sau. Bản script load mới phân biệt OCR_BUSY với MODEL_UNAVAILABLE và timeout bằng error code; các artifact đầu được chạy trước bổ sung phân loại này, lúc đó busy được xác nhận từ API test và log server.

### Benchmark 12 ảnh synthetic riêng lẻ

Scene production: 12/12 thành công, không rỗng; detection P=R=1, IoU OpenCV/threshold 0,5; page CER=0,004/WER=0,0167/exact=11/12. GT-crop recognition-only: CER=0/WER=0/exact=12/12. **Crop GT và ảnh toàn cảnh khác phạm vi**, không so như cùng chất lượng.

Summary/CSV nằm dưới `artifacts/benchmark-scene-2026-10-07/` và `artifacts/benchmark-recognition-2026-10-07/`. Có manifest v2, Python/package/config, fingerprint artifacts; inventory model cục bộ không đồng nghĩa mọi model đó đã dùng, provenance quyết định engine thực. Không có field mapping trong bộ mẫu nên field metric chưa được đánh giá, không báo 100% giả. Breakdown ký tự dấu là NFC count deficits, không phải lỗi edit-aligned.

Scene runner sau lần đo trên đã thêm `end_to_end.field_exact_match` và coverage:
fixture kiểm tra 1/0,5/null, bỏ sót detection của field phải tính sai; region không
liên quan không tăng mẫu số. Bộ synthetic không có field mapping nên chưa chấm
chất lượng field thực tế.

## Còn lại và điều kiện mở khóa

1. **Task 7:** người dùng cung cấp thư mục ảnh có quyền sử dụng; ≥100 real reviewed test, dev riêng, GT/mapping/reading order, validate/audit/freeze mới rồi baseline.
2. **Task 8:** sau baseline dev, thử side/threads/batch theo từng biến, PSM/preprocess và xác nhận holdout. Chưa đổi các default này hoặc thêm hard-timeout worker isolation.
3. **Task 6/9:** Docker daemon sẵn sàng mới nghiệm thu container; setup hybrid từ trống và toàn bộ browser E2E lỗi/drag-drop còn chưa xác minh. CI chỉ thêm/kích hoạt khi người dùng đặt dự án trong Git và xác định runner offline có model.
4. Trước public: gateway body/read/connection timeout, tải upload đồng thời ở hạ tầng triển khai, bảo vệ truy cập/threat model phù hợp; lần triển khai này không cấp quyền public.

## Bảo toàn và rollback

Snapshot source/docs/scripts/tests trước sửa: `C:\Users\PC\AppData\Local\Temp\ocr-vie-before-plan-d0efb58eafef44b08187282ddde4ee06.zip`. Không chứa venv, model weights hay dataset. Cache ModelRegistry và script patch lịch sử không có caller đã bỏ, có thể khôi phục từ snapshot. README/DEMO cũ chuyển vào `docs/HISTORICAL_*_2026-10-07.md`, không xóa ngữ cảnh lịch sử. Không xóa dataset/venv_broken/artifacts người dùng; chỉ loại khỏi Docker context. Không chạy freeze trên dataset thật.

Các skill TDD/systematic-debugging định hướng regression → nguyên nhân → sửa dùng chung → kiểm tra; ponytail giữ kiến trúc hiện tại, không thêm framework/queue/dependency mới. Browser CLI không có nên dùng browser có sẵn để kiểm tra UI thực.
