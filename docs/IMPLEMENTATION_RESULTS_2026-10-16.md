# Tiến độ kế hoạch 16/10/2026

Plan: docs/superpowers/plans/2026-10-16-project-improvements.md

Ruling: repository chưa có Git; triển khai tại workspace được người dùng chỉ định, giữ ledger này thay các script task/commit phụ thuộc Git. Không tự khởi tạo repository.

Pre-flight: A1/A3 dùng chung annotation schema; giữ nhãn draft, chỉ freeze dữ liệu qua validation toàn bộ. A2 thay contract lỗi, phải kiểm tra cả API text và benchmark. A4 phải giữ tương thích test JavaScript hiện có.

## Đã thực hiện

- A1: khóa control/navigation/canvas trong lúc lưu hoặc tải danh sách; chặn duplicate save; snapshot chính xác payload; lỗi mạng giữ dirty; list rỗng vô hiệu editor; bỏ callback ảnh cũ.
- A2: all-failed neural nhận dạng trả lỗi 500 ở cả JSON/text API; lỗi classifier nội bộ được che; partial result và nhận dạng hợp lệ rỗng vẫn giữ contract.
- A3: validator chặn kiểu dữ liệu bắt buộc, tags sai kiểu, split files sai membership/thiếu ID; metadata enum dùng chung với annotator/manifest. Freeze validate toàn dataset trước filter/gate, chặn duplicate image và leakage; không ghi đè version. Lỗi ghi split files trả lỗi rõ ràng, giữ JSONL chính đã lưu, yêu cầu reload trước retry.
- A4 phần timeout: AbortController native, budget server + 60 giây cho upload/phản hồi; luôn mở khóa và giữ ảnh khi abort; thông báo server có thể tiếp tục xử lý.

Ruling: fixture partial-error cũ chỉ có một vùng lỗi; sửa thành hai vùng với một vùng thành công để thực sự kiểm tra partial result. Fixture freeze preliminary dùng ảnh khác nhau và schema hợp lệ, không nới validation để giữ fixture sai.

Ruling: giữ compatibility của OCR_SCENE_DETECTOR/OCR_DEBUG trong đợt này; chưa bật thêm behavior. OCR_SCENE_DETECTOR hiện không đổi detector trong router, OCR_DEBUG không tắt timing của v1; cần deprecate/migration riêng, tránh đổi API không cần thiết.

## Kiểm chứng

Regression API/schema/freeze ban đầu: 9 failed đúng các lỗi dự kiến, sau sửa passed. Regression UI chờ/save ban đầu failed, sau sửa passed. Kiểm tra lỗi ghi split files ban đầu bị disconnect, sau sửa trả 500 có thông báo reload và giữ JSONL.

Final review độc lập theo skill executing-plans phát hiện race khi loading list và freeze filter trước validation. Đã thêm regression RED cho cả hai, sửa khóa loading và metadata validation trước filter; chạy lại full suite sau sửa.

Full pytest cuối: **266 passed**, 32,37 giây, một warning thiếu ccache. Có strict integration tiếng Việt với model cục bộ; `ALLOW_MODEL_DOWNLOAD=false`. Kết quả này không phải benchmark ảnh thật.

Hai dataset hiện có vẫn validate: mỗi bộ 12 records/12 unique images. `pip check`: không broken requirements. Không đổi weights, nhãn thật, cấu hình chất lượng hay dependency.

## Chưa hoàn tất

B1 cần ảnh thật có quyền sử dụng; C1 cần baseline dev thật để chọn default. Chưa đo lại tải/soak, setup từ môi trường trống hoặc browser E2E đầy đủ. A3 chưa hợp nhất toàn bộ schema legacy thành một validator duy nhất; vẫn giữ các mức hợp lệ draft/benchmark hiện có. A4 aliases/readiness/language matrix chưa mở rộng trong lượt này.

C2: thử build `ocr-vie:audit-20261016`; dừng build bằng Ctrl+C trong bước tải Debian do tốc độ chậm, chưa tạo image hoàn chỉnh/chưa chạy container nghiệm thu. Không có build tiếp tục ngầm sau bàn giao. Docker daemon hoạt động. Không bật demo và không can thiệp container khác trên máy.
