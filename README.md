# OCR-Vie

OCR-Vie giúp lấy văn bản tiếng Việt và tiếng Anh từ ảnh. Bạn có thể mở giao diện
trên trình duyệt để chọn ảnh, xem kết quả, sao chép hoặc tải văn bản về dưới dạng
file `.txt`. Dự án cũng có API để tích hợp vào ứng dụng khác.

Ứng dụng chạy trên máy của bạn. Hai model PaddleOCR đã có sẵn trong repository;
bạn cần cài thêm Python và Tesseract để sử dụng.

## Cài đặt và chạy

Hướng dẫn dưới đây dùng Windows và PowerShell.

Trước tiên, cài **Python 3.11**, **Git** và **Tesseract OCR**. Khi cài Tesseract,
hãy thêm dữ liệu ngôn ngữ tiếng Việt (`vie`) và tiếng Anh (`eng`). Mặc định ứng
dụng tìm Tesseract tại `C:\Program Files\Tesseract-OCR\tesseract.exe`.

Mở PowerShell rồi chạy:

```powershell
git clone https://github.com/ElderOne9977/OCR-Vie.git
cd OCR-Vie

$env:MODELS_DIR = (Resolve-Path .\models).Path
$env:ALLOW_MODEL_DOWNLOAD = 'false'
.\scripts\setup.ps1
.\scripts\run_local.ps1
```

Script setup tạo môi trường Python riêng tại `.venv311` và cài các thư viện cần
thiết. Bạn chỉ cần chạy setup khi cài lần đầu hoặc khi dependencies thay đổi.
Model đi kèm chiếm khoảng 92 MiB nên lần clone đầu có thể mất thêm thời gian.

Sau khi server chạy, mở [http://127.0.0.1:8000](http://127.0.0.1:8000).
Để dừng ứng dụng, nhấn `Ctrl+C` trong cửa sổ PowerShell đang chạy server.

Ở những lần sử dụng tiếp theo, mở PowerShell tại thư mục dự án và chạy:

```powershell
$env:MODELS_DIR = (Resolve-Path .\models).Path
$env:ALLOW_MODEL_DOWNLOAD = 'false'
.\scripts\run_local.ps1
```

Nếu bạn cài Tesseract ở vị trí khác, đặt đường dẫn trước khi chạy setup hoặc server:

```powershell
$env:TESSERACT_CMD = 'D:\Tesseract-OCR\tesseract.exe'
```

## Sử dụng

Chọn ảnh, chọn ngôn ngữ rồi bấm **Nhận dạng văn bản**. Có thể giữ loại ảnh ở chế
độ **Tự động**; nếu kết quả chưa tốt, hãy chọn **Tài liệu / trang in** hoặc
**Ảnh thực tế / biển hiệu** phù hợp với ảnh của bạn.

Ứng dụng nhận JPG, PNG, WebP, BMP và TIFF, mặc định tối đa 10 MiB mỗi ảnh. Lần
nhận dạng đầu thường lâu hơn vì cần nạp model. Ảnh rõ nét, đủ sáng và không bị
cắt chữ sẽ cho kết quả tốt hơn. Hãy kiểm tra lại dấu tiếng Việt, số tiền và mã số
trước khi sử dụng kết quả.

Với ảnh thực tế tiếng Việt, PaddleOCR tìm các vùng chữ, sau đó Tesseract đọc
từng vùng. Với trang tài liệu, Tesseract đọc toàn trang. Bạn nên giữ bộ nhận dạng
ở chế độ **Tự động**: model Latin đi kèm chưa có đủ ký tự tiếng Việt.

Hiện dự án chưa hỗ trợ PDF. Tài liệu nhiều cột, bảng phức tạp và chữ viết tay
chưa được xử lý đáng tin cậy. Bộ mẫu đi kèm là ảnh tạo bằng phần mềm, nên kết quả
kiểm thử trên bộ này chưa phản ánh đầy đủ chất lượng với ảnh thực tế.

## Gọi API

Khi server đang chạy, mở [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
để xem các tham số và thử gửi ảnh trực tiếp.

Ví dụ gửi ảnh từ PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/ocr?route=auto&lang=vie%2Beng" -F "file=@anh.png"
```

Thay `anh.png` bằng đường dẫn ảnh của bạn. API trả JSON có văn bản, các vùng chữ
và thông tin xử lý. Nếu chỉ cần văn bản, dùng endpoint `/v1/ocr/text`.

Bạn có thể kiểm tra server qua `/health/live`. Endpoint `/health/ready` cho biết
pipeline ảnh thực tế đã sẵn sàng hay chưa; trước lần nạp model đầu tiên, endpoint
này có thể trả HTTP 503 dù server vẫn đang chạy.

## Phát triển và kiểm thử

Mã ứng dụng nằm trong `app/`, tests trong `tests/`, còn `scripts/` chứa các lệnh
cài đặt, chạy và kiểm tra. Thư mục `benchmarks/` cùng hai bộ `benchmark_data*`
dùng để đánh giá OCR trên ảnh mẫu.

Sau khi setup và đặt `MODELS_DIR` như ở trên, chạy bộ kiểm thử bằng:

```powershell
.\scripts\test.ps1
```

Khi nhập ảnh thật để đánh giá, chỉ dùng dữ liệu bạn có quyền sử dụng. Giữ ảnh
riêng tư trong `benchmark_incoming/` hoặc `data/`, không commit chúng lên Git.
Chỉ dùng các mẫu đã được kiểm tra nhãn để kết luận về chất lượng; bộ lọc
`--reviewed-only` hỗ trợ việc này trong các lệnh benchmark.

## Model và giấy phép

Hai model nằm tại `models/official_models`: `PP-OCRv5_server_det` để phát hiện
chữ và `latin_PP-OCRv5_mobile_rec` để nhận dạng chữ Latin. Các file model được
giữ nguyên từ PaddleOCR, kèm model card dưới tên `MODEL_CARD.txt`.

Thông tin nguồn và giấy phép Apache-2.0 của model nằm trong
[THIRD_PARTY_NOTICES.txt](models/THIRD_PARTY_NOTICES.txt) và
[LICENSE-APACHE-2.0.txt](models/LICENSE-APACHE-2.0.txt). Giấy phép này áp dụng cho
các thành phần PaddleOCR đi kèm; mã nguồn OCR-Vie chưa có giấy phép riêng.

Repository giữ mã nguồn, tests, model và dữ liệu mẫu cần thiết. Môi trường Python,
cache, kết quả chạy và tài liệu làm việc nội bộ được giữ ngoài Git. Dockerfile
cũng có sẵn, nhưng quy trình chạy container chưa được kiểm thử đầy đủ.
