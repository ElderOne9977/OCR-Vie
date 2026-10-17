# data — dataset layout

## Gate dữ liệu thật

Hiện chỉ có 24 ảnh synthetic trong hai bộ benchmark; chưa có real reviewed test.
Cần ít nhất 100 ảnh thật có quyền sử dụng và review: ảnh in/camera/màn hình/biển hiệu,
chữ nhỏ/mờ/xoay/phối cảnh, nhiều cột, dấu và số tiền/mã giả. Giữ dev riêng, tránh
ảnh gần trùng cùng nguồn rơi vào train/test. Không tự tải ảnh riêng tư lên CI.

Import/annotate trong thư mục người dùng chọn. Mỗi vùng ghi raw GT, polygon,
`ignore` khi thực sự không đọc được; `field_key` optional phải ánh xạ một khóa
trong `fields`, không trùng. Validate + audit + review trước freeze. Freeze v2
không ghi đè version, hash cả ảnh thực và nội dung GT; preliminary không được
gọi là chất lượng đại diện. Chưa có ảnh thật thì giữ Task 7 chưa hoàn thành.

```
data/
  raw/         # gốc, không sửa
  processed/   # ảnh đã resize/normalize (tái tạo từ raw)
  train/       # image + sidecar *.txt/*.json
  val/
  test/
```

Sidecar label: `sample_001.txt` chứa ground-truth UTF-8 cùng tên với ảnh `sample_001.jpg`.
`benchmark_data/images` giữ bộ mẫu nhỏ chạy benchmark nhanh, không commit dataset lớn.
