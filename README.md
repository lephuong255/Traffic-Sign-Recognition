# Traffic Sign Recognition

Dự án xây dựng hệ thống phát hiện và phân loại biển báo giao thông, gồm 3 giai đoạn chính:

1. **Phát hiện biển báo giao thông**: sử dụng mô hình YOLO, đầu ra là vị trí các biển báo trên ảnh.
2. **Xử lý ảnh**: cắt vùng ảnh chứa biển báo từ ảnh gốc.
3. **Phân loại biển báo**: sử dụng mô hình CNN, phân loại từng biển báo đã phát hiện được.

## Dataset

Dự án sử dụng dataset GTSRB - German Traffic Sign Recognition Benchmark.

## Hướng dẫn chạy dự án

### 1. Clone dự án

git clone https://github.com/lephuong255/Traffic-Sign-Recognition.git

cd Traffic-Sign-Recognition

### 2. Cài đặt thư viện

pip install -r requirements.txt

### 3. Chạy phát hiện biển báo

python dectection.py


### Sử dụng Docker (tuỳ chọn)

docker build -t traffic-sign-recognition .

docker run -it traffic-sign-recognition


