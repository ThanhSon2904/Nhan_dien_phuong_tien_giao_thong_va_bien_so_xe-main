# Nhận diện phương tiện giao thông và biển số xe

## 1. Giới thiệu hệ thống

Hệ thống này phát hiện phương tiện giao thông (ô tô, xe máy) và trích xuất biển số từ ảnh hoặc video. Ứng dụng gồm hai phần chính:
- Backend: REST API xây dựng bằng FastAPI, xử lý ảnh/video, nhận diện phương tiện và biển số, lưu kết quả vào MongoDB, và phục vụ giao diện frontend.
- Frontend: UI đơn giản (HTML/CSS/JS) để tải ảnh/video lên, bật camera, xem luồng camera và danh sách kết quả phân tích.

Mục tiêu: cung cấp công cụ nhận diện nhanh trên máy cá nhân hoặc máy chủ nhỏ để phân tích video/ảnh phục vụ giám sát giao thông, thử nghiệm và demo.

---

## 2. Công nghệ sử dụng

- Ngôn ngữ: Python 3.10+
- Web API: FastAPI
- Server: Uvicorn
- Xử lý ảnh/Video: OpenCV, Pillow, NumPy
- Nhận diện / OCR: ultralytics (YOLO), EasyOCR (nếu có trong model), các file model được lưu trong `backend/app/models`
- Cơ sở dữ liệu: MongoDB (lưu trữ kết quả phân tích)
- Frontend: HTML/CSS/JavaScript (file tĩnh được serve bởi FastAPI)

Các package chính có trong `backend/requirements.txt`:
- fastapi, uvicorn, pydantic, python-multipart
- opencv-python, numpy, pillow
- easyocr, ultralytics
- pymongo

---

## 3. Hướng dẫn cài đặt và sử dụng (Windows - PowerShell)

Dưới đây là các bước cấu hình và chạy dự án trên Windows (sử dụng PowerShell). Giả định bạn đang ở thư mục gốc của dự án (chứa `backend` và `frontend`).

### 3.1. Chuẩn bị môi trường Python

1. Cài Python 3.10+ nếu chưa có. Kiểm tra:

```powershell
python --version
```

2. Tạo virtual environment và kích hoạt: (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script, cho phép tạm thời (chỉ chạy 1 lần với quyền admin nếu cần):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

3. Cài phụ thuộc Python:

```powershell
pip install -U pip
pip install -r backend\requirements.txt
```

> Lưu ý: Một số package (như `opencv-python`, `easyocr`, `ultralytics`) có thể cần trình biên dịch hoặc phụ thuộc hệ thống; nếu cài bị lỗi, thử cài từng package riêng lẻ và kiểm tra thông báo lỗi.

### 3.2. Chuẩn bị model

Các file model (ví dụ `yolov8n.pt`, `LP_detector.pt`, `LP_ocr.pt`, v.v.) đã có trong `backend/app/models` và thư mục cấp cao `backend`. Bạn có thể thay thế hoặc thêm model mới vào thư mục này.

### 3.3. Cấu hình MongoDB

Hệ thống mặc định sử dụng MongoDB tại `mongodb://localhost:27017` và database tên `traffic`. Có hai tùy chọn:

- Cài MongoDB local: tải và cài đặt từ https://www.mongodb.com/try/download/community
- Hoặc dùng Docker để chạy MongoDB:

```powershell
# Nếu có Docker
docker run -d -p 27017:27017 --name mongo-local mongo:6
```

Nếu MongoDB không cần (chỉ để lưu), dịch vụ vẫn có thể chạy nhưng một số chức năng lưu kết quả sẽ bị vô hiệu (mã đã có try/except cho trường hợp Mongo không sẵn sàng).

Bạn cũng có thể thiết lập biến môi trường để thay đổi URI hoặc tên database:

- MONGODB_URI (ví dụ: mongodb://user:pass@host:27017)
- MONGODB_DB (mặc định: traffic)

### 3.4. Chạy backend (API + serve frontend)

1. Vào thư mục `backend` và chạy Uvicorn:

```powershell
cd backend
# chạy server (sử dụng host 0.0.0.0 nếu muốn truy cập từ máy khác)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

2. Mở trình duyệt và truy cập:

- Giao diện frontend: http://127.0.0.1:8000/  (FastAPI sẽ serve thư mục `frontend`)
- API health: http://127.0.0.1:8000/health

### 3.5. Sử dụng giao diện

- Tải ảnh/video lên bằng nút "Chọn file" và bấm "Tải" để gửi lên endpoint `/analyze/image` hoặc `/analyze/video`.
- Bật/tắt camera bằng nút "Bật/tắt camera" (ứng dụng sẽ gọi `/camera/start`, `/camera/stream` để stream và `/camera/analysis` để xem kết quả phân tích liên tục).
- Kết quả phân tích (danh sách phương tiện + biển số) sẽ hiển thị ở sidebar, và ảnh kết quả có thể được lưu trong `frontend/analysis_results` nếu backend cấu hình `detector.save_dir`.

### 3.6. Các lệnh API hữu ích

Tải ảnh (curl ví dụ):

```powershell
curl -X POST "http://127.0.0.1:8000/analyze/image" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path\to\image.jpg"
```

Tải video:

```powershell
curl -X POST "http://127.0.0.1:8000/analyze/video" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path\to\video.mp4"
```

Bật camera (server-side mở camera và stream):

```powershell
curl -X POST "http://127.0.0.1:8000/camera/start"
# xem stream tại http://127.0.0.1:8000/camera/stream
# xem phân tích ti http://127.0.0.1:8000/camera/analysis
```

Xóa database:

```powershell
curl -X POST "http://127.0.0.1:8000/clear"
```

---

## 4. Lưu ý vận hành và gợi ý

- Camera: Trên Windows, backend dùng `cv2.CAP_DSHOW` để mở camera laptop. Nếu không kết nối được, kiểm tra quyền truy cập Camera trong Windows Settings, ứng dụng khác đang dùng camera, hoặc driver camera.
- GPU: Nếu muốn dùng GPU để tăng tốc inference, cài thêm các driver và phiên bản torch/ultralytics phù hợp; hiện requirements không cố định torch — nếu cần GPU, hãy cài `torch`/`cuda` tương ứng trước khi cài `ultralytics`.
- Hiệu suất: Xử lý video có thể nặng; có thể điều chỉnh số khung (stride) và số khung tối đa (`VIDEO_STRIDE`, `VIDEO_MAX_FRAMES`) thông qua biến môi trường.
- Bảo mật: Mặc định API cho phép CORS từ mọi nguồn (allow_origins=["*"]). Nếu triển khai thật, điều chỉnh lại cài đặt CORS và xác thực.

---

## 5. Cấu trúc dự án (tóm tắt)

- backend/
  - app/
    - main.py         # entrypoint FastAPI
    - detector.py     # logic nhận diện
    - db.py           # MongoStore
    - schemas.py      # Pydantic models
    - models/         # model weights
  - requirements.txt
- frontend/
  - index.html
  - main.js
  - styles.css
  - analysis_results/  # nơi backend lưu ảnh phân tích (nếu bật)


---

Nếu bạn muốn, tôi có thể:
- Thêm phần ví dụ kết quả JSON từ API.
- Tạo script bắt đầu (PowerShell `start.ps1`) để tự động kích hoạt venv và khởi chạy server.
- Cập nhật README theo môi trường bạn muốn (Docker, Linux, hay deploy cloud).

Hoàn thành việc tạo README. Bước tiếp theo: đánh dấu todo hoàn tất.