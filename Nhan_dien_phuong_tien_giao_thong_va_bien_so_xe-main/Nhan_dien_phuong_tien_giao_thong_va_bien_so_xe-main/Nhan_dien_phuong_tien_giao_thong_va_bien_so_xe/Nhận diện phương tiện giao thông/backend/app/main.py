from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from typing import Generator
import os
import time

from app.detector import VehicleDetector
from app.db import MongoStore
from app.schemas import AnalyzeResponse


app = FastAPI(title="Traffic Detection API")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

detector = VehicleDetector()
store = MongoStore()
camera = None  # cv2.VideoCapture

# Simple IoU tracker to deduplicate objects across frames
class SimpleTracker:
	def __init__(self, iou_threshold=0.4, max_lost=5):
		self.next_id = 1
		self.tracks = {}  # id -> {bbox, lost}
		self.iou_threshold = iou_threshold
		self.max_lost = max_lost

	@staticmethod
	def iou(a, b):
		# a,b = (x1,y1,x2,y2)
		x1 = max(a[0], b[0])
		y1 = max(a[1], b[1])
		x2 = min(a[2], b[2])
		y2 = min(a[3], b[3])
		if x2 <= x1 or y2 <= y1:
			return 0.0
		inter = (x2 - x1) * (y2 - y1)
		area_a = (a[2]-a[0])*(a[3]-a[1])
		area_b = (b[2]-b[0])*(b[3]-b[1])
		return inter / float(area_a + area_b - inter)

	def update(self, detections):
		# detections: list of bboxes
		assigned = {}
		for tid, tr in list(self.tracks.items()):
			tr['lost'] += 1

		for bbox in detections:
			best_iou = 0
			best_id = None
			for tid, tr in self.tracks.items():
				i = self.iou(bbox, tr['bbox'])
				if i > best_iou:
					best_iou = i
					best_id = tid
			if best_iou >= self.iou_threshold and best_id is not None:
				# update track
				self.tracks[best_id]['bbox'] = bbox
				self.tracks[best_id]['lost'] = 0
				assigned[best_id] = bbox
			else:
				# new track
				tid = self.next_id
				self.next_id += 1
				self.tracks[tid] = {'bbox': bbox, 'lost': 0}
				assigned[tid] = bbox

		# remove lost tracks
		for tid in list(self.tracks.keys()):
			if self.tracks[tid]['lost'] > self.max_lost:
				del self.tracks[tid]

		return assigned  # mapping id->bbox

video_tracker = SimpleTracker(iou_threshold=0.4, max_lost=5)
camera_tracker = SimpleTracker(iou_threshold=0.4, max_lost=5)

# Prepare analysis results folder inside frontend so saved images are served
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
ANALYSIS_DIR = os.path.join(FRONTEND_DIR, "analysis_results")
try:
	os.makedirs(ANALYSIS_DIR, exist_ok=True)
	detector.save_dir = ANALYSIS_DIR
except Exception:
	detector.save_dir = None


@app.get("/health")
def health() -> dict:
	return {"status": "ok"}


@app.post("/analyze/image", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
	data = await file.read()
	arr = np.frombuffer(data, np.uint8)
	# Fast decode hint
	img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
	if img is None:
		return JSONResponse(status_code=400, content={"summary": "Không đọc được ảnh tải lên", "items": []})
	summary, items = detector.analyze_frame(img)
	try:
		print("Đang lưu kết quả vào database...")
		store.insert_items(items)
		print("Đã lưu thành công vào database")
	except Exception as e:
		print(f"Lỗi khi lưu vào database: {str(e)}")
		pass
	return {"summary": summary, "items": items}


@app.post("/analyze/video", response_model=AnalyzeResponse)
async def analyze_video(file: UploadFile = File(...)):
	# Lưu ra file tạm với đúng phần mở rộng, giúp OpenCV chọn backend phù hợp
	data = await file.read()
	import tempfile, os
	name = file.filename or "upload.mp4"
	_, ext = os.path.splitext(name)
	if not ext:
		ext = ".mp4"
	with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
		tmp.write(data)
		tmp_path = tmp.name

	items_all = []
	# Thử nhiều backend: FFMPEG trước, rồi ANY
	capture = cv2.VideoCapture(tmp_path, cv2.CAP_FFMPEG)
	if not capture.isOpened():
		capture = cv2.VideoCapture(tmp_path, cv2.CAP_ANY)

	if capture.isOpened():
		frame_index = 0
		# Tuneable via env for speed
		max_frames = int(os.getenv("VIDEO_MAX_FRAMES", "120"))
		stride = int(os.getenv("VIDEO_STRIDE", "5"))
		while capture.isOpened() and frame_index < max_frames:
			success, frame = capture.read()
			if not success:
				break
			if frame_index % stride == 0:
				_, items = detector.analyze_frame(frame)
				# Build list of bboxes for tracker
				bboxes = [it.get('bbox') for it in items if it.get('bbox')]
				# existing track ids before update
				existing_ids = set(video_tracker.tracks.keys())
				assigned = video_tracker.update(bboxes)
				# assigned: mapping tid->bbox
				# For each item, find its track id and only append if tid is new (not in existing_ids)
				for it in items:
					bbox = it.get('bbox')
					if not bbox:
						continue
					# find tid for this bbox by matching IoU
					found_tid = None
					for tid, tbbox in assigned.items():
						x1 = max(bbox[0], tbbox[0]); y1 = max(bbox[1], tbbox[1])
						x2 = min(bbox[2], tbbox[2]); y2 = min(bbox[3], tbbox[3])
						inter = max(0, x2-x1) * max(0, y2-y1)
						area_a = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
						area_b = (tbbox[2]-tbbox[0])*(tbbox[3]-tbbox[1])
						iou = inter / float(area_a + area_b - inter) if (area_a+area_b-inter)>0 else 0
						if iou >= 0.3:
							found_tid = tid
							break
					if found_tid is None:
						# couldn't match, skip
						continue
					if found_tid not in existing_ids:
						items_all.append(it)
			frame_index += 1
		capture.release()
	else:
		# Fallback: nếu không mở được dưới dạng video, thử như một ảnh đơn
		arr = np.frombuffer(data, np.uint8)
		img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
		if img is not None:
			_, items = detector.analyze_frame(img)
			items_all.extend(items)
		else:
			return JSONResponse(status_code=400, content={"summary": "Không đọc được video/ảnh tải lên", "items": []})

	try:
		os.remove(tmp_path)
	except Exception:
		pass

	# Tổng hợp
	counts = {"Ô tô": 0, "Xe máy": 0}
	for it in items_all:
		if it["vehicle_type"] in counts:
			counts[it["vehicle_type"]] += 1
	summary = f"Phát hiện có {len(items_all)} phương tiện: gồm {counts['Xe máy']} xe máy và {counts['Ô tô']} ô tô"
	try:
		store.insert_items(items_all)
	except Exception:
		pass
	return {"summary": summary, "items": items_all}


@app.post("/clear")
def clear():
	try:
		store.clear()
	except Exception:
		pass
	return {"status": "cleared"}


@app.post("/camera/start")
def camera_start():
    global camera
    if camera is None:
        try:
            print("Đang thử kết nối với camera laptop...")
            # Sử dụng DirectShow cho Windows với camera laptop
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if camera is not None and camera.isOpened():
                # Thử đọc frame để xác nhận quyền truy cập
                ret, frame = camera.read()
                if ret and frame is not None:
                    print("Đã kết nối thành công với camera laptop")
                    # Thiết lập thông số camera phù hợp với laptop
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Bật auto focus nếu có
                    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Bật auto exposure
                    return {"status": "started", "message": "Đã kết nối camera laptop thành công"}
                else:
                    print("Không thể đọc frame từ camera laptop")
                    camera.release()
                    camera = None
            else:
                print("Không thể mở camera laptop")
                
        except Exception as e:
            print(f"Lỗi khi kết nối camera laptop: {str(e)}")
            if camera:
                camera.release()
                camera = None
        
        # Nếu không thể kết nối camera
        error_message = ("Không thể kết nối với camera laptop. Vui lòng kiểm tra:\n"
                        "1. Camera laptop không bị tắt trong BIOS\n"
                        "2. Không có ứng dụng khác đang sử dụng camera (Zoom, Teams,...)\n"
                        "3. Đã cấp quyền truy cập camera trong Windows Settings:\n"
                        "   Settings > Privacy & Security > Camera\n"
                        "4. Driver camera đã được cài đặt đúng (kiểm tra Device Manager)\n"
                        "5. Thử khởi động lại máy tính")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": error_message
            }
        )
    
    return {"status": "started", "message": "Camera đã được kết nối từ trước"}


@app.post("/camera/stop")
def camera_stop():
	global camera, accumulated_items
	if camera is not None:
		camera.release()
		camera = None
	accumulated_items = []  # Reset danh sách tích lũy
	return {"status": "stopped"}


last_analysis_time = 0
analysis_results = {"summary": "", "items": []}
accumulated_items = []  # Lưu tất cả các kết quả phân tích từ camera

def _camera_frames() -> Generator[bytes, None, None]:
	global last_analysis_time, analysis_results, accumulated_items
	while True:
		if camera is None:
			break
		success, frame = camera.read()
		if not success:
			break
			
		# Phân tích mỗi 5 giây
		current_time = time.time()
		if current_time - last_analysis_time >= 5:  # Đổi lại thành 5 giây
			print("Bắt đầu phân tích frame từ camera...")
			summary, items = detector.analyze_frame(frame)
			last_analysis_time = current_time
			
			if items:  # Chỉ xử lý khi có phát hiện
				# Thêm thời gian phân tích
				for item in items:
					item["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
				
				# Cập nhật danh sách tích lũy with deduplication
				for it in items:
					key = (it.get('plate') or '') + '|' + (it.get('thumbnail') or '')[:128]
					existing_keys = { (x.get('plate') or '') + '|' + (x.get('thumbnail') or '')[:128] for x in accumulated_items }
					if key not in existing_keys:
						accumulated_items.append(it)
				counts = {"Ô tô": 0, "Xe máy": 0}
				for it in accumulated_items:
					if it["vehicle_type"] in counts:
						counts[it["vehicle_type"]] += 1
				summary = f"Phát hiện có {len(accumulated_items)} phương tiện: gồm {counts['Xe máy']} xe máy và {counts['Ô tô']} ô tô"
				
				# Cập nhật kết quả phân tích với toàn bộ items đã tích lũy
				analysis_results = {
					"summary": summary,
					"items": accumulated_items
				}
				
				try:
					print("Đang lưu kết quả vào database...")
					store.insert_items(items)  # Chỉ lưu items mới
					print("Đã lưu thành công vào database")
				except Exception as e:
					print(f"Lỗi khi lưu vào database: {str(e)}")
				
		_, jpg = cv2.imencode('.jpg', frame)
		yield (b"--frame\r\n"
			b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


@app.get("/camera/stream")
def camera_stream():
	return StreamingResponse(_camera_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/camera/analysis")
def get_camera_analysis():
	return analysis_results


# Serve frontend at root so http://127.0.0.1:8000 opens UI
if os.path.isdir(FRONTEND_DIR):
	app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# Also make sure analysis_results is accessible (it's inside frontend already)


