import base64
import io
from datetime import datetime
from typing import List, Tuple
import os

import cv2
import numpy as np
from PIL import Image

try:
	from ultralytics import YOLO
except Exception:  # pragma: no cover
	YOLO = None  # type: ignore

try:
    import torch
except Exception:
    torch = None

try:
	import easyocr  # type: ignore
except Exception:  # pragma: no cover
	easyocr = None  # type: ignore


class VehicleDetector:
	def __init__(self) -> None:
		# Initialize models
		self.vehicle_model = None
		self.plate_detector = None
		self.plate_ocr = None

		if YOLO:
			# Initialize vehicle detection model
			pt_path = os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt")
			pt_path = os.path.abspath(pt_path)
			try:
				self.vehicle_model = YOLO(pt_path if os.path.exists(pt_path) else "yolov8n.pt")
				print("Vehicle detection model loaded successfully")
			except Exception as e:
				print(f"Warning: Failed to load vehicle detection model: {e}")
				self.vehicle_model = None

			# Initialize license plate detector
			# Try multiple likely locations because the repo layout may vary (nested folders or app/models)
			lp_candidates = [
				os.path.join(os.path.dirname(__file__), "..", "License-Plate-Recognition-main", "model", "LP_detector.pt"),
				os.path.join(os.path.dirname(__file__), "..", "License-Plate-Recognition-main", "License-Plate-Recognition-main", "model", "LP_detector.pt"),
				os.path.join(os.path.dirname(__file__), "models", "LP_detector.pt"),
				os.path.join(os.path.dirname(__file__), "..", "models", "LP_detector.pt"),
			]
			lp_detector_path = None
			for p in lp_candidates:
				ap = os.path.abspath(p)
				if os.path.exists(ap):
					lp_detector_path = ap
					break
			if lp_detector_path:
				print(f"Loading license plate detector from {lp_detector_path}")
				try:
					try:
						self.plate_detector = YOLO(lp_detector_path)
					except Exception as e:
						# If model is a YOLOv5 checkpoint, ultralytics YOLOv8 may refuse to load it.
						msg = str(e)
						if torch and ("yolov5" in msg.lower() or "originally trained" in msg.lower() or "v5" in msg.lower()):
							print("Detected YOLOv5 checkpoint; attempting to load via torch.hub (yolov5) as fallback")
							try:
								# load custom yolov5 via torch.hub
								v5 = torch.hub.load('ultralytics/yolov5', 'custom', path=lp_detector_path, force_reload=False)
								# create adapter to mimic ultralytics result interface
								class YOLOv5Adapter:
									def __init__(self, model):
										self.model = model
										self.names = model.names if hasattr(model, 'names') else {}
									def predict(self, img, verbose=False, **kwargs):
										# yolov5 model returns a results object when called
										res = self.model(img)
										# Build a simple namespace with boxes and names compatible with ultralytics v8 usage
										class Box:
											def __init__(self, xyxy, conf, cls_id):
												# store values in similar shape to ultralytics boxes
												self.xyxy = [np.array(xyxy)]
												self.conf = np.array([conf])
												self.cls = np.array([cls_id])
										class Results:
											def __init__(self, boxes, names):
												self.boxes = boxes
												self.names = names
										boxes = []
										# res.xyxy is list of tensors per image (or res.pred for older)
										xy = None
										if hasattr(res, 'xyxy') and len(res.xyxy) > 0:
											xy = res.xyxy[0].cpu().numpy()
										elif hasattr(res, 'pred') and len(res.pred) > 0:
											xy = res.pred[0].cpu().numpy()
										if xy is not None:
											for row in xy:
												x1, y1, x2, y2, conf, cls_id = row
												boxes.append(Box([x1, y1, x2, y2], float(conf), int(cls_id)))
										# Return a list-like object so caller can do [0]
										return [Results(boxes, self.names)]
								self.plate_detector = YOLOv5Adapter(v5)
							except Exception as e2:
								print(f"Warning: torch.hub yolov5 load failed: {e2}")
								self.plate_detector = None
						else:
							print(f"Warning: Failed to load license plate detector: {e}")
							self.plate_detector = None
					else:
						print("License plate detector loaded successfully")
				except Exception as e:
					print(f"Warning: Failed to load license plate detector: {e}")
					self.plate_detector = None
			else:
				print(f"Warning: License plate detector not found. Tried: {', '.join(lp_candidates)}")

			# Initialize license plate OCR model
			ocr_candidates = [
				os.path.join(os.path.dirname(__file__), "..", "License-Plate-Recognition-main", "model", "LP_ocr.pt"),
				os.path.join(os.path.dirname(__file__), "..", "License-Plate-Recognition-main", "License-Plate-Recognition-main", "model", "LP_ocr.pt"),
				os.path.join(os.path.dirname(__file__), "models", "LP_ocr.pt"),
				os.path.join(os.path.dirname(__file__), "..", "models", "LP_ocr.pt"),
			]
			lp_ocr_path = None
			for p in ocr_candidates:
				ap = os.path.abspath(p)
				if os.path.exists(ap):
					lp_ocr_path = ap
					break
			if lp_ocr_path:
				print(f"Loading license plate OCR from {lp_ocr_path}")
				try:
					self.plate_ocr = YOLO(lp_ocr_path)
					print("License plate OCR loaded successfully")
				except Exception as e:
					print(f"Warning: Failed to load license plate OCR: {e}")
					self.plate_ocr = None
			else:
				print(f"Warning: License plate OCR not found. Tried: {', '.join(ocr_candidates)}")

		# Directory to save analyzed crops (can be None)
		self.save_dir = None

		# Default predict settings for speed/quality balance
		default_imgsz = 640
		# Default confidence (slightly lower to improve detection rate on smaller plates)
		self.predict_kwargs = {
			"conf": float(os.getenv("YOLO_CONF", "0.40")),
			"iou": float(os.getenv("YOLO_IOU", "0.5")),
			"imgsz": int(os.getenv("YOLO_IMGSZ", str(default_imgsz))),
			"max_det": int(os.getenv("YOLO_MAXDET", "20")),
		}

		# Post-filtering thresholds to reduce false positives (fractions of frame area / aspect ratio limits)
		# Minimum box area relative to image area (default 0.001 = 0.1%)
		self.min_box_area_frac = float(os.getenv("YOLO_MIN_BOX_AREA", "0.0008"))
		# Aspect ratio (w/h) limits - relaxed to allow motorcycles and various vehicles
		self.min_aspect = float(os.getenv("YOLO_ASPECT_MIN", "0.15"))
		self.max_aspect = float(os.getenv("YOLO_ASPECT_MAX", "8.0"))
		
		# Fallback to EasyOCR if plate OCR model fails
		self.reader = easyocr.Reader(["en", "vi"]) if easyocr else None

	def _to_base64_jpeg(self, image_bgr: np.ndarray) -> str:
		image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		pil = Image.fromarray(image_rgb)
		buf = io.BytesIO()
		pil.save(buf, format="JPEG", quality=85)
		return base64.b64encode(buf.getvalue()).decode("ascii")

	def _save_image(self, image_bgr: np.ndarray, prefix: str = "crop") -> str | None:
		"""Save image to self.save_dir and return relative URL path (under /analysis_results)."""
		if not self.save_dir:
			return None
		try:
			# Ensure dir exists
			os.makedirs(self.save_dir, exist_ok=True)
			ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
			fname = f"{prefix}_{ts}.jpg"
			abs_path = os.path.join(self.save_dir, fname)
			# Write as JPEG
			cv2.imwrite(abs_path, image_bgr)
			# Return web-accessible path relative to frontend root
			return f"/analysis_results/{fname}"
		except Exception:
			return None

	def detect(self, frame_bgr: np.ndarray) -> List[Tuple[str, str, np.ndarray, tuple]]:
		"""Return list of (vehicle_type, plate_text, crop_bgr, bbox).
		bbox is (x1, y1, x2, y2) in original image coordinates.
		"""
		items: List[Tuple[str, str, np.ndarray]] = []
		if self.vehicle_model is None:
			return items
		# Let the model handle resizing internally — pass the original frame
		results = self.vehicle_model.predict(frame_bgr, verbose=False, **self.predict_kwargs)[0]
		# original frame dimensions
		h, w = frame_bgr.shape[:2]
		for box in results.boxes:
			# box.xyxy[0] may be a tensor/array-like
			x1, y1, x2, y2 = map(int, map(float, box.xyxy[0].tolist()))
			cls_id = int(box.cls.item()) if hasattr(box, 'cls') else int(getattr(box, 'class', 0))
			name = results.names.get(cls_id, "vehicle")
			# Only keep vehicle classes of interest
			if name not in ("car", "motorcycle", "bus", "truck", "bicycle"):
				continue
			# Size and aspect ratio filtering to reduce false positives
			bw = max(1, x2 - x1)
			bh = max(1, y2 - y1)
			box_area = bw * bh
			frame_area = w * h
			if box_area < self.min_box_area_frac * frame_area:
				# too small
				continue
			aspect = bw / bh
			if aspect < self.min_aspect or aspect > self.max_aspect:
				continue
			# Crop from original frame using mapped coordinates
			crop = frame_bgr[y1:y2, x1:x2]
			vtype = "Ô tô" if name in ("car", "bus", "truck") else "Xe máy"
			plate_text = self._read_plate(crop)
			items.append((vtype, plate_text, crop, (x1, y1, x2, y2)))
		return items

	def _read_plate(self, vehicle_crop_bgr: np.ndarray) -> str:
		"""Detect and recognize license plate using trained models."""
		if self.plate_detector is None:
			return self._read_plate_easyocr(vehicle_crop_bgr)

		# Detect license plate in vehicle crop
		results = self.plate_detector.predict(vehicle_crop_bgr, verbose=False, **self.predict_kwargs)[0]
		
		# If no plate detected, try EasyOCR on full image
		if len(results.boxes) == 0:
			return self._read_plate_easyocr(vehicle_crop_bgr)

		# Get the plate with highest confidence
		best_box = None
		best_conf = -1
		for box in results.boxes:
			conf = float(box.conf.item())
			if conf > best_conf:
				best_conf = conf
				best_box = box

		if best_box is None:
			return self._read_plate_easyocr(vehicle_crop_bgr)

		# Extract plate region
		x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
		plate_crop = vehicle_crop_bgr[max(0, y1):y2, max(0, x1):x2]

		# Use OCR model if available
		if self.plate_ocr is not None:
			try:
				ocr_results = self.plate_ocr.predict(plate_crop, verbose=False, **self.predict_kwargs)[0]
				if len(ocr_results.boxes) > 0:
					# Combine detected characters into plate text
					chars = []
					for box in sorted(ocr_results.boxes, key=lambda b: b.xyxy[0][0]):  # Sort by x coordinate
						cls_id = int(box.cls.item())
						char = ocr_results.names.get(cls_id, "")
						chars.append(char)
					return "".join(chars)
			except Exception:
				pass

		# Fallback to EasyOCR for this plate crop
		return self._read_plate_easyocr(plate_crop)

	def _read_plate_easyocr(self, image_bgr: np.ndarray) -> str:
		"""Fallback plate reading using EasyOCR with preprocessing."""
		if self.reader is None:
			return ""

		# Downscale slightly for speed
		h, w = image_bgr.shape[:2]
		scale = 0.8 if max(h, w) > 800 else 1.0
		crop = cv2.resize(image_bgr, (int(w*scale), int(h*scale))) if scale != 1.0 else image_bgr
		gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
		# Enhance contrast
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		gray_eq = clahe.apply(gray)
		# Different binarizations
		thr1 = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)
		thr2 = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
		kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		mor = cv2.morphologyEx(thr2, cv2.MORPH_CLOSE, kern, iterations=1)

		candidates = [crop, cv2.cvtColor(thr1, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mor, cv2.COLOR_GRAY2BGR)]
		texts: List[str] = []
		for img in candidates:
			res = self.reader.readtext(img, detail=0, paragraph=True)
			for t in res:
				if isinstance(t, str):
					texts.append(t)

		# Clean and choose by regex pattern similar to VN plates
		import re
		# Biển VN thường: 2 số + chữ (1-2) + nhóm 4-5 số. Hỗ trợ dạng 2 hàng (tách bằng newline)
		pattern = re.compile(r"\b\d{2}[A-Z]{1,2}-?[A-Z0-9]{1,2}\s*\d{3,5}\b")
		best = ""
		for t in texts:
			clean = t.upper().replace(" ", "").replace("|", "1")
			m = pattern.search(clean)
			if m:
				best = m.group(0)
				break
		# Fallback: return the longest token
		if not best and texts:
			best = max(texts, key=len)[:12]
		return best

	def analyze_frame(self, frame_bgr: np.ndarray):
		print("Bắt đầu phân tích khung hình...")
		items = []
		print("Đang phát hiện phương tiện... (25%)")
		dets = self.detect(frame_bgr)
		# Fallback: if no vehicle found, try OCR directly on the full frame (useful for close-up plate photos)
		if not dets:
			print("Không phát hiện phương tiện, thử phân tích biển số trực tiếp... (50%)")
			plate = self._read_plate(frame_bgr)
			if plate:
				print(f"Đã phát hiện biển số: {plate} (75%)")
				file_path = self._save_image(frame_bgr, prefix="frame")
				items.append({
					"date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
					"vehicle_type": "Không rõ",
					"plate": plate,
					"thumbnail": self._to_base64_jpeg(frame_bgr),
					"file_path": file_path,
					"confidence": "100%"
				})
			else:
				print("Không phát hiện được biển số")
		else:
			print(f"Đã phát hiện {len(dets)} phương tiện (50%)")
			for i, (vtype, plate, crop, bbox) in enumerate(dets):
				progress = 50 + (i + 1) * 50 // len(dets)
				print(f"Đang xử lý phương tiện {i+1}/{len(dets)}... ({progress}%)")
				# Save crop if save_dir configured
				file_path = self._save_image(crop, prefix=f"crop_{i+1}")
				items.append({
					"date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
					"vehicle_type": vtype,
					"plate": plate,
					"thumbnail": self._to_base64_jpeg(crop),
					"bbox": bbox,
					"file_path": file_path,
					"confidence": "100%"
				})
		counts = {"Ô tô": 0, "Xe máy": 0, "Khác": 0}
		for it in items:
			vt = it.get("vehicle_type", "Khác")
			if vt not in counts:
				counts["Khác"] += 1
			else:
				counts[vt] += 1
		summary = f"Phát hiện có {len(items)} phương tiện: gồm {counts['Xe máy']} xe máy và {counts['Ô tô']} ô tô"
		return summary, items


