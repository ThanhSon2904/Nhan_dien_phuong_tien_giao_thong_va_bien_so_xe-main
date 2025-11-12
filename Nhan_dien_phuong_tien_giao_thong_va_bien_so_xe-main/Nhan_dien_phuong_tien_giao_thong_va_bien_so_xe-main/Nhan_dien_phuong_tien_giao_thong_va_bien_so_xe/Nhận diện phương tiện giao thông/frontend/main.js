const apiBase = "http://127.0.0.1:8000";

const filePicker = document.getElementById("filePicker");
const uploadBtn = document.getElementById("uploadBtn");
const clearBtn = document.getElementById("clearBtn");
const cameraBtn = document.getElementById("cameraBtn");
const videoPlayer = document.getElementById("videoPlayer");
const imageViewer = document.getElementById("imageViewer");
const cameraViewer = document.getElementById("cameraViewer");
const playPlaceholder = document.getElementById("playPlaceholder");
const detectionsEl = document.getElementById("detections");
const summaryEl = document.getElementById("summary");
const progressEl = document.getElementById("progress");

let pickedFile = null;
let cameraOn = false;

function resetMediaView(){
	videoPlayer.hidden = true;
	imageViewer.hidden = true;
	cameraViewer.hidden = true;
	playPlaceholder.hidden = false;
	progressEl.style.width = "0%";
	progressEl.textContent = "";
	progressEl.style.backgroundColor = "";
}

filePicker.addEventListener("change", () => {
	pickedFile = filePicker.files[0] ?? null;
	uploadBtn.disabled = !pickedFile;
	clearBtn.disabled = !pickedFile;
});

uploadBtn.addEventListener("click", async () => {
	if(!pickedFile) return;
	const form = new FormData();
	form.append("file", pickedFile);
	const isVideo = pickedFile.type.startsWith("video/");
	const url = `${apiBase}/analyze/${isVideo ? "video" : "image"}`;
	try{
		progressEl.style.width = "25%";
		progressEl.textContent = "Đang tải lên...";
		progressEl.style.backgroundColor = "";
		
		const res = await fetch(url, { method: "POST", body: form });
		if(!res.ok){
			const txt = await res.text();
			throw new Error(`Upload failed: ${res.status} ${txt}`);
		}

		progressEl.style.width = "50%";
		progressEl.textContent = "Đang phân tích...";

		const data = await res.json();
		
		// Update progress based on analysis result
		if (data.items && data.items.length > 0) {
			progressEl.style.width = "100%";
			progressEl.textContent = "Phân tích thành công!";
			progressEl.style.backgroundColor = "#4CAF50";
		} else {
			progressEl.style.width = "100%";
			progressEl.textContent = "Không phát hiện được đối tượng";
			progressEl.style.backgroundColor = "#FFA500";
		}
		
		updateUIAfterAnalysis(data, isVideo);
	}catch(err){
		console.error(err);
		progressEl.style.width = "100%";
		progressEl.textContent = "Phân tích thất bại!";
		progressEl.style.backgroundColor = "#f44336";
		alert("Tải và phân tích thất bại. Vui lòng thử ảnh/video khác hoặc kiểm tra server.");
	}
});

clearBtn.addEventListener("click", async () => {
	await fetch(`${apiBase}/clear`, { method: "POST" });
	resetMediaView();
	detectionsEl.innerHTML = "";
	summaryEl.textContent = "—";
	filePicker.value = "";
	pickedFile = null;
	uploadBtn.disabled = true;
	clearBtn.disabled = true;
	
	// Reset camera state
	if (cameraViewer.src) {
		cameraViewer.src = '';
	}
	cameraViewer.style.display = 'none';
});

let analysisInterval = null;

cameraBtn.addEventListener("click", async () => {
	cameraOn = !cameraOn;
	try{
		if(cameraOn){
			const r = await fetch(`${apiBase}/camera/start`, { method: "POST" });
			if(!r.ok) throw new Error("Camera start failed");
			// Đảm bảo camera viewer được hiển thị và thiết lập đúng
			cameraViewer.src = `${apiBase}/camera/stream`;
			cameraViewer.hidden = false;
			cameraViewer.style.display = 'block';  // Đảm bảo hiển thị
			cameraViewer.style.width = '100%';     // Đặt kích thước phù hợp
			cameraViewer.style.height = 'auto';    // Tự động điều chỉnh chiều cao
			
			// Ẩn các phần tử khác
			videoPlayer.hidden = true;
			videoPlayer.style.display = 'none';
			imageViewer.hidden = true;
			imageViewer.style.display = 'none';
			playPlaceholder.hidden = true;
			
			// Cập nhật kết quả phân tích mỗi giây
			progressEl.style.backgroundColor = "";
			progressEl.style.width = "25%";
			progressEl.textContent = "Đang khởi động camera...";
			
			analysisInterval = setInterval(async () => {
				try {
					const res = await fetch(`${apiBase}/camera/analysis`);
					const data = await res.json();
					if (data.items && data.items.length > 0) {
						progressEl.style.width = "100%";
						progressEl.textContent = "Đang phân tích camera (Cập nhật mỗi 5 giây)";
						progressEl.style.backgroundColor = "#4CAF50";
						updateUIAfterAnalysis(data, false);
					}
				} catch (error) {
					console.error("Lỗi khi lấy kết quả phân tích:", error);
				}
			}, 1000);
		} else {
			await fetch(`${apiBase}/camera/stop`, { method: "POST" });
			if (analysisInterval) {
				clearInterval(analysisInterval);
				analysisInterval = null;
			}
			resetMediaView();
		}
	}catch(err){
		console.error(err);
		if (analysisInterval) {
			clearInterval(analysisInterval);
			analysisInterval = null;
		}
		alert("Không bật được camera. Kiểm tra quyền truy cập hoặc thử lại.");
		cameraOn = false;
	}
});

function updateUIAfterAnalysis(data, isVideo){
	// display media
	if (cameraOn) {
		// Don't change media display for camera mode
	} else if(isVideo) {
		videoPlayer.src = URL.createObjectURL(pickedFile);
		videoPlayer.hidden = false;
		imageViewer.hidden = true;
		playPlaceholder.hidden = true;
		try { videoPlayer.load(); videoPlayer.play(); } catch (e) {}
	} else {
		imageViewer.src = URL.createObjectURL(pickedFile);
		imageViewer.hidden = false;
		videoPlayer.hidden = true;
		playPlaceholder.hidden = true;
	}

	// Ensure camera viewer is hidden/cleared when not in camera mode
	if (!cameraOn) {
		cameraViewer.hidden = true;
		try { cameraViewer.src = ''; } catch(e) {}
		if (cameraViewer.style) {
			cameraViewer.style.display = 'none';
		}
	}
	// list
	if (!cameraOn) {
		detectionsEl.innerHTML = ""; // Xóa danh sách cũ nếu không phải camera mode
	}
	// Reverse items array to show newest first
	const items = [...(data.items || [])].reverse();
	items.forEach((it) => {
		const li = document.createElement("li");
		li.className = "item";
	const img = document.createElement("img");
	img.width = 140;
	img.height = 90;
	img.style.objectFit = 'cover';
	img.style.borderRadius = '8px';
	img.alt = it.plate || 'thumbnail';
		// Prefer saved file path (served under /analysis_results) if available
		if (it.file_path) {
			img.src = it.file_path;
			// If loading the saved file fails (missing file or 404), fallback to base64 thumbnail
			img.onerror = () => {
				if (it.thumbnail) {
					img.onerror = null;
					img.src = `data:image/jpeg;base64,${it.thumbnail}`;
				} else {
					// show subtle placeholder (transparent gif data URI)
					img.onerror = null;
					img.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==';
				}
			};
		} else if (it.thumbnail) {
			img.src = `data:image/jpeg;base64,${it.thumbnail}`;
		} else {
			img.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==';
		}
		const info = document.createElement("div");
		info.innerHTML = `<div>${it.date}</div>
						 <div>xe: ${it.vehicle_type}</div>
						 <div class="plate">biển số: ${it.plate || "?"}</div>
						 <div>độ chính xác: ${it.confidence || "N/A"}</div>`;
		li.appendChild(img);
		li.appendChild(info);
		detectionsEl.appendChild(li);
	});
	// summary
	summaryEl.textContent = data.summary || "—";
}


