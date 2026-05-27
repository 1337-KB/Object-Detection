# Real-Time Object Detection System

A YOLOv8-powered object detection system that identifies 80 object classes in real time. Comes with two modes: a browser-based Streamlit app for images, webcam snapshots, and video files — and a dedicated OpenCV script for true 30+ FPS live webcam inference.

---

## Features

- **80 COCO object classes** — people, vehicles, animals, everyday objects and more
- **Three model sizes** — Nano (fastest), Small (balanced), Medium (most accurate)
- **Streamlit UI** — upload images, take webcam snapshots, or process video files in the browser
- **Real-time webcam mode** — OpenCV window with live bounding boxes, FPS counter, and detection summary
- **Live confidence control** — raise or lower the detection threshold on the fly without restarting
- **Per-class color coding** — each class gets a deterministic color so the display is consistent across frames
- **Screenshot saving** — capture annotated frames with a single keypress
- **Dockerized** — runs anywhere with a single command

---

## Tech Stack

| Layer | Technology |
|---|---|
| Detection model | YOLOv8 (Ultralytics) pretrained on COCO |
| Inference | PyTorch (CPU; GPU optional) |
| Browser UI | Streamlit |
| Real-time inference | OpenCV |
| Image processing | Pillow, NumPy |
| Deployment | Docker, Docker Compose |

---

## Project Structure

```
object-detection/
├── app/
│   ├── config.py        # Model options, 80 COCO class names, default thresholds
│   ├── detector.py      # YOLOv8 model loader (cached) and detect() function
│   └── utils.py         # Bounding box drawing, FPS counter, color-per-class, summarize
├── streamlit_app.py     # Browser UI — image, webcam snapshot, and video tabs
├── run_webcam.py        # OpenCV real-time webcam script (30+ FPS)
├── tests/
│   └── test_detector.py # 16 unit tests
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Getting Started

### Option 1 — Run Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Launch the Streamlit app**
```bash
streamlit run streamlit_app.py
```
```
http://localhost:8501
```
The YOLOv8 Nano model (~6MB) downloads automatically on first use and is cached locally.

**3. Or launch the real-time webcam script**
```bash
python run_webcam.py
```

---

### Option 2 — Run with Docker

```bash
docker compose up --build
```
```
http://localhost:8501
```
The model is pre-downloaded during the Docker build so the container starts immediately with no cold-start delay.

---

## Streamlit UI — Three Tabs

### 📁 Image Upload
Upload any JPG, PNG, BMP, or WebP image. The annotated result is shown alongside a per-class detection count and inference time in milliseconds.

### 📷 Webcam
Uses your browser's camera to take a snapshot, which is then run through the detector. For continuous real-time inference, use `run_webcam.py` instead (see below).

### 🎬 Video
Upload an MP4, AVI, MOV, or MKV file. A slider controls how many frames to skip — process every frame for full coverage or every 5th frame for a fast preview. Annotated frames play back live as they're processed.

**Sidebar settings (apply to all tabs):**

| Setting | Description |
|---|---|
| Model size | Nano / Small / Medium — trade speed for accuracy |
| Confidence threshold | Only show detections above this score (default 0.50) |
| NMS IoU threshold | Controls overlap tolerance between boxes (default 0.45) |
| Show confidence scores | Toggle confidence percentages on bounding box labels |

---

## Real-Time Webcam Script

For true 30+ FPS inference, run the dedicated OpenCV script:

```bash
python run_webcam.py
```

**Options:**
```bash
python run_webcam.py --model yolov8s.pt   # use a larger model
python run_webcam.py --conf 0.4           # lower confidence threshold
python run_webcam.py --camera 1           # use a second camera
python run_webcam.py --width 1920 --height 1080  # set resolution
```

**Controls:**

| Key | Action |
|---|---|
| `Q` | Quit |
| `S` | Save annotated screenshot to `screenshots/` |
| `+` or `=` | Raise confidence threshold by 0.05 |
| `-` | Lower confidence threshold by 0.05 |

The current FPS and confidence threshold are displayed live on screen. The detection summary (class name + count) appears in the top-right corner.

---

## Model Sizes

| Model | File | Speed | Accuracy | Best for |
|---|---|---|---|---|
| YOLOv8 Nano | `yolov8n.pt` | Fastest | Good | Real-time on CPU |
| YOLOv8 Small | `yolov8s.pt` | Fast | Better | Balanced CPU use |
| YOLOv8 Medium | `yolov8m.pt` | Moderate | Best of three | GPU or high-end CPU |

All models detect the same 80 COCO classes. Models are downloaded automatically from Ultralytics on first use.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected output: **16 passed**

Tests cover:
- COCO class list (80 classes, known classes present)
- Color generation (RGB tuple, deterministic, unique per class)
- Detection summarisation (counts, ordering, empty input)
- FPS counter (zero on first tick, positive after multiple ticks)
- Model loading
- Detect output structure (class, confidence, box fields)
- Confidence filtering (high threshold ≤ low threshold detections)
- Confidence value range (always 0.0–1.0)

---

## Notes

- **First run** downloads the selected YOLOv8 model weights from Ultralytics and caches them in `~/.cache/`. Subsequent runs load from cache instantly.
- **AMD GPU users** — PyTorch's ROCm backend (required for AMD GPU acceleration) is only officially supported on Linux. On Windows the system runs on CPU, which comfortably handles real-time inference with the Nano model.
- **Multiple models in memory** — switching between model sizes in the Streamlit sidebar caches up to 3 models simultaneously. Switching back is instant; no re-download or reload needed.
- **Video processing speed** — processing every frame of a 30 FPS video at CPU inference speed will be slower than real time. Use the frame-skip slider (every 3rd–5th frame) for a smooth preview experience.
