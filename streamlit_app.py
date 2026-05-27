import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from app.config import DEFAULT_CONFIDENCE, DEFAULT_IOU, MODEL_OPTIONS
from app.detector import detect, load_model
from app.utils import class_color, draw_detections, summarize_detections

st.set_page_config(
    page_title="Object Detection",
    page_icon="🎯",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

model_label = st.sidebar.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
model_path = MODEL_OPTIONS[model_label]

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05, max_value=1.0,
    value=DEFAULT_CONFIDENCE, step=0.05,
    help="Only show detections above this score",
)
iou = st.sidebar.slider(
    "NMS IoU threshold",
    min_value=0.1, max_value=1.0,
    value=DEFAULT_IOU, step=0.05,
    help="Higher = keep more overlapping boxes",
)
show_confidence = st.sidebar.toggle("Show confidence scores", value=True)

st.sidebar.divider()
st.sidebar.caption("Model is cached after first load. Switching models re-downloads once.")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def get_model(path: str):
    return load_model(path)

model = get_model(model_path)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 Real-Time Object Detection")
st.caption("YOLOv8 · 80 COCO classes · Adjust settings in the sidebar")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_image, tab_webcam, tab_video = st.tabs(["📁 Image Upload", "📷 Webcam", "🎬 Video"])


# ── Helper: run detection + display results ───────────────────────────────────
def run_and_display(bgr_frame: np.ndarray, container) -> list[dict]:
    t0 = time.perf_counter()
    detections = detect(model, bgr_frame, confidence, iou)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    annotated = draw_detections(bgr_frame, detections, show_confidence)
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    col_img, col_stats = container.columns([3, 1])
    col_img.image(rgb, use_container_width=True)

    summary = summarize_detections(detections)
    col_stats.metric("Objects found", sum(summary.values()))
    col_stats.metric("Inference time", f"{elapsed_ms:.1f} ms")

    if summary:
        col_stats.divider()
        col_stats.markdown("**Detections**")
        for cls, count in summary.items():
            color = class_color(cls)
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            col_stats.markdown(
                f"<span style='color:{hex_color}'>■</span> **{cls}** × {count}",
                unsafe_allow_html=True,
            )
    else:
        col_stats.info("No objects detected above threshold.")

    return detections


# ── Tab 1: Image Upload ───────────────────────────────────────────────────────
with tab_image:
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        run_and_display(bgr, st)
    else:
        st.info("Upload an image to run detection.")


# ── Tab 2: Webcam Snapshot ────────────────────────────────────────────────────
with tab_webcam:
    st.markdown(
        "Each snapshot is run through the detector. "
        "For true real-time 30+ FPS inference, use the **`run_webcam.py`** script instead."
    )

    cam_image = st.camera_input("Take a snapshot", label_visibility="collapsed")

    if cam_image:
        pil_img = Image.open(cam_image).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        run_and_display(bgr, st)

    st.divider()
    st.markdown("#### 🚀 Real-time mode (terminal)")
    st.code("python run_webcam.py", language="bash")
    st.caption("Opens an OpenCV window — press **Q** to quit, **S** to save a screenshot.")


# ── Tab 3: Video File ─────────────────────────────────────────────────────────
with tab_video:
    video_file = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

    if video_file:
        import tempfile, os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25

        st.info(f"Video: {total_frames} frames @ {fps_src:.0f} FPS")

        frame_step = st.slider(
            "Process every N-th frame (speed vs. coverage)",
            1, 10, 3,
            help="1 = every frame (slow), 10 = every 10th frame (fast preview)",
        )

        if st.button("▶ Run Detection"):
            placeholder = st.empty()
            progress = st.progress(0)
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % frame_step != 0:
                    continue

                detections = detect(model, frame, confidence, iou)
                annotated = draw_detections(frame, detections, show_confidence)
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                placeholder.image(rgb, use_container_width=True)
                progress.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            os.unlink(tmp_path)
            st.success("Done!")
    else:
        st.info("Upload a video file to run detection frame by frame.")
