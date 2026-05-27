"""
Real-time webcam inference script — true 30+ FPS mode.

Usage:
    python run_webcam.py [--model yolov8n.pt] [--conf 0.5] [--camera 0]

Controls:
    Q  — quit
    S  — save screenshot to screenshots/
    +  — increase confidence threshold
    -  — decrease confidence threshold
"""

import argparse
import os
import sys
import time

import cv2

from app.detector import detect, load_model
from app.utils import FPSCounter, draw_detections, draw_fps, summarize_detections


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time YOLO webcam inference")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model file")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    return parser.parse_args()


def main():
    args = parse_args()
    conf = args.conf

    print(f"Loading model: {args.model}")
    model = load_model(args.model)
    print("Model loaded. Opening camera…")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce latency

    os.makedirs("screenshots", exist_ok=True)
    fps_counter = FPSCounter(window=30)
    screenshot_count = 0

    print("\nControls: Q = quit | S = save screenshot | + = raise conf | - = lower conf\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        detections = detect(model, frame, confidence=conf, iou=args.iou)
        fps = fps_counter.tick()

        annotated = draw_detections(frame, detections, show_confidence=True)
        annotated = draw_fps(annotated, fps)

        # Overlay: confidence threshold
        cv2.putText(
            annotated, f"Conf: {conf:.2f}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 220, 255), 2, cv2.LINE_AA,
        )

        # Overlay: detection summary (top-right)
        summary = summarize_detections(detections)
        y_offset = 30
        for cls, count in list(summary.items())[:6]:
            text = f"{cls}: {count}"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            x_pos = annotated.shape[1] - tw - 15
            cv2.putText(
                annotated, text,
                (x_pos, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            y_offset += 28

        cv2.imshow("Object Detection — Q to quit", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            screenshot_count += 1
            path = f"screenshots/capture_{screenshot_count:04d}.jpg"
            cv2.imwrite(path, annotated)
            print(f"Screenshot saved: {path}")
        elif key == ord("+") or key == ord("="):
            conf = min(conf + 0.05, 0.95)
            print(f"Confidence: {conf:.2f}")
        elif key == ord("-"):
            conf = max(conf - 0.05, 0.05)
            print(f"Confidence: {conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
