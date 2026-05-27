import hashlib
import time
from collections import Counter
from typing import Any

import cv2
import numpy as np


def class_color(class_name: str) -> tuple[int, int, int]:
    """Deterministic RGB color per class name via hash."""
    h = hashlib.md5(class_name.encode()).digest()
    r, g, b = h[0], h[1], h[2]
    # Boost saturation: ensure at least one channel is bright
    max_c = max(r, g, b)
    if max_c < 100:
        scale = 180 / max(max_c, 1)
        r, g, b = int(r * scale), int(g * scale), int(b * scale)
    return (r, g, b)


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw bounding boxes and labels onto a frame (BGR)."""
    output = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["class"]
        conf = det["confidence"]
        color = class_color(label)
        bgr = (color[2], color[1], color[0])  # OpenCV is BGR

        cv2.rectangle(output, (x1, y1), (x2, y2), bgr, 2)

        text = f"{label} {conf:.0%}" if show_confidence else label
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(y1 - 6, th + baseline)
        cv2.rectangle(output, (x1, label_y - th - baseline), (x1 + tw + 4, label_y + 2), bgr, -1)
        cv2.putText(
            output, text,
            (x1 + 2, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return output


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return frame


def summarize_detections(detections: list[dict]) -> dict[str, int]:
    """Return count-per-class dict, sorted by count descending."""
    counts = Counter(d["class"] for d in detections)
    return dict(counts.most_common())


class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._times: list[float] = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0
