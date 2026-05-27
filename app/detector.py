from __future__ import annotations

from functools import lru_cache

import numpy as np
from ultralytics import YOLO

from app.config import DEFAULT_CONFIDENCE, DEFAULT_IOU


@lru_cache(maxsize=3)
def load_model(model_path: str) -> YOLO:
    """Load and cache a YOLO model by path/name."""
    return YOLO(model_path)


def detect(
    model: YOLO,
    frame: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
) -> list[dict]:
    """
    Run inference on a single BGR or RGB frame.

    Returns a list of detection dicts:
        {
            "class": str,
            "confidence": float,
            "box": (x1, y1, x2, y2),   # pixel coords, int
        }
    """
    results = model.predict(
        source=frame,
        conf=confidence,
        iou=iou,
        verbose=False,
    )

    detections: list[dict] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id]
            conf = float(box.conf[0].item())
            detections.append({
                "class": cls_name,
                "confidence": conf,
                "box": (int(x1), int(y1), int(x2), int(y2)),
            })

    return detections
