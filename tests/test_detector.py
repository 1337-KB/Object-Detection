import numpy as np
import pytest

from app.config import COCO_CLASSES, MODEL_OPTIONS
from app.utils import FPSCounter, class_color, summarize_detections
from app.detector import detect, load_model


# ── Config tests ───────────────────────────────────────────────────────────────
class TestConfig:
    def test_coco_has_80_classes(self):
        assert len(COCO_CLASSES) == 80

    def test_model_options_not_empty(self):
        assert len(MODEL_OPTIONS) >= 1

    def test_known_classes_present(self):
        assert "person" in COCO_CLASSES
        assert "car" in COCO_CLASSES
        assert "dog" in COCO_CLASSES


# ── Utils tests ────────────────────────────────────────────────────────────────
class TestUtils:
    def test_class_color_returns_rgb_tuple(self):
        color = class_color("person")
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    def test_class_color_deterministic(self):
        assert class_color("cat") == class_color("cat")

    def test_class_color_unique_per_class(self):
        colors = {class_color(cls) for cls in COCO_CLASSES}
        # Should have mostly unique colors (allow a few collisions from hashing)
        assert len(colors) > 60

    def test_summarize_empty(self):
        assert summarize_detections([]) == {}

    def test_summarize_counts(self):
        detections = [
            {"class": "cat", "confidence": 0.9, "box": (0, 0, 10, 10)},
            {"class": "dog", "confidence": 0.8, "box": (0, 0, 10, 10)},
            {"class": "cat", "confidence": 0.7, "box": (0, 0, 10, 10)},
        ]
        summary = summarize_detections(detections)
        assert summary["cat"] == 2
        assert summary["dog"] == 1

    def test_summarize_sorted_by_count(self):
        detections = [
            {"class": "dog", "confidence": 0.9, "box": (0, 0, 10, 10)},
            {"class": "cat", "confidence": 0.8, "box": (0, 0, 10, 10)},
            {"class": "cat", "confidence": 0.7, "box": (0, 0, 10, 10)},
        ]
        keys = list(summarize_detections(detections).keys())
        assert keys[0] == "cat"

    def test_fps_counter_zero_on_first_tick(self):
        counter = FPSCounter()
        assert counter.tick() == 0.0

    def test_fps_counter_positive_after_ticks(self):
        import time
        counter = FPSCounter()
        counter.tick()
        time.sleep(0.05)
        counter.tick()
        assert counter.tick() > 0


# ── Detector tests ─────────────────────────────────────────────────────────────
class TestDetector:
    @pytest.fixture(scope="class")
    def model(self):
        return load_model("yolov8n.pt")

    def test_model_loads(self, model):
        assert model is not None

    def test_detect_blank_frame_returns_list(self, model):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect(model, frame, confidence=0.5)
        assert isinstance(result, list)

    def test_detect_output_structure(self, model):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect(model, frame)
        for det in result:
            assert "class" in det
            assert "confidence" in det
            assert "box" in det
            assert isinstance(det["box"], tuple)
            assert len(det["box"]) == 4

    def test_detect_confidence_filter(self, model):
        """High threshold should return fewer or equal detections than low threshold."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        low = detect(model, frame, confidence=0.1)
        high = detect(model, frame, confidence=0.9)
        assert len(high) <= len(low)

    def test_detect_confidence_values_in_range(self, model):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detect(model, frame, confidence=0.1)
        for det in result:
            assert 0.0 <= det["confidence"] <= 1.0
