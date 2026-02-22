"""
object_detector.py — YOLOv8-Nano (TFLite) with dynamic memory management.

CRITICAL DESIGN:
    This detector is **NOT** kept resident in memory. On a 2 GB Pi,
    keeping YOLO loaded alongside MediaPipe + Vosk + ECAPA would OOM.

    Instead, the detector provides an explicit context-manager and
    load/unload lifecycle:

        detector = ObjectDetector()
        detector.load()         # allocates TFLite interpreter
        results = detector.detect(frame, target_classes=["bottle"])
        detector.unload()       # frees interpreter + forces GC

    Or as a context manager:
        with ObjectDetector() as det:
            results = det.detect(frame, ...)

COCO labels:
    YOLOv8n is trained on 80 COCO classes. We map user-friendly names
    (e.g., "keys") to relevant COCO labels via settings.YOLO_TARGET_CLASSES.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from settings import (
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_INPUT_SIZE,
    YOLO_MODEL_PATH,
)

log = logging.getLogger(__name__)

# Full 80-class COCO label list (YOLOv8 default order)
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    """Single object detection result."""
    class_name: str
    confidence: float
    cx_norm: float  # center x, normalized [0, 1]
    cy_norm: float  # center y, normalized [0, 1]
    w_norm: float   # width,  normalized
    h_norm: float   # height, normalized


@dataclass
class DetectionResults:
    """Container for all detections in a single frame."""
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0


class ObjectDetector:
    """
    Dynamically-loaded YOLOv8-Nano TFLite detector.

    Use `load()` / `unload()` or the context manager to control when the
    ~8 MB model occupies RAM.
    """

    def __init__(self) -> None:
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Allocate TFLite interpreter and tensors."""
        if self._loaded:
            return

        import tflite_runtime.interpreter as tflite

        log.info("Loading YOLOv8n TFLite model: %s", YOLO_MODEL_PATH)
        self._interpreter = tflite.Interpreter(
            model_path=YOLO_MODEL_PATH,
            num_threads=4,
        )
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._loaded = True

        in_shape = self._input_details[0]["shape"]
        log.info("YOLOv8n loaded — input shape: %s", in_shape)

    def unload(self) -> None:
        """Release interpreter and aggressively reclaim memory."""
        if not self._loaded:
            return
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._loaded = False
        gc.collect()
        log.info("YOLOv8n unloaded — memory freed.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc):
        self.unload()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(
        self,
        bgr_frame: np.ndarray,
        target_classes: Optional[List[str]] = None,
        confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    ) -> DetectionResults:
        """
        Run detection on a BGR frame.

        Args:
            bgr_frame: OpenCV BGR image (any resolution).
            target_classes: If given, only return detections whose class
                            name is in this list.
            confidence_threshold: Minimum score to keep a detection.

        Returns:
            DetectionResults with list of Detection objects.
        """
        if not self._loaded:
            raise RuntimeError("Detector not loaded — call load() first.")

        import time as _time

        w_in, h_in = YOLO_INPUT_SIZE
        img = cv2.resize(bgr_frame, (w_in, h_in))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, H, W, 3)

        # If model expects NCHW, transpose
        in_shape = self._input_details[0]["shape"]
        if in_shape[1] == 3:
            img = np.transpose(img, (0, 3, 1, 2))

        self._interpreter.set_tensor(self._input_details[0]["index"], img)

        t0 = _time.monotonic()
        self._interpreter.invoke()
        inference_ms = (_time.monotonic() - t0) * 1000.0

        raw_output = self._interpreter.get_tensor(self._output_details[0]["index"])

        detections = self._parse_yolov8_output(
            raw_output, confidence_threshold, target_classes
        )

        return DetectionResults(detections=detections, inference_ms=inference_ms)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_yolov8_output(
        raw: np.ndarray,
        conf_thr: float,
        target_classes: Optional[List[str]],
    ) -> List[Detection]:
        """
        Parse YOLOv8 TFLite output tensor.

        YOLOv8 outputs shape (1, 84, N) where:
            - rows 0-3: cx, cy, w, h (normalized)
            - rows 4-83: class confidences for 80 COCO classes
        """
        # Squeeze batch dim → (84, N) then transpose → (N, 84)
        out = np.squeeze(raw)
        if out.shape[0] == 84:
            out = out.T  # (N, 84)

        results: List[Detection] = []

        for row in out:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])

            if score < conf_thr:
                continue

            if class_id >= len(COCO_LABELS):
                continue

            class_name = COCO_LABELS[class_id]

            if target_classes and class_name not in target_classes:
                continue

            results.append(
                Detection(
                    class_name=class_name,
                    confidence=score,
                    cx_norm=float(cx),
                    cy_norm=float(cy),
                    w_norm=float(w),
                    h_norm=float(h),
                )
            )

        # NMS: simple sort-by-confidence and keep top-K
        results.sort(key=lambda d: d.confidence, reverse=True)
        return results[:20]
