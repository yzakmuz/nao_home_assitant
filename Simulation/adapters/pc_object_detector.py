"""
pc_object_detector.py -- YOLOv8n via ultralytics (replaces TFLite on PC).

Same interface as vision.object_detector.ObjectDetector:
  load(), unload(), detect(), is_loaded, context manager.

Uses the Detection and DetectionResults dataclasses from the original module
to ensure type compatibility with the rest of the codebase.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import List, Optional

import numpy as np

# Import dataclasses from the ORIGINAL module (this works because
# tflite_runtime is imported inside load(), not at module level)
from vision.object_detector import Detection, DetectionResults

from settings import YOLO_CONFIDENCE_THRESHOLD

log = logging.getLogger(__name__)


class PcObjectDetector:
    """YOLOv8n via ultralytics -- PC replacement for TFLite detector."""

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from ultralytics import YOLO
            log.info("Loading YOLOv8n via ultralytics...")
            self._model = YOLO("yolov8n.pt")
            self._loaded = True
            log.info("YOLOv8n loaded successfully.")
        except ImportError:
            log.error("ultralytics not installed -- YOLO detection unavailable.")
            raise

    def unload(self) -> None:
        if not self._loaded:
            return
        self._model = None
        self._loaded = False
        gc.collect()
        log.info("YOLOv8n unloaded -- memory freed.")

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
        if not self._loaded or self._model is None:
            raise RuntimeError("Detector not loaded -- call load() first.")

        t0 = time.monotonic()
        results = self._model.predict(
            bgr_frame, conf=confidence_threshold, verbose=False
        )
        inference_ms = (time.monotonic() - t0) * 1000.0

        detections: List[Detection] = []
        if results and len(results) > 0:
            result = results[0]
            h_img, w_img = bgr_frame.shape[:2]

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names.get(class_id, "unknown")

                if target_classes and class_name not in target_classes:
                    continue

                # Get xywhn (normalized center-x, center-y, width, height)
                xywhn = box.xywhn[0]
                cx_norm = float(xywhn[0])
                cy_norm = float(xywhn[1])
                w_norm = float(xywhn[2])
                h_norm = float(xywhn[3])

                detections.append(Detection(
                    class_name=class_name,
                    confidence=confidence,
                    cx_norm=cx_norm,
                    cy_norm=cy_norm,
                    w_norm=w_norm,
                    h_norm=h_norm,
                ))

        # Sort by confidence, keep top 20
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return DetectionResults(
            detections=detections[:20],
            inference_ms=inference_ms,
        )
