"""
face_tracker.py — MediaPipe BlazeFace face detection wrapper.

Returns normalized face-center coordinates (0.0–1.0) that the visual
servo controller converts into head yaw/pitch adjustments.

Design:
    - Uses MediaPipe's `FaceDetection` (BlazeFace short-range model)
      which is optimized for < 2 m and runs very fast on ARM.
    - Picks the *largest* (closest) face when multiple are detected.
    - Returns a `FaceDetectionResult` dataclass with center coords,
      bounding-box dimensions, and a confidence score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import mediapipe as mp
import numpy as np

from settings import FACE_MIN_DETECTION_CONFIDENCE, FACE_MODEL_SELECTION

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceDetectionResult:
    """Face detection output in normalized image coordinates [0, 1]."""

    cx: float          # center x (0 = left edge, 1 = right edge)
    cy: float          # center y (0 = top edge, 1 = bottom edge)
    width: float       # bounding-box width  (normalized)
    height: float      # bounding-box height (normalized)
    confidence: float  # detection confidence [0, 1]


class FaceTracker:
    """Thin wrapper around MediaPipe FaceDetection (BlazeFace)."""

    def __init__(self) -> None:
        log.info(
            "Initializing MediaPipe FaceDetection "
            "(model=%d, conf=%.2f)",
            FACE_MODEL_SELECTION,
            FACE_MIN_DETECTION_CONFIDENCE,
        )
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=FACE_MODEL_SELECTION,
            min_detection_confidence=FACE_MIN_DETECTION_CONFIDENCE,
        )

    def detect(self, bgr_frame: np.ndarray) -> Optional[FaceDetectionResult]:
        """
        Run face detection on a BGR OpenCV frame.

        Returns the largest detected face as a `FaceDetectionResult`,
        or None if no face is found.
        """
        rgb = bgr_frame[:, :, ::-1]  # BGR → RGB (zero-copy view)
        results = self._detector.process(rgb)

        if not results.detections:
            return None

        # Pick the detection with the largest bounding box area
        best = max(
            results.detections,
            key=lambda d: (
                d.location_data.relative_bounding_box.width
                * d.location_data.relative_bounding_box.height
            ),
        )

        bb = best.location_data.relative_bounding_box
        cx = bb.xmin + bb.width / 2.0
        cy = bb.ymin + bb.height / 2.0
        confidence = best.score[0] if best.score else 0.0

        return FaceDetectionResult(
            cx=float(np.clip(cx, 0.0, 1.0)),
            cy=float(np.clip(cy, 0.0, 1.0)),
            width=float(bb.width),
            height=float(bb.height),
            confidence=float(confidence),
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._detector.close()
        log.info("FaceTracker closed.")
