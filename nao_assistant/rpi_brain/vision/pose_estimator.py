"""
pose_estimator.py — MediaPipe Pose (BlazePose) wrapper.

Returns normalized body keypoints that the fall detector uses to compute
body height, aspect ratio, and torso angle.

Design:
    - Uses MediaPipe's ``Pose`` (BlazePose Lite, model_complexity=0)
      which is the fastest variant, suitable for RPi 4 at ~30-50 ms.
    - Returns a ``PoseResult`` dataclass with pre-computed body metrics
      so the fall detector only does pure math.
    - Same structural pattern as ``face_tracker.py`` for consistency.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mediapipe as mp
import numpy as np

log = logging.getLogger(__name__)


# ======================================================================
# Data classes
# ======================================================================

@dataclass(frozen=True)
class PoseKeypoint:
    """A single body landmark in normalized image coordinates."""
    x: float            # [0, 1] — left edge to right edge
    y: float            # [0, 1] — top edge to bottom edge
    visibility: float   # [0, 1] — confidence that this point is visible


@dataclass(frozen=True)
class PoseResult:
    """Processed pose output with pre-computed body metrics."""
    keypoints: Dict[str, PoseKeypoint]      # named landmarks
    body_height: float                       # top-to-bottom (normalized)
    body_width: float                        # left-to-right (normalized)
    torso_angle: float                       # degrees from vertical
    hip_center: Tuple[float, float]          # (x, y) midpoint of hips
    shoulder_center: Tuple[float, float]     # (x, y) midpoint of shoulders
    confidence: float                        # average visibility of key landmarks


# Landmark names we extract (MediaPipe Pose indices)
_LANDMARK_NAMES = {
    0:  "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

# Minimum landmarks needed for reliable analysis
_KEY_LANDMARKS = {"nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"}

# Skeleton connections for drawing (pairs of landmark names)
SKELETON_CONNECTIONS = [
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("right_shoulder", "right_hip"),
    ("right_hip", "left_hip"),
    ("left_hip", "left_shoulder"),
    # Left arm (shoulder only — we don't extract elbow/wrist for fall detection)
    # Right arm
    # Left leg
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    # Right leg
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


# ======================================================================
# Pose Estimator
# ======================================================================

class PoseEstimator:
    """Thin wrapper around MediaPipe Pose (BlazePose Lite)."""

    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        log.info(
            "Initializing MediaPipe Pose (complexity=%d, det_conf=%.2f, "
            "track_conf=%.2f)",
            model_complexity,
            min_detection_confidence,
            min_tracking_confidence,
        )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, bgr_frame: np.ndarray) -> Optional[PoseResult]:
        """
        Run pose estimation on a BGR OpenCV frame.

        Returns a ``PoseResult`` with body metrics, or None if no person
        is detected or if too few keypoints are visible.
        """
        rgb = bgr_frame[:, :, ::-1]  # BGR → RGB (zero-copy view)
        results = self._pose.process(rgb)

        if results.pose_landmarks is None:
            return None

        # Extract named landmarks
        landmarks = results.pose_landmarks.landmark
        keypoints: Dict[str, PoseKeypoint] = {}
        for idx, name in _LANDMARK_NAMES.items():
            lm = landmarks[idx]
            keypoints[name] = PoseKeypoint(
                x=float(lm.x),
                y=float(lm.y),
                visibility=float(lm.visibility),
            )

        # Check that key landmarks are sufficiently visible
        visible_key = sum(
            1 for name in _KEY_LANDMARKS
            if keypoints[name].visibility > 0.5
        )
        if visible_key < len(_KEY_LANDMARKS) * 0.5:
            return None  # insufficient data

        # Average confidence of key landmarks
        confidence = sum(
            keypoints[n].visibility for n in _KEY_LANDMARKS
        ) / len(_KEY_LANDMARKS)

        # Compute body metrics
        body_height, body_width = self._compute_body_dimensions(keypoints)
        torso_angle = self._compute_torso_angle(keypoints)
        hip_center = self._midpoint(keypoints["left_hip"], keypoints["right_hip"])
        shoulder_center = self._midpoint(
            keypoints["left_shoulder"], keypoints["right_shoulder"]
        )

        return PoseResult(
            keypoints=keypoints,
            body_height=body_height,
            body_width=body_width,
            torso_angle=torso_angle,
            hip_center=hip_center,
            shoulder_center=shoulder_center,
            confidence=confidence,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()
        log.info("PoseEstimator closed.")

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_body_dimensions(
        kp: Dict[str, PoseKeypoint],
    ) -> Tuple[float, float]:
        """Compute body height and width from visible keypoints."""
        # Collect y-coordinates of visible keypoints for height
        visible_ys = []
        visible_xs = []
        for name, point in kp.items():
            if point.visibility > 0.5:
                visible_ys.append(point.y)
                visible_xs.append(point.x)

        if len(visible_ys) < 2:
            return (0.0, 0.0)

        body_height = max(visible_ys) - min(visible_ys)
        body_width = max(visible_xs) - min(visible_xs)

        return (max(body_height, 0.0), max(body_width, 0.0))

    @staticmethod
    def _compute_torso_angle(kp: Dict[str, PoseKeypoint]) -> float:
        """Compute torso angle in degrees from vertical.

        0° = perfectly upright, 90° = lying flat.
        Uses the vector from hip midpoint to shoulder midpoint.
        """
        ls = kp["left_shoulder"]
        rs = kp["right_shoulder"]
        lh = kp["left_hip"]
        rh = kp["right_hip"]

        # Check visibility
        if min(ls.visibility, rs.visibility, lh.visibility, rh.visibility) < 0.4:
            return 0.0  # not enough data — assume upright

        shoulder_x = (ls.x + rs.x) / 2.0
        shoulder_y = (ls.y + rs.y) / 2.0
        hip_x = (lh.x + rh.x) / 2.0
        hip_y = (lh.y + rh.y) / 2.0

        # Vector from hip to shoulder (in image coords: y increases downward)
        dx = shoulder_x - hip_x
        dy = hip_y - shoulder_y  # flip y so up is positive

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0

        # Angle from vertical: atan2(horizontal, vertical)
        angle_rad = math.atan2(abs(dx), abs(dy))
        return math.degrees(angle_rad)

    @staticmethod
    def _midpoint(
        a: PoseKeypoint, b: PoseKeypoint,
    ) -> Tuple[float, float]:
        return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
