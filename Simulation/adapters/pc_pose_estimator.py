"""
pc_pose_estimator.py -- PC simulation adapter for PoseEstimator.

Wraps the real PoseEstimator and adds simulation-specific features:
    - Feeds skeleton keypoint data to SharedSimState for GUI overlay
    - Supports synthetic "fallen" pose injection via hotkey

The core detection runs the same MediaPipe Pose code as the RPi.
This adapter only adds GUI data and test hooks.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from vision.pose_estimator import PoseEstimator, PoseKeypoint, PoseResult


class PcPoseEstimator(PoseEstimator):
    """PC simulation wrapper around PoseEstimator.

    Adds:
        - ``last_raw_landmarks``: raw keypoint data for skeleton drawing
        - ``inject_fall_pose()``: synthetic fallen pose for ``p`` hotkey
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Raw landmark data for skeleton drawing (set after each detect)
        self.last_raw_landmarks: Optional[list] = None
        # Synthetic fall injection
        self._inject_fall: bool = False

    def detect(self, bgr_frame: np.ndarray) -> Optional[PoseResult]:
        """Run pose detection, optionally returning a synthetic fallen pose."""
        if self._inject_fall:
            return self._make_fallen_pose()

        result = super().detect(bgr_frame)

        # Cache raw keypoint data for skeleton drawing
        if result is not None:
            self.last_raw_landmarks = [
                {"name": name, "x": kp.x, "y": kp.y, "visibility": kp.visibility}
                for name, kp in result.keypoints.items()
            ]
        else:
            self.last_raw_landmarks = None

        return result

    def inject_fall_pose(self) -> None:
        """Enable synthetic fallen pose injection (called by ``p`` hotkey)."""
        self._inject_fall = True

    def clear_fall_pose(self) -> None:
        """Disable synthetic fallen pose injection (called by ``r`` hotkey)."""
        self._inject_fall = False

    @staticmethod
    def _make_fallen_pose() -> PoseResult:
        """Create a synthetic PoseResult representing a fallen person.

        The keypoints describe a person lying horizontally on the ground:
        wide and short, torso angle near 90 degrees.
        """
        # Person lying on the ground: all y-values near bottom (0.85-0.90),
        # spread horizontally (x: 0.2 to 0.8)
        kp = {
            "nose": PoseKeypoint(x=0.25, y=0.85, visibility=0.9),
            "left_shoulder": PoseKeypoint(x=0.30, y=0.87, visibility=0.9),
            "right_shoulder": PoseKeypoint(x=0.40, y=0.87, visibility=0.9),
            "left_hip": PoseKeypoint(x=0.50, y=0.88, visibility=0.9),
            "right_hip": PoseKeypoint(x=0.60, y=0.88, visibility=0.9),
            "left_knee": PoseKeypoint(x=0.65, y=0.89, visibility=0.8),
            "right_knee": PoseKeypoint(x=0.70, y=0.89, visibility=0.8),
            "left_ankle": PoseKeypoint(x=0.75, y=0.90, visibility=0.7),
            "right_ankle": PoseKeypoint(x=0.80, y=0.90, visibility=0.7),
        }
        return PoseResult(
            keypoints=kp,
            body_height=0.05,       # very short (lying flat)
            body_width=0.55,        # very wide (spread out)
            torso_angle=85.0,       # nearly horizontal
            hip_center=(0.55, 0.88),
            shoulder_center=(0.35, 0.87),
            confidence=0.85,
        )
