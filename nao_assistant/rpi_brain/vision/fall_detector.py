"""
fall_detector.py — Person fall detection via temporal multi-signal fusion.

Pure math / logic — no hardware, no MediaPipe, no TCP, no threading.
Receives pre-computed body measurements from PoseEstimator (or face bbox
as fallback) and outputs fall / no-fall decisions.

Detection signals (all extracted from the same pose keypoints):
    1. Height Ratio   — current body height / calibrated baseline
    2. Drop Velocity  — rate of height-ratio change over sliding window
    3. Aspect Ratio   — body width / height (standing=tall, fallen=wide)
    4. Torso Angle    — shoulder-to-hip vector vs vertical (0°=up, 90°=flat)

State machine:
    UNCALIBRATED → CALIBRATING → MONITORING → TRIGGERED → RECOVERY → MONITORING
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, Optional, Tuple

log = logging.getLogger(__name__)


# ======================================================================
# States
# ======================================================================

class FallState(Enum):
    UNCALIBRATED = auto()   # waiting for a person to appear
    CALIBRATING = auto()    # collecting baseline standing height
    MONITORING = auto()     # actively watching for falls
    TRIGGERED = auto()      # fall detected, waiting for recovery
    RECOVERY = auto()       # person appears to be getting up


# ======================================================================
# Measurement snapshot (one per frame)
# ======================================================================

@dataclass(frozen=True)
class BodyMeasurement:
    """Computed body metrics from a single frame."""
    timestamp: float        # time.monotonic()
    body_height: float      # normalized [0, 1] — top-to-bottom keypoints
    body_width: float       # normalized [0, 1] — left-to-right keypoints
    torso_angle: float      # degrees from vertical (0=upright, 90=horizontal)
    has_full_pose: bool     # True if from pose keypoints, False if face-bbox


# ======================================================================
# Person Fall Detector
# ======================================================================

class PersonFallDetector:
    """
    Detects person falls using temporal multi-signal analysis.

    Feed it one ``BodyMeasurement`` per frame via ``update()``.
    Returns True when a fall is first detected.
    """

    def __init__(
        self,
        height_ratio_threshold: float = 0.50,
        velocity_threshold: float = 0.30,
        confirmation_frames: int = 5,
        recovery_ratio: float = 0.80,
        recovery_frames: int = 10,
        calibration_frames: int = 10,
        torso_angle_threshold: float = 75.0,
    ) -> None:
        # Thresholds
        self._height_ratio_thr = height_ratio_threshold
        self._velocity_thr = velocity_threshold
        self._confirmation_frames = confirmation_frames
        self._recovery_ratio = recovery_ratio
        self._recovery_frames = recovery_frames
        self._calibration_frames = calibration_frames
        self._torso_angle_thr = torso_angle_threshold

        # State
        self._state = FallState.UNCALIBRATED
        self._baseline_height: float = 0.0
        self._baseline_samples: list[float] = []

        # Sliding window of recent measurements (~2 seconds)
        self._window: Deque[BodyMeasurement] = deque(maxlen=30)

        # Confirmation / recovery counters
        self._fall_confirm_count: int = 0
        self._recovery_count: int = 0

        # Cached outputs for external queries
        self._fall_score: float = 0.0
        self._height_ratio: float = 1.0

        # Slow-moving baseline for distance adaptation (EMA, tau ~ 10s at 5 Hz)
        self._ema_alpha: float = 1.0 / (10.0 * 5.0)  # 1 / (tau * fps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, measurement: BodyMeasurement) -> bool:
        """Process one frame's body measurement.

        Returns True **once** when a fall is first detected (transition
        from MONITORING → TRIGGERED).  Subsequent calls return False
        until the person recovers and falls again.
        """
        self._window.append(measurement)

        if self._state == FallState.UNCALIBRATED:
            return self._do_uncalibrated(measurement)
        elif self._state == FallState.CALIBRATING:
            return self._do_calibrating(measurement)
        elif self._state == FallState.MONITORING:
            return self._do_monitoring(measurement)
        elif self._state == FallState.TRIGGERED:
            return self._do_triggered(measurement)
        elif self._state == FallState.RECOVERY:
            return self._do_recovery(measurement)
        return False

    def reset(self) -> None:
        """Reset all state.  Used when the monitor re-starts."""
        self._state = FallState.UNCALIBRATED
        self._baseline_height = 0.0
        self._baseline_samples.clear()
        self._window.clear()
        self._fall_confirm_count = 0
        self._recovery_count = 0
        self._fall_score = 0.0
        self._height_ratio = 1.0

    @property
    def state(self) -> FallState:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.name

    @property
    def fall_score(self) -> float:
        return self._fall_score

    @property
    def height_ratio(self) -> float:
        return self._height_ratio

    @property
    def is_triggered(self) -> bool:
        return self._state == FallState.TRIGGERED

    @property
    def baseline_height(self) -> float:
        return self._baseline_height

    def reset_confirmation(self) -> None:
        """Reset the fall confirmation counter (no person visible)."""
        self._fall_confirm_count = 0

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _do_uncalibrated(self, m: BodyMeasurement) -> bool:
        """Wait for a standing person to appear."""
        if m.body_height < 0.01:
            return False  # no person visible

        # Check if person looks like they're standing (taller than wide)
        aspect = m.body_width / max(m.body_height, 1e-6)
        if aspect > 1.2:
            # Person appears horizontal — don't calibrate on this
            log.debug("Fall detector: person appears horizontal (ar=%.2f), "
                      "waiting for standing posture", aspect)
            return False

        # Person looks upright — start calibrating
        self._state = FallState.CALIBRATING
        self._baseline_samples = [m.body_height]
        log.info("Fall detector: person detected, calibrating baseline…")
        return False

    def _do_calibrating(self, m: BodyMeasurement) -> bool:
        """Collect baseline standing height samples."""
        if m.body_height < 0.01:
            # Person disappeared during calibration — restart
            self._state = FallState.UNCALIBRATED
            self._baseline_samples.clear()
            return False

        aspect = m.body_width / max(m.body_height, 1e-6)
        if aspect > 1.2:
            # Person went horizontal during calibration — restart
            self._state = FallState.UNCALIBRATED
            self._baseline_samples.clear()
            return False

        self._baseline_samples.append(m.body_height)

        if len(self._baseline_samples) >= self._calibration_frames:
            self._baseline_height = (
                sum(self._baseline_samples) / len(self._baseline_samples)
            )
            self._state = FallState.MONITORING
            self._fall_confirm_count = 0
            log.info("Fall detector: calibrated — baseline_height=%.3f",
                     self._baseline_height)
        return False

    def _do_monitoring(self, m: BodyMeasurement) -> bool:
        """Active monitoring — compute fall score, check for fall."""
        if m.body_height < 0.01:
            # No person visible — skip, reset confirmation
            self._fall_confirm_count = 0
            return False

        # Compute all 4 signals
        score = self._compute_fall_score(m)
        self._fall_score = score

        # Is this frame a "fall frame"?
        if score > 0.6:
            self._fall_confirm_count += 1
        else:
            self._fall_confirm_count = 0

        # Slow EMA update of baseline for distance adaptation
        # Only update when NOT in a fall-like state
        if score < 0.3 and m.body_height > 0.05:
            self._baseline_height = (
                (1.0 - self._ema_alpha) * self._baseline_height
                + self._ema_alpha * m.body_height
            )

        # Trigger if enough consecutive fall frames
        if self._fall_confirm_count >= self._confirmation_frames:
            self._state = FallState.TRIGGERED
            self._fall_confirm_count = 0
            log.warning("PERSON FALL DETECTED! score=%.2f, height_ratio=%.2f",
                        score, self._height_ratio)
            return True

        return False

    def _do_triggered(self, m: BodyMeasurement) -> bool:
        """Fall was detected.  Watch for recovery."""
        if m.body_height < 0.01:
            return False  # still no person / on floor

        # Check if person is recovering (standing back up)
        hr = m.body_height / max(self._baseline_height, 1e-6)
        self._height_ratio = hr

        if hr > self._recovery_ratio:
            self._recovery_count += 1
        else:
            self._recovery_count = 0

        if self._recovery_count >= self._recovery_frames:
            self._state = FallState.RECOVERY
            log.info("Fall detector: person appears to be recovering…")
        return False

    def _do_recovery(self, m: BodyMeasurement) -> bool:
        """Person is getting up.  Confirm recovery then re-arm."""
        if m.body_height < 0.01:
            # Lost sight during recovery — go back to triggered
            self._state = FallState.TRIGGERED
            self._recovery_count = 0
            return False

        hr = m.body_height / max(self._baseline_height, 1e-6)
        self._height_ratio = hr

        if hr > self._recovery_ratio:
            # Sustained standing — re-arm
            self._state = FallState.MONITORING
            self._recovery_count = 0
            self._fall_confirm_count = 0
            log.info("Fall detector: person recovered — re-armed.")
        else:
            # Dropped back down — still triggered
            self._state = FallState.TRIGGERED
            self._recovery_count = 0
        return False

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_fall_score(self, m: BodyMeasurement) -> float:
        """Fuse the 4 detection signals into a single [0, 1] score."""

        # --- Signal 1: Height Ratio ---
        hr = m.body_height / max(self._baseline_height, 1e-6)
        self._height_ratio = hr
        # Map: ratio >= 1.0 → 0.0 (standing), ratio <= threshold → 1.0 (fallen)
        s_height = _linear_map(hr, low=self._height_ratio_thr, high=1.0,
                               out_low=1.0, out_high=0.0)

        # --- Signal 2: Drop Velocity ---
        s_velocity = self._compute_velocity_signal()

        # --- Signal 3: Aspect Ratio ---
        aspect = m.body_width / max(m.body_height, 1e-6)
        # Map: aspect <= 0.5 → 0.0 (vertical), aspect >= 1.5 → 1.0 (horizontal)
        s_aspect = _linear_map(aspect, low=0.5, high=1.5,
                               out_low=0.0, out_high=1.0)

        # --- Signal 4: Torso Angle ---
        if m.has_full_pose:
            # Map: angle <= 30° → 0.0, angle >= threshold → 1.0
            s_torso = _linear_map(m.torso_angle, low=30.0,
                                  high=self._torso_angle_thr,
                                  out_low=0.0, out_high=1.0)
            # Full pose: standard weights
            score = (
                0.40 * s_height
                + 0.30 * s_velocity
                + 0.15 * s_aspect
                + 0.15 * s_torso
            )
        else:
            # Face-bbox fallback: no torso angle available
            score = (
                0.50 * s_height
                + 0.35 * s_velocity
                + 0.15 * s_aspect
            )

        return max(0.0, min(1.0, score))

    def _compute_velocity_signal(self) -> float:
        """Compute the height-ratio drop velocity over the sliding window."""
        if len(self._window) < 3:
            return 0.0

        oldest = self._window[0]
        newest = self._window[-1]

        dt = newest.timestamp - oldest.timestamp
        if dt < 0.1:
            return 0.0  # too short to measure velocity

        oldest_hr = oldest.body_height / max(self._baseline_height, 1e-6)
        newest_hr = newest.body_height / max(self._baseline_height, 1e-6)

        # Positive velocity = height is dropping (falling)
        velocity = (oldest_hr - newest_hr) / dt

        # Map: velocity <= 0 → 0.0 (not falling), velocity >= threshold → 1.0
        return _linear_map(velocity, low=0.0, high=self._velocity_thr,
                           out_low=0.0, out_high=1.0)


# ======================================================================
# Helpers
# ======================================================================

def _linear_map(
    value: float,
    low: float, high: float,
    out_low: float = 0.0, out_high: float = 1.0,
) -> float:
    """Linearly map *value* from [low, high] to [out_low, out_high], clamped."""
    if high == low:
        return out_high if value >= high else out_low
    t = (value - low) / (high - low)
    t = max(0.0, min(1.0, t))
    return out_low + t * (out_high - out_low)


def measurement_from_pose(pose_result) -> Optional[BodyMeasurement]:
    """Create a BodyMeasurement from a PoseResult (pose_estimator.py).

    Returns None if the pose data is insufficient (too few visible keypoints).
    """
    if pose_result is None:
        return None
    if pose_result.confidence < 0.3:
        return None
    if pose_result.body_height < 0.01:
        return None

    return BodyMeasurement(
        timestamp=time.monotonic(),
        body_height=pose_result.body_height,
        body_width=pose_result.body_width,
        torso_angle=pose_result.torso_angle,
        has_full_pose=True,
    )


def measurement_from_face(face_result) -> Optional[BodyMeasurement]:
    """Create a BodyMeasurement from a FaceDetectionResult (face_tracker.py).

    Fallback mode — uses face bounding box as a rough proxy for body.
    Less accurate than full pose, but works when Pose is unavailable.
    Returns None if face_result is None.
    """
    if face_result is None:
        return None

    # Face bbox gives us rough info:
    # - face height ≈ proportional to body height (head is ~1/7 of body)
    # - face width / height ratio hints at orientation
    # We use the face height scaled by ~7 as an estimate of body height.
    # This is very rough but better than nothing.
    estimated_body_height = face_result.height * 7.0
    estimated_body_width = face_result.width * 3.0  # rough shoulder width estimate

    return BodyMeasurement(
        timestamp=time.monotonic(),
        body_height=min(estimated_body_height, 1.0),
        body_width=min(estimated_body_width, 1.0),
        torso_angle=0.0,  # unknown in face-only mode
        has_full_pose=False,
    )
