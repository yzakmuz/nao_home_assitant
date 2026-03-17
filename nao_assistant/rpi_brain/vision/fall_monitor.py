"""
fall_monitor.py — Always-on person fall detection thread.

Runs as a daemon thread from boot to shutdown, independently of the
visual servo loop.  Reads camera frames at ~5 Hz, runs MediaPipe Pose,
and feeds the PersonFallDetector.  Fires a callback when a fall is
detected.

Architecture:
    - Separate thread from the servo (different rate, different lifecycle).
    - Owns its own PoseEstimator instance (MediaPipe is NOT thread-safe
      across instances of different model types, but separate Pose
      instances in separate threads are fine).
    - Shares the Camera via thread-safe ``Camera.read()`` (returns a copy).
    - Falls back to face-bbox heuristics if Pose cannot be loaded
      (e.g., memory pressure on RPi).

Thread safety:
    - ``_last_pose_result`` and ``_last_fall_state`` are written by this
      thread and read by the GUI updater.  They are atomic reference
      assignments (safe in CPython due to the GIL).
    - The ``on_person_fall`` callback is called from THIS thread.
      The callback should be non-blocking (e.g., set a threading.Event).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from vision.camera import Camera
from vision.fall_detector import (
    BodyMeasurement,
    PersonFallDetector,
    measurement_from_face,
    measurement_from_pose,
)
from settings import (
    FALL_CALIBRATION_FRAMES,
    FALL_CONFIRMATION_FRAMES,
    FALL_DETECTION_ENABLED,
    FALL_HEIGHT_RATIO_THRESHOLD,
    FALL_MONITOR_RATE_HZ,
    FALL_RECOVERY_FRAMES,
    FALL_RECOVERY_RATIO,
    FALL_TORSO_ANGLE_THRESHOLD,
    FALL_VELOCITY_THRESHOLD,
    RAM_WARNING_THRESHOLD_MB,
)

log = logging.getLogger(__name__)


class FallMonitorThread:
    """Always-on daemon thread for person fall detection.

    Starts at boot, stops only at shutdown.  Detects falls regardless
    of which FSM state the main brain is in.
    """

    def __init__(
        self,
        camera: Camera,
        on_person_fall: Optional[Callable[[], None]] = None,
    ) -> None:
        self._camera = camera
        self._callback = on_person_fall

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Pose estimator — created in start() after memory check
        self._pose_estimator = None
        self._use_pose: bool = True  # False = face-bbox fallback

        # Fall detector — core logic
        self._fall_detector = PersonFallDetector(
            height_ratio_threshold=FALL_HEIGHT_RATIO_THRESHOLD,
            velocity_threshold=FALL_VELOCITY_THRESHOLD,
            confirmation_frames=FALL_CONFIRMATION_FRAMES,
            recovery_ratio=FALL_RECOVERY_RATIO,
            recovery_frames=FALL_RECOVERY_FRAMES,
            calibration_frames=FALL_CALIBRATION_FRAMES,
            torso_angle_threshold=FALL_TORSO_ANGLE_THRESHOLD,
        )

        # Cached outputs for GUI (atomic reference assignment = thread-safe)
        self._last_pose_result = None
        self._last_fall_state: str = "INACTIVE"
        self._last_fall_score: float = 0.0

        # Adaptive rate: drop to 1 Hz when no person is seen for 60+ seconds
        self._no_person_count: int = 0
        self._idle_threshold: int = int(60 * FALL_MONITOR_RATE_HZ)

        # Own FaceTracker for fallback mode (separate instance for thread safety).
        # MediaPipe is NOT thread-safe — sharing a FaceDetection instance between
        # the servo thread and this thread causes timestamp mismatch crashes.
        self._face_tracker = None  # created in start() if needed
        self._use_face_fallback: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the fall monitor daemon thread."""
        if self._running:
            return
        if not FALL_DETECTION_ENABLED:
            log.info("Fall detection disabled (FALL_DETECTION_ENABLED=false).")
            return

        # Memory guard: check if we can afford to load Pose
        self._use_pose = self._try_init_pose()

        # If Pose isn't available and face fallback is enabled,
        # create a PRIVATE FaceTracker (separate from servo's instance)
        if not self._use_pose and self._use_face_fallback:
            try:
                from vision.face_tracker import FaceTracker
                self._face_tracker = FaceTracker()
                log.info("Fall monitor: created private FaceTracker for fallback.")
            except Exception as exc:
                log.warning("Fall monitor: could not create FaceTracker: %s", exc)
                self._face_tracker = None

        self._fall_detector.reset()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info(
            "Fall monitor started (%.0f Hz, mode=%s).",
            FALL_MONITOR_RATE_HZ,
            "pose" if self._use_pose else "face-bbox-fallback",
        )

    def stop(self) -> None:
        """Stop the fall monitor thread and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        if self._face_tracker is not None:
            try:
                self._face_tracker.close()
            except Exception:
                pass
            self._face_tracker = None

        if self._pose_estimator is not None:
            try:
                self._pose_estimator.close()
            except Exception:
                pass
            self._pose_estimator = None

        self._last_fall_state = "INACTIVE"
        log.info("Fall monitor stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_pose_result(self):
        """Latest PoseResult from the monitor (read by GUI updater)."""
        return self._last_pose_result

    @property
    def fall_detector_state(self) -> str:
        """Current fall detector state name (for GUI display)."""
        return self._last_fall_state

    @property
    def fall_score(self) -> float:
        """Current fall fusion score (for GUI display)."""
        return self._last_fall_score

    @property
    def fall_detector(self) -> PersonFallDetector:
        """Direct access to the fall detector (for testing)."""
        return self._fall_detector

    def set_face_tracker(self, face_tracker) -> None:
        """Enable face-bbox fallback by creating a PRIVATE FaceTracker.

        Creates a new FaceTracker instance instead of sharing the caller's,
        because MediaPipe is NOT thread-safe — sharing a FaceDetection
        instance between the servo thread and this monitor thread causes
        ``Packet timestamp mismatch`` crashes.

        The *face_tracker* argument is accepted for API compatibility but
        is NOT stored.  A fresh instance is created on start() instead.
        """
        self._use_face_fallback = True

    # ------------------------------------------------------------------
    # Monitor loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main monitoring loop — runs at FALL_MONITOR_RATE_HZ."""
        interval = 1.0 / FALL_MONITOR_RATE_HZ
        idle_interval = 1.0  # 1 Hz when no person visible for a while

        while self._running:
            t_start = time.monotonic()

            fell = False
            try:
                fell = self._process_frame()
            except Exception as exc:
                log.error("Fall monitor error: %s", exc, exc_info=False)

            # Update cached state for GUI
            self._last_fall_state = self._fall_detector.state_name
            self._last_fall_score = self._fall_detector.fall_score

            # Fire callback on fall detection
            if fell and self._callback is not None:
                try:
                    self._callback()
                except Exception as exc:
                    log.error("Fall callback error: %s", exc)

            # Adaptive sleep: slower when no person visible
            if self._no_person_count > self._idle_threshold:
                sleep_target = idle_interval
            else:
                sleep_target = interval

            elapsed = time.monotonic() - t_start
            sleep_time = sleep_target - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_frame(self) -> bool:
        """Process one camera frame.  Returns True if fall just detected."""
        frame = self._camera.read()
        if frame is None:
            return False

        measurement = None

        # Try full pose estimation first
        if self._use_pose and self._pose_estimator is not None:
            try:
                pose = self._pose_estimator.detect(frame)
                self._last_pose_result = pose
                measurement = measurement_from_pose(pose)
            except Exception as exc:
                log.debug("Pose estimation failed: %s", exc)

        # Fallback: use face bbox if pose unavailable or returned None
        if measurement is None and self._face_tracker is not None:
            try:
                face = self._face_tracker.detect(frame)
                measurement = measurement_from_face(face)
            except Exception:
                pass

        if measurement is None:
            self._no_person_count += 1
            # Reset confirmation counter when no person is visible
            self._fall_detector.reset_confirmation()
            return False

        self._no_person_count = 0
        return self._fall_detector.update(measurement)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _try_init_pose(self) -> bool:
        """Try to initialize MediaPipe Pose.  Returns True on success."""
        try:
            from utils.memory import available_mb
            avail = available_mb()
            if avail < RAM_WARNING_THRESHOLD_MB:
                log.warning(
                    "Low memory (%.0f MB available) — fall detection "
                    "using face-bbox fallback (no Pose model loaded).",
                    avail,
                )
                return False
        except Exception:
            pass  # if memory check fails, try loading Pose anyway

        try:
            from vision.pose_estimator import PoseEstimator
            self._pose_estimator = PoseEstimator()
            return True
        except Exception as exc:
            log.warning(
                "Failed to load PoseEstimator: %s — using face-bbox fallback.",
                exc,
            )
            return False
