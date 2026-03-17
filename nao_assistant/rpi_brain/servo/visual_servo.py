"""
visual_servo.py — PID-based visual servoing for face tracking.

Architecture:
    The Pi Camera is physically mounted on the NAO's head and moves
    with it. This module closes the perception-action loop:

    1. FaceTracker gives (cx, cy) in normalized image coords.
    2. Error = (0.5, 0.5) - (cx, cy)  →  how far off-center the face is.
    3. Two independent PID controllers (yaw, pitch) compute angular
       velocity corrections.
    4. Commands are sent to the NAO's head joints via TCP.
    5. If the accumulated Head Yaw exceeds a threshold, a body-turn
       command is issued to realign the torso (only in full-follow mode).

Two tracking modes (Improvement 5):
    - Head-only ("follow me"):  PID head tracking, no body walking.
    - Full-follow ("come here"): PID head tracking + body realignment
      + walking.  When face is lost, searches for the person (head scan
      then 360° body rotation).

The servo loop runs in its own thread at ~15 Hz, started/stopped by
the main state machine.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from comms.tcp_client import NaoTcpClient
from vision.camera import Camera
from vision.face_tracker import FaceDetectionResult, FaceTracker
from settings import (
    BODY_REALIGN_YAW_THRESHOLD_RAD,
    BODY_TURN_SPEED,
    FACE_LOST_FRAME_THRESHOLD,
    NAO_HEAD_PITCH_MAX,
    NAO_HEAD_PITCH_MIN,
    NAO_HEAD_YAW_MAX,
    NAO_HEAD_YAW_MIN,
    SERVO_DEADZONE,
    SERVO_LOOP_INTERVAL_S,
    SERVO_MAX_SPEED,
    SERVO_PITCH_KD,
    SERVO_PITCH_KI,
    SERVO_PITCH_KP,
    SERVO_YAW_KD,
    SERVO_YAW_KI,
    SERVO_YAW_KP,
)

log = logging.getLogger(__name__)


# ======================================================================
# PID Controller (generic single-axis)
# ======================================================================

class PIDController:
    """Discrete PID with anti-windup clamping and output limiting."""

    def __init__(
        self, kp: float, ki: float, kd: float, output_limit: float
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        self._integral += error * dt
        # Anti-windup: clamp integral contribution
        max_integral = self.output_limit / max(self.ki, 1e-6)
        self._integral = float(np.clip(self._integral, -max_integral, max_integral))

        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(output, -self.output_limit, self.output_limit))


# ======================================================================
# Person Search Constants
# ======================================================================

_SEARCH_HEAD_ANGLES = [0.0, 0.6, -0.6, 1.2, -1.2]
_SEARCH_HEAD_HOLD_S = 1.0          # seconds per head angle
_SEARCH_ROTATE_SPEED = 0.3         # rad/s body rotation
_SEARCH_ROTATE_TIMEOUT_S = 22.0    # ~360° at 0.3 rad/s
_SEARCH_TOTAL_TIMEOUT_S = 30.0     # overall safety timeout
_SEARCH_FACE_CONFIRM_FRAMES = 3    # consecutive frames to confirm found


# ======================================================================
# Visual Servo Controller
# ======================================================================

class VisualServoController:
    """
    Runs a face-tracking servo loop in a background thread.

    Supports two modes:
        - ``walk_enabled=False``: head-only tracking ("follow me")
        - ``walk_enabled=True``:  full follow with body walking ("come here")
    """

    def __init__(
        self,
        camera: Camera,
        face_tracker: FaceTracker,
        tcp_client: NaoTcpClient,
    ) -> None:
        self._camera = camera
        self._face_tracker = face_tracker
        self._tcp = tcp_client

        self._pid_yaw = PIDController(
            SERVO_YAW_KP, SERVO_YAW_KI, SERVO_YAW_KD, SERVO_MAX_SPEED
        )
        self._pid_pitch = PIDController(
            SERVO_PITCH_KP, SERVO_PITCH_KI, SERVO_PITCH_KD, SERVO_MAX_SPEED
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Tracked state (estimated current head angles in radians)
        self._head_yaw = 0.0
        self._head_pitch = 0.0

        # Lost-face counter
        self._frames_without_face = 0

        # Protect PID and angle state from race conditions
        self._servo_lock = threading.RLock()

        # BUG 9 fix: periodic sync of head angles from NAO state cache
        self._last_angle_sync = 0.0
        self._angle_sync_interval = 2.0  # seconds

        # Cached face detection result (thread-safe read by dashboard)
        self._last_face_result: Optional[FaceDetectionResult] = None
        self._last_pid_error: tuple[float, float] = (0.0, 0.0)

        # ── Walk mode (Improvement 5) ──
        self._walk_enabled: bool = True

        # ── Person search state ──
        self._search_active: bool = False
        self._search_phase: str = "none"      # "head_scan" | "body_rotate" | "none"
        self._search_start_time: float = 0.0
        self._search_head_idx: int = 0
        self._search_last_move_time: float = 0.0
        self._search_rotate_start: float = 0.0
        self._face_found_count: int = 0

        # ── Face re-acquisition tracking ──
        self._was_face_lost: bool = False

        # ── Periodic feedback (head-only mode) ──
        self._feedback_interval: float = 15.0
        self._last_feedback_time: float = 0.0

        # ── Callbacks (set by main.py) ──
        self.on_tracking_feedback: Optional[Callable[[], None]] = None
        self.on_face_lost_notify: Optional[Callable[[], None]] = None
        self.on_face_reacquired: Optional[Callable[[], None]] = None
        self.on_search_complete: Optional[Callable[[bool], None]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spin up the servo loop thread."""
        if self._running:
            return
        self._running = True
        with self._servo_lock:
            self._pid_yaw.reset()
            self._pid_pitch.reset()
            self._head_yaw = 0.0
            self._head_pitch = 0.0
            self._frames_without_face = 0

        # Reset search and feedback state
        self._search_active = False
        self._search_phase = "none"
        self._face_found_count = 0
        self._was_face_lost = False
        self._last_feedback_time = time.monotonic()

        # Reset FaceTracker to clear stale MediaPipe timestamps.
        # Without this, restarting the servo after a long pause (e.g.,
        # after a bring-object sequence) causes a permanent graph crash:
        # "Packet timestamp mismatch on a calculator"
        self._face_tracker.reset()

        # Command NAO to center head
        self._tcp.send_fire_and_forget(
            {"action": "move_head", "yaw": 0.0, "pitch": 0.0, "speed": 0.2}
        )

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Visual servo started (walk_enabled=%s).", self._walk_enabled)

    def stop(self) -> None:
        """Stop the servo loop and center the head."""
        self._running = False

        # Cancel search if active
        self._search_active = False
        self._search_phase = "none"

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        # Re-center head on stop
        self._tcp.send_fire_and_forget(
            {"action": "move_head", "yaw": 0.0, "pitch": 0.0, "speed": 0.15}
        )
        self._tcp.send_fire_and_forget({"action": "stop_walk"})
        log.info("Visual servo stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def walk_enabled(self) -> bool:
        return self._walk_enabled

    @walk_enabled.setter
    def walk_enabled(self, value: bool) -> None:
        old = self._walk_enabled
        self._walk_enabled = value
        if old and not value:
            # Switching from full-follow to head-only: stop walking
            self._tcp.send_fire_and_forget({"action": "stop_walk"})
            if self._search_active:
                self._end_search(found=False, notify=False)

    @property
    def last_face_result(self) -> Optional[FaceDetectionResult]:
        """Latest face detection from the servo loop (thread-safe read)."""
        return self._last_face_result

    @property
    def last_pid_error(self) -> tuple[float, float]:
        """Latest PID error (error_x, error_y) from the servo loop."""
        return self._last_pid_error

    @property
    def search_active(self) -> bool:
        """Whether the person search is currently running."""
        return self._search_active

    @property
    def search_phase(self) -> str:
        """Current search phase: 'none', 'head_scan', or 'body_rotate'."""
        return self._search_phase

    # ------------------------------------------------------------------
    # Servo loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        t_prev = time.monotonic()

        while self._running:
            t_now = time.monotonic()
            dt = t_now - t_prev
            t_prev = t_now

            frame = self._camera.read()
            if frame is None:
                time.sleep(SERVO_LOOP_INTERVAL_S)
                continue

            try:
                face = self._face_tracker.detect(frame)
            except Exception as exc:
                log.error("Face tracker crashed: %s — skipping frame", exc)
                self._frames_without_face += 1
                time.sleep(SERVO_LOOP_INTERVAL_S)
                continue

            # Cache for dashboard (thread-safe: single writer, atomic ref assign)
            self._last_face_result = face

            if face is not None:
                # ── Face IS visible ──
                self._frames_without_face = 0

                if self._search_active:
                    # During search: require N consecutive frames to confirm
                    self._face_found_count += 1
                    if self._face_found_count >= _SEARCH_FACE_CONFIRM_FRAMES:
                        self._end_search(found=True, notify=True)
                        # Search just ended — fall through to track_face
                    else:
                        # Still confirming — do NOT track (would conflict
                        # with search head/body commands)
                        pass
                        # skip to throttle/sync below

                if not self._search_active:
                    # Normal tracking (or search just ended above)

                    # Fire re-acquired callback if face was previously lost
                    if self._was_face_lost:
                        self._was_face_lost = False
                        if self.on_face_reacquired is not None:
                            try:
                                self.on_face_reacquired()
                            except Exception:
                                pass

                    self._track_face(face, dt)

                # Periodic feedback in head-only mode
                if (not self._walk_enabled
                        and self.on_tracking_feedback is not None
                        and not self._search_active):
                    if t_now - self._last_feedback_time > self._feedback_interval:
                        try:
                            self.on_tracking_feedback()
                        except Exception:
                            pass
                        self._last_feedback_time = t_now
            else:
                # ── Face NOT visible ──
                self._face_found_count = 0
                self._frames_without_face += 1

                if self._search_active:
                    self._continue_search()
                elif self._frames_without_face >= FACE_LOST_FRAME_THRESHOLD:
                    self._was_face_lost = True
                    self._on_face_lost()

            # BUG 9 fix: periodically sync head angles from NAO
            # (skip during search to avoid interference)
            if (not self._search_active
                    and time.monotonic() - self._last_angle_sync > self._angle_sync_interval):
                self._sync_head_angles()
                self._last_angle_sync = time.monotonic()

            # Throttle to target rate
            elapsed = time.monotonic() - t_now
            sleep_time = SERVO_LOOP_INTERVAL_S - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _track_face(self, face: FaceDetectionResult, dt: float) -> None:
        """Compute PID corrections and send head/body commands."""
        # Error: how far the face center is from the image center.
        # Positive error_x → face is to the left → need positive yaw (turn left)
        # Positive error_y → face is above center → need negative pitch (look up)
        error_x = 0.5 - face.cx   # positive = face is left of center
        error_y = 0.5 - face.cy   # positive = face is above center

        # Cache for dashboard
        self._last_pid_error = (error_x, error_y)

        # Deadzone — don't chase tiny jitter
        if abs(error_x) < SERVO_DEADZONE:
            error_x = 0.0
        if abs(error_y) < SERVO_DEADZONE:
            error_y = 0.0

        if error_x == 0.0 and error_y == 0.0:
            return

        # PID outputs (angular velocity adjustments)
        d_yaw = self._pid_yaw.compute(error_x, dt)
        d_pitch = self._pid_pitch.compute(-error_y, dt)  # invert pitch axis

        # Update estimated head angles
        new_yaw = float(np.clip(
            self._head_yaw + d_yaw, NAO_HEAD_YAW_MIN, NAO_HEAD_YAW_MAX
        ))
        new_pitch = float(np.clip(
            self._head_pitch + d_pitch, NAO_HEAD_PITCH_MIN, NAO_HEAD_PITCH_MAX
        ))

        self._head_yaw = new_yaw
        self._head_pitch = new_pitch

        # Send head command
        self._tcp.send_fire_and_forget({
            "action": "move_head",
            "yaw": round(new_yaw, 4),
            "pitch": round(new_pitch, 4),
            "speed": 0.15,
        })

        # Body realignment: ONLY if walking is enabled ("come here" mode)
        if self._walk_enabled and abs(self._head_yaw) > BODY_REALIGN_YAW_THRESHOLD_RAD:
            turn_dir = 1.0 if self._head_yaw > 0 else -1.0
            self._tcp.send_fire_and_forget({
                "action": "walk_toward",
                "x": 0.4,              # slow forward walk
                "y": 0.0,
                "theta": round(turn_dir * BODY_TURN_SPEED, 3),
            })
            # Gradually recenter head as body catches up
            self._head_yaw *= 0.7
            log.debug(
                "Body realign: yaw=%.2f → turn theta=%.2f",
                self._head_yaw, turn_dir * BODY_TURN_SPEED,
            )

    # ------------------------------------------------------------------
    # Face lost handling
    # ------------------------------------------------------------------

    def _on_face_lost(self) -> None:
        """Called when face hasn't been seen for several frames."""
        with self._servo_lock:
            self._pid_yaw.reset()
            self._pid_pitch.reset()
            self._frames_without_face = 0

        if self._walk_enabled and not self._search_active:
            # "Come here" mode: start person search
            self._start_search()
        else:
            # "Follow me" (head-only) or search already active
            self._tcp.send_fire_and_forget({"action": "stop_walk"})
            if not self._walk_enabled and self.on_face_lost_notify is not None:
                try:
                    self.on_face_lost_notify()
                except Exception:
                    pass
            log.debug("Face lost — PIDs reset, body stopped.")

    # ------------------------------------------------------------------
    # Person search (full-follow mode only)
    # ------------------------------------------------------------------

    def _start_search(self) -> None:
        """Begin searching for the person after face is lost."""
        self._search_active = True
        self._search_phase = "head_scan"
        self._search_start_time = time.monotonic()
        self._search_head_idx = 0
        self._search_last_move_time = 0.0
        self._face_found_count = 0

        self._tcp.send_fire_and_forget({"action": "stop_walk"})
        self._tcp.send_fire_and_forget(
            {"action": "say", "text": "I lost you. Let me look around."}
        )
        log.info("Person search started (head scan phase).")

    def _continue_search(self) -> None:
        """Called each frame while search is active and face is NOT visible."""
        now = time.monotonic()

        if self._search_phase == "head_scan":
            # Sweep head through angles, hold each for ~1 second
            if now - self._search_last_move_time > _SEARCH_HEAD_HOLD_S:
                if self._search_head_idx < len(_SEARCH_HEAD_ANGLES):
                    angle = _SEARCH_HEAD_ANGLES[self._search_head_idx]
                    self._tcp.send_fire_and_forget({
                        "action": "move_head",
                        "yaw": angle,
                        "pitch": 0.1,
                        "speed": 0.15,
                    })
                    self._search_head_idx += 1
                    self._search_last_move_time = now
                else:
                    # Head scan complete — transition to body rotation
                    self._search_phase = "body_rotate"
                    self._search_rotate_start = now
                    self._tcp.send_fire_and_forget({
                        "action": "move_head",
                        "yaw": 0.0, "pitch": 0.0, "speed": 0.15,
                    })
                    self._tcp.send_fire_and_forget(
                        {"action": "say",
                         "text": "Let me turn around to find you."}
                    )
                    log.info("Person search: head scan done → body rotation.")

        elif self._search_phase == "body_rotate":
            rotate_elapsed = now - self._search_rotate_start
            if rotate_elapsed < _SEARCH_ROTATE_TIMEOUT_S:
                # Keep rotating in place
                self._tcp.send_fire_and_forget({
                    "action": "set_walk_velocity",
                    "x": 0.0,
                    "y": 0.0,
                    "theta": _SEARCH_ROTATE_SPEED,
                })
            else:
                # Full rotation done — give up
                self._end_search(found=False, notify=True)
                return

        # Overall safety timeout
        if now - self._search_start_time > _SEARCH_TOTAL_TIMEOUT_S:
            self._end_search(found=False, notify=True)

    def _end_search(self, found: bool, notify: bool = True) -> None:
        """End the person search."""
        self._search_active = False
        self._search_phase = "none"
        self._face_found_count = 0

        self._tcp.send_fire_and_forget({"action": "stop_walk"})
        self._tcp.send_fire_and_forget({
            "action": "move_head", "yaw": 0.0, "pitch": 0.0, "speed": 0.15,
        })

        if found:
            self._tcp.send_fire_and_forget(
                {"action": "say",
                 "text": "I found you! I'll keep following."}
            )
            self._was_face_lost = False
            log.info("Person search: FOUND — resuming tracking.")
        else:
            if notify:
                self._tcp.send_fire_and_forget(
                    {"action": "say",
                     "text": "I can't find you. Please come to me."}
                )
            log.info("Person search: NOT FOUND — giving up.")
            if notify and self.on_search_complete is not None:
                try:
                    self.on_search_complete(False)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Head angle sync (BUG 9 fix)
    # ------------------------------------------------------------------

    def _sync_head_angles(self) -> None:
        """Sync local head-angle estimates from NAO (BUG 9 fix).

        Sends a ``query_state`` command to get the actual joint angles
        and overwrites the local PID estimates if drift exceeds 0.02 rad.
        This prevents accumulated integration error from desynchronising
        the head position over time.
        """
        try:
            resp = self._tcp.send_command({"action": "query_state"})
            if resp is None or resp.get("status") != "ok":
                return

            state = resp.get("state", {})
            actual_yaw = state.get("head_yaw")
            actual_pitch = state.get("head_pitch")

            if actual_yaw is None or actual_pitch is None:
                return

            drift = abs(self._head_yaw - actual_yaw) + abs(
                self._head_pitch - actual_pitch
            )
            if drift > 0.02:
                log.debug(
                    "Angle sync: yaw %.3f->%.3f, pitch %.3f->%.3f (drift=%.3f)",
                    self._head_yaw, actual_yaw,
                    self._head_pitch, actual_pitch,
                    drift,
                )
                with self._servo_lock:
                    self._head_yaw = actual_yaw
                    self._head_pitch = actual_pitch
        except Exception:
            pass  # non-critical — will retry next interval
