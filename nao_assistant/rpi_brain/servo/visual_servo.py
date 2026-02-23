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
       command is issued to realign the torso.

The servo loop runs in its own thread at ~15 Hz, started/stopped by
the main state machine.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

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
# Visual Servo Controller
# ======================================================================

class VisualServoController:
    """
    Runs a face-tracking servo loop in a background thread.

    Sends `move_head` and `walk_toward` JSON commands to the NAO via TCP.
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

        # Command NAO to center head
        self._tcp.send_fire_and_forget(
            {"action": "move_head", "yaw": 0.0, "pitch": 0.0, "speed": 0.2}
        )

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Visual servo started.")

    def stop(self) -> None:
        """Stop the servo loop and center the head."""
        self._running = False
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

            face = self._face_tracker.detect(frame)

            if face is not None:
                self._frames_without_face = 0
                self._track_face(face, dt)
            else:
                self._frames_without_face += 1
                if self._frames_without_face >= FACE_LOST_FRAME_THRESHOLD:
                    self._on_face_lost()

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

        # Body realignment: if head is turned too far, rotate the body
        if abs(self._head_yaw) > BODY_REALIGN_YAW_THRESHOLD_RAD:
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

        def _on_face_lost(self) -> None:
            """Called when face hasn't been seen for several frames."""
            with self._servo_lock:
                self._pid_yaw.reset()
                self._pid_pitch.reset()
                self._frames_without_face = 0
            self._tcp.send_fire_and_forget({"action": "stop_walk"})
            log.debug("Face lost — PIDs reset, body stopped.")
