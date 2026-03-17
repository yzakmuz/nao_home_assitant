"""
dashboard.py -- Main OpenCV window compositor + event loop.

Renders all panels into a single 1280x800 window at ~30 FPS.
Handles keyboard events for hotkey command injection.
"""

from __future__ import annotations

import queue
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from gui.panels import (
    draw_audio_bar,
    draw_camera_panel,
    draw_console_panel,
    draw_header,
    draw_hotkey_bar,
    draw_robot_panel,
    draw_state_panel,
)
import gui.robot_visualizer as robot_viz


# ======================================================================
# Hotkey Mapping
# ======================================================================
HOTKEY_COMMANDS = {
    ord("1"): "follow me",
    ord("2"): "stop",
    ord("3"): "find my phone",
    ord("4"): "wave hello",
    ord("5"): "sit down",
    ord("6"): "stand up",
    ord("7"): "what do you see",
    ord("8"): "introduce yourself",
    ord("9"): "dance",
    ord("0"): "come here",
}

BG_COLOR = (30, 30, 30)

# Layout constants
CANVAS_W = 1280
CANVAS_H = 800
HEADER_H = 40
CAM_W = 660
CAM_H = 400
STATE_W = CANVAS_W - CAM_W
STATE_H = CAM_H
ROBOT_W = 350
ROBOT_H = 280
LOG_W = CANVAS_W - ROBOT_W
LOG_H = ROBOT_H
AUDIO_H = 35
HOTKEY_H = 40
BOTTOM_Y = HEADER_H + CAM_H + ROBOT_H


class Dashboard:
    """OpenCV-based live dashboard for the ElderGuard simulation."""

    def __init__(self, shared_state) -> None:
        """
        Args:
            shared_state: SharedSimState instance with all simulation data.
        """
        self._state = shared_state
        self._start_time = time.monotonic()
        self._action_log_display: List[Dict] = []
        self._max_action_log = 50
        # Mouse orbit state for 3D robot camera
        self._orbit_dragging = False
        self._orbit_last_x = 0
        self._orbit_last_y = 0

    def run(self) -> None:
        """Main dashboard loop. Blocks until quit."""
        cv2.namedWindow("ElderGuard Simulation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ElderGuard Simulation", CANVAS_W, CANVAS_H)
        cv2.setMouseCallback("ElderGuard Simulation", self._on_mouse)

        while not self._state.shutdown_event.is_set():
            snap = self._state.snapshot()
            self._drain_action_log()

            canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)

            # Header
            draw_header(canvas, CANVAS_W, snap.get("fsm_state", "IDLE"),
                         self._start_time)

            # Camera panel (top-left)
            draw_camera_panel(
                canvas, 0, HEADER_H, CAM_W, CAM_H,
                frame=snap.get("latest_frame"),
                face_result=snap.get("face_detection"),
                yolo_detections=snap.get("yolo_detections"),
                pose_keypoints=snap.get("pose_keypoints"),
                person_fall_detected=snap.get("person_fall_detected", False),
            )

            # State panel (top-right)
            draw_state_panel(
                canvas, CAM_W, HEADER_H, STATE_W, STATE_H, snap
            )

            # Robot panel (bottom-left)
            draw_robot_panel(
                canvas, 0, HEADER_H + CAM_H, ROBOT_W, ROBOT_H, snap
            )

            # Demo console (bottom-right)
            draw_console_panel(
                canvas, ROBOT_W, HEADER_H + CAM_H, LOG_W, ROBOT_H,
            )

            # Audio bar
            draw_audio_bar(
                canvas, 0, BOTTOM_Y, CANVAS_W, AUDIO_H, snap
            )

            # Hotkey bar
            draw_hotkey_bar(
                canvas, 0, BOTTOM_Y + AUDIO_H, CANVAS_W, HOTKEY_H
            )

            cv2.imshow("ElderGuard Simulation", canvas)

            key = cv2.waitKey(33) & 0xFF  # ~30 FPS
            if key != 255:
                self._handle_key(key)

        cv2.destroyAllWindows()

    def _handle_key(self, key: int) -> None:
        """Process keyboard input."""
        # Safety: release stuck drag on any key press
        self._orbit_dragging = False

        if key == ord("q"):
            self._state.shutdown_event.set()

        elif key == ord("f"):
            # Simulate robot fall (NAO hardware)
            self._state.inject_fall()

        elif key == ord("p"):
            # Simulate person fall (vision-based detection)
            self._state.inject_person_fall()

        elif key == ord("r"):
            # Recover from both fall types
            self._state.clear_fall()
            self._state.clear_person_fall()

        elif key == ord("d"):
            # Simulate disconnect (stop heartbeat for watchdog test)
            self._state.inject_disconnect()

        elif key in HOTKEY_COMMANDS:
            cmd_text = HOTKEY_COMMANDS[key]
            self._state.inject_command(cmd_text)

    def _on_mouse(self, event: int, mx: int, my: int,
                  flags: int, param) -> None:
        """Handle mouse events for 3D robot camera orbit and zoom.

        Left-click drag on the ROBOT PANEL orbits the 3D view.
        Mouse wheel on the robot panel zooms in/out.
        Restricted to robot panel bounds to avoid blocking hotkeys.
        """
        # Robot panel bounds
        rp_x = 0
        rp_y = HEADER_H + CAM_H
        rp_w = ROBOT_W
        rp_h = ROBOT_H
        in_robot_panel = (rp_x <= mx <= rp_x + rp_w
                          and rp_y <= my <= rp_y + rp_h)

        if event == cv2.EVENT_LBUTTONDOWN:
            if in_robot_panel:
                self._orbit_dragging = True
                self._orbit_last_x = mx
                self._orbit_last_y = my

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._orbit_dragging:
                dx = mx - self._orbit_last_x
                dy = my - self._orbit_last_y
                self._orbit_last_x = mx
                self._orbit_last_y = my
                yaw, pitch = robot_viz.get_view_angles()
                yaw += dx * 0.008
                pitch += dy * 0.008
                robot_viz.set_view_angles(yaw, pitch)

        elif event == cv2.EVENT_LBUTTONUP:
            self._orbit_dragging = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            if in_robot_panel:
                if flags > 0:
                    robot_viz.zoom_delta(-0.15)
                else:
                    robot_viz.zoom_delta(0.15)

    def _drain_action_log(self) -> None:
        """Move entries from the action_log queue to display list."""
        while True:
            try:
                entry = self._state.action_log.get_nowait()
                self._action_log_display.append(entry)
                if len(self._action_log_display) > self._max_action_log:
                    self._action_log_display.pop(0)
            except queue.Empty:
                break
