"""
panels.py -- Panel renderers for the OpenCV dashboard.

Each draw_*() function renders into a region of the main canvas.
All coordinates are absolute (caller provides x, y, w, h).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from gui.event_bus import get_event_bus, SEV_WARNING, SEV_ERROR
from gui.robot_visualizer import draw_robot

# ======================================================================
# Color Palette (BGR)
# ======================================================================
BG = (30, 30, 30)
PANEL_BG = (45, 45, 45)
TEXT = (224, 224, 224)
TEXT_DIM = (140, 140, 140)
GREEN = (0, 200, 80)
YELLOW = (0, 200, 255)
RED = (0, 0, 220)
BLUE = (255, 160, 0)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SM = cv2.FONT_HERSHEY_PLAIN

# State color mapping
STATE_COLORS = {
    "IDLE": GREEN,
    "LISTENING": YELLOW,
    "VERIFYING": CYAN,
    "EXECUTING": BLUE,
    "SEARCHING": (0, 165, 255),  # orange
    "SHUTDOWN": RED,
}

CHANNEL_COLORS = {
    "idle": GREEN,
    "walking": YELLOW,
    "speaking": YELLOW,
    "animating": (0, 165, 255),
    "busy": RED,
}


def _fill_panel(canvas: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """Fill a panel region with the panel background color."""
    canvas[y:y + h, x:x + w] = PANEL_BG


def _draw_border(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                  color=(70, 70, 70)) -> None:
    cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), color, 1)


# ======================================================================
# Header Bar
# ======================================================================
def draw_header(canvas: np.ndarray, w: int, fsm_state: str,
                start_time: float) -> None:
    """Draw the title bar at the top."""
    h = 40
    _fill_panel(canvas, 0, 0, w, h)

    cv2.putText(canvas, "ELDERGUARD SIMULATION", (15, 28),
                 FONT, 0.7, WHITE, 2)

    # FSM state badge
    color = STATE_COLORS.get(fsm_state, TEXT)
    badge_text = f"[{fsm_state}]"
    (tw, th), _ = cv2.getTextSize(badge_text, FONT, 0.6, 2)
    bx = w - tw - 120
    cv2.putText(canvas, badge_text, (bx, 28), FONT, 0.6, color, 2)

    # Uptime
    elapsed = time.monotonic() - start_time
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)
    time_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
    cv2.putText(canvas, time_str, (w - 100, 28), FONT, 0.5, TEXT_DIM, 1)

    _draw_border(canvas, 0, 0, w, h)


# ======================================================================
# Camera Panel
# ======================================================================
def draw_camera_panel(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                       frame: Optional[np.ndarray],
                       face_result: Optional[Dict] = None,
                       yolo_detections: Optional[List[Dict]] = None,
                       pose_keypoints: Optional[List[Dict]] = None,
                       person_fall_detected: bool = False) -> None:
    """Draw the live camera feed with overlays."""
    _fill_panel(canvas, x, y, w, h)

    if frame is not None:
        # Resize frame to fit panel (with padding)
        pad = 5
        fw = w - 2 * pad
        fh = h - 25 - pad  # leave room for label
        resized = cv2.resize(frame, (fw, fh))
        canvas[y + 20:y + 20 + fh, x + pad:x + pad + fw] = resized

        # Pose skeleton overlay (drawn UNDER face/yolo boxes)
        if pose_keypoints:
            _draw_skeleton(canvas, pose_keypoints,
                           x + pad, y + 20, fw, fh,
                           color=RED if person_fall_detected else GREEN)

        # Face detection overlay
        if face_result is not None:
            fcx = int(face_result.get("cx", 0.5) * fw) + x + pad
            fcy = int(face_result.get("cy", 0.5) * fh) + y + 20
            fw_box = int(face_result.get("width", 0.1) * fw)
            fh_box = int(face_result.get("height", 0.1) * fh)
            cv2.rectangle(canvas,
                           (fcx - fw_box // 2, fcy - fh_box // 2),
                           (fcx + fw_box // 2, fcy + fh_box // 2),
                           GREEN, 2)
            conf = face_result.get("confidence", 0)
            cv2.putText(canvas, f"Face {conf:.2f}",
                         (fcx - fw_box // 2, fcy - fh_box // 2 - 5),
                         FONT_SM, 1.0, GREEN, 1)

        # YOLO detection overlays
        if yolo_detections:
            for det in yolo_detections[:5]:
                dcx = int(det.get("cx_norm", 0.5) * fw) + x + pad
                dcy = int(det.get("cy_norm", 0.5) * fh) + y + 20
                dw = int(det.get("w_norm", 0.1) * fw)
                dh = int(det.get("h_norm", 0.1) * fh)
                cv2.rectangle(canvas,
                               (dcx - dw // 2, dcy - dh // 2),
                               (dcx + dw // 2, dcy + dh // 2),
                               CYAN, 2)
                label = f"{det.get('class_name', '?')} {det.get('confidence', 0):.2f}"
                cv2.putText(canvas, label,
                             (dcx - dw // 2, dcy - dh // 2 - 5),
                             FONT_SM, 1.0, CYAN, 1)

        # Fall detected banner (flashing red)
        if person_fall_detected:
            flash = int(time.time() * 4) % 2 == 0  # ~4 Hz flash
            if flash:
                banner_h = 30
                overlay = canvas[y + 20:y + 20 + banner_h,
                                 x + pad:x + pad + fw].copy()
                cv2.rectangle(canvas,
                               (x + pad, y + 20),
                               (x + pad + fw, y + 20 + banner_h),
                               (0, 0, 180), -1)
                cv2.putText(canvas, "!! FALL DETECTED !!",
                             (x + pad + fw // 2 - 100, y + 20 + 22),
                             FONT, 0.7, WHITE, 2)
    else:
        cv2.putText(canvas, "No Camera Feed", (x + 20, y + h // 2),
                     FONT, 0.6, TEXT_DIM, 1)

    # Panel label
    cv2.putText(canvas, "CAMERA", (x + 10, y + 15), FONT_SM, 1.2, TEXT_DIM, 1)
    _draw_border(canvas, x, y, w, h)


# Skeleton connections for drawing (pairs of landmark names)
_SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("right_shoulder", "right_hip"),
    ("right_hip", "left_hip"),
    ("left_hip", "left_shoulder"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def _draw_skeleton(canvas: np.ndarray, keypoints: List[Dict],
                   ox: int, oy: int, fw: int, fh: int,
                   color=(0, 200, 80)) -> None:
    """Draw skeleton lines and keypoint dots on the camera panel."""
    kp_map = {kp["name"]: kp for kp in keypoints}

    # Draw connections
    for name_a, name_b in _SKELETON_CONNECTIONS:
        a = kp_map.get(name_a)
        b = kp_map.get(name_b)
        if a is None or b is None:
            continue
        if a.get("visibility", 0) < 0.5 or b.get("visibility", 0) < 0.5:
            continue
        pt1 = (int(a["x"] * fw) + ox, int(a["y"] * fh) + oy)
        pt2 = (int(b["x"] * fw) + ox, int(b["y"] * fh) + oy)
        cv2.line(canvas, pt1, pt2, color, 2)

    # Draw keypoint dots
    for kp in keypoints:
        if kp.get("visibility", 0) < 0.5:
            continue
        pt = (int(kp["x"] * fw) + ox, int(kp["y"] * fh) + oy)
        cv2.circle(canvas, pt, 4, color, -1)
        cv2.circle(canvas, pt, 4, (255, 255, 255), 1)


# ======================================================================
# State Panel
# ======================================================================
def draw_state_panel(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                      snap: Dict[str, Any]) -> None:
    """Draw the FSM state and NAO channel status panel."""
    _fill_panel(canvas, x, y, w, h)

    line_y = y + 25
    spacing = 24

    # FSM State
    fsm = snap.get("fsm_state", "IDLE")
    color = STATE_COLORS.get(fsm, TEXT)
    cv2.putText(canvas, f"FSM State:", (x + 10, line_y), FONT, 0.5, TEXT_DIM, 1)
    cv2.putText(canvas, f"[{fsm}]", (x + 120, line_y), FONT, 0.55, color, 2)
    line_y += spacing

    # Posture
    posture = snap.get("nao_posture", "unknown")
    cv2.putText(canvas, f"Posture:", (x + 10, line_y), FONT, 0.5, TEXT_DIM, 1)
    cv2.putText(canvas, posture, (x + 120, line_y), FONT, 0.5, TEXT, 1)
    line_y += spacing + 5

    # Channel states
    cv2.putText(canvas, "Channels:", (x + 10, line_y), FONT, 0.5, TEXT_DIM, 1)
    line_y += spacing

    channels = [
        ("HEAD", snap.get("nao_head", "idle")),
        ("LEGS", snap.get("nao_legs", "idle")),
        ("SPEECH", snap.get("nao_speech", "idle")),
        ("ARMS", snap.get("nao_arms", "idle")),
    ]
    for name, state in channels:
        ch_color = CHANNEL_COLORS.get(state, TEXT)
        # Status dot
        cv2.circle(canvas, (x + 25, line_y - 4), 5, ch_color, -1)
        cv2.putText(canvas, f"{name}:", (x + 38, line_y), FONT_SM, 1.1, TEXT, 1)
        cv2.putText(canvas, state, (x + 110, line_y), FONT_SM, 1.1, ch_color, 1)
        line_y += spacing - 2

    line_y += 8

    # Head angles
    yaw = snap.get("head_yaw", 0.0)
    pitch = snap.get("head_pitch", 0.0)
    cv2.putText(canvas, f"Head Yaw:  {yaw:+.2f} rad", (x + 10, line_y),
                 FONT, 0.45, TEXT, 1)
    line_y += spacing - 4
    cv2.putText(canvas, f"Head Pitch: {pitch:+.2f} rad", (x + 10, line_y),
                 FONT, 0.45, TEXT, 1)
    line_y += spacing

    # PID error
    ex = snap.get("pid_error_x", 0.0)
    ey = snap.get("pid_error_y", 0.0)
    cv2.putText(canvas, f"PID Err: ({ex:.2f}, {ey:.2f})", (x + 10, line_y),
                 FONT, 0.45, TEXT_DIM, 1)
    line_y += spacing

    # Servo status + mode (Improvement 5)
    servo_mode = snap.get("servo_mode", "off")
    if servo_mode == "head_only":
        servo_text = "HEAD ONLY"
        servo_color = CYAN
    elif servo_mode == "full_follow":
        searching = snap.get("servo_searching", False)
        if searching:
            servo_text = "SEARCHING"
            servo_color = YELLOW
        else:
            servo_text = "FOLLOWING"
            servo_color = GREEN
    else:
        servo_text = "OFF"
        servo_color = RED
    cv2.putText(canvas, f"Servo: {servo_text}", (x + 10, line_y),
                 FONT, 0.5, servo_color, 1)
    line_y += spacing

    # Fall monitor status (Improvement 4)
    fall_state = snap.get("fall_state", "INACTIVE")
    _FALL_STATE_COLORS = {
        "MONITORING": GREEN,
        "CALIBRATING": YELLOW,
        "UNCALIBRATED": TEXT_DIM,
        "TRIGGERED": RED,
        "RECOVERY": YELLOW,
        "INACTIVE": TEXT_DIM,
    }
    fall_color = _FALL_STATE_COLORS.get(fall_state, TEXT_DIM)
    # Flash the dot when triggered
    if fall_state == "TRIGGERED" and int(time.time() * 4) % 2 == 0:
        fall_color = (0, 0, 255)  # bright red flash
    cv2.circle(canvas, (x + 25, line_y - 4), 5, fall_color, -1)
    cv2.putText(canvas, f"Fall:", (x + 38, line_y), FONT_SM, 1.1, TEXT, 1)
    cv2.putText(canvas, fall_state, (x + 80, line_y), FONT_SM, 1.1, fall_color, 1)
    line_y += spacing - 2

    # Fall score (only when monitoring or triggered)
    if fall_state in ("MONITORING", "TRIGGERED", "RECOVERY"):
        fall_score = snap.get("fall_score", 0.0)
        score_color = RED if fall_score > 0.6 else (YELLOW if fall_score > 0.3 else GREEN)
        cv2.putText(canvas, f"Fall Score: {fall_score:.2f}", (x + 10, line_y),
                     FONT, 0.45, score_color, 1)

    # ── Voice Commands Reference — RIGHT column (Improvement 5) ──
    # Positioned in the empty right half of the state panel
    rc_x = x + w // 2 + 10                   # right column start x
    rc_y = y + 25                             # start from top of panel
    cv2.line(canvas, (rc_x - 8, y + 10), (rc_x - 8, y + h - 10),
             (70, 70, 70), 1)                 # vertical divider

    cv2.putText(canvas, "VOICE COMMANDS",
                 (rc_x, rc_y), FONT_SM, 1.0, TEXT_DIM, 1)
    rc_y += 14
    cv2.putText(canvas, '("hey nao" + command)',
                 (rc_x, rc_y), FONT_SM, 0.85, (100, 100, 100), 1)
    rc_y += 18

    _ALL_CMDS = [
        ("follow me",     "track (head only)"),
        ("come here",     "follow + walk"),
        ("stop",          "stop all"),
        ("sit down",      ""),
        ("stand up",      ""),
        ("find my ___",   "keys/phone/..."),
        ("bring me my ___", "pickup + deliver"),
        ("go to it",      "walk to object"),
        ("what do you see", ""),
        ("wave hello",    ""),
        ("say hello",     ""),
        ("dance",         ""),
        ("look left/right", ""),
        ("turn around",   ""),
        ("introduce yourself", ""),
        ("i'm okay",      "after fall alert"),
    ]
    for cmd, hint in _ALL_CMDS:
        if hint:
            line = f"{cmd}  ({hint})"
        else:
            line = cmd
        cv2.putText(canvas, line, (rc_x, rc_y),
                     FONT_SM, 0.85, TEXT_DIM, 1)
        rc_y += 15

    _draw_border(canvas, x, y, w, h)


# ======================================================================
# Robot Visualization Panel
# ======================================================================
def draw_robot_panel(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                      snap: Dict[str, Any]) -> None:
    """Draw the 2D stick-figure robot visualization."""
    _fill_panel(canvas, x, y, w, h)

    cv2.putText(canvas, "ROBOT", (x + 10, y + 15), FONT_SM, 1.2, TEXT_DIM, 1)

    draw_robot(
        canvas, x + 10, y + 20, w - 20, h - 30,
        head_yaw=snap.get("head_yaw", 0.0),
        head_pitch=snap.get("head_pitch", 0.0),
        posture=snap.get("nao_posture", "standing"),
        legs_state=snap.get("nao_legs", "idle"),
        arms_state=snap.get("nao_arms", "idle"),
        speech_state=snap.get("nao_speech", "idle"),
    )

    _draw_border(canvas, x, y, w, h)


# ======================================================================
# Command Log Panel
# ======================================================================
def draw_log_panel(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                    command_log: List[Dict], action_log: List[Dict]) -> None:
    """Draw scrolling command and action logs."""
    _fill_panel(canvas, x, y, w, h)

    cv2.putText(canvas, "COMMAND LOG", (x + 10, y + 15), FONT_SM, 1.2, TEXT_DIM, 1)

    line_y = y + 35
    max_lines = (h - 40) // 18

    # Merge and show recent entries
    all_entries = []
    for entry in command_log[-max_lines:]:
        ts = entry.get("timestamp", "")
        action = entry.get("action", "?")
        text = entry.get("text", "")
        status = entry.get("status", "")
        line = f"{ts} {action}"
        if text:
            line += f' "{text[:25]}"'
        all_entries.append((line, status))

    for entry in action_log[-max(0, max_lines - len(all_entries)):]:
        ts = entry.get("timestamp", "")
        method = entry.get("method", "?")
        proxy = entry.get("proxy", "")
        line = f"{ts} [{proxy}] {method}"
        params = entry.get("params", {})
        if "text" in params:
            line += f' "{str(params["text"])[:20]}"'
        all_entries.append((line, "ok"))

    # Show most recent entries
    for line_text, status in all_entries[-max_lines:]:
        color = GREEN if status in ("ok", "accepted", "") else RED
        cv2.putText(canvas, line_text[:50], (x + 10, line_y),
                     FONT_SM, 0.95, color, 1)
        line_y += 18

    _draw_border(canvas, x, y, w, h)


# ======================================================================
# Demo Console Panel  (Improvement 3 — replaces Command Log in the GUI)
# ======================================================================
_CONSOLE_CAT_COLORS = {
    "STT":    (255, 255, 0),      # cyan
    "VERIFY": (0, 220, 220),      # warm yellow
    "FSM":    (255, 160, 0),      # blue
    "CMD":    (255, 200, 100),    # light cyan
    "NAO":    (100, 220, 100),    # green
    "SYS":    (180, 180, 180),    # light gray
}


def draw_console_panel(canvas: np.ndarray, x: int, y: int,
                       w: int, h: int) -> None:
    """Draw the rich demo console panel with categorized events."""
    _fill_panel(canvas, x, y, w, h)

    cv2.putText(canvas, "DEMO CONSOLE", (x + 10, y + 15),
                 FONT_SM, 1.2, TEXT_DIM, 1)

    line_y = y + 35
    max_lines = (h - 40) // 18

    bus = get_event_bus()
    events = bus.recent(max_lines)

    # Approximate max chars that fit in the panel (HERSHEY_PLAIN ~8px/char)
    max_chars = max(20, (w - 20) // 8)

    for ev in events:
        # Pick color: severity overrides category
        if ev.severity == SEV_ERROR:
            color = RED
        elif ev.severity == SEV_WARNING:
            color = YELLOW
        else:
            color = _CONSOLE_CAT_COLORS.get(ev.category, TEXT)

        # Format: HH:MM:SS [TAG]     message
        ts = ev.timestamp[:8]  # HH:MM:SS (drop millis for compactness)
        tag = f"[{ev.category}]"
        line = f"{ts} {tag:<10s}{ev.message}"

        if len(line) > max_chars:
            line = line[:max_chars - 2] + ".."

        cv2.putText(canvas, line, (x + 10, line_y),
                     FONT_SM, 0.95, color, 1)
        line_y += 18

    _draw_border(canvas, x, y, w, h)


# ======================================================================
# Audio Bar
# ======================================================================
def draw_audio_bar(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                    snap: Dict[str, Any]) -> None:
    """Draw the audio status bar at the bottom."""
    _fill_panel(canvas, x, y, w, h)

    # Audio level
    level = snap.get("audio_level", 0.0)
    bar_w = int(min(level, 1.0) * 120)
    cv2.putText(canvas, "MIC:", (x + 10, y + 22), FONT, 0.45, TEXT_DIM, 1)
    cv2.rectangle(canvas, (x + 60, y + 10), (x + 60 + 120, y + 25),
                   (60, 60, 60), -1)
    if bar_w > 0:
        cv2.rectangle(canvas, (x + 60, y + 10), (x + 60 + bar_w, y + 25),
                       GREEN, -1)

    # STT text
    stt_text = snap.get("stt_text", "")
    cv2.putText(canvas, f'STT: "{stt_text}"', (x + 200, y + 22),
                 FONT, 0.45, TEXT, 1)

    # Speaker verify
    score = snap.get("speaker_score", 0.0)
    accepted = snap.get("speaker_accepted", False)
    verify_color = GREEN if accepted else RED
    verify_text = "OK" if accepted else "REJ"
    cv2.putText(canvas, f"VERIFY: {score:.2f} [{verify_text}]",
                 (x + 480, y + 22), FONT, 0.45, verify_color, 1)

    _draw_border(canvas, x, y, w, h)


# ======================================================================
# Hotkey Bar
# ======================================================================
def draw_hotkey_bar(canvas: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """Draw the hotkey reference bar."""
    _fill_panel(canvas, x, y, w, h)

    line1 = "HOTKEYS: 1=follow 2=stop 3=find 4=wave 5=sit 6=stand 7=see 8=intro 9=dance 0=come"
    line2 = "         f=robot-fall  p=person-fall  r=recover  d=disconnect  q=quit  |  drag=orbit"

    cv2.putText(canvas, line1, (x + 10, y + 16), FONT_SM, 0.95, TEXT_DIM, 1)
    cv2.putText(canvas, line2, (x + 10, y + 32), FONT_SM, 0.95, TEXT_DIM, 1)
    _draw_border(canvas, x, y, w, h)
