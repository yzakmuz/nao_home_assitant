"""
robot_visualizer.py -- 3D wireframe NAO robot renderer (in-place).

Renders a 3D wireframe NAO robot using perspective projection with OpenCV.
The robot stays centered in the panel but all joint animations are visible:
head yaw/pitch rotation, walking leg cycle, arm wave, dance, posture
changes (standing/sitting/resting/fallen), and speech mouth animation.

Features:
  - Smooth posture transitions (lerp with ease-in-out over ~1 second)
  - Interactive camera orbit (mouse drag to rotate, scroll to zoom)

Drop-in replacement for the original 2D stick-figure renderer -- same
function signature, same inputs, same panel integration.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import cv2
import numpy as np

# ======================================================================
# Colors (BGR)
# ======================================================================
BODY_COLOR = (200, 200, 200)
HEAD_COLOR = (220, 220, 230)
EYE_COLOR = (0, 180, 255)
JOINT_COLOR = (120, 120, 120)
WALK_ARROW_COLOR = (0, 200, 80)
FALLEN_COLOR = (0, 0, 220)
GROUND_COLOR = (55, 55, 55)
SHADOW_COLOR = (38, 38, 38)
ANIM_ARM_COLOR = (0, 200, 255)      # animated arms
ANIM_LEG_COLOR = (200, 220, 200)    # animated legs (slight green tint)

# Filled body part colors (Step 2 upgrade)
FILL_BODY = (210, 210, 215)        # torso + neck fill (white-ish)
FILL_LIMB = (195, 195, 200)        # default limb fill (slightly darker)
FILL_LEGS_ACTIVE = (195, 215, 195) # legs during walk/dance (green tint)
FILL_ARMS_ACTIVE = (195, 215, 245) # arms during wave (cyan tint)
OUTLINE = (120, 120, 125)          # body part outlines
FOOT_COLOR = (180, 180, 185)       # feet rectangles
EAR_COLOR = (150, 150, 155)        # head ear bumps
CHEST_DOT = (180, 100, 30)         # chest LED indicator (blue-ish dot)

# ======================================================================
# NAO V5 Skeleton (meters, standing, origin at feet center)
#   X = right,  Y = forward (depth),  Z = up
# ======================================================================
_NAO: Dict[str, np.ndarray] = {
    "head":       np.array([0.0,    0.0, 0.540]),
    "neck":       np.array([0.0,    0.0, 0.440]),
    "l_shoulder": np.array([-0.098, 0.0, 0.440]),
    "r_shoulder": np.array([ 0.098, 0.0, 0.440]),
    "l_elbow":    np.array([-0.105, 0.0, 0.335]),
    "r_elbow":    np.array([ 0.105, 0.0, 0.335]),
    "l_hand":     np.array([-0.105, 0.0, 0.230]),
    "r_hand":     np.array([ 0.105, 0.0, 0.230]),
    "hip":        np.array([0.0,    0.0, 0.220]),
    "l_hip":      np.array([-0.050, 0.0, 0.220]),
    "r_hip":      np.array([ 0.050, 0.0, 0.220]),
    "l_knee":     np.array([-0.050, 0.0, 0.110]),
    "r_knee":     np.array([ 0.050, 0.0, 0.110]),
    "l_foot":     np.array([-0.055, 0.0, 0.000]),
    "r_foot":     np.array([ 0.055, 0.0, 0.000]),
}

# Bone connections  (drawn in this order for basic depth-sort)
_BONES_LEGS = [
    ("hip", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_foot"),
    ("hip", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_foot"),
]
_BONES_TORSO = [("neck", "hip")]
_BONES_ARMS = [
    ("neck", "l_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_hand"),
    ("neck", "r_shoulder"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_hand"),
]
_BONES_HEAD = [("head", "neck")]
_ALL_BONES = _BONES_LEGS + _BONES_TORSO + _BONES_ARMS + _BONES_HEAD

# Joint radii for drawing
_MAJOR_JOINTS = {"l_shoulder", "r_shoulder", "hip", "l_hip", "r_hip", "neck"}

# Body part definitions for filled polygon rendering (Step 2)
# Each limb: (joint_a, joint_b, half_width_meters, color_key)
_LIMB_PARTS = [
    ("l_shoulder", "l_elbow", 0.018, "arms"),   # left upper arm
    ("r_shoulder", "r_elbow", 0.018, "arms"),   # right upper arm
    ("l_elbow",    "l_hand",  0.015, "arms"),   # left lower arm
    ("r_elbow",    "r_hand",  0.015, "arms"),   # right lower arm
    ("l_hip",      "l_knee",  0.022, "legs"),   # left upper leg
    ("r_hip",      "r_knee",  0.022, "legs"),   # right upper leg
    ("l_knee",     "l_foot",  0.020, "legs"),   # left lower leg
    ("r_knee",     "r_foot",  0.020, "legs"),   # right lower leg
    # Neck is handled separately in _draw_filled_body (doesn't follow head rotation)
]
# Torso uses 4 existing joints as corners (trapezoid shape)
_TORSO_CORNERS = ["l_shoulder", "r_shoulder", "r_hip", "l_hip"]
_TORSO_HALF_DEPTH = 0.035     # ~7cm total depth for torso box


# ======================================================================
# Mutable View State (updated by mouse orbit / zoom)
# ======================================================================
_view_yaw = -0.35      # slight turn for 3/4 view, front facing camera
_view_pitch = 0.20     # slight downward tilt
_persp_d = 2.5         # perspective focal distance (larger = less distortion)

# Per-frame shared view rotation matrix (set once in draw_robot, used by all)
_frame_view_rot: np.ndarray = np.eye(3, dtype=np.float64)


def get_view_angles() -> Tuple[float, float]:
    """Return current (yaw, pitch) view angles in radians."""
    return _view_yaw, _view_pitch


def set_view_angles(yaw: float, pitch: float) -> None:
    """Set view orbit angles. Pitch clamped to [-1.2, 1.2] rad."""
    global _view_yaw, _view_pitch
    _view_yaw = yaw
    _view_pitch = max(-1.2, min(1.2, pitch))


def get_zoom() -> float:
    """Return current perspective distance (zoom level)."""
    return _persp_d


def set_zoom(d: float) -> None:
    """Set perspective distance. Clamped to [1.0, 6.0]."""
    global _persp_d
    _persp_d = max(1.0, min(6.0, d))


def zoom_delta(delta: float) -> None:
    """Adjust zoom by delta (positive = zoom out, negative = zoom in)."""
    set_zoom(_persp_d + delta)


# ======================================================================
# Rotation helpers
# ======================================================================

def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# ======================================================================
# 3D  →  2D projection  (recomputes view rotation each frame)
# ======================================================================

def _project(joints: Dict[str, np.ndarray],
             cx: int, cy: int, scale: float) -> Dict[str, Tuple[int, int]]:
    """Project 3D joint dict to 2D canvas coordinates.

    ``cx, cy`` is the on-screen location of the origin (feet center).
    Uses the shared ``_frame_view_rot`` set once per frame in ``draw_robot()``.
    """
    view_rot = _frame_view_rot
    out: Dict[str, Tuple[int, int]] = {}
    for name, pt in joints.items():
        r = view_rot @ pt
        depth = r[1] + _persp_d
        if depth < 0.1:
            depth = 0.1
        f = scale * _persp_d / depth
        out[name] = (int(cx + r[0] * f), int(cy - r[2] * f))
    return out


# ======================================================================
# Smooth Posture Transition System
# ======================================================================

def _ease_in_out(t: float) -> float:
    """Smoothstep ease-in-out: maps [0,1] → [0,1] with smooth acceleration."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp_joints(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray],
                 alpha: float) -> Dict[str, np.ndarray]:
    """Linearly interpolate between two joint dictionaries."""
    result: Dict[str, np.ndarray] = {}
    for name in a:
        if name in b:
            result[name] = a[name] * (1.0 - alpha) + b[name] * alpha
        else:
            result[name] = a[name].copy()
    return result


class _TransitionState:
    """Tracks posture transition animation state."""
    __slots__ = ('prev_posture', 'snapshot_joints', 'snapshot_scale',
                 'start_time', 'duration', 'active')

    def __init__(self) -> None:
        self.prev_posture: str | None = None
        self.snapshot_joints: Dict[str, np.ndarray] | None = None
        self.snapshot_scale: float = 1.0
        self.start_time: float = 0.0
        self.duration: float = 1.0   # seconds for full transition
        self.active: bool = False


_trans = _TransitionState()

# Last-drawn joints (used as snapshot source for mid-transition changes)
_last_drawn_joints: Dict[str, np.ndarray] | None = None
_last_drawn_scale: float = 1.0


# ======================================================================
# Pose builders  (return a dict[str, ndarray] of 3-D joint positions)
# ======================================================================

def _base_joints() -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in _NAO.items()}


def _apply_head_rotation(joints: Dict[str, np.ndarray],
                          yaw: float, pitch: float) -> None:
    """Rotate the head around the neck joint by *yaw* and *pitch*."""
    neck = joints["neck"]
    offset = joints["head"] - neck
    rot = _rot_z(yaw) @ _rot_x(-pitch)
    joints["head"] = neck + rot @ offset


def _apply_walk_animation(joints: Dict[str, np.ndarray], t: float) -> None:
    """Sinusoidal walking gait: leg stride + body sway + counter-arm swing."""
    phase = t * 3.0
    stride = 0.030    # forward/back amplitude
    lift = 0.018      # knee lift

    for prefix, sign in (("l_", 1.0), ("r_", -1.0)):
        s = math.sin(phase + (0 if sign > 0 else math.pi))
        joints[prefix + "knee"][1] += stride * s
        joints[prefix + "foot"][1] += stride * s * 1.2
        joints[prefix + "knee"][2] += lift * max(0.0, s)
        joints[prefix + "foot"][2] += lift * 0.4 * max(0.0, s)

    # Subtle body sway
    sway = 0.005 * math.sin(phase)
    for name in joints:
        if "foot" not in name:
            joints[name][0] += sway

    # Counter-arm swing (subtle, unless arms override it)
    arm_sw = 0.018 * math.sin(phase)
    for part in ("elbow", "hand"):
        joints["l_" + part][1] -= arm_sw
        joints["r_" + part][1] += arm_sw


def _apply_dance_animation(joints: Dict[str, np.ndarray], t: float) -> None:
    """Bouncy dance: lateral hip sway + alternating knee bends."""
    phase = t * 4.0
    # Hip sway
    sway = 0.015 * math.sin(phase)
    for name in joints:
        if "foot" not in name:
            joints[name][0] += sway

    # Bounce (whole body up-down)
    bounce = 0.012 * abs(math.sin(phase * 2))
    for name in joints:
        if "foot" not in name:
            joints[name][2] += bounce

    # Alternating knee bend
    for prefix, sign in (("l_", 1.0), ("r_", -1.0)):
        bend = 0.02 * max(0, math.sin(phase + (0 if sign > 0 else math.pi)))
        joints[prefix + "knee"][1] -= bend
        joints[prefix + "knee"][2] += bend * 0.5


def _apply_arm_animation(joints: Dict[str, np.ndarray], t: float) -> None:
    """Right arm waves up, left arm sways gently."""
    wt = t * 5.0
    upper = 0.105
    lower = 0.105

    # Right arm: wave up
    angle1 = 0.8 + 0.35 * math.sin(wt)
    joints["r_elbow"][0] = joints["r_shoulder"][0] + upper * math.sin(angle1)
    joints["r_elbow"][2] = joints["r_shoulder"][2] + upper * math.cos(angle1)

    angle2 = angle1 + 0.4 * math.sin(wt * 1.3)
    joints["r_hand"][0] = joints["r_elbow"][0] + lower * math.sin(angle2)
    joints["r_hand"][2] = joints["r_elbow"][2] + lower * math.cos(angle2)

    # Left arm: subtle swing
    joints["l_elbow"][1] += 0.015 * math.sin(wt * 0.7)
    joints["l_hand"][1] += 0.025 * math.sin(wt * 0.7)


# --------------- Standing ------------------------------------------------

def _standing_joints(head_yaw: float, head_pitch: float,
                     legs_state: str, arms_state: str,
                     t: float) -> Dict[str, np.ndarray]:
    joints = _base_joints()

    if legs_state == "walking":
        _apply_walk_animation(joints, t)
    elif legs_state == "animating":
        _apply_dance_animation(joints, t)

    if arms_state == "animating":
        _apply_arm_animation(joints, t)

    _apply_head_rotation(joints, head_yaw, head_pitch)
    return joints


# --------------- Sitting --------------------------------------------------

def _sitting_joints(head_yaw: float, head_pitch: float,
                    arms_state: str, t: float) -> Dict[str, np.ndarray]:
    joints = _base_joints()
    drop = 0.11

    # Lower upper body
    for n in ("head", "neck", "l_shoulder", "r_shoulder",
              "l_elbow", "r_elbow", "l_hand", "r_hand", "hip"):
        joints[n][2] -= drop

    hip_z = joints["hip"][2]
    joints["l_hip"][2] = hip_z
    joints["r_hip"][2] = hip_z

    # Knees forward
    joints["l_knee"] = np.array([-0.050, -0.100, hip_z - 0.010])
    joints["r_knee"] = np.array([ 0.050, -0.100, hip_z - 0.010])
    joints["l_foot"] = np.array([-0.050, -0.100, 0.000])
    joints["r_foot"] = np.array([ 0.050, -0.100, 0.000])

    # Hands rest on lap
    joints["l_elbow"] = np.array([-0.080, -0.020, hip_z + 0.060])
    joints["r_elbow"] = np.array([ 0.080, -0.020, hip_z + 0.060])
    joints["l_hand"]  = np.array([-0.060, -0.040, hip_z + 0.020])
    joints["r_hand"]  = np.array([ 0.060, -0.040, hip_z + 0.020])

    if arms_state == "animating":
        _apply_arm_animation(joints, t)

    _apply_head_rotation(joints, head_yaw, head_pitch)
    return joints


# --------------- Resting (crouched) ---------------------------------------

def _resting_joints() -> Dict[str, np.ndarray]:
    joints = _base_joints()
    drop = 0.19

    for n in ("head", "neck", "l_shoulder", "r_shoulder",
              "l_elbow", "r_elbow", "l_hand", "r_hand", "hip"):
        joints[n][2] -= drop

    hip_z = joints["hip"][2]
    joints["l_hip"][2] = hip_z
    joints["r_hip"][2] = hip_z

    joints["l_knee"] = np.array([-0.050, -0.080, 0.055])
    joints["r_knee"] = np.array([ 0.050, -0.080, 0.055])
    joints["l_foot"] = np.array([-0.055, -0.060, 0.000])
    joints["r_foot"] = np.array([ 0.055, -0.060, 0.000])

    joints["l_elbow"] = np.array([-0.070, -0.020, hip_z + 0.055])
    joints["r_elbow"] = np.array([ 0.070, -0.020, hip_z + 0.055])
    joints["l_hand"]  = np.array([-0.060, -0.040, hip_z + 0.015])
    joints["r_hand"]  = np.array([ 0.060, -0.040, hip_z + 0.015])

    joints["head"][1] -= 0.020
    joints["neck"][1] -= 0.010
    return joints


# --------------- Fallen ---------------------------------------------------

def _fallen_joints() -> Dict[str, np.ndarray]:
    """Joints rotated 90 degrees so the robot lies on its side."""
    joints = _base_joints()
    fall_rot = _rot_x(math.pi / 2)
    tilt = _rot_z(0.20)
    for name in joints:
        joints[name] = tilt @ (fall_rot @ joints[name])
        joints[name][2] = max(joints[name][2], 0.0) + 0.015
    return joints


# ======================================================================
# Drawing helpers
# ======================================================================

def _draw_ground(canvas: np.ndarray, cx: int, gy: int, w: int) -> None:
    """Subtle perspective ground grid."""
    hw = w // 3
    # Base line
    cv2.line(canvas, (cx - hw, gy), (cx + hw, gy), GROUND_COLOR, 1)
    # Perspective depth lines
    van_y = gy - 35
    for off in range(-hw, hw + 1, hw // 2):
        ex = cx + int(off * 0.30)
        cv2.line(canvas, (cx + off, gy), (ex, van_y), GROUND_COLOR, 1)
    # Cross lines
    for i in range(3):
        f = 0.25 + i * 0.28
        ly = int(gy - (gy - van_y) * f)
        xr = int(hw * (1.0 - f * 0.70))
        cv2.line(canvas, (cx - xr, ly), (cx + xr, ly), GROUND_COLOR, 1)


def _draw_shadow(canvas: np.ndarray, proj: Dict[str, Tuple[int, int]],
                 gy: int) -> None:
    sx = (proj["l_foot"][0] + proj["r_foot"][0]) // 2
    cv2.ellipse(canvas, (sx, gy), (18, 5), 0, 0, 360, SHADOW_COLOR, -1)


# ======================================================================
# Filled Body Part Helpers (Step 2)
# ======================================================================

_Y_AXIS = np.array([0.0, 1.0, 0.0])
_Z_AXIS = np.array([0.0, 0.0, 1.0])


def _limb_quad_3d(joints_3d: Dict[str, np.ndarray],
                  joint_a: str, joint_b: str,
                  half_width: float):
    """Compute 4 3D corners of a limb rectangle.

    Returns list of 4 np.ndarray (3D points), or None if the bone is
    degenerate (zero length).  Width is perpendicular to the bone axis
    and the Y-axis; falls back to Z-axis if the bone is Y-parallel.
    """
    a = joints_3d[joint_a]
    b = joints_3d[joint_b]
    d = b - a
    d_len = np.linalg.norm(d)

    if d_len < 1e-6:
        return None  # degenerate bone (zero length)

    d_hat = d / d_len

    # Primary perpendicular: cross with Y-axis
    perp = np.cross(d_hat, _Y_AXIS)
    perp_len = np.linalg.norm(perp)

    if perp_len < 1e-4:
        # Bone parallel to Y — use Z-axis fallback
        perp = np.cross(d_hat, _Z_AXIS)
        perp_len = np.linalg.norm(perp)

    if perp_len < 1e-6:
        return None  # both failed (shouldn't happen)

    perp = (perp / perp_len) * half_width

    # 4 corners: A+perp, A-perp, B-perp, B+perp (counter-clockwise)
    return [a + perp, a - perp, b - perp, b + perp]


def _limb_box_faces(joints_3d: Dict[str, np.ndarray],
                    joint_a: str, joint_b: str,
                    half_width: float):
    """Compute 4 side faces of a rectangular prism around a bone.

    Returns a list of face quads (each face is 4 np.ndarray corners),
    or an empty list if the bone is degenerate.  The prism has width
    ``half_width`` in the perp1 direction and ``half_width * 0.7`` in
    the perp2 (depth) direction, giving a slightly flattened cross-section.
    """
    a = joints_3d[joint_a]
    b = joints_3d[joint_b]
    d = b - a
    d_len = np.linalg.norm(d)

    if d_len < 1e-6:
        return []

    d_hat = d / d_len

    # perp1: primary width (visible from front/back)
    perp1 = np.cross(d_hat, _Y_AXIS)
    p1_len = np.linalg.norm(perp1)
    if p1_len < 1e-4:
        perp1 = np.cross(d_hat, _Z_AXIS)
        p1_len = np.linalg.norm(perp1)
    if p1_len < 1e-6:
        return []
    perp1 = (perp1 / p1_len) * half_width

    # perp2: depth (visible from sides) — perpendicular to bone AND perp1
    perp2 = np.cross(d_hat, perp1 / half_width)
    p2_len = np.linalg.norm(perp2)
    if p2_len < 1e-6:
        return []
    perp2 = (perp2 / p2_len) * (half_width * 0.7)

    # 8 corners of the rectangular prism
    a_pp = a + perp1 + perp2
    a_pm = a + perp1 - perp2
    a_mp = a - perp1 + perp2
    a_mm = a - perp1 - perp2
    b_pp = b + perp1 + perp2
    b_pm = b + perp1 - perp2
    b_mp = b - perp1 + perp2
    b_mm = b - perp1 - perp2

    # 4 side faces (skip caps — they are tiny on limbs)
    return [
        [a_pp, a_pm, b_pm, b_pp],  # +perp1 face
        [a_mm, a_mp, b_mp, b_mm],  # -perp1 face
        [a_pm, a_mm, b_mm, b_pm],  # -perp2 face (side)
        [a_mp, a_pp, b_pp, b_mp],  # +perp2 face (side)
    ]


def _torso_box_faces(joints_3d: Dict[str, np.ndarray]):
    """Compute 4 side faces of the torso box.

    Uses the 4 corner joints offset forward/backward by _TORSO_HALF_DEPTH
    along the torso's surface normal (approximately Y-axis).
    Returns a list of face quads, or empty list if joints are missing.
    """
    if not all(j in joints_3d for j in _TORSO_CORNERS):
        return []

    corners = [joints_3d[j] for j in _TORSO_CORNERS]  # ls, rs, rh, lh

    # Compute torso surface normal for depth direction
    v1 = corners[1] - corners[0]   # l_shoulder → r_shoulder
    v2 = corners[3] - corners[0]   # l_shoulder → l_hip
    normal = np.cross(v1, v2)
    n_len = np.linalg.norm(normal)
    if n_len < 1e-6:
        normal = _Y_AXIS.copy()
    else:
        normal = normal / n_len

    offset = normal * _TORSO_HALF_DEPTH

    # Front and back faces
    front = [c + offset for c in corners]
    back = [corners[3] - offset, corners[2] - offset,
            corners[1] - offset, corners[0] - offset]  # reversed winding

    # Side faces connecting front and back edges
    ls_f, rs_f, rh_f, lh_f = front
    ls_b = corners[0] - offset
    rs_b = corners[1] - offset
    rh_b = corners[2] - offset
    lh_b = corners[3] - offset

    left_side = [ls_f, ls_b, lh_b, lh_f]    # left edge
    right_side = [rs_b, rs_f, rh_f, rh_b]   # right edge

    return [front, back, left_side, right_side]


def _project_quad(corners_3d, cx: int, cy: int, scale: float) -> np.ndarray:
    """Project 3D quad corners to 2D canvas coordinates.

    Returns np.array of shape (N, 2) suitable for cv2.fillConvexPoly.
    Uses the shared _frame_view_rot.
    """
    pts_2d = []
    for pt in corners_3d:
        r = _frame_view_rot @ pt
        depth = r[1] + _persp_d
        if depth < 0.1:
            depth = 0.1
        f = scale * _persp_d / depth
        pts_2d.append([int(cx + r[0] * f), int(cy - r[2] * f)])
    return np.array(pts_2d, dtype=np.int32)


def _draw_body_part(canvas: np.ndarray, pts_2d: np.ndarray,
                    fill_color, outline_color) -> None:
    """Draw a filled polygon with outline.

    If the quad has collapsed to near-zero area (edge-on view), draws
    a thin line fallback instead.
    """
    area = cv2.contourArea(pts_2d)
    if area < 2.0:
        # Edge-on: too thin to fill, draw a line instead
        cv2.line(canvas, tuple(pts_2d[0]), tuple(pts_2d[2]),
                 outline_color, 1, cv2.LINE_AA)
        return

    cv2.fillConvexPoly(canvas, pts_2d, fill_color, cv2.LINE_AA)
    cv2.polylines(canvas, [pts_2d], isClosed=True,
                  color=outline_color, thickness=1, lineType=cv2.LINE_AA)


def _part_fill_color(color_key: str, legs_state: str, arms_state: str):
    """Select fill color for a body part based on its key and animation state."""
    if color_key == "legs" and legs_state in ("walking", "animating"):
        return FILL_LEGS_ACTIVE
    if color_key == "arms" and arms_state == "animating":
        return FILL_ARMS_ACTIVE
    if color_key == "body":
        return FILL_BODY
    return FILL_LIMB


def _quad_depth(corners_3d) -> float:
    """Average depth of a quad's corners after view rotation."""
    total = 0.0
    for pt in corners_3d:
        total += (_frame_view_rot @ pt)[1]
    return total / len(corners_3d)


def _draw_filled_body(canvas: np.ndarray,
                      joints_3d: Dict[str, np.ndarray],
                      proj: Dict[str, Tuple[int, int]],
                      cx: int, gy: int, scale: float,
                      legs_state: str, arms_state: str,
                      speech_state: str,
                      head_yaw: float, head_pitch: float,
                      t: float) -> None:
    """Draw the robot with filled polygon body parts, depth-sorted.

    Unified rendering pipeline that replaces _draw_bones + _draw_joints +
    _draw_head.  All body parts (filled quads), the head (filled circle),
    and joint dots are drawn in correct depth order using the painter's
    algorithm.
    """
    # ---- Build draw list: (avg_depth, type_tag, data, fill_color) ----
    draw_list = []

    # Torso box (4 side faces with depth)
    for face in _torso_box_faces(joints_3d):
        draw_list.append((_quad_depth(face), "quad", face, FILL_BODY))

    # Limb boxes (4 side faces each — visible from any angle)
    for ja, jb, hw, color_key in _LIMB_PARTS:
        if ja not in joints_3d or jb not in joints_3d:
            continue
        fill = _part_fill_color(color_key, legs_state, arms_state)
        for face in _limb_box_faces(joints_3d, ja, jb, hw):
            draw_list.append((_quad_depth(face), "quad", face, fill))

    # Neck (special: uses spine direction, NOT rotated head position)
    # This prevents the neck from tilting when the head rotates.
    if all(j in joints_3d for j in ("neck", "hip", "l_shoulder", "r_shoulder")):
        neck_pos = joints_3d["neck"]
        # Body "up" direction from spine (hip → neck)
        spine = neck_pos - joints_3d["hip"]
        spine_len = np.linalg.norm(spine)
        if spine_len > 1e-6:
            up = spine / spine_len
        else:
            up = np.array([0.0, 0.0, 1.0])
        # Body "forward" direction from cross(up, shoulder_line)
        sh_line = joints_3d["r_shoulder"] - joints_3d["l_shoulder"]
        fwd = np.cross(up, sh_line)
        fwd_len = np.linalg.norm(fwd)
        if fwd_len > 1e-6:
            fwd = fwd / fwd_len
        else:
            fwd = np.array([0.0, 1.0, 0.0])
        # Neck top: 0.040m up along spine + 0.015m forward
        neck_top = neck_pos + up * 0.040 + fwd * 0.015
        temp_joints = {"_nb": neck_pos, "_nt": neck_top}
        for face in _limb_box_faces(temp_joints, "_nb", "_nt", 0.009):
            draw_list.append((_quad_depth(face), "quad", face, FILL_BODY))

    # Head (circle — drawn at its depth position)
    if "head" in joints_3d:
        head_depth = (_frame_view_rot @ joints_3d["head"])[1]
        draw_list.append((head_depth, "head", None, None))

    # ---- Sort farthest-first (painter's algorithm) ----
    draw_list.sort(key=lambda item: -item[0])

    # Brightness range for depth cues
    if len(draw_list) > 1:
        min_d = draw_list[-1][0]
        max_d = draw_list[0][0]
        d_range = max_d - min_d if max_d - min_d > 1e-6 else 1.0
    else:
        min_d = 0.0
        d_range = 1.0

    # ---- Draw back-to-front ----
    for depth, dtype, data, fill in draw_list:
        brightness = 1.0 - 0.30 * ((depth - min_d) / d_range)

        if dtype == "quad":
            pts_2d = _project_quad(data, cx, gy, scale)
            adj_fill = tuple(int(c * brightness) for c in fill)
            adj_out = tuple(int(c * brightness) for c in OUTLINE)
            _draw_body_part(canvas, pts_2d, adj_fill, adj_out)

        elif dtype == "head" and "head" in proj:
            hx, hy = proj["head"]
            hr = max(11, int(scale * 0.052 / _persp_d))

            # Head fill with brightness
            hc = tuple(int(c * brightness) for c in HEAD_COLOR)
            oc = tuple(int(c * brightness) for c in JOINT_COLOR)
            cv2.circle(canvas, (hx, hy), hr, hc, -1, cv2.LINE_AA)
            cv2.circle(canvas, (hx, hy), hr, oc, 2, cv2.LINE_AA)

            # Ear bumps (NAO speaker housings)
            ear_off = int(hr * 0.7)
            cv2.circle(canvas, (hx - ear_off, hy), 3,
                       EAR_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (hx + ear_off, hy), 3,
                       EAR_COLOR, -1, cv2.LINE_AA)

            # Eyes (shift with gaze direction)
            ex = int(head_yaw * 5)
            ey = int(head_pitch * 3)
            cv2.circle(canvas, (hx - 4 + ex, hy - 2 + ey), 2,
                       EYE_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (hx + 4 + ex, hy - 2 + ey), 2,
                       EYE_COLOR, -1, cv2.LINE_AA)

            # Mouth
            if speech_state == "speaking":
                mh = int(2 + 2 * abs(math.sin(t * 6.0)))
                cv2.ellipse(canvas, (hx + ex, hy + 5 + ey), (3, mh),
                            0, 0, 360, (100, 100, 255), -1)
            else:
                cv2.line(canvas, (hx - 3, hy + 5), (hx + 3, hy + 5),
                         JOINT_COLOR, 1, cv2.LINE_AA)

    # ---- Chest LED dot (on top of torso) ----
    if "neck" in proj and "hip" in proj:
        tcx = (proj["neck"][0] + proj["hip"][0]) // 2
        tcy = (proj["neck"][1] + proj["hip"][1]) // 2
        cv2.circle(canvas, (tcx, tcy), 3, CHEST_DOT, -1, cv2.LINE_AA)

    # ---- Joint dots on top ("rivet" look) ----
    for name, pt in proj.items():
        if name == "head":
            continue
        if name in ("l_foot", "r_foot"):
            # Feet as small rectangles (NAO flat feet)
            fx, fy = pt
            foot_w, foot_h = 8, 4
            pts = np.array([
                [fx - foot_w, fy - foot_h],
                [fx + foot_w, fy - foot_h],
                [fx + foot_w, fy + foot_h],
                [fx - foot_w, fy + foot_h],
            ], dtype=np.int32)
            f_fill = (FILL_LEGS_ACTIVE if legs_state in ("walking", "animating")
                      else FOOT_COLOR)
            cv2.fillConvexPoly(canvas, pts, f_fill, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], True, OUTLINE, 1, cv2.LINE_AA)
        else:
            r = 4 if name in _MAJOR_JOINTS else 3
            cv2.circle(canvas, pt, r, JOINT_COLOR, -1, cv2.LINE_AA)


# ======================================================================
# Legacy wireframe drawing (kept for rollback — not called)
# ======================================================================

def _draw_bones(canvas: np.ndarray,
                proj: Dict[str, Tuple[int, int]],
                joints_3d: Dict[str, np.ndarray],
                legs_state: str, arms_state: str) -> None:
    """Draw skeleton bones sorted back-to-front by depth (painter's algorithm).

    Each bone's depth is computed as the average Y-coordinate after applying
    the current view rotation.  Bones farther from the camera are drawn first
    so that nearer bones paint over them — giving correct overlap at any orbit
    angle.  A subtle brightness gradient adds depth cues.
    """
    view_rot = _rot_x(_view_pitch) @ _rot_z(_view_yaw)

    # Build draw list: (avg_depth, joint_a, joint_b, color, thickness)
    draw_list = []

    for bone_list, default_color, state, active_color, thick in [
        (_BONES_LEGS,  BODY_COLOR, legs_state, ANIM_LEG_COLOR, 3),
        (_BONES_TORSO, BODY_COLOR, None,       None,           4),
        (_BONES_ARMS,  BODY_COLOR, arms_state, ANIM_ARM_COLOR, 3),
        (_BONES_HEAD,  BODY_COLOR, None,       None,           3),
    ]:
        color = default_color
        if state in ("walking", "animating"):
            color = active_color

        for a_name, b_name in bone_list:
            if a_name not in joints_3d or b_name not in joints_3d:
                continue
            a_rot = view_rot @ joints_3d[a_name]
            b_rot = view_rot @ joints_3d[b_name]
            avg_depth = (a_rot[1] + b_rot[1]) / 2.0

            # Per-bone color override for shoulder→arm connection lines
            bone_color = color
            if bone_list is _BONES_ARMS and "shoulder" in a_name:
                bone_color = BODY_COLOR
                thick_use = 2
            else:
                thick_use = thick

            draw_list.append((avg_depth, a_name, b_name, bone_color, thick_use))

    # Sort by depth descending (farthest first = drawn first = behind)
    draw_list.sort(key=lambda item: -item[0])

    # Compute brightness range for depth cues
    if len(draw_list) > 1:
        min_d = draw_list[-1][0]   # nearest (last after descending sort)
        max_d = draw_list[0][0]    # farthest (first)
        d_range = max_d - min_d if max_d - min_d > 1e-6 else 1.0
    else:
        min_d = 0.0
        d_range = 1.0

    for depth, a_name, b_name, color, thick in draw_list:
        if a_name in proj and b_name in proj:
            # Depth cue: near = full brightness (1.0), far = dimmer (0.65)
            brightness = 1.0 - 0.35 * ((depth - min_d) / d_range)
            adjusted = tuple(int(c * brightness) for c in color)
            cv2.line(canvas, proj[a_name], proj[b_name],
                     adjusted, thick, cv2.LINE_AA)


def _draw_joints(canvas: np.ndarray,
                 proj: Dict[str, Tuple[int, int]]) -> None:
    for name, pt in proj.items():
        if name == "head":
            continue
        r = 4 if name in _MAJOR_JOINTS else 3
        cv2.circle(canvas, pt, r, JOINT_COLOR, -1, cv2.LINE_AA)


def _draw_head(canvas: np.ndarray,
               proj: Dict[str, Tuple[int, int]],
               head_yaw: float, head_pitch: float,
               speech_state: str, scale: float, t: float) -> None:
    """Draw the head sphere, eyes, and mouth."""
    hx, hy = proj["head"]
    hr = max(9, int(scale * 0.040 / _persp_d))

    cv2.circle(canvas, (hx, hy), hr, HEAD_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, (hx, hy), hr, JOINT_COLOR, 2, cv2.LINE_AA)

    # Eyes shift with yaw (gaze direction)
    ex = int(head_yaw * 5)
    ey = int(head_pitch * 3)
    cv2.circle(canvas, (hx - 4 + ex, hy - 2 + ey), 2, EYE_COLOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, (hx + 4 + ex, hy - 2 + ey), 2, EYE_COLOR, -1, cv2.LINE_AA)

    # Mouth
    if speech_state == "speaking":
        mh = int(2 + 2 * abs(math.sin(t * 6.0)))
        cv2.ellipse(canvas, (hx + ex, hy + 5 + ey), (3, mh),
                     0, 0, 360, (100, 100, 255), -1)
    else:
        cv2.line(canvas, (hx - 3, hy + 5), (hx + 3, hy + 5),
                 JOINT_COLOR, 1, cv2.LINE_AA)


# ======================================================================
# Public API  (same signature as the original 2D version)
# ======================================================================

def draw_robot(canvas: np.ndarray, x: int, y: int, w: int, h: int,
               head_yaw: float = 0.0, head_pitch: float = 0.0,
               posture: str = "standing", legs_state: str = "idle",
               arms_state: str = "idle", speech_state: str = "idle") -> None:
    """Draw a 3D wireframe NAO robot on the canvas.

    Drop-in replacement for the original 2D renderer.  Same arguments,
    same panel integration -- only the visual output changes.

    Includes smooth posture transitions (lerp over ~1 second) and
    interactive camera orbit (via set_view_angles / zoom_delta).

    Args:
        canvas: BGR image to draw on.
        x, y: Top-left corner of the drawing area.
        w, h: Width and height of the drawing area.
        head_yaw: Head yaw in radians (positive = robot looks left).
        head_pitch: Head pitch in radians (positive = robot looks down).
        posture: "standing" | "sitting" | "resting" | "fallen"
        legs_state: "idle" | "walking" | "animating"
        arms_state: "idle" | "animating"
        speech_state: "idle" | "speaking"
    """
    global _last_drawn_joints, _last_drawn_scale, _frame_view_rot

    cx = x + w // 2
    ground_y = y + h - 25
    t = time.monotonic()

    # Compute view rotation once per frame (shared by _project, _draw_filled_body)
    _frame_view_rot = _rot_x(_view_pitch) @ _rot_z(_view_yaw)

    # Scale so the 0.51m tall robot fills ~60% of the panel height
    usable_h = h - 50
    if usable_h < 40:
        usable_h = 40
    scale = usable_h * 0.60 / 0.51

    # Ground reference grid
    _draw_ground(canvas, cx, ground_y, w)

    # --- Compute TARGET joints for current posture -----------------------
    if posture == "fallen":
        target_joints = _fallen_joints()
        target_scale = scale * 0.80
    elif posture == "resting":
        target_joints = _resting_joints()
        target_scale = scale
    elif posture == "sitting":
        target_joints = _sitting_joints(head_yaw, head_pitch, arms_state, t)
        target_scale = scale
    else:  # standing (default)
        target_joints = _standing_joints(head_yaw, head_pitch,
                                          legs_state, arms_state, t)
        target_scale = scale

    # --- Posture transition handling -------------------------------------
    if _trans.prev_posture is None:
        # First frame — no transition
        _trans.prev_posture = posture

    if posture != _trans.prev_posture:
        # Posture changed — start a new transition
        if _last_drawn_joints is not None:
            # Snapshot the current displayed joints (handles mid-transition)
            _trans.snapshot_joints = {k: v.copy()
                                      for k, v in _last_drawn_joints.items()}
            _trans.snapshot_scale = _last_drawn_scale
        else:
            _trans.snapshot_joints = {k: v.copy()
                                      for k, v in target_joints.items()}
            _trans.snapshot_scale = target_scale
        _trans.start_time = t
        _trans.active = True
        _trans.prev_posture = posture

    # Apply transition interpolation
    if _trans.active:
        elapsed = t - _trans.start_time
        progress = elapsed / _trans.duration
        if progress >= 1.0:
            # Transition complete
            _trans.active = False
            joints = target_joints
            use_scale = target_scale
        else:
            alpha = _ease_in_out(progress)
            joints = _lerp_joints(_trans.snapshot_joints, target_joints, alpha)
            use_scale = (_trans.snapshot_scale
                         + alpha * (target_scale - _trans.snapshot_scale))
    else:
        joints = target_joints
        use_scale = target_scale

    # Save for next frame (snapshot source for future transitions)
    _last_drawn_joints = {k: v.copy() for k, v in joints.items()}
    _last_drawn_scale = use_scale

    # --- Project and draw ------------------------------------------------
    proj = _project(joints, cx, ground_y, use_scale)

    _draw_shadow(canvas, proj, ground_y)

    # During transition, use target posture's visual style (colors change
    # immediately, positions transition smoothly)
    draw_legs = legs_state if not _trans.active else "idle"
    draw_arms = arms_state if not _trans.active else "idle"

    # For fallen/resting without active transition, suppress animation colors
    if posture in ("fallen", "resting") and not _trans.active:
        draw_legs = "idle"
        draw_arms = "idle"

    # Determine head drawing params
    if posture in ("fallen", "resting") and not _trans.active:
        draw_yaw, draw_pitch, draw_speech = 0.0, 0.0, "idle"
    else:
        draw_yaw, draw_pitch, draw_speech = head_yaw, head_pitch, speech_state

    _draw_filled_body(canvas, joints, proj, cx, ground_y, use_scale,
                      draw_legs, draw_arms, draw_speech,
                      draw_yaw, draw_pitch, t)

    # Status labels (only when transition is complete)
    if not _trans.active:
        if posture == "fallen":
            alpha = 0.7 + 0.3 * math.sin(t * 3.0)
            color = (0, 0, int(220 * alpha))
            cv2.putText(canvas, "FALLEN!", (cx - 35, ground_y - 65),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif posture == "resting":
            cv2.putText(canvas, "RESTING", (cx - 32, ground_y - 70),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Walk direction arrow (projects robot's forward direction to screen)
    if legs_state == "walking" and not _trans.active:
        # Project the +Y axis (robot forward) through the view rotation
        fwd = _frame_view_rot @ np.array([0.0, 1.0, 0.0])
        # fwd[0] = rightward on screen, fwd[2] = upward (negated for cv2)
        dx = fwd[0]
        dy = -fwd[2]
        d_len = math.sqrt(dx * dx + dy * dy)
        if d_len < 1e-6:
            dx, dy, d_len = 1.0, 0.0, 1.0
        arrow_len = 15
        dx = dx / d_len * arrow_len
        dy = dy / d_len * arrow_len * 0.5  # compress vertical
        ay = ground_y + 12
        cv2.arrowedLine(canvas,
                         (int(cx - dx), int(ay - dy)),
                         (int(cx + dx), int(ay + dy)),
                         WALK_ARROW_COLOR, 2, tipLength=0.3)
