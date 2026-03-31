"""
elderguard_nao.py — Webots controller for NAO V5 (ElderGuard simulation).

Step 1: Basic setup — keyboard-controlled movements.
Step 2: TCP server for brain commands — full command dispatch with
        state tracking, two-phase ACK protocol, and motion management.

The controller receives the exact same JSON commands as the real NAO
server (nao_body/server.py) via TCP on port 5555.  It translates them
to Webots motor / motion-file calls and returns responses that match
the real protocol, including channel state snapshots.

Usage:
  1. Open elderguard_room.wbt in Webots R2023b
  2. Controller starts automatically (TCP server on port 5555)
  3. Brain (main.py) connects and sends JSON commands
  4. Keyboard still works for manual testing (click 3D view first)
"""

from controller import Supervisor, Keyboard, Motion
from tcp_handler import TcpHandler
from camera_server import CameraStreamServer
from postures import (POSTURES, STAND_INIT, CROUCH,
                       ARM_CARRY, ARM_REACH_DOWN, ARM_OFFER, ARM_REST)
import json
import math
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# NAO Joint Names (identical to NAOqi)
# ---------------------------------------------------------------------------
HEAD_JOINTS = ["HeadYaw", "HeadPitch"]

ARM_JOINTS_L = [
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
]
ARM_JOINTS_R = [
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw",
]
LEG_JOINTS_L = [
    "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch",
    "LAnklePitch", "LAnkleRoll",
]
LEG_JOINTS_R = [
    "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch",
    "RAnklePitch", "RAnkleRoll",
]
HAND_JOINTS = ["LHand", "RHand"]

ALL_JOINTS = (HEAD_JOINTS + ARM_JOINTS_L + ARM_JOINTS_R
              + LEG_JOINTS_L + LEG_JOINTS_R + HAND_JOINTS)

# ---------------------------------------------------------------------------
# Action routing (mirrors real server channel routing)
# ---------------------------------------------------------------------------

# Actions handled inline (immediate response, no ack/done split)
INLINE_ACTIONS = {
    "move_head", "move_head_relative",
    "set_walk_velocity", "stop_walk", "stop_all",
    "query_state", "heartbeat", "get_posture",
    "get_person_position", "follow_person", "stop_follow",
    "get_object_position", "stop_navigate",
}

# Actions whose responses include head angles at the top level
ANGLE_ACTIONS = {
    "move_head", "move_head_relative",
    "query_state", "heartbeat", "get_posture",
}

# Motion keys used for walking (for interrupt logic)
WALK_MOTION_KEYS = (
    "forwards", "backwards", "side_left", "side_right",
    "turn_left_40", "turn_right_40", "turn_left_60",
    "turn_right_60", "turn_left_180",
)

# Head joint limits (radians) — same as real NAO
HEAD_YAW_MIN, HEAD_YAW_MAX = -2.0857, 2.0857
HEAD_PITCH_MIN, HEAD_PITCH_MAX = -0.6719, 0.5149


# ===================================================================
# State Machine
# ===================================================================

class NaoState:
    """Track NAO channel states (mirrors NaoStateMachine from real server)."""

    def __init__(self, initial_posture="standing"):
        self.posture = initial_posture   # standing / sitting / resting
        self.head = "idle"
        self.legs = "idle"               # idle / walking / animating
        self.speech = "idle"             # idle / speaking
        self.arms = "idle"               # idle / animating

    def snapshot(self, head_yaw=0.0, head_pitch=0.0):
        """Return state dict matching the real server format."""
        return {
            "posture": self.posture,
            "head": self.head,
            "legs": self.legs,
            "speech": self.speech,
            "arms": self.arms,
            "head_yaw": round(head_yaw, 4),
            "head_pitch": round(head_pitch, 4),
        }

    def can_execute(self, action, params=None):
        """Validate state before command. Returns (ok, reason_string)."""
        params = params or {}

        if action in ("walk_toward", "set_walk_velocity"):
            if self.posture != "standing":
                return False, "must_stand_first"
            if self.legs == "animating":
                return False, "legs_busy_dancing"

        elif action == "animate":
            name = params.get("name", "")
            if name == "dance":
                if self.posture != "standing":
                    return False, "must_stand_first"
                if self.legs in ("walking", "animating"):
                    return False, "legs_busy"
                if self.arms == "animating":
                    return False, "arms_busy"
            else:  # wave
                if self.arms == "animating":
                    return False, "arms_busy"

        elif action == "pickup_sequence":
            if self.posture != "standing":
                return False, "must_stand_first"

        elif action == "navigate_to":
            if self.posture != "standing":
                return False, "must_stand_first"
            if self.legs == "animating":
                return False, "legs_busy_dancing"

        return True, ""

    def reset_channels(self):
        """Reset all channels to idle (preserve posture)."""
        self.head = "idle"
        self.legs = "idle"
        self.speech = "idle"
        self.arms = "idle"


# ===================================================================
# Async Task Types
# ===================================================================

class MotionTask:
    """Plays a sequence of motion files one after the other."""

    def __init__(self, request_id, action, motion_keys, controller, cmd=None):
        self.request_id = request_id
        self.action = action
        self.cmd = cmd or {}
        self.motion_keys = list(motion_keys)
        self.current_index = 0
        self._motion_obj = None
        self._play_next(controller)

    def _play_next(self, controller):
        """Start the next motion in the sequence."""
        while self.current_index < len(self.motion_keys):
            key = self.motion_keys[self.current_index]
            motion = controller.motions.get(key)
            if motion:
                if self._motion_obj:
                    self._motion_obj.stop()
                motion.play()
                self._motion_obj = motion
                return
            else:
                print("[WARN] Motion '%s' not found, skipping" % key)
                self.current_index += 1
        self._motion_obj = None

    def is_complete(self, controller):
        if self._motion_obj is None:
            return True
        if self._motion_obj.isOver():
            self.current_index += 1
            if self.current_index < len(self.motion_keys):
                self._play_next(controller)
                return False
            self._motion_obj = None
            return True
        return False

    def stop(self):
        if self._motion_obj:
            self._motion_obj.stop()
            self._motion_obj = None


class TimedTask:
    """Completes after a fixed duration of simulation time."""

    def __init__(self, request_id, action, duration_s, controller, cmd=None):
        self.request_id = request_id
        self.action = action
        self.cmd = cmd or {}
        self.end_time = controller.getTime() + duration_s

    def is_complete(self, controller):
        return controller.getTime() >= self.end_time

    def stop(self):
        pass


class MultiPhaseTask:
    """Executes a series of callables at specified time offsets."""

    def __init__(self, request_id, action, phases, controller, cmd=None):
        """
        phases: list of (time_offset_from_start, callable).
                Must be sorted by time. Last entry sets total duration.
        """
        self.request_id = request_id
        self.action = action
        self.cmd = cmd or {}
        self.phases = phases
        self.start_time = controller.getTime()
        self.current_phase = 0
        self.total_duration = phases[-1][0] if phases else 0

    def is_complete(self, controller):
        elapsed = controller.getTime() - self.start_time
        # Fire any phases whose time has come
        while self.current_phase < len(self.phases):
            t, fn = self.phases[self.current_phase]
            if elapsed >= t:
                fn()
                self.current_phase += 1
            else:
                break
        return elapsed >= self.total_duration

    def stop(self):
        pass  # Unfired phases are simply skipped


# ===================================================================
# Main Controller
# ===================================================================

class ElderGuardNao(Supervisor):
    """Webots NAO V5 controller with full TCP command dispatch.

    Extends Supervisor (which extends Robot) so we can read the
    virtual person's position for simulated face detection (Option B).
    """

    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())

        # -- devices --
        self._init_motors()
        self._init_sensors()
        self._init_cameras()
        self._load_motions()

        # Go to standing posture
        self._go_to_posture(STAND_INIT)

        # -- TCP server --
        port = int(os.environ.get("NAO_PORT", "5555"))
        self.tcp = TcpHandler(port=port)
        self.tcp.start()

        # -- Camera stream server --
        cam_port = int(os.environ.get("NAO_CAM_PORT", "5556"))
        self.cam_server = CameraStreamServer(port=cam_port)
        self.cam_server.start()
        self._cam_frame_counter = 0

        # -- state --
        self.state = NaoState("standing")

        # -- async tasks (one per channel) --
        self._legs_task = None
        self._speech_task = None
        self._arms_task = None

        # -- velocity walk (set_walk_velocity loop) --
        self._velocity_walk_active = False
        self._velocity_motion = None

        # -- head tracking: override motion-file head keyframes --
        self._desired_head_yaw = 0.0
        self._desired_head_pitch = -0.17

        # -- keyboard motion (not TCP-tracked) --
        self._current_kb_motion = None

        # -- connection state --
        self._was_connected = False

        # -- person tracking --
        self._tracking_active = False
        self._tracking_walk = False
        self._tracking_step = 0

        # -- navigate_to tracking --
        self._nav_active = False
        self._nav_target = None          # [x, y, z] world coordinates
        self._nav_stop_dist = 0.6        # stop when within this distance (meters)
        self._nav_request_id = None      # for sending async "done" response
        self._nav_step = 0               # update counter

        # -- object carrying (Supervisor-based fake grasping) --
        self._carrying_object = None     # Webots Node reference
        self._carry_active = False

        # -- virtual person (Supervisor access) --
        self._person_node = self.getFromDef("PERSON")
        if self._person_node:
            self._person_pos_field = self._person_node.getField("translation")
            print("[ElderGuard NAO] Virtual person found (DEF PERSON)")
        else:
            self._person_pos_field = None
            print("[ElderGuard NAO] No virtual person (DEF PERSON) in world")

        # -- object nodes (Supervisor access for DEF'd objects) --
        self._object_nodes = {}
        for def_name in ("PHONE", "BOTTLE"):
            node = self.getFromDef(def_name)
            if node:
                self._object_nodes[def_name] = node
                print("[ElderGuard NAO] Object found: DEF %s" % def_name)

        print("[ElderGuard NAO] Controller initialized (TCP + Camera + Supervisor)")
        print("[ElderGuard NAO] TCP command server on port %d" % port)
        print("[ElderGuard NAO] Camera stream server on port %d" % cam_port)
        print("[ElderGuard NAO] Webots timestep: %d ms" % self.time_step)
        print("[ElderGuard NAO] Motors: %d  Motions: %d" % (
            len(self.motors), len(self.motions)))
        print("")
        self._print_help()

    # ==================================================================
    # Device Initialization
    # ==================================================================

    def _init_motors(self):
        self.motors = {}
        for name in ALL_JOINTS:
            motor = self.getDevice(name)
            if motor is not None:
                self.motors[name] = motor
            else:
                print("[WARN] Motor not found: %s" % name)

        self.lphalanx = []
        self.rphalanx = []
        for i in range(1, 9):
            lp = self.getDevice("LPhalanx%d" % i)
            rp = self.getDevice("RPhalanx%d" % i)
            if lp:
                self.lphalanx.append(lp)
            if rp:
                self.rphalanx.append(rp)

    def _init_sensors(self):
        self.position_sensors = {}
        for name in ALL_JOINTS:
            sensor_name = name + "S"
            sensor = self.getDevice(sensor_name)
            if sensor is not None:
                sensor.enable(self.time_step)
                self.position_sensors[name] = sensor

        self.accelerometer = self.getDevice("accelerometer")
        if self.accelerometer:
            self.accelerometer.enable(4 * self.time_step)

        self.gyro = self.getDevice("gyro")
        if self.gyro:
            self.gyro.enable(4 * self.time_step)

        self.gps = self.getDevice("gps")
        if self.gps:
            self.gps.enable(4 * self.time_step)

        self.inertial_unit = self.getDevice("inertial unit")
        if self.inertial_unit:
            self.inertial_unit.enable(self.time_step)

        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.time_step)

    def _init_cameras(self):
        self.camera_top = self.getDevice("CameraTop")
        self.camera_bottom = self.getDevice("CameraBottom")
        if self.camera_top:
            self.camera_top.enable(4 * self.time_step)
        if self.camera_bottom:
            self.camera_bottom.enable(4 * self.time_step)

    def _load_motions(self):
        webots_home = os.environ.get("WEBOTS_HOME", "")
        motion_dir = os.path.join(
            webots_home, "projects", "robots", "softbank", "nao", "motions"
        )

        if not os.path.isdir(motion_dir):
            for candidate in [
                "X:/Apps - Work/Webots/webots R2023b/install/Webots",
                "C:/Program Files/Webots",
                "C:/Webots",
            ]:
                test_dir = os.path.join(
                    candidate, "projects", "robots", "softbank", "nao", "motions"
                )
                if os.path.isdir(test_dir):
                    motion_dir = test_dir
                    break

        self.motions = {}
        motion_files = {
            "forwards": "Forwards50.motion",
            "backwards": "Backwards.motion",
            "side_left": "SideStepLeft.motion",
            "side_right": "SideStepRight.motion",
            "turn_left_40": "TurnLeft40.motion",
            "turn_right_40": "TurnRight40.motion",
            "turn_left_60": "TurnLeft60.motion",
            "turn_right_60": "TurnRight60.motion",
            "turn_left_180": "TurnLeft180.motion",
            "hand_wave": "HandWave.motion",
            "stand_up_front": "StandUpFromFront.motion",
            "tai_chi": "TaiChi.motion",
        }

        for key, filename in motion_files.items():
            path = os.path.join(motion_dir, filename)
            if os.path.isfile(path):
                try:
                    self.motions[key] = Motion(path)
                except Exception as e:
                    print("[WARN] Failed to load %s: %s" % (filename, e))
            else:
                print("[WARN] Motion file not found: %s" % path)

        if not self.motions:
            print("[ERROR] No motion files loaded! Check WEBOTS_HOME.")

    # ==================================================================
    # Low-level Motor Control
    # ==================================================================

    def _go_to_posture(self, joint_angles, speed=None):
        """Set all joints in the dict to their target angles.

        Args:
            joint_angles: dict of joint_name -> target_angle (radians)
            speed: fraction of max motor velocity (0.0-1.0).
                   None = full speed. 0.3 = 30% of max = safe transition.
        """
        for name, angle in joint_angles.items():
            motor = self.motors.get(name)
            if motor:
                if speed is not None:
                    max_vel = motor.getMaxVelocity()
                    motor.setVelocity(max(0.1, speed * max_vel))
                motor.setPosition(float(angle))

    def _restore_motor_speeds(self):
        """Reset all motor velocities to their maximum (undo slow transitions)."""
        for motor in self.motors.values():
            motor.setVelocity(motor.getMaxVelocity())

    def _set_joint(self, name, angle):
        motor = self.motors.get(name)
        if motor:
            motor.setPosition(float(angle))

    def _get_joint_angle(self, name):
        sensor = self.position_sensors.get(name)
        if sensor:
            return sensor.getValue()
        return 0.0

    def _open_hand(self, hand="right"):
        phalanxes = self.rphalanx if hand == "right" else self.lphalanx
        hand_name = "RHand" if hand == "right" else "LHand"
        for motor in phalanxes:
            motor.setPosition(motor.getMaxPosition())
        if hand_name in self.motors:
            self.motors[hand_name].setPosition(1.0)

    def _close_hand(self, hand="right"):
        phalanxes = self.rphalanx if hand == "right" else self.lphalanx
        hand_name = "RHand" if hand == "right" else "LHand"
        for motor in phalanxes:
            motor.setPosition(motor.getMinPosition())
        if hand_name in self.motors:
            self.motors[hand_name].setPosition(0.0)

    def _open_both_hands(self):
        self._open_hand("left")
        self._open_hand("right")

    def _close_both_hands(self):
        self._close_hand("left")
        self._close_hand("right")

    # ==================================================================
    # State / Response Helpers
    # ==================================================================

    def _state_snapshot(self):
        """Build state dict with current head angles."""
        yaw = self._get_joint_angle("HeadYaw")
        pitch = self._get_joint_angle("HeadPitch")
        return self.state.snapshot(yaw, pitch)

    def _build_response(self, action, request_id=None, status="ok",
                        resp_type=None, **extra):
        """Build a JSON response dict matching the real server format."""
        resp = {
            "status": status,
            "action": action,
            "state": self._state_snapshot(),
        }
        if request_id is not None:
            resp["id"] = request_id
        if resp_type:
            resp["type"] = resp_type
        if action in ANGLE_ACTIONS:
            resp["head_yaw"] = round(self._get_joint_angle("HeadYaw"), 4)
            resp["head_pitch"] = round(self._get_joint_angle("HeadPitch"), 4)
        resp.update(extra)
        return resp

    # ==================================================================
    # TCP Command Dispatch
    # ==================================================================

    def _dispatch(self, cmd):
        """Route a parsed JSON command to the appropriate handler."""
        action = cmd.get("action", "")
        request_id = cmd.get("id")
        no_ack = cmd.get("no_ack", False)

        if not action:
            return

        # State validation
        ok, reason = self.state.can_execute(action, cmd)
        if not ok:
            if not no_ack:
                resp = self._build_response(
                    action, request_id, status="rejected",
                    resp_type="ack" if request_id else None,
                    reason=reason)
                self.tcp.send_response(resp)
            return

        if action in INLINE_ACTIONS:
            self._dispatch_inline(cmd, action, request_id, no_ack)
        else:
            self._dispatch_async(cmd, action, request_id, no_ack)

    def _dispatch_inline(self, cmd, action, request_id, no_ack):
        """Execute an inline action and send a single response."""
        handler = self._get_handler(action)
        if handler:
            handler(cmd)

        if no_ack:
            return

        resp = self._build_response(
            action, request_id,
            resp_type="done" if request_id else None)

        # Add person + NAO position data for get_person_position
        if action == "get_person_position":
            pos = self._get_person_position()
            if pos:
                resp["person_position"] = {
                    "x": round(pos[0], 3),
                    "y": round(pos[1], 3),
                    "z": round(pos[2], 3),
                }
                resp["person_found"] = True
            else:
                resp["person_found"] = False
            # Include NAO's own position and orientation
            if self.gps:
                gp = self.gps.getValues()
                resp["nao_position"] = {
                    "x": round(gp[0], 3),
                    "y": round(gp[1], 3),
                    "z": round(gp[2], 3),
                }
            if self.inertial_unit:
                rpy = self.inertial_unit.getRollPitchYaw()
                resp["nao_orientation"] = {
                    "roll": round(rpy[0], 4),
                    "pitch": round(rpy[1], 4),
                    "yaw": round(rpy[2], 4),
                }

        # Add object position data for get_object_position
        if action == "get_object_position":
            obj_name = cmd.get("name", "").upper()
            node = self._object_nodes.get(obj_name)
            if node:
                try:
                    pos = node.getPosition()
                    resp["object_position"] = {
                        "x": round(pos[0], 3),
                        "y": round(pos[1], 3),
                        "z": round(pos[2], 3),
                    }
                    resp["object_found"] = True
                    resp["object_name"] = obj_name
                except Exception:
                    resp["object_found"] = False
                    resp["object_name"] = obj_name
            else:
                resp["object_found"] = False
                resp["object_name"] = obj_name
            # Include NAO position for reference
            if self.gps:
                gp = self.gps.getValues()
                resp["nao_position"] = {
                    "x": round(gp[0], 3),
                    "y": round(gp[1], 3),
                    "z": round(gp[2], 3),
                }

        self.tcp.send_response(resp)

    def _dispatch_async(self, cmd, action, request_id, no_ack):
        """Start an async action: send ack, track task, done on completion."""
        # Check handler exists before sending ack
        handler = self._get_handler(action)
        if not handler:
            if not no_ack:
                resp = self._build_response(
                    action, request_id, status="error",
                    resp_type="done" if request_id else None,
                    message="unknown action: %s" % action)
                self.tcp.send_response(resp)
            return

        # Interrupt current legs task for walk/pose/rest/wake_up/pickup
        if action in ("walk_toward", "pose", "rest", "wake_up", "pickup_sequence",
                      "navigate_to"):
            self._interrupt_legs()

        # Complete existing speech/arms task before replacing
        self._complete_channel_if_active(action, cmd)

        # Pre-state update (before ack)
        self._pre_state_update(action, cmd)

        # Send ack
        if not no_ack:
            resp = self._build_response(
                action, request_id,
                status="accepted" if request_id else "ok",
                resp_type="ack" if request_id else None)
            self.tcp.send_response(resp)

        # Start the async work
        handler(cmd, request_id)

    def _get_handler(self, action):
        """Map action name to handler method."""
        return {
            # Inline
            "move_head":          self._handle_move_head,
            "move_head_relative": self._handle_move_head_relative,
            "set_walk_velocity":  self._handle_set_walk_velocity,
            "stop_walk":          self._handle_stop_walk,
            "stop_all":           self._handle_stop_all,
            "query_state":        self._handle_noop,
            "heartbeat":          self._handle_noop,
            "get_posture":        self._handle_noop,
            "get_person_position": self._handle_get_person_position,
            "get_object_position": self._handle_get_object_position,
            "follow_person":      self._handle_follow_person,
            "stop_follow":        self._handle_stop_follow,
            "stop_navigate":      self._handle_stop_navigate,
            # Async (LEGS)
            "navigate_to":        self._handle_navigate_to,
            "walk_toward":        self._handle_walk_toward,
            "pose":               self._handle_pose,
            "rest":               self._handle_rest,
            "wake_up":            self._handle_wake_up,
            "pickup_sequence":    self._handle_pickup_sequence,
            # Async (SPEECH)
            "say":                self._handle_say,
            "animated_say":       self._handle_animated_say,
            # Async (ARMS)
            "animate":            self._handle_animate,
            "open_hand":          self._handle_open_hand,
            "close_hand":         self._handle_close_hand,
            "arm_carry_position": self._handle_arm_carry,
            "arm_reach_down":     self._handle_arm_reach_down,
            "arm_offer_position": self._handle_arm_offer,
            "arm_rest_position":  self._handle_arm_rest,
            "offer_and_release":  self._handle_offer_and_release,
            "start_carrying":     self._handle_start_carrying,
            "stop_carrying":      self._handle_stop_carrying,
        }.get(action)

    # ==================================================================
    # Interrupt / State Update Helpers
    # ==================================================================

    def _interrupt_legs(self):
        """Stop current legs activity (walk interruption)."""
        if self._legs_task:
            # If the interrupted task was a dance, also clear arms channel
            if (self._legs_task.action == "animate"
                    and self._legs_task.cmd.get("name") == "dance"):
                self.state.arms = "idle"
            self._legs_task.stop()
            self._legs_task = None
        self._stop_velocity_walk()
        self.state.legs = "idle"
        # Stop any playing walk/turn motions
        for key in WALK_MOTION_KEYS:
            motion = self.motions.get(key)
            if motion and not motion.isOver():
                motion.stop()

    def _stop_velocity_walk(self):
        """Stop continuous velocity-mode walking."""
        if self._velocity_walk_active:
            if self._velocity_motion:
                self._velocity_motion.setLoop(False)
                self._velocity_motion.stop()
                self._velocity_motion = None
            self._velocity_walk_active = False

    def _stop_all_activity(self):
        """Emergency stop: all motions, all tasks, reset channels."""
        # Stop every motion file
        for motion in self.motions.values():
            if not motion.isOver():
                motion.stop()
        # Cancel all tasks
        if self._legs_task:
            self._legs_task.stop()
            self._legs_task = None
        if self._arms_task:
            self._arms_task.stop()
            self._arms_task = None
        self._speech_task = None
        self._stop_velocity_walk()
        self._stop_person_tracking()
        self._stop_navigation()
        self._stop_carrying()
        self.state.reset_channels()

    def _complete_channel_if_active(self, action, cmd):
        """If the new action's channel already has a task, complete it first."""
        # SPEECH channel
        if action in ("say", "animated_say") and self._speech_task:
            task = self._speech_task
            self._speech_task = None
            self._post_state_update(task.action, task.cmd)
            if task.request_id:
                resp = self._build_response(
                    task.action, task.request_id, resp_type="done")
                self.tcp.send_response(resp)

        # ARMS channel
        arms_actions = ("animate", "open_hand", "close_hand",
                        "arm_carry_position", "arm_reach_down",
                        "arm_offer_position", "arm_rest_position",
                        "offer_and_release",
                        "start_carrying", "stop_carrying")
        if action in arms_actions and self._arms_task:
            task = self._arms_task
            self._arms_task = None
            task.stop()
            self._post_state_update(task.action, task.cmd)
            if task.request_id:
                resp = self._build_response(
                    task.action, task.request_id, resp_type="done")
                self.tcp.send_response(resp)

    def _pre_state_update(self, action, cmd):
        """Update channel state immediately (before ack is sent)."""
        if action == "walk_toward":
            self.state.legs = "walking"
        elif action == "navigate_to":
            self.state.legs = "walking"
        elif action in ("say", "animated_say"):
            self.state.speech = "speaking"
        elif action == "animate":
            name = cmd.get("name", "")
            if name == "dance":
                self.state.legs = "animating"
                self.state.arms = "animating"
            else:
                self.state.arms = "animating"
        elif action in ("pose", "rest", "wake_up"):
            self.state.legs = "idle"  # walk was interrupted
        elif action == "pickup_sequence":
            self.state.legs = "walking"
            self.state.arms = "animating"
        elif action in ("open_hand", "close_hand",
                         "arm_carry_position", "arm_reach_down",
                         "arm_offer_position", "arm_rest_position",
                         "offer_and_release",
                         "start_carrying", "stop_carrying"):
            self.state.arms = "animating"

    def _post_state_update(self, action, cmd):
        """Update channel state after a task completes."""
        if action == "walk_toward":
            self.state.legs = "idle"
        elif action in ("say", "animated_say"):
            self.state.speech = "idle"
        elif action == "animate":
            name = cmd.get("name", "")
            if name == "dance":
                self.state.legs = "idle"
                self.state.arms = "idle"
            else:
                self.state.arms = "idle"
        elif action == "pose":
            name = cmd.get("name", "").lower()
            if name in ("sit", "crouch"):
                self.state.posture = "sitting"
            else:
                self.state.posture = "standing"
                # Re-apply StandInit at full speed to guarantee exact position
                self._restore_motor_speeds()
                self._go_to_posture(STAND_INIT)
            self.state.legs = "idle"
        elif action == "rest":
            self.state.posture = "resting"
            self.state.legs = "idle"
            self._restore_motor_speeds()
        elif action == "wake_up":
            self.state.posture = "standing"
            self.state.legs = "idle"
            # Re-apply StandInit at full speed to guarantee exact position
            self._restore_motor_speeds()
            self._go_to_posture(STAND_INIT)
        elif action == "pickup_sequence":
            self.state.posture = "standing"
            self.state.legs = "idle"
            self.state.arms = "idle"
            self._restore_motor_speeds()
            self._go_to_posture(STAND_INIT)
        elif action in ("open_hand", "close_hand",
                         "arm_carry_position", "arm_reach_down",
                         "arm_offer_position", "arm_rest_position",
                         "offer_and_release",
                         "start_carrying", "stop_carrying"):
            self.state.arms = "idle"

    # ==================================================================
    # Inline Action Handlers
    # ==================================================================

    def _handle_noop(self, cmd):
        """No-op handler for query_state / heartbeat / get_posture."""
        pass

    def _handle_get_person_position(self, cmd):
        """No-op — position data added in _dispatch_inline."""
        pass

    def _handle_get_object_position(self, cmd):
        """No-op — position data added in _dispatch_inline."""
        pass

    def _handle_stop_navigate(self, cmd):
        """Stop navigate_to tracking."""
        self._stop_navigation()

    def _handle_follow_person(self, cmd):
        """Start built-in person tracking (head + optional walk)."""
        walk = cmd.get("walk", False)
        self._start_person_tracking(walk_enabled=walk)

    def _handle_stop_follow(self, cmd):
        """Stop built-in person tracking."""
        self._stop_person_tracking()

    def _get_person_position(self):
        """Read the virtual person's world position via Supervisor API.

        Returns [x, y, z] or None if no person in world.
        """
        if self._person_node:
            try:
                return self._person_node.getPosition()
            except Exception:
                pass
        return None

    def _handle_move_head(self, cmd):
        yaw = float(cmd.get("yaw", 0.0))
        pitch = float(cmd.get("pitch", 0.0))
        yaw = max(HEAD_YAW_MIN, min(HEAD_YAW_MAX, yaw))
        pitch = max(HEAD_PITCH_MIN, min(HEAD_PITCH_MAX, pitch))
        self._desired_head_yaw = yaw
        self._desired_head_pitch = pitch
        self._set_joint("HeadYaw", yaw)
        self._set_joint("HeadPitch", pitch)

    def _handle_move_head_relative(self, cmd):
        d_yaw = float(cmd.get("d_yaw", 0.0))
        d_pitch = float(cmd.get("d_pitch", 0.0))
        cur_yaw = self._get_joint_angle("HeadYaw")
        cur_pitch = self._get_joint_angle("HeadPitch")
        new_yaw = max(HEAD_YAW_MIN, min(HEAD_YAW_MAX, cur_yaw + d_yaw))
        new_pitch = max(HEAD_PITCH_MIN, min(HEAD_PITCH_MAX, cur_pitch + d_pitch))
        self._desired_head_yaw = new_yaw
        self._desired_head_pitch = new_pitch
        self._set_joint("HeadYaw", new_yaw)
        self._set_joint("HeadPitch", new_pitch)

    def _handle_set_walk_velocity(self, cmd):
        x = float(cmd.get("x", 0))
        theta = float(cmd.get("theta", 0))

        if x > 0.01:
            # Interrupt any legs task (walk_toward etc.)
            if self._legs_task:
                self._interrupt_legs()
            if not self._velocity_walk_active:
                motion = self.motions.get("forwards")
                if motion:
                    motion.setLoop(True)
                    motion.play()
                    self._velocity_motion = motion
                    self._velocity_walk_active = True
            self.state.legs = "walking"
        elif x <= 0.01 and abs(theta) < 0.1:
            self._stop_velocity_walk()
            self.state.legs = "idle"

    def _handle_stop_walk(self, cmd):
        self._interrupt_legs()
        self.state.legs = "idle"

    def _handle_stop_all(self, cmd):
        self._stop_all_activity()
        print("[NAO] Stop all — channels reset")

    # ==================================================================
    # Async Action Handlers — LEGS channel
    # ==================================================================

    def _handle_walk_toward(self, cmd, request_id=None):
        x = float(cmd.get("x", 0))
        y = float(cmd.get("y", 0))
        theta = float(cmd.get("theta", 0))

        motion_keys = self._plan_walk_sequence(x, y, theta)

        if not motion_keys:
            # Nothing to walk — send done immediately
            self._post_state_update("walk_toward", cmd)
            if request_id:
                resp = self._build_response(
                    "walk_toward", request_id, resp_type="done")
                self.tcp.send_response(resp)
            return

        self._legs_task = MotionTask(
            request_id, "walk_toward", motion_keys, self, cmd)

    def _plan_walk_sequence(self, x, y, theta):
        """Choose motion files for a walk_toward command."""
        keys = []
        abs_theta = abs(theta)

        # Turning
        if abs_theta >= 0.35:
            if theta > 0:                          # turn left
                if abs_theta >= 2.5:
                    keys.append("turn_left_180")
                elif abs_theta >= 0.8:
                    keys.append("turn_left_60")
                else:
                    keys.append("turn_left_40")
            else:                                   # turn right
                if abs_theta >= 2.5:
                    # No turn_right_180; chain three right_60 turns
                    keys.extend(["turn_right_60"] * 3)
                elif abs_theta >= 0.8:
                    keys.append("turn_right_60")
                else:
                    keys.append("turn_right_40")

        # Forward / backward
        if x > 0.1:
            n = max(1, int(round(x / 0.5)))
            keys.extend(["forwards"] * n)
        elif x < -0.1:
            keys.append("backwards")

        # Side step
        if abs(y) > 0.1:
            keys.append("side_left" if y > 0 else "side_right")

        return keys

    def _handle_pose(self, cmd, request_id=None):
        name = cmd.get("name", "stand")
        posture_angles = POSTURES.get(name) or POSTURES.get(name.lower())

        if posture_angles:
            self._go_to_posture(posture_angles, speed=0.3)
            self._legs_task = TimedTask(
                request_id, "pose", 3.0, self, cmd)
            print("[POSTURE] Going to: %s (slow transition)" % name)
        else:
            print("[WARN] Unknown posture: %s" % name)
            self._post_state_update("pose", cmd)
            if request_id:
                resp = self._build_response("pose", request_id, resp_type="done")
                self.tcp.send_response(resp)

    def _handle_rest(self, cmd, request_id=None):
        self._go_to_posture(CROUCH, speed=0.3)
        self._legs_task = TimedTask(request_id, "rest", 3.0, self, cmd)
        print("[POSTURE] Resting (crouch, slow transition)")

    def _handle_wake_up(self, cmd, request_id=None):
        self._go_to_posture(STAND_INIT, speed=0.3)
        self._desired_head_yaw = 0.0
        self._desired_head_pitch = -0.17
        self._legs_task = TimedTask(request_id, "wake_up", 3.0, self, cmd)
        print("[POSTURE] Waking up -> StandInit (slow transition)")

    def _handle_pickup_sequence(self, cmd, request_id=None):
        hand = cmd.get("hand", "right")

        phases = [
            (0.0,  lambda: self._open_hand(hand)),
            (0.5,  lambda: self._go_to_posture(CROUCH, speed=0.3)),
            (3.5,  lambda: self._go_to_posture(ARM_REACH_DOWN)),
            (4.5,  lambda: self._close_hand(hand)),
            (5.5,  lambda: self._go_to_posture(ARM_CARRY)),
            (6.5,  lambda: self._go_to_posture(STAND_INIT, speed=0.3)),
            (10.0, lambda: None),   # done sentinel
        ]

        self._legs_task = MultiPhaseTask(
            request_id, "pickup_sequence", phases, self, cmd)
        print("[PICKUP] Starting sequence (hand=%s)" % hand)

    def _handle_navigate_to(self, cmd, request_id=None):
        """Start navigating to a target position (head tracking + walk)."""
        x = float(cmd.get("x", 0))
        y = float(cmd.get("y", 0))
        z = float(cmd.get("z", 0))
        stop_dist = float(cmd.get("stop_distance", 0.6))

        # Stop any existing person tracking or navigation
        self._stop_person_tracking()
        self._nav_active = True
        self._nav_target = [x, y, z]
        self._nav_stop_dist = stop_dist
        self._nav_request_id = request_id
        self._nav_step = 0
        print("[NAV] Navigate to (%.2f, %.2f, %.2f) stop_dist=%.2f" % (
            x, y, z, stop_dist))

    # ==================================================================
    # Async Action Handlers — ARMS (carrying)
    # ==================================================================

    def _handle_start_carrying(self, cmd, request_id=None):
        """Start carrying a DEF'd object (Supervisor teleport to hand)."""
        obj_name = cmd.get("name", "PHONE").upper()
        self._start_carrying(obj_name)
        self._arms_task = TimedTask(
            request_id, "start_carrying", 0.3, self, cmd)

    def _handle_stop_carrying(self, cmd, request_id=None):
        """Stop carrying (release object)."""
        self._stop_carrying()
        self._arms_task = TimedTask(
            request_id, "stop_carrying", 0.3, self, cmd)

    # ==================================================================
    # Async Action Handlers — SPEECH channel
    # ==================================================================

    def _handle_say(self, cmd, request_id=None):
        text = cmd.get("text", "")
        duration = max(0.5, len(text) * 0.05)
        print("[SAY] %s" % text)
        self._speech_task = TimedTask(request_id, "say", duration, self, cmd)

    def _handle_animated_say(self, cmd, request_id=None):
        text = cmd.get("text", "")
        duration = max(0.5, len(text) * 0.06)
        print("[ANIMATED_SAY] %s" % text)
        self._speech_task = TimedTask(
            request_id, "animated_say", duration, self, cmd)

    # ==================================================================
    # Async Action Handlers — ARMS channel
    # ==================================================================

    def _handle_animate(self, cmd, request_id=None):
        name = cmd.get("name", "")

        if name == "wave":
            if "hand_wave" in self.motions:
                self._arms_task = MotionTask(
                    request_id, "animate", ["hand_wave"], self, cmd)
            else:
                self._arms_task = TimedTask(
                    request_id, "animate", 3.0, self, cmd)
            print("[ANIMATE] Wave")

        elif name == "dance":
            # Use tai_chi as dance; it goes on _legs_task because
            # dance claims both legs + arms channels.
            if "tai_chi" in self.motions:
                self._legs_task = MotionTask(
                    request_id, "animate", ["tai_chi"], self, cmd)
            else:
                self._legs_task = TimedTask(
                    request_id, "animate", 8.0, self, cmd)
            print("[ANIMATE] Dance")

        else:
            print("[WARN] Unknown animation: %s" % name)
            # Reset state since no task was created
            self.state.arms = "idle"
            if request_id:
                resp = self._build_response(
                    "animate", request_id, resp_type="done")
                self.tcp.send_response(resp)

    def _handle_open_hand(self, cmd, request_id=None):
        hand = cmd.get("hand", "right")
        self._open_hand(hand)
        self._arms_task = TimedTask(
            request_id, "open_hand", 0.5, self, cmd)
        print("[HAND] Open %s" % hand)

    def _handle_close_hand(self, cmd, request_id=None):
        hand = cmd.get("hand", "right")
        self._close_hand(hand)
        self._arms_task = TimedTask(
            request_id, "close_hand", 0.5, self, cmd)
        print("[HAND] Close %s" % hand)

    def _handle_arm_carry(self, cmd, request_id=None):
        self._go_to_posture(ARM_CARRY)
        self._arms_task = TimedTask(
            request_id, "arm_carry_position", 1.0, self, cmd)

    def _handle_arm_reach_down(self, cmd, request_id=None):
        self._go_to_posture(ARM_REACH_DOWN)
        self._arms_task = TimedTask(
            request_id, "arm_reach_down", 1.0, self, cmd)

    def _handle_arm_offer(self, cmd, request_id=None):
        self._go_to_posture(ARM_OFFER)
        self._arms_task = TimedTask(
            request_id, "arm_offer_position", 1.0, self, cmd)

    def _handle_arm_rest(self, cmd, request_id=None):
        self._go_to_posture(ARM_REST)
        self._arms_task = TimedTask(
            request_id, "arm_rest_position", 1.0, self, cmd)

    def _handle_offer_and_release(self, cmd, request_id=None):
        hand = cmd.get("hand", "right")
        phases = [
            (0.0, lambda: self._go_to_posture(ARM_OFFER)),
            (3.0, lambda: self._open_hand(hand)),
            (3.5, lambda: self._go_to_posture(ARM_REST)),
            (5.0, lambda: None),   # done sentinel
        ]
        self._arms_task = MultiPhaseTask(
            request_id, "offer_and_release", phases, self, cmd)
        print("[OFFER] Starting offer-and-release (hand=%s)" % hand)

    # ==================================================================
    # Task Update Loop
    # ==================================================================

    def _update_tasks(self):
        """Check and advance all async tasks. Called every robot.step."""

        # --- LEGS ---
        if self._legs_task and self._legs_task.is_complete(self):
            task = self._legs_task
            self._legs_task = None
            self._post_state_update(task.action, task.cmd)
            if task.request_id:
                resp = self._build_response(
                    task.action, task.request_id, resp_type="done")
                self.tcp.send_response(resp)

        # --- SPEECH ---
        if self._speech_task and self._speech_task.is_complete(self):
            task = self._speech_task
            self._speech_task = None
            self._post_state_update(task.action, task.cmd)
            if task.request_id:
                resp = self._build_response(
                    task.action, task.request_id, resp_type="done")
                self.tcp.send_response(resp)

        # --- ARMS ---
        if self._arms_task and self._arms_task.is_complete(self):
            task = self._arms_task
            self._arms_task = None
            self._post_state_update(task.action, task.cmd)
            if task.request_id:
                resp = self._build_response(
                    task.action, task.request_id, resp_type="done")
                self.tcp.send_response(resp)

        # Override head position every tick to prevent walk-motion
        # keyframes from fighting the brain's PID servo commands.
        self._set_joint("HeadYaw", self._desired_head_yaw)
        self._set_joint("HeadPitch", self._desired_head_pitch)

    # ==================================================================
    # Connection State
    # ==================================================================

    def _check_connection(self):
        """Detect connect / disconnect events."""
        connected = self.tcp.is_connected

        if not self._was_connected and connected:
            print("[TCP] Brain connected!")
            self._was_connected = True

        elif self._was_connected and not connected:
            print("[TCP] Brain disconnected — stopping all motion")
            self._stop_all_activity()
            self._was_connected = False

    # ==================================================================
    # Keyboard (kept for manual testing)
    # ==================================================================

    def _handle_keyboard(self):
        key = self.keyboard.getKey()
        if key == -1:
            return

        # Walking
        if key == Keyboard.UP:
            self._start_kb_motion("forwards")
        elif key == Keyboard.DOWN:
            self._start_kb_motion("backwards")
        elif key == Keyboard.LEFT:
            self._start_kb_motion("side_left")
        elif key == Keyboard.RIGHT:
            self._start_kb_motion("side_right")

        # Body rotation
        elif key == ord("Q") or key == ord("q"):
            self._start_kb_motion("turn_left_40")
        elif key == ord("E") or key == ord("e"):
            self._start_kb_motion("turn_right_40")
        elif key == (Keyboard.LEFT | Keyboard.SHIFT):
            self._start_kb_motion("turn_left_60")
        elif key == (Keyboard.RIGHT | Keyboard.SHIFT):
            self._start_kb_motion("turn_right_60")
        elif key == ord("T") or key == ord("t"):
            self._start_kb_motion("turn_left_180")

        # Head control
        elif key == ord("H") or key == ord("h"):
            self._desired_head_yaw = min(
                self._desired_head_yaw + 0.2, HEAD_YAW_MAX)
            self._set_joint("HeadYaw", self._desired_head_yaw)
        elif key == ord("J") or key == ord("j"):
            self._desired_head_yaw = max(
                self._desired_head_yaw - 0.2, HEAD_YAW_MIN)
            self._set_joint("HeadYaw", self._desired_head_yaw)
        elif key == ord("U") or key == ord("u"):
            self._desired_head_pitch = max(
                self._desired_head_pitch - 0.15, HEAD_PITCH_MIN)
            self._set_joint("HeadPitch", self._desired_head_pitch)
        elif key == ord("N") or key == ord("n"):
            self._desired_head_pitch = min(
                self._desired_head_pitch + 0.15, HEAD_PITCH_MAX)
            self._set_joint("HeadPitch", self._desired_head_pitch)
        elif key == ord("0"):
            self._desired_head_yaw = 0.0
            self._desired_head_pitch = -0.17
            self._set_joint("HeadYaw", 0.0)
            self._set_joint("HeadPitch", -0.17)
            print("[Head] Centered")

        # Hands
        elif key == ord("O") or key == ord("o"):
            self._open_both_hands()
            print("[Hands] Opened")
        elif key == ord("C") or key == ord("c"):
            self._close_both_hands()
            print("[Hands] Closed")

        # Animations
        elif key == ord("W") or key == ord("w"):
            self._start_kb_motion("hand_wave")

        # Stop
        elif key == ord(" "):
            self._stop_all_activity()
            self._go_to_posture(STAND_INIT)
            self._desired_head_yaw = 0.0
            self._desired_head_pitch = -0.17
            self.state.posture = "standing"
            print("[Motion] All stopped, StandInit")

        # Status
        elif key == ord("P") or key == ord("p"):
            self._print_status()
        elif key == ord("?") or key == ord("/"):
            self._print_help()

    def _start_kb_motion(self, key):
        """Start a motion from keyboard (not tracked as TCP task)."""
        if self._current_kb_motion:
            self._current_kb_motion.stop()
        motion = self.motions.get(key)
        if motion:
            motion.play()
            self._current_kb_motion = motion
            print("[KB] %s" % key)

    # ==================================================================
    # Status & Help
    # ==================================================================

    def _print_status(self):
        print("=" * 55)
        print("  ElderGuard NAO — Status Report")
        print("=" * 55)
        yaw = self._get_joint_angle("HeadYaw")
        pitch = self._get_joint_angle("HeadPitch")
        print("  Head:      yaw=%.3f  pitch=%.3f rad" % (yaw, pitch))

        if self.gps:
            pos = self.gps.getValues()
            print("  Position:  x=%.2f  y=%.2f  z=%.2f" % (
                pos[0], pos[1], pos[2]))
        if self.inertial_unit:
            rpy = self.inertial_unit.getRollPitchYaw()
            print("  Orient:    roll=%.2f  pitch=%.2f  yaw=%.2f" % (
                rpy[0], rpy[1], rpy[2]))

        print("  State:     posture=%s  legs=%s  speech=%s  arms=%s" % (
            self.state.posture, self.state.legs,
            self.state.speech, self.state.arms))
        print("  TCP:       %s" % (
            "connected" if self.tcp.is_connected else "not connected"))

        tasks = []
        if self._legs_task:
            tasks.append("legs:%s" % self._legs_task.action)
        if self._speech_task:
            tasks.append("speech:%s" % self._speech_task.action)
        if self._arms_task:
            tasks.append("arms:%s" % self._arms_task.action)
        if self._velocity_walk_active:
            tasks.append("velocity_walk")
        print("  Tasks:     %s" % (", ".join(tasks) if tasks else "none"))

        if self.camera_top:
            img = self.camera_top.getImage()
            if img:
                print("  Camera:    %dx%d (%d bytes)" % (
                    self.camera_top.getWidth(),
                    self.camera_top.getHeight(), len(img)))
        print("=" * 55)

    def _print_help(self):
        print("=" * 55)
        print("  ElderGuard NAO — Keyboard Controls")
        print("=" * 55)
        print("  [Up/Down]       Walk forward / backward")
        print("  [Left/Right]    Side step left / right")
        print("  [Q/E]           Rotate body left / right (40 deg)")
        print("  [Shift+Left]    Rotate body left (60 deg)")
        print("  [Shift+Right]   Rotate body right (60 deg)")
        print("  [T]             Turn around (180 deg)")
        print("  [H/J]           Head left / right")
        print("  [U/N]           Head up / down")
        print("  [0]             Center head")
        print("  [O/C]           Open / close hands")
        print("  [W]             Wave hello")
        print("  [Space]         Stop all motion")
        print("  [P]             Print status")
        print("  [?]             Print this help")
        print("")
        print("  TCP: port 5555 (commands)  port 5556 (camera)")
        print("=" * 55)

    # ==================================================================
    # Built-in Person Tracking (Option B — uses Supervisor 3D positions)
    # ==================================================================

    def _start_person_tracking(self, walk_enabled=False):
        """Start tracking the virtual person with head (and optionally body)."""
        self._stop_navigation()  # mutual exclusion with navigate_to
        self._tracking_active = True
        self._tracking_walk = walk_enabled
        self._tracking_step = 0
        mode = "HEAD+WALK" if walk_enabled else "HEAD ONLY"
        print("[TRACK] Person tracking started (%s)" % mode)

    def _stop_person_tracking(self):
        """Stop person tracking."""
        if getattr(self, '_tracking_active', False):
            self._tracking_active = False
            self._tracking_walk = False
            print("[TRACK] Person tracking stopped")

    def _update_person_tracking(self):
        """Called every step — compute direction to person and move head/body."""
        if not getattr(self, '_tracking_active', False):
            return
        if not self._person_node:
            return

        self._tracking_step += 1

        # Only update every 4 steps (~60ms) to avoid jitter
        if self._tracking_step % 4 != 0:
            return

        # Get person head position (person origin + head height offset)
        person_pos = self._get_person_position()
        if not person_pos:
            return
        # Person head is ~1.5m above their base Y=0, but position
        # reports the Pedestrian center, so head is near that Y value
        target = np.array([person_pos[0], person_pos[1], person_pos[2]])

        # Get NAO position from GPS
        if not self.gps:
            return
        gps = self.gps.getValues()
        nao_pos = np.array([gps[0], gps[1], gps[2]])

        # Get NAO orientation from inertial unit
        if not self.inertial_unit:
            return
        rpy = self.inertial_unit.getRollPitchYaw()
        nao_yaw = rpy[2]  # Yaw around Z axis (Z-up world)

        # Vector from NAO to person in world frame
        delta = target - nao_pos
        dx, dy, dz = delta[0], delta[1], delta[2]

        # Horizontal distance (XY plane — Z is up in this world)
        horiz_dist = math.sqrt(dx * dx + dy * dy)

        # Angle to person in world frame (XY plane, from +X axis CCW)
        world_angle = math.atan2(dy, dx)

        # Relative angle = how much NAO needs to rotate to face person
        rel_angle = world_angle - nao_yaw

        # Normalize to [-pi, pi]
        rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))

        # Head yaw: directly use relative angle (clamped to head limits)
        head_yaw = max(HEAD_YAW_MIN, min(HEAD_YAW_MAX, rel_angle))

        # Head pitch: look up/down based on height difference (Z is vertical)
        head_pitch = -math.atan2(dz - 0.5, max(horiz_dist, 0.3))
        head_pitch = max(HEAD_PITCH_MIN, min(HEAD_PITCH_MAX, head_pitch))

        # Apply head tracking
        self._desired_head_yaw = head_yaw
        self._desired_head_pitch = head_pitch
        self._set_joint("HeadYaw", head_yaw)
        self._set_joint("HeadPitch", head_pitch)

        # Body walking (if enabled and far enough)
        if not self._tracking_walk:
            return
        if self.state.posture != "standing":
            return

        # Only update walk every 20 steps (~400ms)
        if self._tracking_step % 20 != 0:
            return

        if horiz_dist < 0.6:
            # Close enough — stop walking
            if self._velocity_walk_active:
                self._stop_velocity_walk()
                self.state.legs = "idle"
            return

        # If head is turned far, body needs to turn too
        if abs(head_yaw) > 0.5 and horiz_dist > 0.6:
            # Turn body toward person by playing turn motion
            if not self._velocity_walk_active and not self._legs_task:
                turn = head_yaw * 0.8  # Don't over-turn
                if abs(turn) >= 0.35:
                    keys = []
                    if turn > 0.8:
                        keys.append("turn_left_60")
                    elif turn > 0.35:
                        keys.append("turn_left_40")
                    elif turn < -0.8:
                        keys.append("turn_right_60")
                    elif turn < -0.35:
                        keys.append("turn_right_40")
                    if keys:
                        self._legs_task = MotionTask(
                            None, "walk_toward", keys, self,
                            {"action": "walk_toward"})
                        self.state.legs = "walking"
        elif abs(head_yaw) < 0.4 and horiz_dist > 0.6:
            # Roughly facing person — walk forward
            if not self._legs_task:
                self._legs_task = MotionTask(
                    None, "walk_toward", ["forwards"], self,
                    {"action": "walk_toward"})
                self.state.legs = "walking"

    # ==================================================================
    # Navigate-To Tracking (generalized position-based navigation)
    # ==================================================================

    def _stop_navigation(self):
        """Stop navigate_to loop and clean up."""
        if self._nav_active:
            self._nav_active = False
            self._interrupt_legs()
            # Send cancelled done response if there's a pending request
            if self._nav_request_id:
                resp = self._build_response(
                    "navigate_to", self._nav_request_id,
                    resp_type="done", status="ok")
                resp["arrived"] = False
                resp["cancelled"] = True
                self.tcp.send_response(resp)
                self._nav_request_id = None
            print("[NAV] Navigation stopped")

    def _update_navigation(self):
        """Called every step — navigate toward self._nav_target."""
        if not self._nav_active:
            return
        if self._nav_target is None:
            return

        self._nav_step += 1

        # Only update every 4 steps (~80ms) to avoid jitter
        if self._nav_step % 4 != 0:
            return

        target = np.array(self._nav_target)

        # Get NAO position from GPS
        if not self.gps:
            return
        gps = self.gps.getValues()
        nao_pos = np.array([gps[0], gps[1], gps[2]])

        # Get NAO orientation from inertial unit
        if not self.inertial_unit:
            return
        rpy = self.inertial_unit.getRollPitchYaw()
        nao_yaw = rpy[2]

        # Vector from NAO to target in world frame
        delta = target - nao_pos
        dx, dy, dz = delta[0], delta[1], delta[2]

        # Horizontal distance (XY plane — Z is up in this world)
        horiz_dist = math.sqrt(dx * dx + dy * dy)

        # Angle to target in world frame (XY plane, from +X axis CCW)
        world_angle = math.atan2(dy, dx)

        # Relative angle = how much NAO needs to rotate
        rel_angle = world_angle - nao_yaw
        rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))

        # Head yaw: directly use relative angle (clamped)
        head_yaw = max(HEAD_YAW_MIN, min(HEAD_YAW_MAX, rel_angle))

        # Head pitch: look down for ground objects, up for person-height targets (Z is vertical)
        head_pitch = -math.atan2(dz - 0.5, max(horiz_dist, 0.3))
        head_pitch = max(HEAD_PITCH_MIN, min(HEAD_PITCH_MAX, head_pitch))

        # Apply head tracking
        self._desired_head_yaw = head_yaw
        self._desired_head_pitch = head_pitch
        self._set_joint("HeadYaw", head_yaw)
        self._set_joint("HeadPitch", head_pitch)

        # Check if arrived
        if horiz_dist < self._nav_stop_dist:
            self._nav_active = False
            if self._velocity_walk_active:
                self._stop_velocity_walk()
                self.state.legs = "idle"
            if self._legs_task:
                self._legs_task.stop()
                self._legs_task = None
                self.state.legs = "idle"
            # Send done response
            if self._nav_request_id:
                resp = self._build_response(
                    "navigate_to", self._nav_request_id,
                    resp_type="done")
                resp["arrived"] = True
                resp["final_distance"] = round(horiz_dist, 3)
                self.tcp.send_response(resp)
                self._nav_request_id = None
            print("[NAV] Arrived (dist=%.3f)" % horiz_dist)
            return

        # Only update walk every 20 steps (~400ms)
        if self._nav_step % 20 != 0:
            return

        if self.state.posture != "standing":
            return

        # Turn body if needed
        if abs(head_yaw) > 0.5 and horiz_dist > self._nav_stop_dist:
            if not self._velocity_walk_active and not self._legs_task:
                turn = head_yaw * 0.8  # Don't over-turn
                if abs(turn) >= 0.35:
                    keys = []
                    if turn > 0.8:
                        keys.append("turn_left_60")
                    elif turn > 0.35:
                        keys.append("turn_left_40")
                    elif turn < -0.8:
                        keys.append("turn_right_60")
                    elif turn < -0.35:
                        keys.append("turn_right_40")
                    if keys:
                        self._legs_task = MotionTask(
                            None, "walk_toward", keys, self,
                            {"action": "walk_toward"})
                        self.state.legs = "walking"
        elif abs(head_yaw) < 0.4 and horiz_dist > self._nav_stop_dist:
            # Roughly facing target — walk forward
            if not self._legs_task:
                self._legs_task = MotionTask(
                    None, "walk_toward", ["forwards"], self,
                    {"action": "walk_toward"})
                self.state.legs = "walking"

    # ==================================================================
    # Object Carrying (Supervisor-based fake grasping)
    # ==================================================================

    def _start_carrying(self, def_name):
        """Start teleporting an object to track NAO's right hand."""
        node = self._object_nodes.get(def_name.upper())
        if not node:
            print("[CARRY] Object '%s' not found" % def_name)
            return False
        self._carrying_object = node
        self._carry_active = True
        print("[CARRY] Now carrying: %s" % def_name)
        return True

    def _stop_carrying(self):
        """Stop teleporting the carried object (release it)."""
        if self._carry_active and self._carrying_object:
            try:
                self._carrying_object.resetPhysics()
            except Exception:
                pass
            print("[CARRY] Released object")
        self._carrying_object = None
        self._carry_active = False

    def _update_carrying(self):
        """Teleport carried object to NAO's right hand position each step."""
        if not self._carry_active or not self._carrying_object:
            return
        if not self.gps or not self.inertial_unit:
            return

        gps = self.gps.getValues()
        rpy = self.inertial_unit.getRollPitchYaw()
        nao_yaw = rpy[2]

        # Hand offset from NAO torso in NAO's local frame (tunable)
        # forward = along NAO's facing direction, up = vertical (Z), right = NAO's right
        local_forward = 0.05   # slightly in front of torso
        local_up = 0.05        # slightly above GPS position (Z axis)
        local_right = -0.12    # to the right side

        # Rotate local offset by NAO yaw to get world offset
        # Z-up world: forward = (sin(yaw), cos(yaw), 0), right = (cos(yaw), -sin(yaw), 0)
        cos_y = math.cos(nao_yaw)
        sin_y = math.sin(nao_yaw)

        world_dx = local_forward * sin_y + local_right * cos_y
        world_dy = local_forward * cos_y - local_right * sin_y
        world_dz = local_up

        target_pos = [
            gps[0] + world_dx,
            gps[1] + world_dy,
            gps[2] + world_dz,
        ]

        try:
            trans_field = self._carrying_object.getField("translation")
            trans_field.setSFVec3f(target_pos)
            self._carrying_object.resetPhysics()
        except Exception as e:
            print("[CARRY] Teleport failed: %s" % e)

    # ==================================================================
    # Camera Streaming
    # ==================================================================

    def _stream_camera_frame(self):
        """Grab a frame from CameraTop and send it to the camera client."""
        self.cam_server.poll()  # Accept new camera clients (non-blocking)

        if not self.cam_server.is_connected:
            return
        if not self.camera_top:
            return

        image = self.camera_top.getImage()
        if image:
            w = self.camera_top.getWidth()
            h = self.camera_top.getHeight()
            self.cam_server.send_frame(image, w, h)

    # ==================================================================
    # Main Loop
    # ==================================================================

    def run(self):
        """Main control loop — processes TCP commands + keyboard input."""
        while self.step(self.time_step) != -1:
            # 1. Check connection state
            self._check_connection()

            # 2. Process all pending TCP commands
            for cmd in self.tcp.poll_commands():
                self._dispatch(cmd)

            # 3. Advance async tasks
            self._update_tasks()

            # 4. Person tracking (if active)
            self._update_person_tracking()

            # 4b. Navigate-to tracking (if active)
            self._update_navigation()

            # 4c. Object carrying (if active)
            self._update_carrying()

            # 5. Stream camera frame (~15 FPS: every 4th step at 16ms)
            self._cam_frame_counter += 1
            if self._cam_frame_counter % 4 == 0:
                self._stream_camera_frame()

            # 6. Handle keyboard input
            self._handle_keyboard()

        # Clean shutdown
        self.cam_server.stop()
        self.tcp.stop()
        print("[ElderGuard NAO] Controller shutdown.")


# ===================================================================
# Entry Point
# ===================================================================
if __name__ == "__main__":
    print("[ElderGuard NAO] Starting controller...")
    nao = ElderGuardNao()
    nao.run()
