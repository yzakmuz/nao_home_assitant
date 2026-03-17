#!/usr/bin/env python3
"""
run_simulation.py -- Single entry point for the ElderGuard PC simulation.

Boots the entire system on a single PC:
  1. Patches imports (bootstrap)
  2. Starts mock NAO server (real server.py with mock proxies)
  3. Starts the brain (real main.py AssistantApp)
  4. Runs the OpenCV dashboard (main thread)

Usage:
    python run_simulation.py                          # Full mode (mic + camera)
    python run_simulation.py --keyboard --no-camera   # Keyboard + synthetic camera
    python run_simulation.py --keyboard               # Keyboard + real camera
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


# ======================================================================
# Shared Simulation State
# ======================================================================
@dataclass
class SharedSimState:
    """Thread-safe container shared between brain, server, and GUI."""

    lock: threading.Lock = field(default_factory=threading.Lock)

    # Camera + Vision
    latest_frame: Optional[np.ndarray] = None
    face_detection: Optional[Dict[str, Any]] = None
    yolo_detections: List[Dict[str, Any]] = field(default_factory=list)

    # FSM
    fsm_state: str = "IDLE"

    # NAO State (from NaoStateCache)
    nao_posture: str = "standing"
    nao_head: str = "idle"
    nao_legs: str = "idle"
    nao_speech: str = "idle"
    nao_arms: str = "idle"
    head_yaw: float = 0.0
    head_pitch: float = 0.0

    # Servo
    servo_running: bool = False
    servo_mode: str = "off"          # "off", "head_only", "full_follow"
    servo_searching: bool = False    # True during person search
    servo_search_phase: str = "none" # "none", "head_scan", "body_rotate"
    pid_error_x: float = 0.0
    pid_error_y: float = 0.0

    # Audio
    audio_level: float = 0.0
    stt_text: str = ""
    speaker_score: float = 0.0
    speaker_accepted: bool = False
    speaker_name: str = ""

    # Fall Detection (Improvement 4)
    pose_keypoints: Optional[List[Dict[str, Any]]] = None
    fall_state: str = "INACTIVE"
    fall_score: float = 0.0
    person_fall_detected: bool = False

    # Logs
    command_log: List[Dict[str, Any]] = field(default_factory=list)
    action_log: queue.Queue = field(default_factory=queue.Queue)

    # Control
    command_queue: queue.Queue = field(default_factory=queue.Queue)
    shutdown_event: threading.Event = field(default_factory=threading.Event)

    # References (set after server starts)
    _mock_proxies: Any = None
    _tcp_client: Any = None
    _fall_monitor: Any = None

    def snapshot(self) -> Dict[str, Any]:
        """Return a frozen copy of all display-relevant state."""
        with self.lock:
            return {
                "latest_frame": self.latest_frame.copy() if self.latest_frame is not None else None,
                "face_detection": self.face_detection,
                "yolo_detections": list(self.yolo_detections),
                "fsm_state": self.fsm_state,
                "nao_posture": self.nao_posture,
                "nao_head": self.nao_head,
                "nao_legs": self.nao_legs,
                "nao_speech": self.nao_speech,
                "nao_arms": self.nao_arms,
                "head_yaw": self.head_yaw,
                "head_pitch": self.head_pitch,
                "servo_running": self.servo_running,
                "servo_mode": self.servo_mode,
                "servo_searching": self.servo_searching,
                "servo_search_phase": self.servo_search_phase,
                "pid_error_x": self.pid_error_x,
                "pid_error_y": self.pid_error_y,
                "audio_level": self.audio_level,
                "stt_text": self.stt_text,
                "speaker_score": self.speaker_score,
                "speaker_accepted": self.speaker_accepted,
                "speaker_name": self.speaker_name,
                "command_log": list(self.command_log[-20:]),
                "pose_keypoints": self.pose_keypoints,
                "fall_state": self.fall_state,
                "fall_score": self.fall_score,
                "person_fall_detected": self.person_fall_detected,
            }

    def inject_command(self, text: str) -> None:
        """Inject a command from hotkey (bypasses voice pipeline)."""
        try:
            self.command_queue.put_nowait(("inject_command", text))
        except queue.Full:
            pass

    def inject_fall(self) -> None:
        """Simulate a robot fall event."""
        if self._mock_proxies is not None:
            self._mock_proxies.memory.simulate_fall()

    def clear_fall(self) -> None:
        """Clear a simulated fall."""
        if self._mock_proxies is not None:
            self._mock_proxies.memory.clear_fall()

    def inject_person_fall(self) -> None:
        """Simulate a person fall (inject synthetic pose into fall monitor).

        Sets the pose estimator to return fallen-pose data.  The fall
        detector will naturally trigger after FALL_CONFIRMATION_FRAMES.
        """
        if self._fall_monitor is not None:
            pose_est = getattr(self._fall_monitor, '_pose_estimator', None)
            if pose_est is not None and hasattr(pose_est, 'inject_fall_pose'):
                pose_est.inject_fall_pose()

    def clear_person_fall(self) -> None:
        """Clear a simulated person fall."""
        if self._fall_monitor is not None:
            pose_est = getattr(self._fall_monitor, '_pose_estimator', None)
            if pose_est is not None and hasattr(pose_est, 'clear_fall_pose'):
                pose_est.clear_fall_pose()
            # Reset the fall detector state
            self._fall_monitor.fall_detector.reset()
            self.person_fall_detected = False

    def inject_disconnect(self) -> None:
        """Simulate a connection disconnect (for watchdog testing)."""
        if self._tcp_client is not None:
            try:
                self._tcp_client.disconnect()
            except Exception:
                pass

    def add_command_log(self, action: str, text: str = "",
                        status: str = "ok") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        entry = {"timestamp": ts, "action": action, "text": text, "status": status}
        with self.lock:
            self.command_log.append(entry)
            if len(self.command_log) > 100:
                self.command_log = self.command_log[-50:]


# ======================================================================
# CLI Arguments
# ======================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ElderGuard PC Simulation")
    p.add_argument("--keyboard", action="store_true",
                    help="Type commands instead of speaking (alias for --no-mic --no-verify)")
    p.add_argument("--no-camera", action="store_true",
                    help="Use synthetic camera frames")
    p.add_argument("--no-mic", action="store_true",
                    help="Disable microphone")
    p.add_argument("--no-verify", action="store_true",
                    help="Skip speaker verification")
    p.add_argument("--no-tts", action="store_true",
                    help="Disable PC text-to-speech output")
    p.add_argument("--no-yolo", action="store_true",
                    help="Disable object detection")
    p.add_argument("--camera", type=int, default=0,
                    help="Camera device index (default: 0)")
    p.add_argument("--speed", type=float, default=1.0,
                    help="Speed multiplier (0.5=fast, 1.0=real-time, 2.0=slow)")
    p.add_argument("--master", type=str, default=None,
                    help="Set master speaker for this session (e.g., david, itzhak)")
    p.add_argument("--verbose", action="store_true",
                    help="Debug-level logging")
    return p.parse_args()


# ======================================================================
# Configuration
# ======================================================================
def configure_sim_settings(args: argparse.Namespace) -> None:
    """Apply CLI args to sim_config BEFORE bootstrap."""
    # We need to modify sim_config before it gets installed as "settings"
    import sim_config

    if args.keyboard:
        args.no_mic = True
        args.no_verify = True

    sim_config.SIM_NO_MIC = args.no_mic
    sim_config.SIM_NO_CAMERA = args.no_camera
    sim_config.SIM_SKIP_VERIFY = args.no_verify
    sim_config.SIM_NO_YOLO = args.no_yolo
    sim_config.SIM_SPEED_MULTIPLIER = args.speed
    sim_config.CAMERA_INDEX = args.camera

    if args.no_tts:
        sim_config.SIM_PC_TTS = False

    if args.verbose:
        sim_config.LOG_LEVEL = "DEBUG"

    # Runtime master selection: copy named embedding to master_embedding.npy
    # In multi-speaker mode, --master restricts acceptance to ONE person only
    if args.master:
        sim_config.MULTI_SPEAKER_MODE = False  # restrict to single master
        _models_dir = os.path.dirname(sim_config.MASTER_EMBEDDING_PATH)
        named_emb = os.path.join(_models_dir, f"{args.master.lower()}_embedding.npy")
        if os.path.isfile(named_emb):
            import shutil
            shutil.copy2(named_emb, sim_config.MASTER_EMBEDDING_PATH)
            print(f"[OK] Master speaker set to: {args.master.upper()} (single-master mode)")
        else:
            print(f"[ERROR] No embedding found for '{args.master}' at {named_emb}")
            print("  Run: python enroll_from_data.py --set-master " + args.master)
            sys.exit(1)


# ======================================================================
# Dependency Checks
# ======================================================================
def check_dependencies(sim_config) -> None:
    """Check for required dependencies and print helpful messages."""
    issues = []

    # Check Vosk model
    if not os.path.isdir(sim_config.VOSK_MODEL_PATH):
        print(f"\n[WARNING] Vosk model not found at: {sim_config.VOSK_MODEL_PATH}")
        print("  Download from: https://alphacephei.com/vosk/models")
        print("  Extract to: nao_assistant/rpi_brain/models/vosk-model-small-en-us-0.15/")
        print("  -> Falling back to keyboard input mode.\n")
        sim_config.SIM_NO_MIC = True

    # Check ECAPA model
    if not os.path.isfile(sim_config.SPEAKER_MODEL_PATH):
        print(f"\n[INFO] ECAPA model not found at: {sim_config.SPEAKER_MODEL_PATH}")
        print("  -> Skipping speaker verification.\n")
        sim_config.SIM_SKIP_VERIFY = True

    # Check master embedding
    if not os.path.isfile(sim_config.MASTER_EMBEDDING_PATH):
        if not sim_config.SIM_SKIP_VERIFY:
            print(f"\n[INFO] Master embedding not found at: {sim_config.MASTER_EMBEDDING_PATH}")
            print("  -> Skipping speaker verification.\n")
            sim_config.SIM_SKIP_VERIFY = True

    # Check microphone
    if not sim_config.SIM_NO_MIC:
        try:
            import sounddevice as sd
            dev = sd.query_devices(kind="input")
            print(f"[OK] Microphone detected: {dev['name']} "
                  f"(native {int(dev['default_samplerate'])} Hz)")
        except Exception as e:
            print(f"\n[WARNING] No microphone detected: {e}")
            print("  -> Falling back to keyboard input mode.\n")
            sim_config.SIM_NO_MIC = True

    # Check ultralytics
    if not sim_config.SIM_NO_YOLO:
        try:
            import ultralytics
        except ImportError:
            print("\n[WARNING] ultralytics not installed.")
            print("  pip install ultralytics")
            print("  -> YOLO object detection disabled.\n")
            sim_config.SIM_NO_YOLO = True


# ======================================================================
# Server Readiness Check
# ======================================================================
def wait_for_server(port: int, timeout: float = 10.0) -> bool:
    """Wait for the mock server to start accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=1.0)
            s.close()
            return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.2)
    return False


# ======================================================================
# Brain Hooks -- Feed simulation state to the GUI
# ======================================================================
def inject_sim_hooks(app, shared_state: SharedSimState) -> None:
    """Monkey-patch the brain to feed state updates to the GUI."""
    from gui.event_bus import (
        get_event_bus, SimEventBus,
        CAT_STT, CAT_VERIFY, CAT_FSM, CAT_CMD, CAT_SYS,
        SEV_INFO, SEV_COMMAND, SEV_RESPONSE, SEV_WARNING, SEV_ERROR,
    )
    bus = get_event_bus()

    # Store reference to command queue for hotkey injection
    app._sim_command_queue = shared_state.command_queue
    app._sim_shared_state = shared_state

    # Store TCP client reference for disconnect simulation
    shared_state._tcp_client = app._tcp

    # Store fall monitor reference for person fall simulation
    if hasattr(app, '_fall_monitor'):
        shared_state._fall_monitor = app._fall_monitor

    # -- Hook 1: Patch _handle_idle to check hotkey command queue --
    # The original _handle_idle blocks indefinitely in wait_for_wake_word.
    # This patched version polls both the hotkey command queue AND the
    # wake-word source in a non-blocking loop so hotkeys always work.
    original_handle_idle = app._handle_idle.__func__

    def patched_handle_idle(self):
        from command.parser import parse_command
        from main import State
        from utils.memory import check_pressure, emergency_cleanup

        # Memory housekeeping (from original _handle_idle)
        pressure = check_pressure()
        if pressure == "critical":
            emergency_cleanup()

        is_keyboard = hasattr(self._stt, '_input_queue')
        has_vosk = hasattr(self._stt, '_new_recognizer')

        # Fallback: unknown STT type — use original (blocking) behavior
        if not is_keyboard and not has_vosk:
            try:
                cmd_type, cmd_text = self._sim_command_queue.get_nowait()
                if cmd_type == "inject_command":
                    self._pending_text = cmd_text
                    self._current_intent = parse_command(cmd_text)
                    self._state = State.EXECUTING
                    shared_state.add_command_log(
                        self._current_intent.type.name, cmd_text, "injected"
                    )
                    bus.emit(CAT_CMD, f'Hotkey: "{cmd_text}"', SEV_COMMAND)
                    return
            except queue.Empty:
                pass
            original_handle_idle(self)
            return

        # Create Vosk recognizer for voice mode (keyboard mode skips this)
        vosk_rec = None
        if has_vosk and not is_keyboard:
            vosk_rec = self._stt._new_recognizer()

        # Non-blocking loop: poll BOTH hotkey queue and wake-word source
        while True:
            # --- Priority 1: hotkey command injection ---
            try:
                cmd_type, cmd_text = self._sim_command_queue.get_nowait()
                if cmd_type == "inject_command":
                    self._pending_text = cmd_text
                    self._current_intent = parse_command(cmd_text)
                    self._state = State.EXECUTING
                    shared_state.add_command_log(
                        self._current_intent.type.name, cmd_text, "injected"
                    )
                    bus.emit(CAT_CMD, f'Hotkey: "{cmd_text}"', SEV_COMMAND)
                    return
            except queue.Empty:
                pass

            # --- Priority 2: wake-word detection (short timeout) ---
            if is_keyboard:
                # Keyboard mode: check the STT's stdin queue
                try:
                    text = self._stt._input_queue.get(timeout=0.15)
                    if text:
                        # User typed a command — put it back for listen_command
                        self._stt._input_queue.put(text)
                    # Wake word — proceed to LISTENING
                    self._tcp.send_fire_and_forget(
                        {"action": "say", "text": "Yes?"}
                    )
                    bus.emit(CAT_STT, 'Wake word (keyboard)', SEV_INFO)
                    self._mic.start_recording()
                    self._state = State.LISTENING
                    return
                except queue.Empty:
                    continue
            else:
                # Voice mode: read one mic chunk and feed to Vosk
                import settings as _settings
                chunk = self._mic.read(timeout=0.2)
                if chunk is None:
                    continue
                if vosk_rec.AcceptWaveform(chunk):
                    result = json.loads(vosk_rec.Result())
                    text = result.get("text", "").strip().lower()
                    if _settings.WAKE_WORD in text:
                        # Extract any command spoken AFTER the wake word
                        # e.g. "hey nao follow me" → remainder = "follow me"
                        wk_end = (text.index(_settings.WAKE_WORD)
                                  + len(_settings.WAKE_WORD))
                        remainder = text[wk_end:].strip()
                        self._tcp.send_fire_and_forget(
                            {"action": "say", "text": "Yes?"}
                        )
                        bus.emit(CAT_STT,
                                 f'Wake word: "{_settings.WAKE_WORD}"',
                                 SEV_INFO)
                        if remainder:
                            # Command included with wake word — execute
                            # directly (skip LISTENING timeout)
                            bus.emit(CAT_STT,
                                     f'Command (inline): "{remainder}"',
                                     SEV_INFO)
                            self._pending_text = remainder
                            self._current_intent = parse_command(remainder)
                            self._state = State.EXECUTING
                            shared_state.add_command_log(
                                self._current_intent.type.name,
                                remainder, "voice"
                            )
                            return
                        # Just wake word, no command — proceed to LISTENING
                        self._mic.start_recording()
                        self._state = State.LISTENING
                        return
                    vosk_rec.Reset()
                else:
                    partial = json.loads(vosk_rec.PartialResult())
                    self._stt.last_partial = partial.get(
                        "partial", ""
                    ).strip().lower()

    import types
    app._handle_idle = types.MethodType(patched_handle_idle, app)

    # -- Hook 1b: Patch _handle_listening to capture STT command result --
    original_handle_listening = app._handle_listening.__func__

    def patched_handle_listening(self):
        original_handle_listening(self)
        from main import State
        if self._state == State.VERIFYING or self._state == State.EXECUTING:
            text = getattr(self, '_pending_text', '')
            bus.emit(CAT_STT, f'Command: "{text}"', SEV_INFO)
        elif self._state == State.IDLE:
            bus.emit(CAT_STT, "No command recognized (timeout)", SEV_WARNING)

    app._handle_listening = types.MethodType(patched_handle_listening, app)

    # -- Hook 1c: Patch _handle_verifying to capture verify result --
    original_handle_verifying = app._handle_verifying.__func__

    def patched_handle_verifying(self):
        original_handle_verifying(self)
        from main import State
        name = getattr(self, '_last_verified_speaker', 'unknown')
        score = getattr(self, '_last_verify_score', 0.0)
        if self._state == State.EXECUTING:
            bus.emit(CAT_VERIFY,
                     f"Accepted: {name} (score={score:.2f})",
                     SEV_RESPONSE)
        elif self._state == State.IDLE:
            bus.emit(CAT_VERIFY,
                     f"Rejected (score={score:.2f})",
                     SEV_WARNING)

    app._handle_verifying = types.MethodType(patched_handle_verifying, app)

    # -- Hook 2: Patch _main_loop with FSM transition tracking --
    original_main_loop = app._main_loop.__func__

    def patched_main_loop(self):
        from main import State
        prev_state = self._state
        while self._running:
            # Priority: check for person fall BEFORE any state logic
            if hasattr(self, '_fall_event') and self._fall_event.is_set():
                self._fall_event.clear()
                self._respond_to_person_fall()
                bus.emit(CAT_SYS, "Person fall response triggered", SEV_ERROR)
                continue

            # Check if person search failed (from servo thread)
            if hasattr(self, '_search_failed_event') and self._search_failed_event.is_set():
                self._search_failed_event.clear()
                self._servo.stop()
                self._state = State.IDLE
                bus.emit(CAT_SYS, "Person search failed — returning to IDLE", SEV_WARNING)
                continue

            # Detect FSM transitions
            current = self._state
            if current != prev_state:
                bus.emit(CAT_FSM,
                         f"{prev_state.name} -> {current.name}",
                         SEV_INFO)
                # Emit extra system events for notable transitions
                if current.name == "SEARCHING":
                    bus.emit(CAT_SYS, "YOLO search started", SEV_INFO)
                elif prev_state.name == "SEARCHING":
                    bus.emit(CAT_SYS, "YOLO search ended", SEV_INFO)
                prev_state = current

            try:
                if current == State.IDLE:
                    self._handle_idle()
                elif current == State.LISTENING:
                    self._handle_listening()
                elif current == State.VERIFYING:
                    self._handle_verifying()
                elif current == State.EXECUTING:
                    self._handle_executing()
                elif current == State.SEARCHING:
                    self._handle_searching()
                else:
                    break
            except Exception:
                logging.getLogger("main").exception(
                    "Unhandled error in state %s", self._state.name
                )
                bus.emit(CAT_SYS,
                         f"Error in state {self._state.name}",
                         SEV_ERROR)
                self._state = State.IDLE
                time.sleep(1.0)

    app._main_loop = types.MethodType(patched_main_loop, app)

    # -- Hook 3: Patch _send_checked to log commands --
    original_send_checked = app._send_checked.__func__

    def patched_send_checked(self, command, speak_error=True):
        result = original_send_checked(self, command, speak_error)
        action = command.get("action", "?")
        text = command.get("text", command.get("name", ""))
        status = "ok" if result else "error"
        shared_state.add_command_log(action, str(text), status)

        # Emit to event bus (skip high-frequency actions)
        if not SimEventBus.should_skip_action(action):
            desc = action
            if text:
                desc += f' "{str(text)[:30]}"'
            sev = SEV_COMMAND if result else SEV_ERROR
            bus.emit(CAT_CMD, desc, sev)

        return result

    app._send_checked = types.MethodType(patched_send_checked, app)


def _update_shared_state(app, shared_state: SharedSimState) -> None:
    """Pull current state from the brain into SharedSimState."""
    with shared_state.lock:
        shared_state.fsm_state = app._state.name

        # Camera frame — grab latest from camera (thread-safe)
        try:
            frame = app._camera.read()
            if frame is not None:
                shared_state.latest_frame = frame
        except Exception:
            pass

        # Face detection — read CACHED result from servo loop
        # NEVER call app._face_tracker.detect() here: MediaPipe is not
        # thread-safe, and calling from two threads causes timestamp
        # mismatch errors that permanently crash the graph.
        try:
            face = app._servo.last_face_result
            if face is not None:
                shared_state.face_detection = {
                    "cx": face.cx, "cy": face.cy,
                    "width": face.width, "height": face.height,
                    "confidence": face.confidence,
                }
            else:
                shared_state.face_detection = None
            # PID error from servo
            shared_state.pid_error_x, shared_state.pid_error_y = \
                app._servo.last_pid_error
        except Exception:
            pass

        # NAO state from cache
        try:
            ns = app._tcp.nao_state
            shared_state.nao_posture = ns.posture
            shared_state.nao_head = ns.head
            shared_state.nao_legs = ns.legs
            shared_state.nao_speech = ns.speech
            shared_state.nao_arms = ns.arms
            shared_state.head_yaw = ns.head_yaw
            shared_state.head_pitch = ns.head_pitch
        except Exception:
            pass

        # Servo state
        shared_state.servo_running = app._servo.is_running
        if app._servo.is_running:
            shared_state.servo_mode = (
                "full_follow" if app._servo.walk_enabled else "head_only"
            )
            shared_state.servo_searching = app._servo.search_active
            shared_state.servo_search_phase = app._servo.search_phase
        else:
            shared_state.servo_mode = "off"
            shared_state.servo_searching = False
            shared_state.servo_search_phase = "none"

        # Audio level from mic (if real mic is active)
        try:
            if hasattr(app._mic, 'audio_level'):
                shared_state.audio_level = app._mic.audio_level
        except Exception:
            pass

        # STT partial text (if available)
        try:
            if hasattr(app._stt, 'last_partial'):
                shared_state.stt_text = app._stt.last_partial
        except Exception:
            pass

        # Speaker verification result (multi-speaker identity)
        try:
            shared_state.speaker_name = app._last_verified_speaker
            shared_state.speaker_score = app._last_verify_score
        except Exception:
            pass

        # Fall detection state (Improvement 4)
        try:
            fm = app._fall_monitor
            if fm.is_running:
                shared_state.fall_state = fm.fall_detector_state
                shared_state.fall_score = fm.fall_score
                shared_state.person_fall_detected = fm.fall_detector.is_triggered

                # Pose keypoints for skeleton overlay
                pose = fm.last_pose_result
                if pose is not None:
                    shared_state.pose_keypoints = [
                        {"name": n, "x": kp.x, "y": kp.y,
                         "visibility": kp.visibility}
                        for n, kp in pose.keypoints.items()
                    ]
                else:
                    shared_state.pose_keypoints = None
            else:
                shared_state.fall_state = "INACTIVE"
                shared_state.pose_keypoints = None
        except Exception:
            pass

    # --- Event bus: track NAO state changes (outside lock) ---
    _track_state_changes(shared_state)


def _track_state_changes(shared_state: SharedSimState) -> None:
    """Emit event-bus events when NAO channel states or servo change."""
    prev = getattr(shared_state, '_prev_nao', None)
    if prev is None:
        return

    from gui.event_bus import get_event_bus, CAT_NAO, CAT_SYS
    bus = get_event_bus()

    # Channel state transitions
    _NAO_FIELDS = [
        ("nao_posture", "posture", "Posture"),
        ("nao_legs",    "legs",    "Legs"),
        ("nao_speech",  "speech",  "Speech"),
        ("nao_arms",    "arms",    "Arms"),
    ]
    for attr, key, label in _NAO_FIELDS:
        curr = getattr(shared_state, attr, None)
        if curr is not None and curr != prev.get(key):
            bus.emit(CAT_NAO, f"{label}: {prev.get(key, '?')} -> {curr}")
            prev[key] = curr

    # Servo start/stop + mode
    servo = shared_state.servo_running
    servo_mode = shared_state.servo_mode
    if servo != prev.get("servo") or servo_mode != prev.get("servo_mode"):
        if servo:
            mode_label = "head-only" if servo_mode == "head_only" else "full follow"
            bus.emit(CAT_SYS, f"Servo started ({mode_label})")
        else:
            bus.emit(CAT_SYS, "Servo stopped")
        prev["servo"] = servo
        prev["servo_mode"] = servo_mode

    # Person search state
    searching = shared_state.servo_searching
    if searching != prev.get("servo_searching"):
        if searching:
            phase = shared_state.servo_search_phase
            bus.emit(CAT_SYS, f"Person search started ({phase})", SEV_WARNING)
        else:
            bus.emit(CAT_SYS, "Person search ended")
        prev["servo_searching"] = searching

    # Fall detection state changes
    from gui.event_bus import SEV_ERROR, SEV_WARNING
    fall_state = shared_state.fall_state
    if fall_state != prev.get("fall_state"):
        if fall_state == "TRIGGERED":
            bus.emit(CAT_SYS, "PERSON FALL DETECTED!", SEV_ERROR)
        elif fall_state == "MONITORING" and prev.get("fall_state") in ("RECOVERY", "TRIGGERED"):
            bus.emit(CAT_SYS, "Person recovered — fall detector re-armed")
        elif fall_state == "CALIBRATING":
            bus.emit(CAT_SYS, "Fall detector calibrating...")
        elif fall_state == "MONITORING" and prev.get("fall_state") == "CALIBRATING":
            bus.emit(CAT_SYS, "Fall detector calibrated — monitoring")
        prev["fall_state"] = fall_state


# ======================================================================
# JSONL Event Logger
# ======================================================================
class EventLogger:
    """Logs simulation events to a JSONL file."""

    def __init__(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(log_dir, f"sim_{ts}.jsonl")
        self._file = open(self._path, "w", encoding="utf-8")
        print(f"[logger] Event log: {self._path}")

    def log(self, event: str, **data) -> None:
        entry = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            **data,
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    # -- 1. Parse CLI args --
    args = parse_args()

    # -- 2. Add Simulation/ to sys.path so our modules are importable --
    sim_dir = os.path.dirname(os.path.abspath(__file__))
    if sim_dir not in sys.path:
        sys.path.insert(0, sim_dir)

    # -- 3. Configure sim_config based on args (BEFORE bootstrap) --
    configure_sim_settings(args)

    # -- 4. Run bootstrap (MUST be before any rpi_brain/nao_body import) --
    import sim_config
    from adapters.bootstrap import bootstrap
    check_dependencies(sim_config)
    bootstrap(sim_config)

    # -- 5. Setup logging --
    log_level = "DEBUG" if args.verbose else sim_config.LOG_LEVEL
    logging.basicConfig(
        level=log_level,
        format=sim_config.LOG_FORMAT,
    )
    log = logging.getLogger("simulation")

    # -- 6. Print banner --
    print("=" * 60)
    print("  ElderGuard Simulation")
    print("=" * 60)
    mode_parts = []
    if sim_config.SIM_NO_MIC:
        mode_parts.append("keyboard")
    else:
        mode_parts.append("microphone")
    if sim_config.SIM_NO_CAMERA:
        mode_parts.append("synthetic-camera")
    else:
        mode_parts.append("webcam")
    if sim_config.SIM_SKIP_VERIFY:
        mode_parts.append("no-verify")
    if sim_config.SIM_NO_YOLO:
        mode_parts.append("no-yolo")
    print(f"  Mode: {' + '.join(mode_parts)}")
    print(f"  Speed: {sim_config.SIM_SPEED_MULTIPLIER}x")
    print("=" * 60)

    # -- 7. Create shared state --
    shared_state = SharedSimState()

    # Initialize state-change tracking for the event bus
    shared_state._prev_nao = {
        "posture": "standing", "legs": "idle", "speech": "idle",
        "arms": "idle", "servo": False, "servo_mode": "off",
        "servo_searching": False, "fall_state": "INACTIVE",
    }

    # -- 7b. Initialize event bus and emit startup event --
    from gui.event_bus import get_event_bus, CAT_SYS
    sim_bus = get_event_bus()
    sim_bus.emit(CAT_SYS, "Simulation starting...")

    # -- 8. Start event logger --
    event_logger = EventLogger(sim_config.SIM_LOG_DIR)

    # -- 9. Create mock proxies and start server in daemon thread --
    from mock_nao.mock_server import create_mock_proxies, start_mock_server

    mock_proxies = create_mock_proxies(
        shared_state.action_log,
        speed_mult=sim_config.SIM_SPEED_MULTIPLIER,
    )
    shared_state._mock_proxies = mock_proxies  # for fall/disconnect sim

    server_thread = threading.Thread(
        target=start_mock_server,
        args=(mock_proxies, sim_config.NAO_PORT),
        daemon=True,
    )
    server_thread.start()

    # Wait for server to be ready
    if not wait_for_server(sim_config.NAO_PORT, timeout=10.0):
        print("[ERROR] Mock server failed to start within 10 seconds.")
        sys.exit(1)
    print("[OK] Mock NAO server ready on port %d." % sim_config.NAO_PORT)
    event_logger.log("server_started", port=sim_config.NAO_PORT)

    # -- 10. Create and configure the brain --
    from main import AssistantApp
    app = AssistantApp()
    inject_sim_hooks(app, shared_state)

    event_logger.log("brain_created")

    # -- 11. Start brain in daemon thread --
    def brain_thread_fn():
        try:
            app.start()
        except Exception as exc:
            log.exception("Brain thread crashed: %s", exc)
        finally:
            shared_state.shutdown_event.set()

    brain_thread = threading.Thread(target=brain_thread_fn, daemon=True)
    brain_thread.start()

    # Give brain a moment to connect
    time.sleep(1.0)
    event_logger.log("brain_started")
    print("[OK] Brain started.")

    # -- 11b. Start a fast state-update thread for camera + dashboard --
    # The FSM main loop blocks on wait_for_wake_word, so camera frames
    # would freeze during IDLE. This thread updates independently at ~30 Hz.
    def state_updater_fn():
        while not shared_state.shutdown_event.is_set():
            try:
                _update_shared_state(app, shared_state)
            except Exception:
                pass
            time.sleep(0.033)  # ~30 Hz

    state_updater = threading.Thread(target=state_updater_fn, daemon=True)
    state_updater.start()

    # -- 12. Signal handler --
    def signal_handler(sig, frame):
        print("\n[SIGNAL] Shutting down...")
        shared_state.shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    # -- 13. Run dashboard (main thread -- OpenCV requires it) --
    print("[OK] Launching dashboard...")
    event_logger.log("dashboard_started")

    from gui.dashboard import Dashboard
    dashboard = Dashboard(shared_state)

    try:
        dashboard.run()  # blocks until quit
    except KeyboardInterrupt:
        pass

    # -- 14. Graceful shutdown --
    print("\n[SHUTDOWN] Stopping simulation...")
    shared_state.shutdown_event.set()
    event_logger.log("shutdown_initiated")

    try:
        app.shutdown()
    except Exception:
        pass

    # Wait for threads to finish
    brain_thread.join(timeout=5.0)

    event_logger.log("shutdown_complete")
    event_logger.close()
    print("[DONE] Simulation ended.")


if __name__ == "__main__":
    main()
