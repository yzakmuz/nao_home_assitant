#!/usr/bin/env python3
"""
main.py — Raspberry Pi Brain entry point.

Finite-State-Machine orchestrator that ties together every subsystem:
    Audio (Vosk STT, Speaker Verification) ←→ Command Parser
    Vision (Camera, FaceTracker, YOLO)     ←→ Visual Servo
    Comms (TCP Client)                     ←→ NAO Body
"""

from __future__ import annotations

import logging
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import numpy as np  # חובה בשביל אתחול השמע

import settings
from audio.mic_stream import MicStream
from audio.speaker_verify import SpeakerVerifier
from audio.stt_engine import SttEngine
from command.parser import Intent, IntentType, parse_command
from comms.tcp_client import NaoTcpClient
from servo.visual_servo import VisualServoController
from utils.memory import (
    available_mb,
    check_pressure,
    emergency_cleanup,
    force_gc,
    log_memory,
)
from vision.camera import Camera
from vision.face_tracker import FaceTracker
from vision.fall_monitor import FallMonitorThread
from vision.object_detector import ObjectDetector

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
log = logging.getLogger("main")


# ---------------------------------------------------------------------------
# FSM States
# ---------------------------------------------------------------------------
class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    VERIFYING = auto()
    EXECUTING = auto()
    SEARCHING = auto()
    SHUTDOWN = auto()


@dataclass
class FoundObject:
    """Information about the last found object."""
    target_name: str        # "phone", "keys", "bottle", etc.
    class_name: str         # COCO class: "cell phone", "bottle", etc.
    confidence: float       # detection confidence
    cx_norm: float          # center x in frame (normalized 0-1)
    cy_norm: float          # center y in frame (normalized 0-1)
    head_yaw: float         # head yaw angle at time of detection (radians)
    head_pitch: float       # head pitch at detection
    timestamp: float        # time.monotonic() when found


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
class AssistantApp:
    """Top-level orchestrator for the robot brain."""

    def __init__(self) -> None:
        self._state = State.IDLE
        self._running = False

        # -- Comms --
        self._tcp = NaoTcpClient()
        self._tcp.on_event = self._handle_nao_event
        self._tcp.on_reconnect = self._handle_nao_reconnect

        # -- Audio --
        self._mic = MicStream()
        self._stt = SttEngine()
        self._speaker = SpeakerVerifier()

        # -- Vision --
        self._camera = Camera()
        self._face_tracker = FaceTracker()

        # -- Servo --
        self._servo = VisualServoController(
            self._camera, self._face_tracker, self._tcp
        )

        # -- Fall Monitor (always-on, Improvement 4) --
        self._fall_monitor = FallMonitorThread(
            self._camera,
            on_person_fall=self._handle_person_fall_callback,
        )
        self._fall_monitor.set_face_tracker(self._face_tracker)
        self._fall_event = threading.Event()
        self._search_failed_event = threading.Event()  # set by servo thread

        # -- Object Detector (lazy) --
        self._detector: Optional[ObjectDetector] = None

        # --- FIX #3: אתחול מראש כדי למנוע קריסת חסר-משתנה ---
        self._recorded_audio: np.ndarray = np.array([], dtype=np.float32)
        self._pending_text: str = ""
        self._current_intent: Intent = Intent(
            type=IntentType.UNKNOWN,
            raw_text=""
        )

        # Multi-speaker: last verified speaker identity
        self._last_verified_speaker: str = ""
        self._last_verify_score: float = 0.0

        # Last found object (Improvement 6)
        self._last_found_object: Optional[FoundObject] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize all subsystems and enter the main loop."""
        log.info("=" * 60)
        log.info("  NAO Assistant Brain — Starting Up")
        log.info("=" * 60)
        log_memory("startup")

        # Connect to NAO
        log.info("Connecting to NAO at %s:%d …", settings.NAO_IP, settings.NAO_PORT)
        self._tcp.ensure_connected()
        self._tcp.send_command({"action": "say", "text": "Brain connected. I am ready."})

        # Start sensors
        self._mic.start()
        self._camera.start()

        # Servo starts only when user says "follow me" — not at boot

        # Start fall monitor (always-on, runs at ~5 Hz)
        self._fall_monitor.start()

        self._running = True
        log_memory("ready")
        log.info("All systems nominal. Entering main loop.\n")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt received.")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Graceful shutdown of all subsystems."""
        if not self._running:
            return
        self._running = False
        self._state = State.SHUTDOWN

        log.info("Shutting down …")
        self._fall_monitor.stop()
        self._servo.stop()
        self._camera.stop()
        self._mic.stop()
        self._face_tracker.close()

        if self._detector is not None:
            self._detector.unload()
            self._detector = None

        self._tcp.send_command({"action": "say", "text": "Going to sleep. Goodbye."})
        self._tcp.send_command({"action": "rest"})
        self._tcp.disconnect()
        log_memory("shutdown")
        log.info("Shutdown complete.")

    # ------------------------------------------------------------------
    # Main Loop (FSM)
    # ------------------------------------------------------------------

    def _main_loop(self) -> None:
        while self._running:
            # Priority: check for person fall BEFORE any state logic
            if self._fall_event.is_set():
                self._fall_event.clear()
                self._respond_to_person_fall()
                continue

            # Check if person search failed (set from servo thread)
            if self._search_failed_event.is_set():
                self._search_failed_event.clear()
                log.info("Person search failed — stopping servo, returning to IDLE.")
                self._servo.stop()
                self._state = State.IDLE
                continue

            try:
                if self._state == State.IDLE:
                    self._handle_idle()
                elif self._state == State.LISTENING:
                    self._handle_listening()
                elif self._state == State.VERIFYING:
                    self._handle_verifying()
                elif self._state == State.EXECUTING:
                    self._handle_executing()
                elif self._state == State.SEARCHING:
                    self._handle_searching()
                else:
                    break
            except Exception:
                log.exception("Unhandled error in state %s", self._state.name)
                self._state = State.IDLE
                time.sleep(1.0)

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_idle(self) -> None:
        """Wait for the wake word."""
        # Memory housekeeping between interactions
        pressure = check_pressure()
        if pressure == "critical":
            emergency_cleanup()

        self._stt.wait_for_wake_word(self._mic)

        # Acknowledge wake word via NAO
        self._tcp.send_fire_and_forget(
            {"action": "say", "text": "Yes?"}
        )

        # Begin recording for speaker verification
        self._mic.start_recording()
        self._state = State.LISTENING

    def _handle_listening(self) -> None:
        """Capture the command utterance after wake word."""
        command_text = self._stt.listen_command(self._mic)

        # Stop recording (retrieves the audio buffer for speaker verification)
        self._recorded_audio = self._mic.stop_recording()

        if command_text is None:
            self._tcp.send_fire_and_forget(
                {"action": "say", "text": "I did not catch that."}
            )
            self._state = State.IDLE
            return

        self._pending_text = command_text
        self._state = State.VERIFYING

    def _handle_verifying(self) -> None:
        """Verify the speaker is enrolled (multi-speaker or master-only)."""
        if settings.MULTI_SPEAKER_MODE and hasattr(self._speaker, 'verify_multi'):
            accepted, speaker_name, score = self._speaker.verify_multi(
                self._recorded_audio
            )
            self._last_verified_speaker = speaker_name
            self._last_verify_score = score
        else:
            accepted, score = self._speaker.verify(self._recorded_audio)
            self._last_verified_speaker = "master" if accepted else "unknown"
            self._last_verify_score = score

        if not accepted:
            log.warning("Speaker rejected (score=%.3f).", score)
            self._tcp.send_fire_and_forget(
                {"action": "say", "text": "Sorry, I don't recognize your voice."}
            )
            self._state = State.IDLE
            return

        log.info(
            "Speaker accepted: %s (score=%.3f). Command: '%s'",
            self._last_verified_speaker, score, self._pending_text,
        )
        # Parse the command
        self._current_intent = parse_command(self._pending_text)
        self._state = State.EXECUTING

    def _handle_executing(self) -> None:
        """Dispatch the parsed intent to the appropriate handler."""
        intent: Intent = self._current_intent
        log.info("Executing intent: %s", intent.type.name)

        dispatch = {
            IntentType.FOLLOW_ME: self._exec_follow_me,
            IntentType.STOP: self._exec_stop,
            IntentType.FIND_OBJECT: self._exec_find_object,
            IntentType.WAVE: self._exec_wave,
            IntentType.SAY_HELLO: self._exec_say_hello,
            IntentType.SIT_DOWN: self._exec_sit_down,
            IntentType.STAND_UP: self._exec_stand_up,
            IntentType.WHAT_DO_YOU_SEE: self._exec_what_do_you_see,
            IntentType.LOOK_LEFT: self._exec_look_direction,
            IntentType.LOOK_RIGHT: self._exec_look_direction,
            IntentType.TURN_AROUND: self._exec_turn_around,
            IntentType.COME_HERE: self._exec_come_here,
            IntentType.INTRODUCE: self._exec_introduce,
            IntentType.DANCE: self._exec_dance,
            IntentType.IM_OKAY: self._exec_im_okay,
            IntentType.BRING_OBJECT: self._exec_bring_object,
            IntentType.GO_TO_OBJECT: self._exec_go_to_object,
            IntentType.UNKNOWN: self._exec_unknown,
        }

        handler = dispatch.get(intent.type, self._exec_unknown)
        handler(intent)

        # Return to idle unless we're in a continuous mode (SEARCHING)
        if self._state == State.EXECUTING:
            self._state = State.IDLE

    def _handle_searching(self) -> None:
        """YOLO-powered object search loop."""
        intent: Intent = self._current_intent
        target_name = intent.params.get("target_name", "object")
        coco_classes: List[str] = intent.params.get("coco_classes", [])

        if self._detector is not None:
            log.warning("Previous YOLO detector still loaded — force unloading.")
            try:
                self._detector.unload()
            except Exception as exc:
                log.error("Error unloading previous detector: %s", exc)
            self._detector = None
            force_gc()

        self._send_checked(
            {"action": "say", "text": f"Looking for your {target_name}."},
            speak_error=False,
        )

        # Check available memory before loading YOLO
        log_memory("pre-yolo-load")
        if check_pressure() == "critical" or available_mb() < 150:
            log.error("Insufficient memory to load YOLO (need 150 MB available, have %.0f MB).", available_mb())
            self._send_checked(
                {"action": "say", "text": f"Sorry, not enough memory to search for your {target_name} right now."},
                speak_error=False,
            )
            self._state = State.IDLE
            return

        # Dynamically load YOLO
        detector = ObjectDetector()
        detector.load()
        self._detector = detector
        log_memory("post-yolo-load")

        found = False
        deadline = time.monotonic() + settings.YOLO_MAX_SEARCH_SECONDS
        scan_angles = [0.0, 0.5, -0.5, 1.0, -1.0]  # yaw angles to scan
        scan_idx = 0

        try:
            while time.monotonic() < deadline and self._running:
                frame = self._camera.read()
                if frame is None:
                    time.sleep(0.1)
                    continue

                results = detector.detect(
                    frame,
                    target_classes=coco_classes or None,
                )

                if results.detections:
                    best = results.detections[0]
                    log.info(
                        "Found '%s' (conf=%.2f) at (%.2f, %.2f)",
                        best.class_name, best.confidence,
                        best.cx_norm, best.cy_norm,
                    )
                    self._send_checked(
                        {"action": "say", "text": f"I found your {target_name}!"}
                    )
                    # Point head toward the detected object
                    error_x = 0.5 - best.cx_norm
                    error_y = 0.5 - best.cy_norm
                    self._tcp.send_fire_and_forget({
                        "action": "move_head_relative",
                        "d_yaw": round(error_x * 0.4, 3),
                        "d_pitch": round(-error_y * 0.3, 3),
                    })
                    # Store found object for "bring me" command
                    self._last_found_object = FoundObject(
                        target_name=target_name,
                        class_name=best.class_name,
                        confidence=best.confidence,
                        cx_norm=best.cx_norm,
                        cy_norm=best.cy_norm,
                        head_yaw=self._tcp.nao_state.head_yaw,
                        head_pitch=self._tcp.nao_state.head_pitch,
                        timestamp=time.monotonic(),
                    )
                    log.info("Object stored: %s at yaw=%.2f",
                             target_name, self._last_found_object.head_yaw)
                    found = True
                    break

                # Scan: periodically rotate head to look around
                if int((time.monotonic() - deadline + settings.YOLO_MAX_SEARCH_SECONDS) / 3) > scan_idx:
                    scan_idx = min(scan_idx + 1, len(scan_angles) - 1)
                    self._tcp.send_fire_and_forget({
                        "action": "move_head",
                        "yaw": scan_angles[scan_idx % len(scan_angles)],
                        "pitch": 0.15,  # slight downward gaze
                        "speed": 0.12,
                    })

                time.sleep(0.15)  # ~6-7 FPS for YOLO on Pi

            if not found:
                self._send_checked(
                    {"action": "say", "text": f"Sorry, I could not find your {target_name}."}
                )

        finally:
            # CRITICAL: always unload YOLO to free RAM
            detector.unload()
            self._detector = None
            force_gc()
            log_memory("post-yolo-unload")

            # Re-center head
            self._tcp.send_fire_and_forget(
                {"action": "move_head", "yaw": 0.0, "pitch": 0.0, "speed": 0.15}
            )

        self._state = State.IDLE

    # ------------------------------------------------------------------
    # Command helpers (Phase 3)
    # ------------------------------------------------------------------

    def _send_checked(
        self, command: dict, speak_error: bool = True
    ) -> Optional[dict]:
        """Send a command and check the response for errors (BUG 5 fix).

        Returns the response dict on success, or None on failure.
        If *speak_error* is True, informs the user via speech.
        """
        resp = self._tcp.send_command(command)

        if resp is None:
            log.error(
                "No response for '%s' — connection lost?",
                command.get("action"),
            )
            if speak_error:
                self._tcp.send_fire_and_forget(
                    {"action": "say", "text": "I lost connection. Please try again."}
                )
            return None

        status = resp.get("status", "")
        if status in ("error", "rejected"):
            reason = resp.get("reason", resp.get("message", "unknown"))
            log.warning(
                "Command '%s' %s: %s",
                command.get("action"), status, reason,
            )
            if speak_error:
                self._tcp.send_fire_and_forget(
                    {"action": "say", "text": "Sorry, I cannot do that right now."}
                )
            return None

        return resp

    def _ensure_standing(self) -> bool:
        """Ensure NAO is standing before a movement command (BUG 8 fix).

        Checks ``nao_state.posture`` and auto-sends pose:stand or
        wake_up as needed.  Returns True when standing, False on failure.
        """
        posture = self._tcp.nao_state.posture

        # If unknown, query first
        if posture == "unknown":
            resp = self._tcp.send_command({"action": "query_state"})
            if resp is not None:
                posture = self._tcp.nao_state.posture

        if posture == "standing":
            return True

        if posture == "sitting":
            log.info("NAO is sitting — auto-standing before command.")
            self._tcp.send_command(
                {"action": "say", "text": "Let me stand up first."}
            )
            resp = self._tcp.send_command_and_wait_done(
                {"action": "pose", "name": "stand"}, timeout=15.0
            )
            if resp and resp.get("status") == "ok":
                time.sleep(1.0)  # stabilization
                return True
            log.error("Failed to auto-stand: %s", resp)
            self._tcp.send_fire_and_forget(
                {"action": "say", "text": "Sorry, I could not stand up."}
            )
            return False

        if posture == "resting":
            log.info("NAO is resting — auto-waking before command.")
            resp = self._tcp.send_command_and_wait_done(
                {"action": "wake_up"}, timeout=15.0
            )
            if resp and resp.get("status") == "ok":
                self._tcp.send_command(
                    {"action": "say", "text": "I am awake and ready."}
                )
                time.sleep(1.0)
                return True
            log.error("Failed to auto-wake: %s", resp)
            self._tcp.send_fire_and_forget(
                {"action": "say", "text": "Sorry, I could not wake up."}
            )
            return False

        # fallen or other unknown state
        log.warning("NAO posture is '%s' — cannot ensure standing.", posture)
        self._tcp.send_fire_and_forget(
            {"action": "say", "text": "I am not in a position to do that."}
        )
        return False

    def _walk_to_target(self, timeout: float = 15.0, forward_speed: float = 0.4) -> None:
        """Walk toward a tracked target using head angle for steering.

        Uses ``set_walk_velocity`` for smooth continuous walking.
        Requires the visual servo to be running for head tracking.
        Blocks the FSM for up to *timeout* seconds.
        """
        if not self._servo.is_running:
            log.warning("_walk_to_target: servo not running — falling back to simple walk.")
            self._tcp.send_fire_and_forget(
                {"action": "walk_toward", "x": forward_speed, "y": 0.0, "theta": 0.0}
            )
            return

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline and self._running:
            head_yaw = self._tcp.nao_state.head_yaw

            # Steer toward face (proportional to head yaw)
            theta = head_yaw * 0.4

            # Slow down when turning sharply
            if abs(head_yaw) > 0.4:
                vx = 0.1
            else:
                vx = forward_speed

            self._tcp.send_fire_and_forget({
                "action": "set_walk_velocity",
                "x": round(vx, 2),
                "y": 0.0,
                "theta": round(theta, 3),
            })

            time.sleep(0.4)

        # Stop walking when done
        self._tcp.send_fire_and_forget({"action": "stop_walk"})

    # ------------------------------------------------------------------
    # Servo tracking callbacks (Improvement 5)
    # ------------------------------------------------------------------

    def _tracking_feedback(self) -> None:
        """Called every ~15s during head-only tracking when face is visible."""
        phrases = [
            "I can see you.",
            "Still watching.",
            "I'm keeping an eye on you.",
            "I see you there.",
        ]
        self._tcp.send_fire_and_forget(
            {"action": "say", "text": random.choice(phrases)}
        )

    def _on_follow_face_lost(self) -> None:
        """Face lost in head-only mode — notify, no search."""
        self._tcp.send_fire_and_forget(
            {"action": "say",
             "text": "I lost sight of you. Please come back in front of me."}
        )

    def _on_follow_face_reacquired(self) -> None:
        """Face re-acquired in head-only mode."""
        self._tcp.send_fire_and_forget(
            {"action": "say", "text": "I can see you again."}
        )

    def _on_search_complete(self, found: bool) -> None:
        """Called from servo thread when person search ends (come here mode).

        MUST NOT call servo.stop() here — that would deadlock (thread
        joining itself).  Instead, set an event that the main loop checks.
        """
        if not found:
            log.info("Person search failed — signaling main loop.")
            self._search_failed_event.set()

    # ------------------------------------------------------------------
    # Person fall detection callbacks (Improvement 4)
    # ------------------------------------------------------------------

    def _handle_person_fall_callback(self) -> None:
        """Called from the fall monitor thread when a fall is detected.

        Sets a threading.Event that the main loop checks at the top of
        every cycle.  This is non-blocking (safe to call from any thread).
        """
        log.warning("Person fall callback fired — setting fall event.")
        self._fall_event.set()

    def _respond_to_person_fall(self) -> None:
        """Handle a detected person fall — runs in the main FSM thread.

        Stops all activity, safely releases any held object, alerts the
        person, waits for "I'm okay" voice response, and escalates.
        """
        log.error("RESPONDING TO PERSON FALL — stopping all activity.")

        # 1. Stop servo if running
        if self._servo.is_running:
            self._servo.stop()

        # 2. Stop all NAO activity
        self._tcp.send_fire_and_forget({"action": "stop_all"})

        # 3. Safe release — open hand in case holding something
        self._tcp.send_fire_and_forget({"action": "open_hand", "hand": "right"})
        self._tcp.send_fire_and_forget({"action": "arm_rest"})

        # 3. Alert the person
        self._tcp.send_command(
            {"action": "say",
             "text": "Are you okay? I think you may have fallen!"}
        )

        # 4. Wait up to 10 seconds for "I'm okay" response
        response_heard = self._listen_for_okay(timeout=10.0)

        if response_heard:
            log.info("Person responded 'I'm okay' — re-arming fall detector.")
            self._tcp.send_command(
                {"action": "say", "text": "I'm glad you're alright."}
            )
        else:
            log.critical("No response after fall — escalating!")
            self._tcp.send_command(
                {"action": "say", "text": "I'm calling for help!"}
            )

        # 5. Return to IDLE (servo does NOT auto-restart)
        self._state = State.IDLE

    def _listen_for_okay(self, timeout: float = 10.0) -> bool:
        """Listen for 'I'm okay' voice command within *timeout* seconds.

        Uses a fresh Vosk recognizer to listen for the response.
        Also checks if visual recovery occurs (person stands back up).
        Returns True if the phrase was heard or visual recovery detected.
        """
        deadline = time.monotonic() + timeout

        try:
            rec = self._stt._new_recognizer()
        except Exception:
            # If STT is unavailable (keyboard mode, etc.), just wait
            while time.monotonic() < deadline and self._running:
                if not self._fall_monitor.fall_detector.is_triggered:
                    return True
                time.sleep(0.5)
            return False

        import json as _json
        while time.monotonic() < deadline and self._running:
            try:
                # Check visual recovery (person stood back up)
                if not self._fall_monitor.fall_detector.is_triggered:
                    return True

                chunk = self._mic.read(timeout=0.5)
                if chunk is None:
                    continue

                if rec.AcceptWaveform(chunk):
                    result = _json.loads(rec.Result())
                    text = result.get("text", "").strip().lower()
                    if "okay" in text:
                        return True
                    rec.Reset()
            except Exception as exc:
                log.debug("Transient error in listen_for_okay: %s", exc)
                continue  # keep listening, don't abort

        return False

    # ------------------------------------------------------------------
    # NAO event / reconnect callbacks (Phase 5)
    # ------------------------------------------------------------------

    def _handle_nao_event(self, event: dict) -> None:
        """Handle async events from the NAO server (called from reader thread)."""
        event_type = event.get("event")
        if event_type == "fallen":
            log.error("NAO HAS FALLEN — stopping servo, informing user.")
            self._servo.stop()
            # Speech might still work (robot is on the ground)
            self._tcp.send_fire_and_forget(
                {"action": "say", "text": "I have fallen! Please help me up."}
            )
        else:
            log.warning("Unknown NAO event: %s", event_type)

    def _handle_nao_reconnect(self) -> None:
        """Called after successful auto-reconnect (from reconnect thread)."""
        log.info("Reconnected to NAO — state resynced (posture=%s).",
                 self._tcp.nao_state.posture)

    # ------------------------------------------------------------------
    # Intent executors
    # ------------------------------------------------------------------

    def _exec_follow_me(self, intent: Intent) -> None:
        """Head-only tracking. Robot watches the person without walking."""
        # Already in head-only mode?
        if self._servo.is_running and not self._servo.walk_enabled:
            self._send_checked(
                {"action": "say", "text": "I'm already watching you."}
            )
            return

        # Stop existing servo if in a different mode (e.g., full follow)
        if self._servo.is_running:
            self._servo.stop()

        self._send_checked(
            {"action": "say", "text": "I'll keep my eyes on you."}
        )

        # Configure for head-only mode
        self._servo.walk_enabled = False
        self._servo.on_tracking_feedback = self._tracking_feedback
        self._servo.on_face_lost_notify = self._on_follow_face_lost
        self._servo.on_face_reacquired = self._on_follow_face_reacquired
        self._servo.on_search_complete = None
        self._servo.start()

    def _exec_stop(self, intent: Intent) -> None:
        self._servo.stop()
        self._send_checked({"action": "stop_all"}, speak_error=False)
        self._send_checked({"action": "say", "text": "Stopping."})

    def _exec_find_object(self, intent: Intent) -> None:
        if not self._ensure_standing():
            return
        self._servo.stop()
        self._state = State.SEARCHING

    def _exec_wave(self, intent: Intent) -> None:
        self._send_checked({"action": "animate", "name": "wave"})

    def _exec_say_hello(self, intent: Intent) -> None:
        self._send_checked(
            {"action": "say", "text": "Hello! Nice to see you."}
        )

    def _exec_sit_down(self, intent: Intent) -> None:
        self._servo.stop()
        self._send_checked({"action": "say", "text": "Sitting down."})
        self._send_checked({"action": "pose", "name": "sit"})

    def _exec_stand_up(self, intent: Intent) -> None:
        self._send_checked({"action": "say", "text": "Standing up."})
        resp = self._tcp.send_command_and_wait_done(
            {"action": "pose", "name": "stand"}, timeout=15.0
        )
        if resp is None or resp.get("status") != "ok":
            log.warning("Stand-up may have failed: %s", resp)

    def _exec_what_do_you_see(self, intent: Intent) -> None:
        self._tcp.send_fire_and_forget(
            {"action": "say", "text": "Let me take a look."}
        )

        detector = ObjectDetector()
        log_memory("pre-yolo-see")
        try:
            detector.load()
            frame = self._camera.read()
            if frame is not None:
                results = detector.detect(frame)
                if results.detections:
                    names = list(set(d.class_name for d in results.detections[:5]))
                    description = ", ".join(names)
                    self._send_checked(
                        {"action": "say", "text": f"I can see: {description}."}
                    )
                else:
                    self._send_checked(
                        {"action": "say", "text": "I don't see anything notable."}
                    )
            else:
                self._send_checked(
                    {"action": "say", "text": "My camera is not available right now."}
                )
        finally:
            detector.unload()
            force_gc()
            log_memory("post-yolo-see")

    def _exec_look_direction(self, intent: Intent) -> None:
        if intent.type == IntentType.LOOK_LEFT:
            yaw, text = 0.8, "Looking left."
        else:
            yaw, text = -0.8, "Looking right."
        self._send_checked({"action": "say", "text": text})
        self._send_checked(
            {"action": "move_head", "yaw": yaw, "pitch": 0.0, "speed": 0.12}
        )
        time.sleep(2.0)
        self._tcp.send_fire_and_forget(
            {"action": "move_head", "yaw": 0.0, "pitch": 0.0, "speed": 0.1}
        )

    def _exec_turn_around(self, intent: Intent) -> None:
        if not self._ensure_standing():
            return
        self._send_checked({"action": "say", "text": "Turning around."})
        self._send_checked(
            {"action": "walk_toward", "x": 0.0, "y": 0.0, "theta": 3.14}
        )

    def _exec_come_here(self, intent: Intent) -> None:
        """Full follow mode. Robot tracks AND walks toward the person."""
        # Already in full follow mode?
        if self._servo.is_running and self._servo.walk_enabled:
            self._send_checked(
                {"action": "say", "text": "I'm already following you."}
            )
            return

        if not self._ensure_standing():
            return

        # Stop existing servo if in a different mode (e.g., head-only)
        if self._servo.is_running:
            self._servo.stop()

        self._send_checked(
            {"action": "say", "text": "I'll follow you wherever you go."}
        )

        # Configure for full follow mode with person search
        self._servo.walk_enabled = True
        self._servo.on_tracking_feedback = None
        self._servo.on_face_lost_notify = None
        self._servo.on_face_reacquired = None
        self._servo.on_search_complete = self._on_search_complete
        self._servo.start()

    def _exec_introduce(self, intent: Intent) -> None:
        self._send_checked({
            "action": "animated_say",
            "text": (
                "Hello! I am NAO, your personal assistant robot. "
                "My brain runs on a Raspberry Pi mounted on my back. "
                "I can follow you, find objects, and do many things!"
            ),
        })

    def _exec_dance(self, intent: Intent) -> None:
        if not self._ensure_standing():
            return
        self._send_checked({"action": "say", "text": "Watch my moves!"})
        self._send_checked({"action": "animate", "name": "dance"})

    def _exec_im_okay(self, intent: Intent) -> None:
        self._send_checked(
            {"action": "say", "text": "Good to hear you're okay!"}
        )

    def _exec_bring_object(self, intent: Intent) -> None:
        """Full pick-up-and-deliver sequence with fall interrupt checks."""

        # ── PHASE 0: VALIDATION ──
        obj = self._last_found_object
        if obj is None:
            self._send_checked({"action": "say",
                "text": "I haven't found anything yet. Tell me what to look for."})
            return

        # Check name match (skip for "bring it to me" / "pick it up")
        requested_name = intent.params.get("target_name", "")
        if requested_name and requested_name != obj.target_name:
            self._send_checked({"action": "say",
                "text": f"I last found a {obj.target_name}, not a {requested_name}."})
            return

        # Check staleness
        age = time.monotonic() - obj.timestamp
        if age > 60.0:
            self._send_checked({"action": "say",
                "text": f"It's been a while since I found your {obj.target_name}. "
                        "Let me search again."})
            self._current_intent = Intent(
                type=IntentType.FIND_OBJECT, raw_text="",
                params={"target_name": obj.target_name,
                        "coco_classes": [obj.class_name]})
            self._state = State.SEARCHING
            return

        if not self._ensure_standing():
            return

        name = obj.target_name

        # ── PHASE 1: APPROACH OBJECT ──
        self._send_checked({"action": "say", "text": f"Let me get your {name}."})
        self._send_checked({"action": "move_head",
            "yaw": obj.head_yaw, "pitch": obj.head_pitch, "speed": 0.15})
        time.sleep(1.0)
        if self._check_fall_interrupt():
            return
        theta = obj.head_yaw * 0.3
        self._tcp.send_command_and_wait_done(
            {"action": "walk_toward", "x": 0.5, "y": 0.0, "theta": round(theta, 3)},
            timeout=8.0)
        if self._check_fall_interrupt():
            return

        # ── PHASE 2: PICKUP ──
        self._send_checked({"action": "say", "text": "Let me pick it up."})
        self._tcp.send_command_and_wait_done(
            {"action": "pickup_object"}, timeout=12.0)
        if self._check_fall_interrupt():
            return
        self._send_checked({"action": "say", "text": "Got it!"})

        # ── PHASE 3: FIND PERSON AND RETURN ──
        self._return_to_person(name)
        if self._check_fall_interrupt():
            return

        # ── PHASE 4: DELIVER ──
        self._send_checked({"action": "say", "text": f"Here is your {name}."})
        self._tcp.send_command_and_wait_done(
            {"action": "offer_object"}, timeout=8.0)
        if self._check_fall_interrupt():
            return
        self._send_checked({"action": "say", "text": "There you go!"})

    def _exec_go_to_object(self, intent: Intent) -> None:
        """Walk to the last found object and point it out (no pickup)."""
        obj = self._last_found_object
        if obj is None:
            self._send_checked({"action": "say",
                "text": "I haven't found anything yet. Tell me what to look for."})
            return

        if not self._ensure_standing():
            return

        name = obj.target_name
        self._send_checked({"action": "say", "text": f"Going to your {name}."})
        self._send_checked({"action": "move_head",
            "yaw": obj.head_yaw, "pitch": obj.head_pitch, "speed": 0.15})
        time.sleep(1.0)
        theta = obj.head_yaw * 0.3
        self._tcp.send_command_and_wait_done(
            {"action": "walk_toward", "x": 0.5, "y": 0.0, "theta": round(theta, 3)},
            timeout=8.0)
        self._send_checked({"action": "say",
            "text": f"Here is your {name}. It's right here."})

    def _check_fall_interrupt(self) -> bool:
        """Check if a person fall was detected during a bring sequence.

        Safely releases any held object then triggers the fall response.
        Returns True if a fall was handled (caller should abort).
        """
        if not self._fall_event.is_set():
            return False

        self._fall_event.clear()
        log.warning("Fall detected during bring sequence — safe release.")

        if self._servo.is_running:
            self._servo.stop()

        # Safe release: open hand so object drops gently
        self._tcp.send_fire_and_forget(
            {"action": "open_hand", "hand": "right"}
        )
        time.sleep(0.3)
        self._tcp.send_fire_and_forget({"action": "arm_rest"})

        self._respond_to_person_fall()
        return True

    def _return_to_person(self, object_name: str, timeout: float = 30.0) -> None:
        """Find the person and walk to them while holding an object.

        Uses the servo in full-follow mode with person search.
        Stops when the person's face is large enough (close proximity).
        """
        self._send_checked(
            {"action": "say", "text": "Let me bring it to you."})

        # Set arm to carry position
        self._tcp.send_fire_and_forget({"action": "arm_carry"})

        # Start servo in full-follow mode (reuses person search)
        self._servo.walk_enabled = True
        self._servo.on_search_complete = self._on_search_complete
        self._servo.on_tracking_feedback = None
        self._servo.on_face_lost_notify = None
        self._servo.on_face_reacquired = None
        self._servo.start()

        deadline = time.monotonic() + timeout
        last_arm_refresh = time.monotonic()
        CLOSE_ENOUGH_AREA = 0.04  # face area ~1m distance

        while time.monotonic() < deadline and self._running:
            # Fall check every iteration (~0.3s)
            if self._fall_event.is_set():
                self._servo.stop()
                return  # caller's _check_fall_interrupt handles release

            # Person close enough?
            face = self._servo.last_face_result
            if face is not None:
                face_area = face.width * face.height
                if face_area > CLOSE_ENOUGH_AREA:
                    break

            # Search failed?
            if self._search_failed_event.is_set():
                self._search_failed_event.clear()
                self._send_checked({"action": "say",
                    "text": "I can't find you. Please come to me."})
                time.sleep(5.0)
                break

            # Refresh carry arm position (walking may shift arm)
            if time.monotonic() - last_arm_refresh > 3.0:
                self._tcp.send_fire_and_forget({"action": "arm_carry"})
                last_arm_refresh = time.monotonic()

            time.sleep(0.3)

        self._servo.stop()

    def _exec_unknown(self, intent: Intent) -> None:
        self._send_checked(
            {"action": "say", "text": "Sorry, I did not understand that command."}
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _signal_handler(sig, frame):
    log.info("Signal %s received — shutting down.", sig)
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    app = AssistantApp()
    app.start()


if __name__ == "__main__":
    main()