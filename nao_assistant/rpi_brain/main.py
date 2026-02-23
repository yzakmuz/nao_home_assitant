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
import signal
import sys
import time
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
    check_pressure,
    emergency_cleanup,
    force_gc,
    log_memory,
)
from vision.camera import Camera
from vision.face_tracker import FaceTracker
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

        # -- Object Detector (lazy) --
        self._detector: Optional[ObjectDetector] = None

        # --- FIX #3: אתחול מראש כדי למנוע קריסת חסר-משתנה ---
        self._recorded_audio: np.ndarray = np.array([], dtype=np.float32)
        self._pending_text: str = ""
        self._current_intent: Intent = Intent(
            type=IntentType.UNKNOWN,
            raw_text=""
        )

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

        # Start face-tracking servo (runs in background)
        self._servo.start()

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
        """Verify the speaker is the enrolled master."""
        is_master, score = self._speaker.verify(self._recorded_audio)

        if not is_master:
            log.warning("Speaker rejected (score=%.3f).", score)
            self._tcp.send_fire_and_forget(
                {"action": "say", "text": "Sorry, I only listen to my owner."}
            )
            self._state = State.IDLE
            return

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

            self._tcp.send_fire_and_forget(
                {"action": "say", "text": f"Looking for your {target_name}."}
            )

            # Dynamically load YOLO
            log_memory("pre-yolo-load")
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
                    self._tcp.send_command(
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
                self._tcp.send_command(
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
    # Intent executors
    # ------------------------------------------------------------------

    def _exec_follow_me(self, intent: Intent) -> None:
        self._tcp.send_command(
            {"action": "say", "text": "Okay, I will follow you."}
        )
        if not self._servo.is_running:
            self._servo.start()

    def _exec_stop(self, intent: Intent) -> None:
        self._servo.stop()
        self._tcp.send_command({"action": "stop_walk"})
        self._tcp.send_command({"action": "say", "text": "Stopping."})
        time.sleep(0.5)
        self._servo.start()

    def _exec_find_object(self, intent: Intent) -> None:
        self._servo.stop()
        self._state = State.SEARCHING

    def _exec_wave(self, intent: Intent) -> None:
        self._tcp.send_command({"action": "animate", "name": "wave"})

    def _exec_say_hello(self, intent: Intent) -> None:
        self._tcp.send_command(
            {"action": "say", "text": "Hello! Nice to see you."}
        )

    def _exec_sit_down(self, intent: Intent) -> None:
        self._servo.stop()
        self._tcp.send_command({"action": "pose", "name": "sit"})
        self._tcp.send_command({"action": "say", "text": "Sitting down."})

    def _exec_stand_up(self, intent: Intent) -> None:
        self._tcp.send_command({"action": "pose", "name": "stand"})
        self._tcp.send_command({"action": "say", "text": "Standing up."})
        time.sleep(2.0)
        self._servo.start()

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
                    self._tcp.send_command(
                        {"action": "say", "text": f"I can see: {description}."}
                    )
                else:
                    self._tcp.send_command(
                        {"action": "say", "text": "I don't see anything notable."}
                    )
            else:
                self._tcp.send_command(
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
        self._tcp.send_command({"action": "say", "text": text})
        self._tcp.send_command(
            {"action": "move_head", "yaw": yaw, "pitch": 0.0, "speed": 0.12}
        )
        time.sleep(2.0)
        self._tcp.send_fire_and_forget(
            {"action": "move_head", "yaw": 0.0, "pitch": 0.0, "speed": 0.1}
        )

    def _exec_turn_around(self, intent: Intent) -> None:
        self._tcp.send_command({"action": "say", "text": "Turning around."})
        self._tcp.send_command(
            {"action": "walk_toward", "x": 0.0, "y": 0.0, "theta": 3.14}
        )

    def _exec_come_here(self, intent: Intent) -> None:
        self._tcp.send_command({"action": "say", "text": "Coming to you."})
        self._tcp.send_command(
            {"action": "walk_toward", "x": 0.6, "y": 0.0, "theta": 0.0}
        )

    def _exec_introduce(self, intent: Intent) -> None:
        self._tcp.send_command({
            "action": "animated_say",
            "text": (
                "Hello! I am NAO, your personal assistant robot. "
                "My brain runs on a Raspberry Pi mounted on my back. "
                "I can follow you, find objects, and do many things!"
            ),
        })

    def _exec_dance(self, intent: Intent) -> None:
        self._tcp.send_command({"action": "say", "text": "Watch my moves!"})
        self._tcp.send_command({"action": "animate", "name": "dance"})

    def _exec_unknown(self, intent: Intent) -> None:
        self._tcp.send_command(
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