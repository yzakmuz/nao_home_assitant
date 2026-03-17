"""
mock_proxies.py -- Mock ALProxy classes for PC simulation.

Each mock simulates realistic NAO timing, tracks internal state, and
pushes structured action-log entries to a shared queue for the GUI.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Speed multiplier -- set by run_simulation.py from sim_config
_speed_mult = 1.0


def set_speed_multiplier(mult: float) -> None:
    global _speed_mult
    _speed_mult = mult


def _sim_sleep(seconds: float) -> None:
    """Sleep for a simulated duration (scaled by speed multiplier)."""
    time.sleep(max(0.01, seconds * _speed_mult))


# ======================================================================
# Shared Action Log
# ======================================================================
_action_log: Optional[queue.Queue] = None


def set_action_log(q: queue.Queue) -> None:
    global _action_log
    _action_log = q


def _log_action(proxy: str, method: str, params: Dict[str, Any] = None,
                duration_ms: float = 0) -> None:
    if _action_log is None:
        return
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "proxy": proxy,
        "method": method,
        "params": params or {},
        "duration_ms": round(duration_ms, 1),
    }
    try:
        _action_log.put_nowait(entry)
    except queue.Full:
        pass


# ======================================================================
# MockMotion
# ======================================================================
class MockMotion:
    """Simulates ALMotion proxy."""

    def __init__(self) -> None:
        self._joints: Dict[str, float] = {
            "HeadYaw": 0.0, "HeadPitch": 0.0,
            "LShoulderPitch": 1.4, "LShoulderRoll": 0.2, "LElbowYaw": -1.2,
            "LElbowRoll": -0.5, "LWristYaw": 0.0, "LHand": 0.3,
            "RShoulderPitch": 1.4, "RShoulderRoll": -0.2, "RElbowYaw": 1.2,
            "RElbowRoll": 0.5, "RWristYaw": 0.0, "RHand": 0.3,
            "LHipYawPitch": 0.0, "LHipRoll": 0.0, "LHipPitch": 0.0,
            "LKneePitch": 0.0, "LAnklePitch": 0.0, "LAnkleRoll": 0.0,
            "RHipYawPitch": 0.0, "RHipRoll": 0.0, "RHipPitch": 0.0,
            "RKneePitch": 0.0, "RAnklePitch": 0.0, "RAnkleRoll": 0.0,
        }
        self._lock = threading.Lock()
        self._velocity = (0.0, 0.0, 0.0)
        self._moving = False
        self._stop_flag = threading.Event()
        self._stiffness = True

    def setAngles(self, names: Any, angles: Any, speed: float) -> None:
        """Set joint angles (non-blocking -- updates state immediately)."""
        if isinstance(names, str):
            names = [names]
            angles = [angles]
        with self._lock:
            for n, a in zip(names, angles):
                if n in self._joints:
                    self._joints[n] = float(a)
        _log_action("motion", "setAngles",
                     {"names": list(names), "angles": [round(a, 3) for a in angles],
                      "speed": speed})

    def getAngles(self, names: Any, use_sensors: bool = True) -> List[float]:
        if isinstance(names, str):
            names = [names]
        with self._lock:
            return [self._joints.get(n, 0.0) for n in names]

    def moveTo(self, x: float, y: float, theta: float) -> None:
        """Blocking walk to target -- simulates NAO walk timing."""
        self._stop_flag.clear()
        self._moving = True
        dist = math.sqrt(x * x + y * y)
        duration = dist / 0.1 + abs(theta) / 0.3
        duration = max(0.5, duration)
        _log_action("motion", "moveTo",
                     {"x": round(x, 2), "y": round(y, 2),
                      "theta": round(theta, 2)},
                     duration * 1000)

        # Simulate blocking walk with early stop support
        deadline = time.monotonic() + duration * _speed_mult
        while time.monotonic() < deadline:
            if self._stop_flag.is_set():
                break
            time.sleep(0.05)
        self._moving = False

    def moveToward(self, x: float, y: float, theta: float) -> None:
        """Non-blocking velocity mode."""
        with self._lock:
            self._velocity = (x, y, theta)
            self._moving = (x != 0.0 or y != 0.0 or theta != 0.0)
        _log_action("motion", "moveToward",
                     {"x": round(x, 2), "y": round(y, 2),
                      "theta": round(theta, 2)})

    def moveIsActive(self) -> bool:
        return self._moving

    def stopMove(self) -> None:
        self._stop_flag.set()
        with self._lock:
            self._velocity = (0.0, 0.0, 0.0)
            self._moving = False
        _log_action("motion", "stopMove")

    def wakeUp(self) -> None:
        self._stiffness = True
        _log_action("motion", "wakeUp", duration_ms=1500)
        _sim_sleep(1.5)

    def rest(self) -> None:
        self._stiffness = False
        _log_action("motion", "rest", duration_ms=1500)
        _sim_sleep(1.5)

    @property
    def velocity(self):
        with self._lock:
            return self._velocity


# ======================================================================
# MockTts
# ======================================================================
class MockTts:
    """Simulates ALTextToSpeech proxy."""

    def __init__(self) -> None:
        self._speaking = False
        self._cancel = threading.Event()
        self._engine = None
        self._tts_lock = threading.Lock()
        # Try to load pyttsx3 for actual PC speech
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 150)
        except Exception:
            pass

    def say(self, text: str) -> None:
        text = str(text)
        self._cancel.clear()
        self._speaking = True
        duration = max(0.5, len(text) * 0.05)
        _log_action("tts", "say", {"text": text}, duration * 1000)

        # Use pyttsx3 if available and enabled
        try:
            import sim_config
            if sim_config.SIM_PC_TTS and self._engine is not None:
                with self._tts_lock:
                    self._engine.say(text)
                    self._engine.runAndWait()
                self._speaking = False
                return
        except Exception:
            pass

        # Fallback: just simulate timing
        deadline = time.monotonic() + duration * _speed_mult
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                break
            time.sleep(0.05)
        self._speaking = False

    def stopAll(self) -> None:
        self._cancel.set()
        _log_action("tts", "stopAll")

    @property
    def is_speaking(self) -> bool:
        return self._speaking


# ======================================================================
# MockAnimatedTts
# ======================================================================
class MockAnimatedTts:
    """Simulates ALAnimatedSpeech proxy."""

    def __init__(self, tts: MockTts) -> None:
        self._tts = tts

    def say(self, text: str) -> None:
        text = str(text)
        duration = max(0.5, len(text) * 0.06)
        _log_action("animated_tts", "say", {"text": text}, duration * 1000)
        # Delegate to real tts for actual speech
        self._tts.say(text)


# ======================================================================
# MockPosture
# ======================================================================
class MockPosture:
    """Simulates ALRobotPosture proxy."""

    def __init__(self) -> None:
        self._posture = "Stand"
        self._lock = threading.Lock()

    def goToPosture(self, name: str, speed: float) -> None:
        duration = 2.0 / max(speed, 0.1)
        _log_action("posture", "goToPosture",
                     {"name": name, "speed": speed},
                     duration * 1000)
        _sim_sleep(duration)
        with self._lock:
            self._posture = name

    def getPosture(self) -> str:
        with self._lock:
            return self._posture


# ======================================================================
# MockMemory
# ======================================================================
class MockMemory:
    """Simulates ALMemory proxy -- supports fall simulation."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {
            "robotHasFallen": 0,
        }
        self._lock = threading.Lock()

    def getData(self, key: str) -> Any:
        with self._lock:
            return self._data.get(key, 0)

    def simulate_fall(self) -> None:
        with self._lock:
            self._data["robotHasFallen"] = 1
        _log_action("memory", "simulate_fall")

    def clear_fall(self) -> None:
        with self._lock:
            self._data["robotHasFallen"] = 0
        _log_action("memory", "clear_fall")


# ======================================================================
# MockLeds
# ======================================================================
class MockLeds:
    """No-op LED proxy."""
    def fadeRGB(self, *args, **kwargs): pass
    def setIntensity(self, *args, **kwargs): pass
    def on(self, *args, **kwargs): pass
    def off(self, *args, **kwargs): pass


# ======================================================================
# MockNaoProxies -- same property API as NaoProxies
# ======================================================================
class MockNaoProxies:
    """Drop-in replacement for NaoProxies using mock proxy objects."""

    def __init__(self, action_log_queue: queue.Queue) -> None:
        set_action_log(action_log_queue)
        self.ip = "127.0.0.1"
        self._motion = MockMotion()
        self._tts = MockTts()
        self._animated_tts = MockAnimatedTts(self._tts)
        self._posture = MockPosture()
        self._leds = MockLeds()
        self._memory = MockMemory()

    @property
    def motion(self) -> MockMotion:
        return self._motion

    @property
    def tts(self) -> MockTts:
        return self._tts

    @property
    def animated_tts(self) -> MockAnimatedTts:
        return self._animated_tts

    @property
    def posture(self) -> MockPosture:
        return self._posture

    @property
    def leds(self) -> MockLeds:
        return self._leds

    @property
    def memory(self) -> MockMemory:
        return self._memory
