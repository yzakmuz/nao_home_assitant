"""
keyboard_input.py -- Typed-command fallback when no mic/vosk is available.

Provides:
  - KeyboardSttEngine: replaces SttEngine with keyboard input
  - DummyMicStream: replaces MicStream with a no-op that returns silence
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class DummyMicStream:
    """No-op microphone stream that returns silence."""

    def __init__(self) -> None:
        self._running = False

    def start(self) -> None:
        self._running = True
        log.info("DummyMicStream started (no real microphone).")

    def stop(self) -> None:
        self._running = False

    def read(self, timeout: float = 0.5) -> Optional[bytes]:
        time.sleep(timeout)
        return b"\x00" * 8000

    def drain(self) -> None:
        pass

    def start_recording(self) -> None:
        pass

    def stop_recording(self) -> np.ndarray:
        # Return 2 seconds of silence (enough for speaker verify to accept)
        return np.zeros(32000, dtype=np.float32)


class KeyboardSttEngine:
    """Replaces SttEngine for keyboard input mode.

    Provides the same interface used by main.py:
      - wait_for_wake_word(mic) -> str
      - listen_command(mic, timeout) -> Optional[str]
    """

    def __init__(self) -> None:
        log.info("KeyboardSttEngine initialized (type commands in console).")
        self._input_queue: queue.Queue = queue.Queue()
        self._input_thread: Optional[threading.Thread] = None
        self._running = True
        # Start a background thread for non-blocking stdin reads
        self._input_thread = threading.Thread(
            target=self._stdin_reader, daemon=True
        )
        self._input_thread.start()

    def _stdin_reader(self) -> None:
        """Read lines from stdin in a background thread."""
        while self._running:
            try:
                line = input()
                self._input_queue.put(line.strip())
            except (EOFError, KeyboardInterrupt):
                break

    def wait_for_wake_word(self, mic) -> str:
        """Block until Enter is pressed (simulates wake word detection)."""
        print("\n[SIM] Press Enter (or type a command directly) to simulate 'hey nao'...")
        while True:
            try:
                text = self._input_queue.get(timeout=0.5)
                if text:
                    # If user typed a command directly, put it back for listen_command
                    self._input_queue.put(text)
                return "hey nao"
            except queue.Empty:
                continue

    def listen_command(
        self,
        mic,
        timeout: float = 5.0,
    ) -> Optional[str]:
        """Read a command from the input queue."""
        print("[SIM] Type command (e.g., 'follow me', 'sit down'): ", end="", flush=True)
        try:
            text = self._input_queue.get(timeout=timeout)
            text = text.strip().lower()
            if text:
                log.info("Keyboard command: '%s'", text)
                return text
        except queue.Empty:
            pass
        return None
