"""
mic_stream.py — Thread-safe ring-buffer microphone capture.

Uses `sounddevice` for low-latency PortAudio access. Audio is written
into a thread-safe queue that the STT engine drains.

Design notes:
    * The callback runs on a real-time audio thread — keep it minimal.
    * Frames are 16-bit signed PCM, 16 kHz mono (matching Vosk).
    * A secondary `recording_buffer` accumulates raw audio when the
      system enters LISTENING state, so we can later hand the full
      utterance to the speaker-verification model.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from settings import (
    MIC_CHANNELS,
    MIC_CHUNK_FRAMES,
    MIC_DEVICE_INDEX,
    MIC_SAMPLE_RATE,
)

log = logging.getLogger(__name__)


class MicStream:
    """Continuous microphone capture with a frame queue and optional recorder."""

    def __init__(self) -> None:
        self._frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=50)
        self._stream: Optional[sd.RawInputStream] = None
        self._recording = False
        self._rec_lock = threading.Lock()
        self._rec_chunks: list[bytes] = []
        self._rec_chunks_max = int(MIC_SAMPLE_RATE * 5.0 / MIC_CHUNK_FRAMES)  # max 5 sec worth

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the microphone and begin streaming."""
        if self._stream is not None:
            return
        self._stream = sd.RawInputStream(
            samplerate=MIC_SAMPLE_RATE,
            channels=MIC_CHANNELS,
            dtype="int16",
            blocksize=MIC_CHUNK_FRAMES,
            device=MIC_DEVICE_INDEX,
            callback=self._audio_callback,
        )
        self._stream.start()
        log.info(
            "Mic opened — %d Hz, chunk=%d frames", MIC_SAMPLE_RATE, MIC_CHUNK_FRAMES
        )

    def stop(self) -> None:
        """Close the microphone stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("Mic closed.")

    # ------------------------------------------------------------------
    # Frame access (used by STT engine)
    # ------------------------------------------------------------------

    def read(self, timeout: float = 0.5) -> Optional[bytes]:
        """
        Return the next audio chunk (bytes, int16 PCM) or None on timeout.
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Discard any buffered frames."""
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Recording mode — capture full utterance for speaker verification
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Begin accumulating audio chunks into a buffer."""
        with self._rec_lock:
            self._rec_chunks.clear()
            self._recording = True
        log.debug("Recording started.")

    def stop_recording(self) -> np.ndarray:
        """
        Stop recording and return the captured audio as a float32 numpy
        array normalized to [-1.0, 1.0].
        """
        with self._rec_lock:
            self._recording = False
            chunks = list(self._rec_chunks)
            self._rec_chunks.clear()

        if not chunks:
            return np.array([], dtype=np.float32)

        raw = b"".join(chunks)
        pcm = np.frombuffer(raw, dtype=np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        log.debug("Recording stopped — %.2f s captured.", len(audio) / MIC_SAMPLE_RATE)
        return audio

    # ------------------------------------------------------------------
    # PortAudio callback (real-time thread — no allocations!)
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: bytes,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            log.warning("Mic status: %s", status)

        raw = bytes(indata)

        # Push to STT queue (drop if full — better than blocking RT thread)
        try:
            self._frame_queue.put_nowait(raw)
        except queue.Full:
            pass

        # Accumulate for speaker verification with buffer limit
        if self._recording:
            with self._rec_lock:
                if self._recording:
                    if len(self._rec_chunks) < self._rec_chunks_max:
                        self._rec_chunks.append(raw)
                    else:
                        log.warning("Recording buffer at limit (5s) — dropping oldest chunks.")
                        self._rec_chunks.pop(0)
                        self._rec_chunks.append(raw)
