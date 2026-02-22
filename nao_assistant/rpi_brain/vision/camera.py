"""
camera.py — Thread-safe, continuous Pi Camera frame grabber.

Uses OpenCV's V4L2 backend to read from /dev/video0 (the Pi Camera
exposed by the `bcm2835-v4l2` or `libcamera` stack).

Design:
    A background thread grabs frames as fast as the camera allows.
    Consumers call `read()` to get the *latest* frame (skipping stale ones).
    This avoids the classic problem of OpenCV's internal buffer causing
    multi-second lag on a slow consumer.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from settings import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH

log = logging.getLogger(__name__)


class Camera:
    """Low-latency Pi Camera reader (always serves the latest frame)."""

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Open the camera and spawn the capture thread."""
        if self._running:
            return True

        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        if not cap.isOpened():
            log.error("Cannot open camera index %d", CAMERA_INDEX)
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize internal buffering

        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        log.info(
            "Camera started — %dx%d @ %d FPS",
            CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
        )
        return True

    def stop(self) -> None:
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frame = None
        log.info("Camera stopped.")

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read(self) -> Optional[np.ndarray]:
        """Return the latest BGR frame, or None if unavailable."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_size(self) -> tuple[int, int]:
        """(width, height) of the camera frames."""
        return (CAMERA_WIDTH, CAMERA_HEIGHT)

    # ------------------------------------------------------------------
    # Background grab loop
    # ------------------------------------------------------------------

    def _grab_loop(self) -> None:
        """Continuously grab frames, keeping only the latest."""
        assert self._cap is not None
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                log.warning("Camera read failed — retrying in 0.1 s")
                time.sleep(0.1)
                continue
            with self._lock:
                self._frame = frame
        log.debug("Camera grab loop exited.")
