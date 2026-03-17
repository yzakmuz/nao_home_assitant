"""
pc_camera.py -- PC-compatible camera adapter (no V4L2).

Drop-in replacement for vision.camera.Camera.  Differences:
  - Uses cv2.VideoCapture(index) without CAP_V4L2 flag
  - Falls back to synthetic frames if no webcam available
  - Higher resolution: 640x480 @ 30fps (vs 320x240 @ 15 on Pi)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Optional

import cv2
import numpy as np

from settings import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH

log = logging.getLogger(__name__)


class PcCamera:
    """PC webcam reader -- same interface as vision.camera.Camera."""

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._synthetic = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        if self._running:
            return True

        try:
            import sim_config
            use_synthetic = sim_config.SIM_NO_CAMERA
        except (ImportError, AttributeError):
            use_synthetic = False

        if not use_synthetic:
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._cap = cap
                self._synthetic = False
                log.info(
                    "PC Camera started -- %dx%d @ %d FPS (device %d)",
                    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_INDEX,
                )
            else:
                cap.release()
                log.warning("No webcam found (device %d) -- using synthetic frames.", CAMERA_INDEX)
                self._synthetic = True
        else:
            self._synthetic = True
            log.info("Synthetic camera mode (--no-camera).")

        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
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
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_size(self) -> tuple:
        return (CAMERA_WIDTH, CAMERA_HEIGHT)

    # ------------------------------------------------------------------
    # Background grab loop
    # ------------------------------------------------------------------

    def _grab_loop(self) -> None:
        while self._running:
            if self._synthetic:
                frame = self._generate_synthetic_frame()
                with self._lock:
                    self._frame = frame
                time.sleep(1.0 / max(CAMERA_FPS, 1))
            else:
                assert self._cap is not None
                ok, frame = self._cap.read()
                if not ok:
                    time.sleep(0.1)
                    continue
                with self._lock:
                    self._frame = frame

    # ------------------------------------------------------------------
    # Synthetic frame generator
    # ------------------------------------------------------------------

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate a dark frame with a moving skin-colored ellipse (face-like)."""
        w, h = CAMERA_WIDTH, CAMERA_HEIGHT
        frame = np.full((h, w, 3), 40, dtype=np.uint8)

        t = time.monotonic()
        # Figure-8 pattern
        cx = int(w / 2 + w * 0.2 * math.sin(t * 0.5))
        cy = int(h / 2 + h * 0.1 * math.sin(t * 1.0))

        # Skin-colored ellipse (face-like for MediaPipe)
        face_w, face_h = 60, 80
        cv2.ellipse(frame, (cx, cy), (face_w, face_h), 0, 0, 360,
                     (180, 200, 220), -1)

        # Simple eye dots
        eye_offset = 20
        cv2.circle(frame, (cx - eye_offset, cy - 15), 5, (60, 60, 60), -1)
        cv2.circle(frame, (cx + eye_offset, cy - 15), 5, (60, 60, 60), -1)

        # Mouth
        cv2.ellipse(frame, (cx, cy + 25), (15, 8), 0, 0, 180, (100, 100, 120), 2)

        # Label
        cv2.putText(frame, "SYNTHETIC CAMERA", (10, h - 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

        return frame
