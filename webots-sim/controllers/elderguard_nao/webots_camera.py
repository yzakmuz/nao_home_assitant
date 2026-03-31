"""
webots_camera.py — Camera adapter that receives frames from the Webots
NAO CameraTop via TCP stream.

Drop-in replacement for vision.camera.Camera and PcCamera.
Exposes the same interface: start(), stop(), read(), is_running, frame_size.

Usage (standalone test):
    cam = WebotsCamera("127.0.0.1", 5556)
    cam.start()
    frame = cam.read()   # np.ndarray BGR or None
    cam.stop()

Protocol (per frame received):
    [2 bytes uint16 BE: width]
    [2 bytes uint16 BE: height]
    [width * height * 3 bytes: raw BGR pixel data]
"""

import socket
import struct
import threading
import time

import numpy as np


class WebotsCamera:
    """Receives camera frames from the Webots NAO controller via TCP."""

    def __init__(self, host="127.0.0.1", port=5556):
        self._host = host
        self._port = port
        self._sock = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._width = 320
        self._height = 240
        self._fps_count = 0
        self._fps_time = 0.0
        self._fps = 0.0

    def start(self):
        """Connect to the Webots camera server and begin receiving frames."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5.0)
            self._sock.connect((self._host, self._port))
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except (ConnectionRefusedError, OSError) as e:
            print("[WebotsCamera] Cannot connect to %s:%d — %s" % (
                self._host, self._port, e))
            self._sock = None
            return False

        self._running = True
        self._fps_time = time.monotonic()
        self._thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="WebotsCamera")
        self._thread.start()
        print("[WebotsCamera] Connected to %s:%d" % (self._host, self._port))
        return True

    def stop(self):
        """Stop receiving and close connection."""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self._frame = None
        print("[WebotsCamera] Stopped.")

    def read(self):
        """Return the latest BGR frame as a numpy array, or None."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    @property
    def is_running(self):
        return self._running

    @property
    def frame_size(self):
        return (self._width, self._height)

    @property
    def fps(self):
        """Approximate frames-per-second being received."""
        return self._fps

    def _receive_loop(self):
        """Background thread: continuously receive frames from server."""
        while self._running:
            try:
                # Read 4-byte header: width (uint16) + height (uint16)
                header = self._recv_exact(4)
                if header is None:
                    break
                w, h = struct.unpack(">HH", header)

                # Read pixel data
                data_len = w * h * 3
                pixel_data = self._recv_exact(data_len)
                if pixel_data is None:
                    break

                # Convert to numpy BGR array
                frame = np.frombuffer(
                    pixel_data, dtype=np.uint8).reshape(h, w, 3)

                with self._lock:
                    self._frame = frame
                    self._width = w
                    self._height = h

                # FPS tracking
                self._fps_count += 1
                now = time.monotonic()
                elapsed = now - self._fps_time
                if elapsed >= 1.0:
                    self._fps = self._fps_count / elapsed
                    self._fps_count = 0
                    self._fps_time = now

            except (ConnectionResetError, ConnectionAbortedError, OSError):
                break
            except Exception as e:
                print("[WebotsCamera] Error: %s" % e)
                break

        self._running = False
        print("[WebotsCamera] Receive loop ended.")

    def _recv_exact(self, n):
        """Read exactly n bytes from socket. Returns bytes or None."""
        data = b""
        while len(data) < n:
            try:
                chunk = self._sock.recv(min(n - len(data), 65536))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                if not self._running:
                    return None
                continue
            except (ConnectionResetError, OSError):
                return None
        return data
