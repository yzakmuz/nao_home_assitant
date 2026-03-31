"""
camera_server.py — Non-blocking camera frame streaming server.

Streams the Webots NAO CameraTop frames to the brain process via TCP.
The brain receives frames through a WebotsCamera adapter that provides
the same Camera.read() interface as the real Pi camera.

Protocol (per frame):
    [2 bytes uint16 BE: width]
    [2 bytes uint16 BE: height]
    [width * height * 3 bytes: raw BGR pixel data]

Design:
    Non-blocking I/O via select(), polled from the robot.step() loop.
    Single client connection (like tcp_handler.py).
    BGRA → BGR conversion (Webots cameras return 4-channel BGRA).
"""

import select
import socket
import struct

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class CameraStreamServer:
    """Streams camera frames to a connected brain client."""

    def __init__(self, port=5556):
        self._port = port
        self._server_sock = None
        self._client_sock = None
        self._client_addr = None
        self._running = False

    def start(self):
        """Bind and listen (non-blocking)."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.setblocking(False)
        try:
            self._server_sock.bind(("0.0.0.0", self._port))
            self._server_sock.listen(1)
            self._running = True
            print("[CAM] Stream server listening on port %d" % self._port)
        except OSError as e:
            print("[CAM] ERROR: Could not bind port %d: %s" % (self._port, e))
            self._server_sock = None

    def stop(self):
        """Close all sockets."""
        self._running = False
        self._disconnect_client()
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None
        print("[CAM] Stream server stopped.")

    @property
    def is_connected(self):
        return self._client_sock is not None

    def poll(self):
        """Check for new client connections (non-blocking). Call every step."""
        if not self._running or not self._server_sock:
            return
        try:
            readable, _, _ = select.select([self._server_sock], [], [], 0)
            if readable:
                conn, addr = self._server_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(0.1)  # Short timeout for sendall
                if self._client_sock:
                    print("[CAM] Replacing existing camera client.")
                    self._disconnect_client()
                self._client_sock = conn
                self._client_addr = addr
                print("[CAM] Client connected from %s:%d" % addr)
        except Exception:
            pass

    def send_frame(self, webots_image, width, height):
        """Convert BGRA image from Webots camera and send as BGR.

        Args:
            webots_image: raw bytes from camera.getImage() (BGRA format)
            width: image width in pixels
            height: image height in pixels
        """
        if self._client_sock is None or webots_image is None:
            return

        bgr_bytes = _bgra_to_bgr(webots_image, width, height)
        if bgr_bytes is None:
            return

        header = struct.pack(">HH", width, height)
        try:
            self._client_sock.sendall(header + bgr_bytes)
        except (socket.timeout, BrokenPipeError,
                ConnectionResetError, OSError):
            print("[CAM] Client disconnected (send failed).")
            self._disconnect_client()

    def _disconnect_client(self):
        if self._client_sock:
            try:
                self._client_sock.close()
            except Exception:
                pass
        self._client_sock = None
        self._client_addr = None


def _bgra_to_bgr(image_bytes, width, height):
    """Convert Webots BGRA image to BGR bytes."""
    expected = width * height * 4
    if len(image_bytes) != expected:
        return None

    if _HAS_NUMPY:
        bgra = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
            height, width, 4)
        bgr = np.ascontiguousarray(bgra[:, :, :3])
        return bgr.tobytes()
    else:
        # Pure Python fallback (slower but works without numpy)
        bgr = bytearray(width * height * 3)
        si, di = 0, 0
        for _ in range(width * height):
            bgr[di] = image_bytes[si]
            bgr[di + 1] = image_bytes[si + 1]
            bgr[di + 2] = image_bytes[si + 2]
            si += 4
            di += 3
        return bytes(bgr)
