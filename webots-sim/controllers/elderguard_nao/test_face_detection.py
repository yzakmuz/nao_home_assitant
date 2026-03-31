"""
test_face_detection.py — Test virtual person face detection.

Tests two approaches for detecting the virtual person:
  Option A: MediaPipe BlazeFace on camera stream frames
  Option B: Geometric position from Supervisor (get_person_position)

Run this while Webots simulation is running with a Pedestrian in the world.

Usage:
    python test_face_detection.py
    python test_face_detection.py --no-display    (headless)
    python test_face_detection.py --position-only  (skip MediaPipe, test position query only)
"""

import json
import socket
import struct
import sys
import threading
import time

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

HAS_MEDIAPIPE = False  # Not used — OpenCV Haar cascade used instead


# ===================================================================
# TCP Client for person position query
# ===================================================================

class SimpleTcpClient:
    """Minimal TCP client for querying the Webots controller."""

    def __init__(self, host="127.0.0.1", port=5555):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        self.sock.connect((host, port))
        self._buf = ""
        self._counter = 0

    def query_person_position(self):
        self._counter += 1
        rid = "face_%d" % self._counter
        cmd = {"action": "get_person_position", "id": rid}
        self.sock.sendall((json.dumps(cmd) + "\n").encode("utf-8"))

        resp = self._recv()
        if resp and resp.get("person_found"):
            return resp.get("person_position")
        return None

    def _recv(self):
        while "\n" not in self._buf:
            data = self.sock.recv(4096)
            if not data:
                return None
            self._buf += data.decode("utf-8")
        line, self._buf = self._buf.split("\n", 1)
        return json.loads(line.strip()) if line.strip() else None

    def close(self):
        self.sock.close()


# ===================================================================
# Camera receiver (inline, not using WebotsCamera for simplicity)
# ===================================================================

def receive_camera_frame(sock):
    """Receive one frame from camera stream. Returns BGR numpy array or None."""
    try:
        header = _recv_exact(sock, 4)
        if header is None:
            return None
        w, h = struct.unpack(">HH", header)
        data = _recv_exact(sock, w * h * 3)
        if data is None:
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
    except Exception:
        return None


def _recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(min(n - len(data), 65536))
        if not chunk:
            return None
        data += chunk
    return data


# ===================================================================
# Test: Person Position Query (Option B)
# ===================================================================

def test_person_position():
    """Test that get_person_position returns valid coordinates."""
    print("\n--- Test: Person Position Query (Option B) ---")
    try:
        tcp = SimpleTcpClient("127.0.0.1", 5555)
    except ConnectionRefusedError:
        print("[ERROR] Cannot connect to TCP port 5555")
        return False

    pos = tcp.query_person_position()
    tcp.close()

    if pos:
        print("[OK] Person found at: x=%.3f  y=%.3f  z=%.3f" % (
            pos["x"], pos["y"], pos["z"]))
        return True
    else:
        print("[WARN] No person found (DEF PERSON missing from world?)")
        return False


# ===================================================================
# Test: MediaPipe Face Detection (Option A)
# ===================================================================

def test_face_detection_camera(no_display=False):
    """Test face detection on camera stream frames using OpenCV Haar cascade."""
    print("\n--- Test: Face Detection on Camera Stream (Option A) ---")

    if not HAS_CV2:
        print("[SKIP] OpenCV not installed")
        return False

    # Connect to camera stream
    try:
        cam_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cam_sock.settimeout(5.0)
        cam_sock.connect(("127.0.0.1", 5556))
        print("[OK] Connected to camera stream (port 5556)")
    except ConnectionRefusedError:
        print("[ERROR] Cannot connect to camera stream port 5556")
        return False

    # Initialize OpenCV Haar cascade face detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("[ERROR] Could not load Haar cascade")
        return False
    print("[OK] OpenCV Haar cascade face detector loaded")

    detected_count = 0
    total_frames = 0
    max_frames = 150  # Test ~10 seconds at 15 FPS
    start = time.monotonic()

    print("[INFO] Testing %d frames... (press 'q' to stop)" % max_frames)

    try:
        for _ in range(max_frames):
            frame = receive_camera_frame(cam_sock)
            if frame is None:
                break

            total_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

            has_face = len(faces) > 0
            if has_face:
                detected_count += 1

            if not no_display:
                display = frame.copy()

                for (x, y, w, h) in faces:
                    cv2.rectangle(display, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    cv2.putText(display, "Face", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 1)

                label = "Face: %s | %d/%d detected" % (
                    "YES" if has_face else "no",
                    detected_count, total_frames)
                cv2.putText(display, label, (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1)
                cv2.imshow("Face Detection Test", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if total_frames % 15 == 0:
                    print("  Frame %d — face: %s  (%d/%d)" % (
                        total_frames, "YES" if has_face else "no",
                        detected_count, total_frames))

    except KeyboardInterrupt:
        pass

    elapsed = time.monotonic() - start
    cam_sock.close()
    if HAS_CV2:
        cv2.destroyAllWindows()

    rate = (detected_count / total_frames * 100) if total_frames > 0 else 0
    print("\nResults:")
    print("  Frames:   %d" % total_frames)
    print("  Detected: %d (%.1f%%)" % (detected_count, rate))
    print("  Duration: %.1f s" % elapsed)

    if rate > 50:
        print("[PASS] Face detector sees the virtual person (%.0f%% rate)" % rate)
        return True
    elif rate > 10:
        print("[PARTIAL] Face detected sometimes (%.0f%%) — may need adjustment" % rate)
        return True
    else:
        print("[INFO] Face detector cannot detect the 3D rendered face (%.0f%%)" % rate)
        print("       This is expected — Webots faces are too low-poly")
        print("       → Use Option B (geometric projection from 3D position)")
        return False


# ===================================================================
# Main
# ===================================================================

def main():
    no_display = "--no-display" in sys.argv
    position_only = "--position-only" in sys.argv

    print("=" * 55)
    print("  ElderGuard — Virtual Person Face Detection Test")
    print("=" * 55)

    # Test 1: Person position query
    pos_ok = test_person_position()

    # Test 2: MediaPipe face detection
    if not position_only:
        mp_ok = test_face_detection_camera(no_display)
    else:
        mp_ok = False
        print("\n[SKIP] MediaPipe test (--position-only)")

    # Summary
    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    print("  Person position (Option B): %s" % (
        "WORKING" if pos_ok else "NOT AVAILABLE"))
    print("  Face detection  (Option A): %s" % (
        "WORKING" if mp_ok else "NOT DETECTED"))
    print("")

    if mp_ok:
        print("  Recommendation: Use Option A (camera-based face detection)")
    elif pos_ok:
        print("  Recommendation: Use Option B (geometric projection from 3D position)")
        print("  Note: Option B is 100%% reliable for Webots simulation")
    else:
        print("  [ERROR] Neither option works — check world file has DEF PERSON")

    print("=" * 55)


if __name__ == "__main__":
    main()
