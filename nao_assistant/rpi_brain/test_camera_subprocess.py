#!/usr/bin/env python3
"""Test camera capture using rpicam-still (no picamera2 needed)."""
import subprocess
import time
import cv2
import numpy as np

print("=" * 50)
print("  Camera Test via rpicam-still")
print("=" * 50)

# Method 1: Capture single frame to file
print("\n[1/2] Single frame capture...")
try:
    t = time.time()
    subprocess.run([
        "rpicam-still",
        "-o", "/tmp/frame.jpg",
        "--width", "320",
        "--height", "240",
        "--timeout", "500",
        "--nopreview",
        "-q", "80"
    ], capture_output=True, timeout=5)
    frame = cv2.imread("/tmp/frame.jpg")
    print(f"  ✓ Captured {frame.shape} in {time.time()-t:.2f}s")
    print(f"  Mean brightness: {frame.mean():.1f}")
except Exception as e:
    print(f"  ✗ {e}")

# Method 2: Capture to stdout (faster, no disk I/O)
print("\n[2/2] Stream capture via stdout...")
try:
    t = time.time()
    result = subprocess.run([
        "rpicam-still",
        "-o", "-",
        "--width", "320",
        "--height", "240",
        "--timeout", "500",
        "--nopreview",
        "--encoding", "jpg",
        "-q", "80"
    ], capture_output=True, timeout=5)
    arr = np.frombuffer(result.stdout, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    print(f"  ✓ Captured {frame.shape} in {time.time()-t:.2f}s")

    # Quick MediaPipe face test
    import mediapipe as mp
    fd = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det = fd.process(rgb)
    faces = det.detections if det.detections else []
    print(f"  ✓ MediaPipe found {len(faces)} face(s)")
    fd.close()
except Exception as e:
    print(f"  ✗ {e}")

print("\n" + "=" * 50)
print("  DONE")
print("=" * 50)
