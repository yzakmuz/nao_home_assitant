#!/usr/bin/env python3
"""Find which /dev/video* is the Pi camera."""
import cv2

for i in range(32):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        ret, frame = cap.read()
        if ret and frame is not None:
            brightness = frame.mean()
            print(f"  /dev/video{i} — ✓ frame {frame.shape}, brightness={brightness:.1f}")
        else:
            print(f"  /dev/video{i} — opened but no frame")
        cap.release()
