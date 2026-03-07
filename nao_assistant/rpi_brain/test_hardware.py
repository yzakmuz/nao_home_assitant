#!/usr/bin/env python3
"""test_hardware.py — Test USB mic, Pi Camera, and TCP to NAO."""
import time
import numpy as np

print("=" * 50)
print("  RPI Brain — Hardware Test")
print("=" * 50)

# ── 1. USB Microphone ──
print("\n[1/3] USB Microphone...")
try:
    import sounddevice as sd
    devices = sd.query_devices()
    # Find input devices
    input_devs = [d for d in devices if d['max_input_channels'] > 0]
    print(f"  Found {len(input_devs)} input device(s):")
    for d in input_devs:
        print(f"    - {d['name']} (channels: {d['max_input_channels']})")
    
    # Record 2 seconds
    print("  Recording 2 seconds... SPEAK NOW!")
    audio = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype='int16')
    sd.wait()
    peak = np.max(np.abs(audio))
    print(f"  ✓ Recorded {len(audio)} samples, peak amplitude: {peak}")
    if peak > 500:
        print(f"  ✓ Mic is picking up sound!")
    else:
        print(f"  ⚠ Very quiet — check mic connection or speak louder")
except Exception as e:
    print(f"  ✗ {e}")

# ── 2. Pi Camera ──
print("\n[2/3] Pi Camera...")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Try libcamera index
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"  ✓ Frame captured: {frame.shape} (H x W x channels)")
        print(f"  Mean brightness: {frame.mean():.1f} (0=black, 255=white)")
        cv2.imwrite("/tmp/test_frame.jpg", frame)
        print(f"  ✓ Saved test frame to /tmp/test_frame.jpg")
    else:
        print(f"  ✗ Could not read frame from camera")
    cap.release()
except Exception as e:
    print(f"  ✗ {e}")

# Try picamera2 as fallback
try:
    if not ret:
        print("  Trying picamera2 instead...")
        from picamera2 import Picamera2
        picam = Picamera2()
        config = picam.create_still_configuration(main={"size": (320, 240)})
        picam.configure(config)
        picam.start()
        time.sleep(1)
        frame = picam.capture_array()
        print(f"  ✓ picamera2 frame: {frame.shape}")
        picam.stop()
except:
    pass

# ── 3. TCP to NAO ──
print("\n[3/3] TCP Connection to NAO...")
print("  Enter NAO IP (or press Enter to skip): ", end="")
nao_ip = input().strip()
if nao_ip:
    try:
        import socket, json
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        t = time.time()
        sock.connect((nao_ip, 9559))
        print(f"  ✓ Connected in {time.time()-t:.2f}s")
        
        # Send a test command
        cmd = {"action": "say", "text": "Brain connected. Hardware test passed."}
        sock.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
        print(f"  ✓ Sent: {cmd}")
        
        # Wait for response
        data = sock.recv(4096).decode("utf-8")
        print(f"  ✓ NAO responded: {data.strip()}")
        sock.close()
    except Exception as e:
        print(f"  ✗ {e}")
else:
    print("  ⊘ Skipped (no IP entered)")

print("\n" + "=" * 50)
print("  HARDWARE TEST DONE")
print("=" * 50)
