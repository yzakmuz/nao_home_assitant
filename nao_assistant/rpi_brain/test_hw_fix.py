#!/usr/bin/env python3
"""Test mic at 44100Hz + resample, and camera via picamera2."""
import time
import numpy as np

print("=" * 50)
print("  Hardware Fix Test")
print("=" * 50)

# ── 1. Mic at 44100 + resample to 16000 ──
print("\n[1/2] USB Mic (44100Hz → resample to 16000Hz)...")
try:
    import sounddevice as sd

    print("  Recording 3 seconds at 44100Hz... SPEAK NOW!")
    audio = sd.rec(int(3 * 44100), samplerate=44100, channels=1, dtype='int16')
    sd.wait()
    audio = audio.flatten()
    peak_raw = np.max(np.abs(audio))
    print(f"  ✓ Recorded {len(audio)} samples, peak: {peak_raw}")

    # Resample 44100 → 16000
    ratio = 16000 / 44100
    new_len = int(len(audio) * ratio)
    indices = np.arange(new_len) / ratio
    indices = indices.astype(int)
    indices = np.clip(indices, 0, len(audio) - 1)
    audio_16k = audio[indices]
    print(f"  ✓ Resampled: {len(audio_16k)} samples at 16000Hz")

    # Quick Vosk test with real audio
    from vosk import Model, KaldiRecognizer
    import json
    model = Model("models/vosk-model-small-en-us-0.15")
    grammar = '["hey nao", "follow me", "stop", "sit down", "stand up", "dance", "wave", "rest", "find my keys", "hello", "test", "[unk]"]'
    rec = KaldiRecognizer(model, 16000, grammar)
    rec.AcceptWaveform(audio_16k.tobytes())
    result = json.loads(rec.FinalResult())
    print(f"  ✓ Vosk heard: \"{result.get('text', '')}\"")
    if peak_raw > 500:
        print(f"  ✓ Mic is working!")
    else:
        print(f"  ⚠ Very quiet — speak louder or check mic")
    del model, rec
except Exception as e:
    print(f"  ✗ {e}")

# ── 2. Pi Camera via picamera2 ──
print("\n[2/2] Pi Camera (picamera2)...")
try:
    from picamera2 import Picamera2
    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={"size": (320, 240), "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()
    time.sleep(1)  # let camera warm up

    frame = picam.capture_array()
    print(f"  ✓ Frame captured: {frame.shape}")
    print(f"  Mean brightness: {frame.mean():.1f}")

    # Save test image
    import cv2
    cv2.imwrite("/tmp/camera_test.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"  ✓ Saved to /tmp/camera_test.jpg")

    # FPS test - 10 frames
    t = time.time()
    for _ in range(10):
        picam.capture_array()
    fps = 10 / (time.time() - t)
    print(f"  ✓ Capture speed: {fps:.1f} FPS")

    picam.stop()
    picam.close()
except Exception as e:
    print(f"  ✗ {e}")

print("\n" + "=" * 50)
print("  DONE")
print("=" * 50)
