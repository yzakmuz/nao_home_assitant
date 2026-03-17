#!/usr/bin/env python3
"""Find working sample rate for USB mic."""
import sounddevice as sd

dev = sd.query_devices(kind='input')
print(f"Device: {dev['name']}")
print(f"Default rate: {dev['default_samplerate']}")

for rate in [48000, 44100, 32000, 16000, 8000]:
    try:
        sd.rec(int(0.1 * rate), samplerate=rate, channels=1, dtype='int16')
        sd.wait()
        print(f"  {rate} Hz — ✓ works")
    except Exception as e:
        print(f"  {rate} Hz — ✗ failed")
