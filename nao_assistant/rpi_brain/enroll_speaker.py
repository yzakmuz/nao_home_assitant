#!/usr/bin/env python3
"""
enroll_speaker.py — One-time utility to enroll the master's voice.

Records a short sample from the microphone, computes an ECAPA-TDNN
embedding, and saves it to disk for use by the speaker verification
system at runtime.

Usage:
    cd rpi_brain/
    python enroll_speaker.py

    # Optionally specify duration:
    python enroll_speaker.py --duration 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import sounddevice as sd

import settings
from audio.speaker_verify import SpeakerVerifier

logging.basicConfig(level="INFO", format=settings.LOG_FORMAT)
log = logging.getLogger("enroll")


def record_audio(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Record audio from the default microphone."""
    log.info(
        "Recording for %.1f seconds — please speak clearly …", duration
    )
    time.sleep(0.5)  # small delay so the user is ready

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    log.info("Recording complete.")
    return audio.flatten()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll master speaker voice.")
    parser.add_argument(
        "--duration", type=float, default=4.0,
        help="Recording duration in seconds (default: 4.0).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  Speaker Enrollment")
    print("=" * 50)
    print(
        "\nThis will record your voice and create an embedding\n"
        "file that the robot uses to verify your identity.\n"
    )
    print(f"  Model:      {settings.SPEAKER_MODEL_PATH}")
    print(f"  Output:     {settings.MASTER_EMBEDDING_PATH}")
    print(f"  Duration:   {args.duration:.1f} seconds")
    print(f"  Threshold:  {settings.SPEAKER_VERIFY_THRESHOLD}")
    print()

    input("Press ENTER when ready to record … ")

    audio = record_audio(args.duration)

    # Compute & save embedding
    verifier = SpeakerVerifier()
    success = verifier.enroll(audio)

    if success:
        print("\n✓ Enrollment successful!")
        print(f"  Embedding saved to: {settings.MASTER_EMBEDDING_PATH}")

        # Quick self-test
        print("\nRunning self-verification test …")
        is_match, score = verifier.verify(audio)
        print(f"  Self-test score: {score:.4f} (threshold: {settings.SPEAKER_VERIFY_THRESHOLD})")
        if is_match:
            print("  ✓ Self-test PASSED — you are recognized.\n")
        else:
            print("  ✗ Self-test FAILED — try recording again with clearer speech.\n")
            sys.exit(1)
    else:
        print("\n✗ Enrollment failed. Ensure the recording is long enough.")
        sys.exit(1)


if __name__ == "__main__":
    main()
