#!/usr/bin/env python3
"""
enroll_speaker.py -- Utility to enroll speaker voice(s) on the RPi.

Records multiple samples from the microphone, computes ECAPA-TDNN
embeddings for each, averages them for robustness, and saves the
embedding to disk for use by the speaker verification system.

Supports multi-speaker enrollment: each person gets a named embedding
file ({name}_embedding.npy) and the system can verify against all of them.

Usage:
    cd rpi_brain/
    python enroll_speaker.py                              # enroll master
    python enroll_speaker.py --person david               # enroll named person
    python enroll_speaker.py --person david --set-master  # enroll + set as master
    python enroll_speaker.py --duration 5 --samples 5
    python enroll_speaker.py --test-only                  # test existing enrollment
    python enroll_speaker.py --list                       # list enrolled speakers
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


def record_audio(duration: float, sample_rate: int) -> np.ndarray:
    """Record audio from the default microphone at the specified rate.

    Returns float32 mono array normalized to [-1.0, 1.0] at 16 kHz
    (resampled from native rate if needed).
    """
    log.info(
        "Recording for %.1f seconds at %d Hz -- speak clearly ...",
        duration, sample_rate,
    )
    time.sleep(0.3)  # small delay so the user is ready

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=settings.MIC_DEVICE_INDEX,
    )
    sd.wait()
    audio = audio.flatten()

    # Resample to 16 kHz if the mic was opened at a different rate
    if sample_rate != settings.MIC_SAMPLE_RATE:
        n_target = int(len(audio) * settings.MIC_SAMPLE_RATE / sample_rate)
        audio = np.interp(
            np.linspace(0, len(audio), n_target, endpoint=False),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    log.info("Recording complete (%.2f s at 16 kHz).", len(audio) / settings.MIC_SAMPLE_RATE)
    return audio


def enroll_multi_sample(
    verifier: SpeakerVerifier,
    num_samples: int,
    duration: float,
    sample_rate: int,
) -> bool:
    """Record multiple samples and average their embeddings for robust enrollment.

    Returns True if enrollment succeeded.
    """
    embeddings = []

    for i in range(num_samples):
        print(f"\n--- Sample {i + 1}/{num_samples} ---")
        if i > 0:
            input("Press ENTER when ready for the next sample ... ")

        audio = record_audio(duration, sample_rate)

        # Check audio level (detect silence / bad mic)
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        print(f"  Audio level: RMS={rms:.4f}, peak={peak:.4f}")

        if rms < 0.005:
            print("  WARNING: Very low audio level -- is the mic working?")
            print("  Skipping this sample.")
            continue

        if peak > 0.99:
            print("  WARNING: Audio is clipping -- move further from the mic.")

        emb = verifier.compute_embedding(audio)
        if emb is None:
            print("  WARNING: Could not compute embedding (audio too short?).")
            continue

        embeddings.append(emb)
        print(f"  Embedding computed (dim={emb.shape[0]}, norm={np.linalg.norm(emb):.4f})")

    if len(embeddings) < 1:
        print("\nERROR: No valid embeddings collected. Enrollment failed.")
        return False

    if len(embeddings) < num_samples:
        print(f"\nWARNING: Only {len(embeddings)}/{num_samples} samples were valid.")

    # Average embeddings and L2-normalize
    avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg_embedding)
    if norm > 1e-8:
        avg_embedding = avg_embedding / norm

    print(f"\nAveraged {len(embeddings)} embeddings (dim={avg_embedding.shape[0]})")

    # Save
    import os
    os.makedirs(os.path.dirname(settings.MASTER_EMBEDDING_PATH) or ".", exist_ok=True)
    np.save(settings.MASTER_EMBEDDING_PATH, avg_embedding)
    verifier._master_emb = avg_embedding

    return True


def run_self_test(verifier: SpeakerVerifier, sample_rate: int, duration: float) -> bool:
    """Record a new sample and verify it matches the enrolled master."""
    print("\n--- Self-Verification Test ---")
    input("Press ENTER to record a test sample ... ")

    audio = record_audio(duration, sample_rate)
    is_match, score = verifier.verify(audio)

    print(f"  Score: {score:.4f} (threshold: {settings.SPEAKER_VERIFY_THRESHOLD})")

    if is_match:
        print("  PASSED -- you are recognized as the owner.")
        return True
    else:
        print("  FAILED -- you are NOT recognized.")
        print("  Try: speak more clearly, reduce background noise, or lower the threshold.")
        return False


def list_enrolled() -> None:
    """List enrolled speakers."""
    import os
    models_dir = os.path.dirname(settings.MASTER_EMBEDDING_PATH) or "."
    print(f"\nEnrolled speakers (in {models_dir}):\n")
    found = False
    for f in sorted(os.listdir(models_dir)):
        if f.endswith("_embedding.npy") and f != "master_embedding.npy":
            name = f.replace("_embedding.npy", "")
            emb = np.load(os.path.join(models_dir, f))
            print(f"  {name:15s}  dim={emb.shape[0]}")
            found = True
    if not found:
        print("  (none)")

    if os.path.isfile(settings.MASTER_EMBEDDING_PATH):
        master_emb = np.load(settings.MASTER_EMBEDDING_PATH)
        for f in os.listdir(models_dir):
            if f.endswith("_embedding.npy") and f != "master_embedding.npy":
                emb = np.load(os.path.join(models_dir, f))
                if emb.shape == master_emb.shape and np.allclose(emb, master_emb):
                    name = f.replace("_embedding.npy", "")
                    print(f"\n  Current master: {name.upper()}")
                    break
    print()


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description="Enroll speaker voice(s) on RPi.")
    parser.add_argument(
        "--person", type=str, default=None,
        help="Name of the person to enroll (saves {name}_embedding.npy).",
    )
    parser.add_argument(
        "--set-master", action="store_true",
        help="Also set this person as master (copy to master_embedding.npy).",
    )
    parser.add_argument(
        "--duration", type=float, default=4.0,
        help="Recording duration per sample in seconds (default: 4.0).",
    )
    parser.add_argument(
        "--samples", type=int, default=3,
        help="Number of voice samples to record and average (default: 3).",
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="Skip enrollment, just test existing enrollment.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all enrolled speakers and exit.",
    )
    args = parser.parse_args()

    # Determine recording sample rate -- use native mic rate
    # The mic hardware may not support 16kHz directly
    sample_rate = settings.MIC_NATIVE_RATE

    print("\n" + "=" * 60)
    print("  Speaker Enrollment — ElderGuard")
    print("=" * 60)
    print()
    print(f"  Model:         {settings.SPEAKER_MODEL_PATH}")
    print(f"  Embedding:     {settings.MASTER_EMBEDDING_PATH}")
    print(f"  Mic rate:      {sample_rate} Hz (resampled to {settings.MIC_SAMPLE_RATE} Hz)")
    print(f"  Duration:      {args.duration:.1f} s per sample")
    print(f"  Samples:       {args.samples}")
    print(f"  Threshold:     {settings.SPEAKER_VERIFY_THRESHOLD}")
    print(f"  Multi-speaker: {settings.MULTI_SPEAKER_MODE}")
    print()

    if args.list:
        list_enrolled()
        return

    # Load the verifier
    verifier = SpeakerVerifier()

    if args.test_only:
        if settings.MULTI_SPEAKER_MODE and verifier.list_enrolled():
            # Test against all enrolled speakers
            print("Testing against all enrolled speakers...")
            input("Press ENTER to record test sample ... ")
            audio = record_audio(args.duration, sample_rate)
            accepted, name, score = verifier.verify_multi(audio)
            print(f"\n  Result: {'ACCEPTED' if accepted else 'REJECTED'}")
            print(f"  Speaker: {name}")
            print(f"  Score:   {score:.4f}")
        else:
            if verifier._master_emb is None:
                print("ERROR: No master embedding found. Run enrollment first.")
                sys.exit(1)
            ok = run_self_test(verifier, sample_rate, args.duration)
            sys.exit(0 if ok else 1)
        return

    # Multi-sample enrollment
    person_name = args.person
    if person_name:
        print(f"Enrolling: {person_name.upper()}")
    else:
        print("Enrolling master speaker.")
    print("Speak naturally -- say a few sentences each time.")
    print("For best results: speak in a quiet environment.\n")

    input("Press ENTER when ready to begin enrollment ... ")

    success = enroll_multi_sample(
        verifier, args.samples, args.duration, sample_rate,
    )

    if not success:
        print("\nEnrollment failed. Please check your microphone and try again.")
        sys.exit(1)

    # If --person was given, also save as named embedding
    if person_name:
        models_dir = os.path.dirname(settings.MASTER_EMBEDDING_PATH) or "."
        named_path = os.path.join(models_dir, f"{person_name.lower()}_embedding.npy")
        master_emb = np.load(settings.MASTER_EMBEDDING_PATH)
        np.save(named_path, master_emb)
        print(f"\nNamed embedding saved to: {named_path}")
        verifier._enrolled[person_name.lower()] = master_emb

        if not args.set_master:
            print(f"\nNote: {person_name} is enrolled but NOT set as master.")
            print(f"  To set as master: python enroll_speaker.py --person {person_name} --set-master")
        else:
            print(f"\n{person_name.upper()} set as master speaker.")
    else:
        print(f"\nMaster embedding saved to: {settings.MASTER_EMBEDDING_PATH}")

    # Self-test
    print("\nRunning self-verification test ...")
    ok = run_self_test(verifier, sample_rate, args.duration)

    if ok:
        print("\n" + "=" * 60)
        print("  Enrollment complete!")
        print("=" * 60)
        print()
    else:
        print("\nSelf-test failed. Consider re-enrolling with more/clearer samples.")
        sys.exit(1)


if __name__ == "__main__":
    main()
