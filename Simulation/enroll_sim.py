#!/usr/bin/env python3
"""
enroll_sim.py -- Multi-person voice enrollment for the ElderGuard PC simulation.

Records voice samples for multiple people, computes ECAPA-TDNN embeddings,
and saves master embeddings for use by the speaker verification system.

Usage:
    # Enroll David (3 samples, set as master):
    python enroll_sim.py --person david --set-master

    # Enroll Itzhak (3 samples):
    python enroll_sim.py --person itzhak

    # Enroll both people interactively:
    python enroll_sim.py --enroll-both --set-master david

    # Test existing enrollment:
    python enroll_sim.py --test-only

    # List enrolled speakers:
    python enroll_sim.py --list

    # Custom settings:
    python enroll_sim.py --person david --samples 5 --duration 5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap — MUST happen before any rpi_brain import
# ---------------------------------------------------------------------------
_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

import sim_config
# Do NOT install fake audio stubs — we need real sounddevice for recording
sim_config.SIM_NO_MIC = False
sim_config.SIM_SKIP_VERIFY = False

from adapters.bootstrap import bootstrap
bootstrap(sim_config)

# Now safe to import rpi_brain modules
import sounddevice as sd
from audio.speaker_verify import SpeakerVerifier

logging.basicConfig(level="INFO", format=sim_config.LOG_FORMAT)
log = logging.getLogger("enroll_sim")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODELS_DIR = os.path.dirname(sim_config.MASTER_EMBEDDING_PATH)


def _embedding_path(person_name: str) -> str:
    """Return the file path for a named person's embedding."""
    return os.path.join(_MODELS_DIR, f"{person_name.lower()}_embedding.npy")


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def record_audio(duration: float) -> np.ndarray:
    """Record audio from the PC microphone.

    Returns float32 mono array normalized to [-1.0, 1.0] at 16 kHz.
    """
    sample_rate = sim_config.MIC_NATIVE_RATE
    print(f"  Recording for {duration:.1f}s at {sample_rate} Hz — speak now ...")
    time.sleep(0.3)

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=sim_config.MIC_DEVICE_INDEX,
    )
    sd.wait()
    audio = audio.flatten()

    # Resample to 16 kHz if needed
    if sample_rate != sim_config.MIC_SAMPLE_RATE:
        n_target = int(len(audio) * sim_config.MIC_SAMPLE_RATE / sample_rate)
        audio = np.interp(
            np.linspace(0, len(audio), n_target, endpoint=False),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    return audio


def check_audio_quality(audio: np.ndarray) -> bool:
    """Check audio quality. Returns True if OK."""
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    print(f"  Audio level: RMS={rms:.4f}, peak={peak:.4f}")

    if rms < 0.005:
        print("  WARNING: Very low audio level — is the mic working?")
        return False
    if peak > 0.99:
        print("  WARNING: Audio is clipping — move further from the mic.")
    return True


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------
def enroll_person(
    verifier: SpeakerVerifier,
    person_name: str,
    num_samples: int = 3,
    duration: float = 4.0,
) -> bool:
    """Record voice samples and save an averaged embedding for one person.

    Returns True on success.
    """
    print(f"\n{'='*50}")
    print(f"  Enrolling: {person_name.upper()}")
    print(f"  Samples: {num_samples}, Duration: {duration:.1f}s each")
    print(f"{'='*50}")
    print()
    print("Speak naturally — say a few sentences each time.")
    print("For best results: speak in a quiet environment.\n")

    embeddings = []

    for i in range(num_samples):
        print(f"\n--- {person_name}: Sample {i+1}/{num_samples} ---")
        if i > 0:
            input("Press ENTER when ready for the next sample ... ")

        audio = record_audio(duration)

        if not check_audio_quality(audio):
            print("  Skipping this sample.")
            continue

        emb = verifier.compute_embedding(audio)
        if emb is None:
            print("  WARNING: Could not compute embedding.")
            continue

        embeddings.append(emb)
        print(f"  Embedding computed (dim={emb.shape[0]})")

    if len(embeddings) < 1:
        print(f"\nERROR: No valid embeddings for {person_name}. Enrollment failed.")
        return False

    if len(embeddings) < num_samples:
        print(f"\nWARNING: Only {len(embeddings)}/{num_samples} samples were valid.")

    # Average and L2-normalize
    avg_emb = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg_emb)
    if norm > 1e-8:
        avg_emb = avg_emb / norm

    # Save named embedding
    os.makedirs(_MODELS_DIR, exist_ok=True)
    emb_path = _embedding_path(person_name)
    np.save(emb_path, avg_emb)
    print(f"\nSaved {person_name}'s embedding to: {emb_path}")
    print(f"  dim={avg_emb.shape[0]}, norm={np.linalg.norm(avg_emb):.4f}")

    return True


def set_master(person_name: str) -> None:
    """Copy a person's embedding to master_embedding.npy (used by the verifier)."""
    src = _embedding_path(person_name)
    if not os.path.isfile(src):
        print(f"ERROR: No embedding found for '{person_name}' at {src}")
        sys.exit(1)

    emb = np.load(src)
    np.save(sim_config.MASTER_EMBEDDING_PATH, emb)
    print(f"\nMaster speaker set to: {person_name.upper()}")
    print(f"  Copied {src} -> {sim_config.MASTER_EMBEDDING_PATH}")


def list_enrolled() -> None:
    """List all enrolled speaker embeddings."""
    print(f"\nEnrolled speakers (in {_MODELS_DIR}):\n")
    found = False
    for f in sorted(os.listdir(_MODELS_DIR)):
        if f.endswith("_embedding.npy") and f != "master_embedding.npy":
            name = f.replace("_embedding.npy", "")
            emb = np.load(os.path.join(_MODELS_DIR, f))
            print(f"  {name:15s}  dim={emb.shape[0]}")
            found = True

    if not found:
        print("  (none)")

    # Show who is master
    if os.path.isfile(sim_config.MASTER_EMBEDDING_PATH):
        master_emb = np.load(sim_config.MASTER_EMBEDDING_PATH)
        # Find which person matches
        for f in os.listdir(_MODELS_DIR):
            if f.endswith("_embedding.npy") and f != "master_embedding.npy":
                emb = np.load(os.path.join(_MODELS_DIR, f))
                if np.allclose(emb, master_emb):
                    name = f.replace("_embedding.npy", "")
                    print(f"\n  Current master: {name.upper()}")
                    break
    print()


def run_test(verifier: SpeakerVerifier) -> None:
    """Record a test sample and compare against all enrolled speakers."""
    print("\n--- Speaker Verification Test ---")
    print("This will record a short sample and compare against all enrolled profiles.\n")
    input("Press ENTER to record test sample ... ")

    audio = record_audio(4.0)
    if not check_audio_quality(audio):
        print("Audio quality too low for testing.")
        return

    test_emb = verifier.compute_embedding(audio)
    if test_emb is None:
        print("Could not compute embedding for test sample.")
        return

    # L2-normalize
    norm = np.linalg.norm(test_emb)
    if norm > 1e-8:
        test_emb = test_emb / norm

    print(f"\n{'Person':15s} {'Score':>8s} {'Result':>10s}")
    print("-" * 35)

    for f in sorted(os.listdir(_MODELS_DIR)):
        if f.endswith("_embedding.npy") and f != "master_embedding.npy":
            name = f.replace("_embedding.npy", "")
            emb = np.load(os.path.join(_MODELS_DIR, f))
            score = float(np.dot(test_emb, emb))
            accepted = score >= sim_config.SPEAKER_VERIFY_THRESHOLD
            tag = "MATCH" if accepted else "reject"
            print(f"  {name:13s} {score:8.4f} {tag:>10s}")

    # Also test against master
    if os.path.isfile(sim_config.MASTER_EMBEDDING_PATH):
        master = np.load(sim_config.MASTER_EMBEDDING_PATH)
        score = float(np.dot(test_emb, master))
        accepted = score >= sim_config.SPEAKER_VERIFY_THRESHOLD
        tag = "MATCH" if accepted else "reject"
        print(f"  {'(master)':13s} {score:8.4f} {tag:>10s}")

    print(f"\n  Threshold: {sim_config.SPEAKER_VERIFY_THRESHOLD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-person voice enrollment for ElderGuard simulation."
    )
    parser.add_argument(
        "--person", type=str,
        help="Name of the person to enroll (e.g., david, itzhak).",
    )
    parser.add_argument(
        "--enroll-both", action="store_true",
        help="Enroll both David and Itzhak interactively.",
    )
    parser.add_argument(
        "--samples", type=int, default=3,
        help="Number of voice samples per person (default: 3).",
    )
    parser.add_argument(
        "--duration", type=float, default=4.0,
        help="Recording duration per sample in seconds (default: 4.0).",
    )
    parser.add_argument(
        "--set-master", nargs="?", const="__auto__", default=None,
        help="Set the enrolled person as master (or specify name).",
    )
    parser.add_argument(
        "--test-only", action="store_true",
        help="Skip enrollment, test against all enrolled profiles.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all enrolled speakers and exit.",
    )
    args = parser.parse_args()

    # Header
    print("\n" + "=" * 55)
    print("  Speaker Enrollment — ElderGuard Simulation")
    print("=" * 55)
    print(f"  Model:     {sim_config.SPEAKER_MODEL_PATH}")
    print(f"  Models dir: {_MODELS_DIR}")
    print(f"  Mic rate:  {sim_config.MIC_NATIVE_RATE} Hz "
          f"(resampled to {sim_config.MIC_SAMPLE_RATE} Hz)")
    print(f"  Threshold: {sim_config.SPEAKER_VERIFY_THRESHOLD}")

    # Quick actions
    if args.list:
        list_enrolled()
        return

    # Load verifier
    verifier = SpeakerVerifier()

    if args.test_only:
        run_test(verifier)
        return

    # Enrollment
    if args.enroll_both:
        people = ["david", "itzhak"]
    elif args.person:
        people = [args.person.lower()]
    else:
        parser.error("Specify --person NAME or --enroll-both.")

    for person in people:
        input(f"\nPress ENTER to start enrolling {person.upper()} ... ")
        ok = enroll_person(verifier, person, args.samples, args.duration)
        if not ok:
            print(f"Enrollment failed for {person}.")
            sys.exit(1)

    # Set master
    master_name = None
    if args.set_master:
        if args.set_master == "__auto__":
            master_name = people[0]  # default to first enrolled person
        else:
            master_name = args.set_master.lower()
        set_master(master_name)

    # Run self-test
    print("\n" + "=" * 55)
    print("  Enrollment complete! Running verification test...")
    print("=" * 55)
    run_test(verifier)

    print("\nDone! You can now run the simulation with speaker verification:")
    if master_name:
        print(f"  Master speaker: {master_name.upper()}")
    print("  python run_simulation.py")


if __name__ == "__main__":
    main()
