#!/usr/bin/env python3
"""
enroll_from_data.py -- Compute speaker embeddings from pre-recorded voice data.

Processes WAV files in data/processed/ (or data/augmented/) for each person,
computes ECAPA-TDNN embeddings, averages them, and saves the result.
At runtime, the simulation (or RPi) loads the embedding for the active master.

This script works for BOTH the PC simulation AND the real RPi deployment.
The output .npy files are portable -- just copy them to the RPi's models/ folder.

Architecture:
    OFFLINE (this script, on PC):
        data/processed/david/*.wav  --> ECAPA-TDNN --> david_embedding.npy
        data/processed/itzhak/*.wav --> ECAPA-TDNN --> itzhak_embedding.npy

    RUNTIME (run_simulation.py or main.py on RPi):
        --master david  --> loads david_embedding.npy as master
        Voice command audio --> ECAPA embedding --> cosine vs master --> accept/reject

Usage:
    # Process all people from processed data:
    python enroll_from_data.py

    # Process only david:
    python enroll_from_data.py --person david

    # Use augmented data instead of processed:
    python enroll_from_data.py --source augmented

    # Set david as master:
    python enroll_from_data.py --set-master david

    # Verify embeddings against each other:
    python enroll_from_data.py --verify

    # List enrolled speakers:
    python enroll_from_data.py --list
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import wave

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

import sim_config

sim_config.SIM_NO_MIC = True  # don't need mic for file-based enrollment
sim_config.SIM_SKIP_VERIFY = False  # need the real verifier
sim_config.SIM_USE_SPEECHBRAIN_VERIFY = True  # use SpeechBrain on PC

from adapters.bootstrap import bootstrap
bootstrap(sim_config)

from audio.speaker_verify import SpeakerVerifier

logging.basicConfig(level="INFO", format=sim_config.LOG_FORMAT)
log = logging.getLogger("enroll_data")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(_SIM_DIR, "data")
_MODELS_DIR = os.path.dirname(sim_config.MASTER_EMBEDDING_PATH)


def _embedding_path(person_name: str) -> str:
    """Return path for a named person's embedding."""
    return os.path.join(_MODELS_DIR, f"{person_name.lower()}_embedding.npy")


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_wav(path: str) -> np.ndarray:
    """Load a WAV file as float32 mono at 16 kHz.

    Handles mono/stereo, any sample rate (resamples to 16 kHz).
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # Convert to numpy
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Stereo → mono
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    elif n_channels > 2:
        audio = audio.reshape(-1, n_channels)[:, 0]  # take first channel

    # Resample to 16 kHz if needed
    target_rate = sim_config.MIC_SAMPLE_RATE
    if framerate != target_rate:
        n_target = int(len(audio) * target_rate / framerate)
        audio = np.interp(
            np.linspace(0, len(audio), n_target, endpoint=False),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    return audio


# ---------------------------------------------------------------------------
# Enrollment from files
# ---------------------------------------------------------------------------
def enroll_from_files(
    verifier: SpeakerVerifier,
    person_name: str,
    wav_files: list[str],
    max_files: int = 0,
) -> bool:
    """Compute averaged embedding from a list of WAV files.

    Returns True on success.
    """
    if max_files > 0:
        wav_files = wav_files[:max_files]

    print(f"\n{'='*55}")
    print(f"  Enrolling: {person_name.upper()} ({len(wav_files)} files)")
    print(f"{'='*55}")

    embeddings = []
    skipped = 0

    for i, path in enumerate(wav_files):
        fname = os.path.basename(path)
        try:
            audio = load_wav(path)
        except Exception as e:
            print(f"  [{i+1:3d}] {fname:30s} ERROR: {e}")
            skipped += 1
            continue

        duration = len(audio) / sim_config.MIC_SAMPLE_RATE
        rms = np.sqrt(np.mean(audio ** 2))

        if rms < 0.003:
            print(f"  [{i+1:3d}] {fname:30s} SKIP (silent, RMS={rms:.4f})")
            skipped += 1
            continue

        emb = verifier.compute_embedding(audio)
        if emb is None:
            print(f"  [{i+1:3d}] {fname:30s} SKIP (too short: {duration:.1f}s)")
            skipped += 1
            continue

        embeddings.append(emb)
        print(f"  [{i+1:3d}] {fname:30s} OK  ({duration:.1f}s, RMS={rms:.4f})")

    if len(embeddings) < 1:
        print(f"\nERROR: No valid embeddings for {person_name}.")
        return False

    print(f"\n  Valid: {len(embeddings)}/{len(wav_files)} "
          f"(skipped: {skipped})")

    # Average and L2-normalize
    avg_emb = np.mean(embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg_emb)
    if norm > 1e-8:
        avg_emb = avg_emb / norm

    # Save
    os.makedirs(_MODELS_DIR, exist_ok=True)
    emb_path = _embedding_path(person_name)
    np.save(emb_path, avg_emb)
    print(f"\n  Saved: {emb_path}")
    print(f"  dim={avg_emb.shape[0]}, norm={np.linalg.norm(avg_emb):.4f}")

    return True


def set_master(person_name: str) -> None:
    """Copy a person's embedding to master_embedding.npy."""
    src = _embedding_path(person_name)
    if not os.path.isfile(src):
        print(f"ERROR: No embedding for '{person_name}' at {src}")
        sys.exit(1)

    emb = np.load(src)
    np.save(sim_config.MASTER_EMBEDDING_PATH, emb)
    print(f"\nMaster set to: {person_name.upper()}")
    print(f"  {src} -> {sim_config.MASTER_EMBEDDING_PATH}")


def list_enrolled() -> None:
    """List enrolled speakers and show which is master."""
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

    if os.path.isfile(sim_config.MASTER_EMBEDDING_PATH):
        master_emb = np.load(sim_config.MASTER_EMBEDDING_PATH)
        for f in os.listdir(_MODELS_DIR):
            if f.endswith("_embedding.npy") and f != "master_embedding.npy":
                emb = np.load(os.path.join(_MODELS_DIR, f))
                if emb.shape == master_emb.shape and np.allclose(emb, master_emb):
                    name = f.replace("_embedding.npy", "")
                    print(f"\n  Current master: {name.upper()}")
                    break
    print()


def verify_embeddings() -> None:
    """Print cross-similarity matrix between all enrolled speakers."""
    print(f"\nCross-Similarity Matrix:")
    print(f"  (cosine similarity, threshold={sim_config.SPEAKER_VERIFY_THRESHOLD})\n")

    # Load all embeddings
    speakers = {}
    for f in sorted(os.listdir(_MODELS_DIR)):
        if f.endswith("_embedding.npy") and f != "master_embedding.npy":
            name = f.replace("_embedding.npy", "")
            speakers[name] = np.load(os.path.join(_MODELS_DIR, f))

    if len(speakers) < 2:
        print("  Need at least 2 enrolled speakers for cross-comparison.")
        return

    # Header
    names = list(speakers.keys())
    header = f"  {'':15s}" + "".join(f"{n:>12s}" for n in names)
    print(header)
    print("  " + "-" * (15 + 12 * len(names)))

    for n1 in names:
        row = f"  {n1:15s}"
        for n2 in names:
            score = float(np.dot(speakers[n1], speakers[n2]))
            tag = " *" if n1 == n2 else ""
            row += f"{score:10.4f}{tag:>2s}"
        print(row)

    print()
    print("  * = self-similarity (should be ~1.0)")
    print("  Cross-speaker scores should be < threshold for good separation")


# ---------------------------------------------------------------------------
# Find WAV files for a person
# ---------------------------------------------------------------------------
def find_wav_files(person_name: str, source: str = "processed") -> list[str]:
    """Find all WAV files for a person in the specified source directory."""
    person_dir = os.path.join(_DATA_DIR, source, person_name.lower())
    if not os.path.isdir(person_dir):
        print(f"ERROR: Directory not found: {person_dir}")
        sys.exit(1)

    files = sorted([
        os.path.join(person_dir, f)
        for f in os.listdir(person_dir)
        if f.lower().endswith(".wav")
    ])

    if not files:
        print(f"ERROR: No WAV files in {person_dir}")
        sys.exit(1)

    return files


# ---------------------------------------------------------------------------
# Discover all people in the data directory
# ---------------------------------------------------------------------------
def discover_people(source: str = "processed") -> list[str]:
    """Return list of person names found in the data source directory."""
    source_dir = os.path.join(_DATA_DIR, source)
    if not os.path.isdir(source_dir):
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    people = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ])
    return people


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute speaker embeddings from pre-recorded voice data."
    )
    parser.add_argument(
        "--person", type=str, default=None,
        help="Enroll only this person (default: all people in data/).",
    )
    parser.add_argument(
        "--source", type=str, default="processed",
        choices=["raw", "processed", "augmented"],
        help="Which data folder to use (default: processed).",
    )
    parser.add_argument(
        "--max-files", type=int, default=0,
        help="Max WAV files per person (0=all, default: all).",
    )
    parser.add_argument(
        "--set-master", type=str, default=None,
        help="Set this person as the master speaker.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Print cross-similarity matrix between enrolled speakers.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List enrolled speakers and exit.",
    )
    args = parser.parse_args()

    # Header
    print("\n" + "=" * 55)
    print("  Speaker Enrollment from Data — ElderGuard")
    print("=" * 55)
    print(f"  Model:      {sim_config.SPEAKER_MODEL_PATH}")
    print(f"  Models dir: {_MODELS_DIR}")
    print(f"  Data dir:   {_DATA_DIR}")
    print(f"  Source:     {args.source}")

    # Quick actions
    if args.list:
        list_enrolled()
        return

    if args.verify:
        verify_embeddings()
        return

    # Just set master (no enrollment)
    if args.set_master and not args.person:
        set_master(args.set_master)
        return

    # Load verifier (needs the real ECAPA model)
    model_size = os.path.getsize(sim_config.SPEAKER_MODEL_PATH) / 1024
    print(f"  Model size: {model_size:.0f} KB")
    if model_size < 500:
        print("\n  WARNING: Model appears to be the 328 KB stub!")
        print("  Run: python export_ecapa_model.py  (to get the real 25 MB model)")
        print("  Proceeding anyway (embeddings will be low quality)...\n")

    verifier = SpeakerVerifier()

    # Determine who to enroll
    if args.person:
        people = [args.person.lower()]
    else:
        people = discover_people(args.source)
        print(f"  Found people: {', '.join(people)}")

    # Enroll each person
    for person in people:
        wav_files = find_wav_files(person, args.source)
        print(f"\n  {person}: {len(wav_files)} WAV files in {args.source}/")

        ok = enroll_from_files(verifier, person, wav_files, args.max_files)
        if not ok:
            print(f"\n  FAILED for {person}")
            sys.exit(1)

    # Set master if requested
    if args.set_master:
        set_master(args.set_master)

    # Print cross-similarity
    print()
    verify_embeddings()

    print("\nDone! Embeddings saved to:", _MODELS_DIR)
    print("\nNext steps:")
    print("  # Set master for simulation:")
    print("  python enroll_from_data.py --set-master david")
    print()
    print("  # Run simulation with verification:")
    print("  python run_simulation.py")
    print()
    print("  # For RPi deployment, copy the .npy files:")
    print(f"  scp {_MODELS_DIR}/*_embedding.npy pi@<RPI_IP>:~/rpi_brain/models/")
    print(f"  scp {_MODELS_DIR}/master_embedding.npy pi@<RPI_IP>:~/rpi_brain/models/")


if __name__ == "__main__":
    main()
