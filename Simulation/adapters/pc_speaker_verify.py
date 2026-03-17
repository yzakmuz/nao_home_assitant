"""
pc_speaker_verify.py -- SpeechBrain-based speaker verifier for PC simulation.

On the real RPi, speaker verification uses ONNX Runtime with a pre-exported
ECAPA-TDNN model. On PC, we use SpeechBrain directly (with PyTorch) for
higher quality embeddings and to avoid ONNX export compatibility issues.

The interface matches the original SpeakerVerifier exactly, so it's a
drop-in replacement via monkey-patching in bootstrap.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch

from settings import (
    MASTER_EMBEDDING_PATH,
    MIC_SAMPLE_RATE,
    MULTI_SPEAKER_MODE,
    SPEAKER_MIN_AUDIO_S,
    SPEAKER_VERIFY_THRESHOLD,
)

log = logging.getLogger(__name__)

# Module-level classifier (loaded once, shared across instances)
_classifier = None
_classifier_lock = None


def _get_classifier():
    """Lazy-load the SpeechBrain ECAPA-TDNN classifier."""
    global _classifier
    if _classifier is not None:
        return _classifier

    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    # Fix Windows symlink issue
    os.environ.setdefault("SB_FETCH_STRATEGY", "copy")
    import speechbrain.utils.fetching as sb_fetch
    import shutil
    from pathlib import Path

    _orig_link = sb_fetch.link_with_strategy
    def _copy_strategy(src, dst, *args, **kwargs):
        dst = Path(dst)
        src = Path(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(str(src), str(dst))
        return dst
    sb_fetch.link_with_strategy = _copy_strategy

    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    savedir = os.path.join(os.path.expanduser("~"), "sb_ecapa")
    log.info("Loading SpeechBrain ECAPA-TDNN (first time may download ~100MB)...")
    _classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
    )
    log.info("SpeechBrain ECAPA-TDNN loaded.")
    return _classifier


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized 1-D vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        return 0.0
    return float(dot / norm)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


class PcSpeakerVerifier:
    """SpeechBrain-based speaker verifier (PC simulation replacement)."""

    def __init__(self) -> None:
        self._classifier = _get_classifier()

        # Load master embedding if available
        self._master_emb: Optional[np.ndarray] = None
        if os.path.isfile(MASTER_EMBEDDING_PATH):
            self._master_emb = np.load(MASTER_EMBEDDING_PATH)
            log.info("Master embedding loaded (dim=%d).", self._master_emb.shape[-1])
        else:
            log.warning("No master embedding at '%s'. Run enrollment first.",
                        MASTER_EMBEDDING_PATH)

        # Multi-speaker: load all enrolled speaker embeddings
        self._enrolled: dict[str, np.ndarray] = {}
        if MULTI_SPEAKER_MODE:
            self._load_enrolled_speakers()

    def compute_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Compute speaker embedding from float32 mono waveform at 16kHz."""
        min_samples = int(SPEAKER_MIN_AUDIO_S * MIC_SAMPLE_RATE)
        if audio.shape[0] < min_samples:
            log.warning("Audio too short: %.2fs (need %.2fs).",
                        audio.shape[0] / MIC_SAMPLE_RATE, SPEAKER_MIN_AUDIO_S)
            return None

        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            emb = self._classifier.encode_batch(waveform)
            emb = emb.squeeze(0).squeeze(0).numpy()

        return _l2_normalize(emb)

    def verify(self, audio: np.ndarray) -> tuple[bool, float]:
        """Verify whether audio belongs to the enrolled master."""
        if self._master_emb is None:
            log.error("Cannot verify — no master embedding enrolled.")
            return False, 0.0

        emb = self.compute_embedding(audio)
        if emb is None:
            return False, 0.0

        if emb.shape[0] != self._master_emb.shape[0]:
            log.error("Dimension mismatch: %d vs %d. Re-run enrollment!",
                      emb.shape[0], self._master_emb.shape[0])
            return False, 0.0

        score = _cosine_similarity(emb, self._master_emb)
        is_master = score >= SPEAKER_VERIFY_THRESHOLD
        log.info("Speaker verify: score=%.3f threshold=%.3f -> %s",
                 score, SPEAKER_VERIFY_THRESHOLD,
                 "ACCEPTED" if is_master else "REJECTED")
        return is_master, score

    def enroll(self, audio: np.ndarray) -> bool:
        """Enroll master speaker from audio."""
        emb = self.compute_embedding(audio)
        if emb is None:
            return False
        os.makedirs(os.path.dirname(MASTER_EMBEDDING_PATH) or ".", exist_ok=True)
        np.save(MASTER_EMBEDDING_PATH, emb)
        self._master_emb = emb
        log.info("Master enrolled (dim=%d).", emb.shape[0])
        return True

    # ------------------------------------------------------------------
    # Multi-speaker support
    # ------------------------------------------------------------------

    def _load_enrolled_speakers(self) -> None:
        """Scan the models directory for *_embedding.npy files and load them."""
        models_dir = os.path.dirname(MASTER_EMBEDDING_PATH) or "."
        self._enrolled.clear()

        if not os.path.isdir(models_dir):
            log.warning("Models directory not found: %s", models_dir)
            return

        for fname in sorted(os.listdir(models_dir)):
            if not fname.endswith("_embedding.npy"):
                continue
            if fname == "master_embedding.npy":
                continue
            name = fname.replace("_embedding.npy", "")
            fpath = os.path.join(models_dir, fname)
            try:
                emb = np.load(fpath)
                self._enrolled[name] = emb
                log.info("Enrolled speaker loaded: %s (dim=%d)", name, emb.shape[-1])
            except Exception:
                log.exception("Failed to load embedding: %s", fpath)

        if not self._enrolled:
            log.warning("No enrolled speaker embeddings found in %s", models_dir)

    def reload_speakers(self) -> None:
        """Re-scan the models directory for enrolled speakers."""
        self._load_enrolled_speakers()
        if os.path.isfile(MASTER_EMBEDDING_PATH):
            self._master_emb = np.load(MASTER_EMBEDDING_PATH)

    def verify_multi(self, audio: np.ndarray) -> tuple[bool, str, float]:
        """Verify against ALL enrolled speakers.

        Returns:
            (accepted, speaker_name, best_score)
        """
        if not self._enrolled:
            accepted, score = self.verify(audio)
            return accepted, "master" if accepted else "unknown", score

        emb = self.compute_embedding(audio)
        if emb is None:
            return False, "unknown", 0.0

        best_name = "unknown"
        best_score = -1.0

        for name, enrolled_emb in self._enrolled.items():
            if emb.shape[0] != enrolled_emb.shape[0]:
                log.warning(
                    "Dimension mismatch for '%s': model=%d, enrolled=%d — skipping.",
                    name, emb.shape[0], enrolled_emb.shape[0],
                )
                continue
            score = _cosine_similarity(emb, enrolled_emb)
            if score > best_score:
                best_score = score
                best_name = name

        accepted = best_score >= SPEAKER_VERIFY_THRESHOLD
        log.info(
            "Speaker verify (multi): best=%s score=%.3f threshold=%.3f -> %s",
            best_name, best_score, SPEAKER_VERIFY_THRESHOLD,
            "ACCEPTED" if accepted else "REJECTED",
        )
        return accepted, best_name if accepted else "unknown", best_score

    def enroll_speaker(self, name: str, audio: np.ndarray) -> bool:
        """Enroll a named speaker from audio."""
        emb = self.compute_embedding(audio)
        if emb is None:
            log.error("Enrollment failed for '%s' -- audio too short.", name)
            return False

        models_dir = os.path.dirname(MASTER_EMBEDDING_PATH) or "."
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, f"{name.lower()}_embedding.npy")
        np.save(path, emb)
        self._enrolled[name.lower()] = emb
        log.info("Speaker '%s' enrolled (dim=%d) -> %s", name, emb.shape[0], path)
        return True

    def list_enrolled(self) -> list[str]:
        """Return names of all enrolled speakers."""
        return sorted(self._enrolled.keys())
