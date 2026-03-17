"""
speaker_verify.py -- ECAPA-TDNN speaker verification via ONNX Runtime.

Workflow:
    1. Load a pre-exported ONNX model of ECAPA-TDNN (exported from
       SpeechBrain via tools/export_ecapa_onnx.py).
    2. On enrollment, compute and save the master's embedding to disk.
    3. At runtime, compute an embedding from a new utterance and compare
       it to the stored master embedding via cosine similarity.

Audio expectations:
    - Input: float32 numpy array, mono, 16 kHz, normalized to [-1, 1].
    - Minimum duration: SPEAKER_MIN_AUDIO_S (default 1.0 s).
    - Embeddings are L2-normalized before comparison.

Model details (real ECAPA-TDNN):
    - Input:  "waveform" -- (batch, samples) float32
    - Output: "embedding" -- (batch, 192) float32, L2-normalized
    - Size:   ~25 MB
    - Source:  speechbrain/spkrec-ecapa-voxceleb (HuggingFace)

Memory note:
    The ONNX model is kept resident (~20 MB) because verification runs
    on every accepted command -- too frequent to justify dynamic loading.

IMPORTANT: After replacing the stub model with the real ECAPA-TDNN,
you MUST re-run enrollment (python enroll_speaker.py). The old
master_embedding.npy is incompatible with the new model.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from settings import (
    MASTER_EMBEDDING_PATH,
    MIC_SAMPLE_RATE,
    MULTI_SPEAKER_MODE,
    SPEAKER_MIN_AUDIO_S,
    SPEAKER_MODEL_PATH,
    SPEAKER_VERIFY_THRESHOLD,
)

log = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized 1-D vectors.

    For L2-normalized vectors, cosine similarity equals the dot product.
    We compute the full formula for safety (handles un-normalized inputs).
    """
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        return 0.0
    return float(dot / norm)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector. Returns zero vector if norm is near zero."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


class SpeakerVerifier:
    """Lightweight ONNX-based speaker embedding & cosine verification."""

    def __init__(self) -> None:
        import onnxruntime as ort

        log.info("Loading speaker model: %s", SPEAKER_MODEL_PATH)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            SPEAKER_MODEL_PATH,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        # Load master embedding if available
        self._master_emb: Optional[np.ndarray] = None
        if os.path.isfile(MASTER_EMBEDDING_PATH):
            self._master_emb = np.load(MASTER_EMBEDDING_PATH)
            log.info(
                "Master embedding loaded (dim=%d).", self._master_emb.shape[-1]
            )
        else:
            log.warning(
                "No master embedding at '%s'. Run enrollment first.",
                MASTER_EMBEDDING_PATH,
            )

        # Multi-speaker: load all enrolled speaker embeddings
        self._enrolled: dict[str, np.ndarray] = {}
        if MULTI_SPEAKER_MODE:
            self._load_enrolled_speakers()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def compute_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute a speaker embedding from a float32 mono waveform.

        Returns a 1-D L2-normalized numpy array (the embedding) or None
        if the audio is too short.
        """
        min_samples = int(SPEAKER_MIN_AUDIO_S * MIC_SAMPLE_RATE)
        if audio.shape[0] < min_samples:
            log.warning(
                "Audio too short for verification: %.2f s (need %.2f s).",
                audio.shape[0] / MIC_SAMPLE_RATE,
                SPEAKER_MIN_AUDIO_S,
            )
            return None

        # Model expects (batch, samples) -- float32
        waveform = audio.astype(np.float32).reshape(1, -1)
        outputs = self._session.run(None, {self._input_name: waveform})
        embedding = outputs[0].flatten().astype(np.float32)

        # L2-normalize the embedding for consistent cosine similarity
        embedding = _l2_normalize(embedding)
        return embedding

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self, audio: np.ndarray) -> tuple[bool, float]:
        """
        Verify whether *audio* belongs to the enrolled master.

        Returns:
            (is_master, similarity_score)
        """
        if self._master_emb is None:
            log.error("Cannot verify -- no master embedding enrolled.")
            return False, 0.0

        emb = self.compute_embedding(audio)
        if emb is None:
            return False, 0.0

        # Check dimension mismatch (e.g., stub model vs real model)
        if emb.shape[0] != self._master_emb.shape[0]:
            log.error(
                "Embedding dimension mismatch: model produces %d-dim but "
                "master embedding is %d-dim. Re-run enrollment!",
                emb.shape[0], self._master_emb.shape[0],
            )
            return False, 0.0

        score = _cosine_similarity(emb, self._master_emb)
        is_master = score >= SPEAKER_VERIFY_THRESHOLD
        log.info(
            "Speaker verify: score=%.3f threshold=%.3f -> %s",
            score,
            SPEAKER_VERIFY_THRESHOLD,
            "ACCEPTED" if is_master else "REJECTED",
        )
        return is_master, score

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll(self, audio: np.ndarray) -> bool:
        """
        Compute and persist the master's embedding from a sample utterance.
        Call this once during setup with a clean recording of the owner.

        For better accuracy, use enroll_speaker.py which records multiple
        samples and averages the embeddings.
        """
        emb = self.compute_embedding(audio)
        if emb is None:
            log.error("Enrollment failed -- audio too short.")
            return False

        os.makedirs(os.path.dirname(MASTER_EMBEDDING_PATH) or ".", exist_ok=True)
        np.save(MASTER_EMBEDDING_PATH, emb)
        self._master_emb = emb
        log.info(
            "Master enrolled -- embedding saved to '%s' (dim=%d).",
            MASTER_EMBEDDING_PATH,
            emb.shape[0],
        )
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
        """Re-scan the models directory for enrolled speakers.

        Call this after adding a new person's embedding at runtime.
        """
        self._load_enrolled_speakers()
        # Also reload master in case it changed
        if os.path.isfile(MASTER_EMBEDDING_PATH):
            self._master_emb = np.load(MASTER_EMBEDDING_PATH)

    def verify_multi(self, audio: np.ndarray) -> tuple[bool, str, float]:
        """Verify against ALL enrolled speakers.

        Returns:
            (accepted, speaker_name, best_score)

        If no enrolled speakers are loaded, falls back to master-only
        verification (returns speaker_name="master").
        """
        if not self._enrolled:
            # Fallback to master-only
            accepted, score = self.verify(audio)
            return accepted, "master" if accepted else "unknown", score

        emb = self.compute_embedding(audio)
        if emb is None:
            return False, "unknown", 0.0

        best_name = "unknown"
        best_score = -1.0

        for name, enrolled_emb in self._enrolled.items():
            # Skip dimension mismatches (e.g., stub vs real model)
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
        """Enroll a named speaker from audio. Saves {name}_embedding.npy.

        The embedding is added to the in-memory enrolled dict immediately.
        """
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
