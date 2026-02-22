"""
speaker_verify.py — ECAPA-TDNN speaker verification via ONNX Runtime.

Workflow:
    1. Load a pre-exported ONNX model of ECAPA-TDNN (or any speaker
       embedding network that takes variable-length waveforms).
    2. On enrollment, compute and save the master's embedding to disk.
    3. At runtime, compute an embedding from a new utterance and compare
       it to the stored master embedding via cosine similarity.

Audio expectations:
    - Input: float32 numpy array, mono, 16 kHz, normalized to [-1, 1].
    - Minimum duration: SPEAKER_MIN_AUDIO_S (default 1.0 s).

Memory note:
    The ONNX model is kept resident (~20 MB) because verification runs
    on every accepted command — too frequent to justify dynamic loading.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from settings import (
    MASTER_EMBEDDING_PATH,
    MIC_SAMPLE_RATE,
    SPEAKER_MIN_AUDIO_S,
    SPEAKER_MODEL_PATH,
    SPEAKER_VERIFY_THRESHOLD,
)

log = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        return 0.0
    return float(dot / norm)


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

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def compute_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute a speaker embedding from a float32 mono waveform.

        Returns a 1-D numpy array (the embedding) or None if the audio
        is too short.
        """
        min_samples = int(SPEAKER_MIN_AUDIO_S * MIC_SAMPLE_RATE)
        if audio.shape[0] < min_samples:
            log.warning(
                "Audio too short for verification: %.2f s (need %.2f s).",
                audio.shape[0] / MIC_SAMPLE_RATE,
                SPEAKER_MIN_AUDIO_S,
            )
            return None

        # Model expects (batch, samples) — float32
        waveform = audio.astype(np.float32).reshape(1, -1)
        outputs = self._session.run(None, {self._input_name: waveform})
        embedding = outputs[0].flatten().astype(np.float32)
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
            log.error("Cannot verify — no master embedding enrolled.")
            return False, 0.0

        emb = self.compute_embedding(audio)
        if emb is None:
            return False, 0.0

        score = _cosine_similarity(emb, self._master_emb)
        is_master = score >= SPEAKER_VERIFY_THRESHOLD
        log.info(
            "Speaker verify: score=%.3f threshold=%.3f → %s",
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
        """
        emb = self.compute_embedding(audio)
        if emb is None:
            log.error("Enrollment failed — audio too short.")
            return False

        os.makedirs(os.path.dirname(MASTER_EMBEDDING_PATH) or ".", exist_ok=True)
        np.save(MASTER_EMBEDDING_PATH, emb)
        self._master_emb = emb
        log.info(
            "Master enrolled — embedding saved to '%s' (dim=%d).",
            MASTER_EMBEDDING_PATH,
            emb.shape[0],
        )
        return True
