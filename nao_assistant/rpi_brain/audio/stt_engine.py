"""
stt_engine.py — Grammar-constrained offline speech-to-text with Vosk.

The engine exposes a simple blocking interface:
    result = stt.listen_once(mic, timeout=5.0)
which drains audio from the MicStream until Vosk returns a final result,
a silence timeout expires, or the caller's deadline is hit.

Grammar mode restricts the recognizer to a predefined phrase list,
dramatically improving accuracy on a small vocabulary.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from vosk import KaldiRecognizer, Model, SetLogLevel

from settings import (
    GRAMMAR_PHRASES,
    LISTEN_TIMEOUT_S,
    MIC_SAMPLE_RATE,
    VOSK_MODEL_PATH,
    WAKE_WORD,
)

log = logging.getLogger(__name__)

# Suppress Vosk's own verbose logging
SetLogLevel(-1)


class SttEngine:
    """Vosk-based grammar STT — small footprint, offline, Pi-friendly."""

    def __init__(self) -> None:
        log.info("Loading Vosk model from '%s' …", VOSK_MODEL_PATH)
        self._model = Model(VOSK_MODEL_PATH)

        # Build grammar JSON string
        self._grammar = json.dumps(GRAMMAR_PHRASES)
        log.info("Grammar loaded: %d phrases.", len(GRAMMAR_PHRASES))

    def _new_recognizer(self) -> KaldiRecognizer:
        """Create a fresh recognizer with the grammar vocabulary."""
        rec = KaldiRecognizer(self._model, MIC_SAMPLE_RATE, self._grammar)
        rec.SetWords(True)
        return rec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wait_for_wake_word(self, mic) -> str:
        """
        Block until the wake word is detected (in a final result).
        Returns the full final text that contained the wake word.
        """
        rec = self._new_recognizer()
        log.info("Listening for wake word '%s' …", WAKE_WORD)

        while True:
            chunk = mic.read(timeout=1.0)
            if chunk is None:
                continue

            if rec.AcceptWaveform(chunk):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip().lower()
                if WAKE_WORD in text:
                    log.info("Wake word detected (final): '%s'", text)
                    return text  # ✅ Final result — safe to return
                else:
                    # Final result but no wake word — reset and continue listening
                    log.debug("Got final result but no wake word: '%s'", text)
                    rec.Reset()
            else:
                # Partial result — check but don't return
                partial = json.loads(rec.PartialResult())
                ptext = partial.get("partial", "").strip().lower()
                if WAKE_WORD in ptext:
                    log.debug("Wake word (partial): '%s' — continuing to listen...", ptext)
                    # ✅ Keep recording; don't return or reset yet!
                    # This allows the user to continue speaking their command

    def listen_command(
        self,
        mic,
        timeout: float = LISTEN_TIMEOUT_S,
    ) -> Optional[str]:
        """
        After wake word, listen for a single command utterance.

        Returns the recognized phrase (str) or None on timeout / empty.
        """
        rec = self._new_recognizer()
        deadline = time.monotonic() + timeout
        log.info("Listening for command (%.1f s timeout) …", timeout)

        last_partial = ""
        silence_start: Optional[float] = None
        SILENCE_THRESHOLD_S = 1.5  # finalize after 1.5 s of no new partials

        while time.monotonic() < deadline:
            chunk = mic.read(timeout=0.3)
            if chunk is None:
                continue

            if rec.AcceptWaveform(chunk):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip().lower()
                if text and text != "[unk]":
                    log.info("Command recognized: '%s'", text)
                    return text
            else:
                partial = json.loads(rec.PartialResult())
                ptext = partial.get("partial", "").strip().lower()
                if ptext and ptext != last_partial:
                    last_partial = ptext
                    silence_start = None
                elif ptext and silence_start is None:
                    silence_start = time.monotonic()
                elif silence_start and (time.monotonic() - silence_start > SILENCE_THRESHOLD_S):
                    # Force finalization
                    final = json.loads(rec.FinalResult())
                    text = final.get("text", "").strip().lower()
                    if text and text != "[unk]":
                        log.info("Command (silence-finalized): '%s'", text)
                        return text
                    break

        # Last chance — finalize whatever is buffered
        final = json.loads(rec.FinalResult())
        text = final.get("text", "").strip().lower()
        if text and text != "[unk]":
            log.info("Command (deadline-finalized): '%s'", text)
            return text

        log.info("No command recognized within timeout.")
        return None