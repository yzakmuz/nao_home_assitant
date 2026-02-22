"""
parser.py — Command intent parser.

Since we use a grammar-constrained recognizer, the input text is
already from a small vocabulary. This parser maps recognized phrases
to structured `Intent` objects that the main state machine dispatches.

This is intentionally simple — no NLP model needed when Vosk grammar
constrains the output to a known set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional

from settings import WAKE_WORD, YOLO_TARGET_CLASSES

log = logging.getLogger(__name__)


class IntentType(Enum):
    """Exhaustive set of intents the robot can handle."""
    FOLLOW_ME = auto()
    STOP = auto()
    FIND_OBJECT = auto()
    WAVE = auto()
    SAY_HELLO = auto()
    SIT_DOWN = auto()
    STAND_UP = auto()
    WHAT_DO_YOU_SEE = auto()
    LOOK_LEFT = auto()
    LOOK_RIGHT = auto()
    TURN_AROUND = auto()
    COME_HERE = auto()
    INTRODUCE = auto()
    DANCE = auto()
    UNKNOWN = auto()


@dataclass
class Intent:
    """Parsed command intent with optional parameters."""
    type: IntentType
    raw_text: str
    params: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Mapping table: phrase → (IntentType, params_extractor)
# ------------------------------------------------------------------

def _extract_find_target(text: str) -> Dict[str, Any]:
    """Extract the object name from 'find my <object>' phrases."""
    # Try each known target alias
    for friendly_name, coco_classes in YOLO_TARGET_CLASSES.items():
        if friendly_name in text:
            return {
                "target_name": friendly_name,
                "coco_classes": coco_classes,
            }
    # Fallback: look for any word after "find"
    parts = text.split("find")
    if len(parts) > 1:
        remainder = parts[1].strip().replace("my ", "")
        return {
            "target_name": remainder or "object",
            "coco_classes": [],  # empty = detect everything
        }
    return {"target_name": "object", "coco_classes": []}


# Static phrase → intent mappings
_PHRASE_MAP = {
    "follow me":            (IntentType.FOLLOW_ME, None),
    "stop":                 (IntentType.STOP, None),
    "wave hello":           (IntentType.WAVE, None),
    "say hello":            (IntentType.SAY_HELLO, None),
    "sit down":             (IntentType.SIT_DOWN, None),
    "stand up":             (IntentType.STAND_UP, None),
    "what do you see":      (IntentType.WHAT_DO_YOU_SEE, None),
    "look left":            (IntentType.LOOK_LEFT, None),
    "look right":           (IntentType.LOOK_RIGHT, None),
    "turn around":          (IntentType.TURN_AROUND, None),
    "come here":            (IntentType.COME_HERE, None),
    "introduce yourself":   (IntentType.INTRODUCE, None),
    "dance":                (IntentType.DANCE, None),
}


def parse_command(raw_text: str) -> Intent:
    """
    Parse a recognized phrase into a structured Intent.

    The input has already been lowercased by the STT engine.
    """
    # Strip wake word if the user said it again in the command
    text = raw_text.replace(WAKE_WORD, "").strip()

    if not text:
        return Intent(type=IntentType.UNKNOWN, raw_text=raw_text)

    # Check exact matches first
    for phrase, (intent_type, _) in _PHRASE_MAP.items():
        if phrase in text:
            log.info("Parsed intent: %s (from '%s')", intent_type.name, text)
            return Intent(type=intent_type, raw_text=raw_text)

    # Check "find" pattern
    if "find" in text:
        params = _extract_find_target(text)
        log.info(
            "Parsed intent: FIND_OBJECT target='%s' (from '%s')",
            params.get("target_name"), text,
        )
        return Intent(type=IntentType.FIND_OBJECT, raw_text=raw_text, params=params)

    log.info("Unknown command: '%s'", text)
    return Intent(type=IntentType.UNKNOWN, raw_text=raw_text)
