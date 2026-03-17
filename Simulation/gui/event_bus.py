"""
event_bus.py -- Thread-safe event bus for the simulation demo console.

Provides a centralized, thread-safe event stream that all simulation
components push to.  The GUI console panel reads from this bus.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
from typing import List, Optional


# ======================================================================
# Event Categories
# ======================================================================
CAT_STT = "STT"
CAT_VERIFY = "VERIFY"
CAT_FSM = "FSM"
CAT_CMD = "CMD"
CAT_NAO = "NAO"
CAT_SYS = "SYS"

# ======================================================================
# Severity Levels
# ======================================================================
SEV_INFO = "info"
SEV_COMMAND = "command"
SEV_RESPONSE = "response"
SEV_WARNING = "warning"
SEV_ERROR = "error"


class SimEvent:
    """A single event in the simulation event stream."""

    __slots__ = ("timestamp", "category", "message", "severity")

    def __init__(self, category: str, message: str,
                 severity: str = SEV_INFO) -> None:
        now = datetime.now()
        self.timestamp = (now.strftime("%H:%M:%S.")
                          + f"{now.microsecond // 1000:03d}")
        self.category = category
        self.message = message
        self.severity = severity


class SimEventBus:
    """Thread-safe event bus backed by a bounded deque."""

    # Actions to skip (high-frequency, not useful in console)
    _SKIP_ACTIONS = frozenset({
        "move_head", "move_head_relative", "heartbeat",
    })

    def __init__(self, max_events: int = 300) -> None:
        self._events: deque = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def emit(self, category: str, message: str,
             severity: str = SEV_INFO) -> None:
        """Push a new event onto the bus."""
        event = SimEvent(category, message, severity)
        with self._lock:
            self._events.append(event)

    def recent(self, count: int = 20) -> List[SimEvent]:
        """Return the *count* most recent events."""
        with self._lock:
            items = list(self._events)
        return items[-count:]

    @classmethod
    def should_skip_action(cls, action: str) -> bool:
        """Return True if this action should NOT be logged to the console."""
        return action in cls._SKIP_ACTIONS


# ======================================================================
# Global Singleton
# ======================================================================
_bus: Optional[SimEventBus] = None


def get_event_bus() -> SimEventBus:
    """Return (and lazily create) the global event bus singleton."""
    global _bus
    if _bus is None:
        _bus = SimEventBus()
    return _bus
