"""
memory.py — RAM monitoring and garbage-collection utilities.

On a 2 GB Raspberry Pi running multiple AI models, memory pressure is
a first-class concern. This module provides:

    - Real-time RSS / available-RAM checks.
    - Logging warnings when thresholds are crossed.
    - A `force_gc()` helper that runs a full collection cycle.
    - A decorator `@guard_memory` that logs before/after RSS for
      any function expected to allocate heavily.
"""

from __future__ import annotations

import functools
import gc
import logging
import os
from typing import Callable, TypeVar

import psutil

from settings import RAM_CRITICAL_THRESHOLD_MB, RAM_WARNING_THRESHOLD_MB

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)

_process = psutil.Process(os.getpid())


# ------------------------------------------------------------------
# Queries
# ------------------------------------------------------------------

def rss_mb() -> float:
    """Return current process RSS in megabytes."""
    return _process.memory_info().rss / (1024 * 1024)


def available_mb() -> float:
    """Return system-wide available memory in megabytes."""
    return psutil.virtual_memory().available / (1024 * 1024)


def log_memory(label: str = "") -> None:
    """Log current memory stats at INFO level."""
    rss = rss_mb()
    avail = available_mb()
    prefix = f"[{label}] " if label else ""
    log.info("%sRSS=%.0f MB | Available=%.0f MB", prefix, rss, avail)


def check_pressure() -> str:
    """
    Check memory pressure. Returns:
        "ok"       — plenty of headroom
        "warning"  — getting tight
        "critical" — dangerously low
    """
    avail = available_mb()
    if avail < RAM_CRITICAL_THRESHOLD_MB:
        log.warning("CRITICAL memory: %.0f MB available!", avail)
        return "critical"
    if avail < RAM_WARNING_THRESHOLD_MB:
        log.warning("Low memory warning: %.0f MB available.", avail)
        return "warning"
    return "ok"


# ------------------------------------------------------------------
# Actions
# ------------------------------------------------------------------

def force_gc() -> int:
    """Run full garbage collection. Returns number of objects freed."""
    gc.collect()
    gc.collect()  # second pass catches reference cycles
    freed = gc.collect()
    log.debug("GC collected %d objects.", freed)
    return freed


def emergency_cleanup() -> None:
    """
    Last-resort cleanup when memory is critical.
    Runs aggressive GC.
    """
    log.warning("Emergency memory cleanup triggered.")
    force_gc()
    log_memory("post-emergency")


# ------------------------------------------------------------------
# Decorator
# ------------------------------------------------------------------

def guard_memory(label: str = "") -> Callable[[F], F]:
    """
    Decorator that logs RSS before and after a function call,
    and runs GC after heavy allocators.

    Usage:
        @guard_memory("yolo_detect")
        def detect(self, frame):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tag = label or func.__qualname__
            rss_before = rss_mb()
            log.debug("[%s] RSS before: %.0f MB", tag, rss_before)

            result = func(*args, **kwargs)

            rss_after = rss_mb()
            delta = rss_after - rss_before
            if abs(delta) > 10:
                log.info(
                    "[%s] RSS: %.0f → %.0f MB (Δ%+.0f)",
                    tag, rss_before, rss_after, delta,
                )
            return result

        return wrapper  # type: ignore[return-value]
    return decorator
