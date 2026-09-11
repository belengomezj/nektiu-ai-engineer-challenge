"""Small latency logger used without an external observability service."""

import logging
import time
from contextlib import contextmanager

LOG = logging.getLogger("nektibot")


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def log_duration(stage: str, duration_ms: float) -> None:
    LOG.info("timing stage=%s duration_ms=%.2f", stage, duration_ms)


@contextmanager
def measure(stage: str):
    """Measure one operation without logging inputs or document content."""
    started = time.perf_counter()
    try:
        yield
    finally:
        log_duration(stage, elapsed_ms(started))
