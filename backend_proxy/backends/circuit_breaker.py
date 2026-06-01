"""Per-backend circuit breaker.

States:
    closed   -> normal: all requests go through.
    open     -> short-circuited: pick() skips this backend until cooldown elapses.
    half-open-> next request is allowed; success closes, failure re-opens.

This is intentionally minimal — no exponential backoff, no rolling window.
Tuned for "vLLM crashed, drain to a sibling for a minute" rather than
full-blown service-mesh resiliency.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass


class State(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    failures_threshold: int = 5
    cooldown_s: float = 30.0

    state: State = State.CLOSED
    _failures: int = 0
    _opened_at: float = 0.0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    # ---- queries ----------------------------------------------------
    def is_available(self) -> bool:
        with self._lock:
            if self.state is State.CLOSED:
                return True
            if self.state is State.OPEN:
                if time.monotonic() - self._opened_at >= self.cooldown_s:
                    self.state = State.HALF_OPEN
                    return True
                return False
            # HALF_OPEN: allow exactly one probe
            return True

    # ---- callbacks --------------------------------------------------
    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self.state = State.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self.state is State.HALF_OPEN or self._failures >= self.failures_threshold:
                self.state = State.OPEN
                self._opened_at = time.monotonic()
