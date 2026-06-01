"""Routing strategies. Pluggable: name -> Strategy class. The pool consults
the active strategy when picking a backend; if the strategy yields nothing
useful (e.g. nothing matches the requested model) the fallback strategy is
consulted with the same set of healthy backends."""

from __future__ import annotations

import itertools
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

from .base import Backend
from .openai_compat import OpenAICompatBackend


class Strategy(ABC):
    name: str

    @abstractmethod
    def pick(self, candidates: Sequence[Backend], *, model: Optional[str]) -> Optional[Backend]: ...


class RoundRobin(Strategy):
    name = "round_robin"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cursor = 0

    def pick(self, candidates: Sequence[Backend], *, model: Optional[str]) -> Optional[Backend]:
        if not candidates:
            return None
        with self._lock:
            i = self._cursor % len(candidates)
            self._cursor += 1
        return candidates[i]


class LeastBusy(Strategy):
    """Pick the backend with the fewest in-flight requests; ties broken by RR."""
    name = "least_busy"

    def __init__(self) -> None:
        self._rr = RoundRobin()

    def pick(self, candidates: Sequence[Backend], *, model: Optional[str]) -> Optional[Backend]:
        if not candidates:
            return None
        # _inflight counter is set on the VLLMBackend instance by the pool
        def busy(b: Backend) -> int:
            return getattr(b, "_inflight", 0)
        min_busy = min(busy(b) for b in candidates)
        winners = [b for b in candidates if busy(b) == min_busy]
        return self._rr.pick(winners, model=model)


class ModelAffinity(Strategy):
    """Route to the backend(s) that explicitly declare the requested model;
    when several match, fall through to round-robin among them."""
    name = "model_affinity"

    def __init__(self) -> None:
        self._rr = RoundRobin()

    def pick(self, candidates: Sequence[Backend], *, model: Optional[str]) -> Optional[Backend]:
        if not candidates:
            return None
        if not model:
            return None  # let the fallback strategy handle it
        matches = [
            b for b in candidates
            if isinstance(b, OpenAICompatBackend) and model in (b.cfg.models or [])
        ]
        if not matches:
            return None
        return self._rr.pick(matches, model=model)


class Sticky(Strategy):
    """Hash on a request key (e.g. user id, conversation id) for stickiness.
    Today the key isn't plumbed through context — falls back to RR. Hook is
    in place so it can be wired without touching the pool."""
    name = "sticky"

    def __init__(self) -> None:
        self._rr = RoundRobin()

    def pick(self, candidates: Sequence[Backend], *, model: Optional[str]) -> Optional[Backend]:
        return self._rr.pick(candidates, model=model)


STRATEGIES: Dict[str, type[Strategy]] = {
    s.name: s for s in (RoundRobin, LeastBusy, ModelAffinity, Sticky)
}


def make_strategy(name: str) -> Strategy:
    if name == "none":
        # treat as "never picks anything"
        class _NoneStrategy(Strategy):
            name = "none"
            def pick(self, candidates, *, model): return None
        return _NoneStrategy()
    cls = STRATEGIES.get(name)
    if cls is None:
        raise ValueError(f"unknown routing strategy: {name}")
    return cls()
