"""Unit tests for routing strategies and the circuit breaker."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import pytest

from backend_proxy.backends.circuit_breaker import CircuitBreaker, State
from backend_proxy.backends.strategies import (
    LeastBusy,
    ModelAffinity,
    RoundRobin,
    make_strategy,
)
from backend_proxy.config.schema import BackendConfig
from backend_proxy.backends.openai_compat import VLLMBackend


def _make_backend(name: str, models: Optional[List[str]] = None) -> VLLMBackend:
    return VLLMBackend(BackendConfig(name=name, base_url="http://example",
                                     models=models or []))


# ---- RoundRobin ------------------------------------------------------

def test_round_robin_cycles():
    s = RoundRobin()
    a, b, c = _make_backend("a"), _make_backend("b"), _make_backend("c")
    seq = [s.pick([a, b, c], model=None).name for _ in range(7)]
    assert seq == ["a", "b", "c", "a", "b", "c", "a"]


def test_round_robin_empty():
    assert RoundRobin().pick([], model=None) is None


# ---- LeastBusy -------------------------------------------------------

def test_least_busy_picks_lowest_inflight():
    a, b, c = _make_backend("a"), _make_backend("b"), _make_backend("c")
    a._inflight = 5
    b._inflight = 2
    c._inflight = 3
    assert LeastBusy().pick([a, b, c], model=None).name == "b"


def test_least_busy_ties_break_round_robin():
    a, b = _make_backend("a"), _make_backend("b")
    a._inflight = b._inflight = 0
    s = LeastBusy()
    seq = [s.pick([a, b], model=None).name for _ in range(4)]
    assert seq == ["a", "b", "a", "b"]


# ---- ModelAffinity ---------------------------------------------------

def test_model_affinity_routes_by_model():
    a = _make_backend("a", ["llama-3"])
    b = _make_backend("b", ["qwen"])
    s = ModelAffinity()
    assert s.pick([a, b], model="qwen").name == "b"
    assert s.pick([a, b], model="missing") is None


def test_model_affinity_round_robin_among_matches():
    a = _make_backend("a", ["m"])
    b = _make_backend("b", ["m"])
    s = ModelAffinity()
    picks = [s.pick([a, b], model="m").name for _ in range(4)]
    assert picks == ["a", "b", "a", "b"]


# ---- make_strategy ---------------------------------------------------

def test_make_strategy_unknown_raises():
    with pytest.raises(ValueError):
        make_strategy("nope")


def test_make_strategy_none_picks_nothing():
    s = make_strategy("none")
    a = _make_backend("a")
    assert s.pick([a], model=None) is None


# ---- CircuitBreaker --------------------------------------------------

def test_breaker_opens_after_threshold():
    cb = CircuitBreaker(failures_threshold=3, cooldown_s=10)
    for _ in range(2):
        cb.record_failure()
    assert cb.is_available() is True
    cb.record_failure()
    assert cb.state is State.OPEN
    assert cb.is_available() is False


def test_breaker_half_open_after_cooldown():
    cb = CircuitBreaker(failures_threshold=1, cooldown_s=0.05)
    cb.record_failure()
    assert cb.is_available() is False
    time.sleep(0.06)
    assert cb.is_available() is True   # transitions to HALF_OPEN
    assert cb.state is State.HALF_OPEN


def test_breaker_half_open_failure_reopens():
    cb = CircuitBreaker(failures_threshold=1, cooldown_s=0.01)
    cb.record_failure()
    time.sleep(0.02)
    cb.is_available()  # -> HALF_OPEN
    cb.record_failure()
    assert cb.state is State.OPEN


def test_breaker_half_open_success_closes():
    cb = CircuitBreaker(failures_threshold=1, cooldown_s=0.01)
    cb.record_failure()
    time.sleep(0.02)
    cb.is_available()  # -> HALF_OPEN
    cb.record_success()
    assert cb.state is State.CLOSED


# ---- HealthWatcher hysteresis ---------------------------------------
import asyncio  # noqa: E402

from backend_proxy.backends.health import HealthWatcher  # noqa: E402
from backend_proxy.config.schema import HealthCheckConfig  # noqa: E402


class _FakeBackend:
    """Minimal Backend stand-in for HealthWatcher unit tests."""
    def __init__(self, name: str) -> None:
        self.name = name
        self._scripted: List[bool] = []
        self.calls = 0

    def script(self, results: List[bool]) -> None:
        self._scripted = list(results)

    async def health(self) -> bool:
        self.calls += 1
        if self._scripted:
            return self._scripted.pop(0)
        return True

    # Some HealthWatcher code calls set_health_probe if present; provide it.
    def set_health_probe(self, *, timeout_s: float) -> None:  # noqa: ARG002
        pass


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def test_health_single_failure_does_not_flip_to_unhealthy():
    """The fix for the user-visible 503 bug: one transient probe failure
    must keep the backend marked healthy."""
    b = _FakeBackend("a")
    b.script([False])  # single probe fails
    breakers = {"a": CircuitBreaker()}
    hw = HealthWatcher([b], breakers, config=HealthCheckConfig(unhealthy_threshold=2))
    _run(hw._probe(b))
    assert hw.healthy["a"] is True   # still healthy after 1 failure


def test_health_two_consecutive_failures_flip_to_unhealthy():
    b = _FakeBackend("a")
    b.script([False, False])
    breakers = {"a": CircuitBreaker(failures_threshold=10)}
    hw = HealthWatcher([b], breakers, config=HealthCheckConfig(unhealthy_threshold=2))
    _run(hw._probe(b))
    _run(hw._probe(b))
    assert hw.healthy["a"] is False


def test_health_failure_then_success_stays_healthy():
    """A flap (fail, then ok, then ok) must NEVER flip to unhealthy."""
    b = _FakeBackend("a")
    b.script([False, True, True])
    breakers = {"a": CircuitBreaker()}
    hw = HealthWatcher([b], breakers, config=HealthCheckConfig(unhealthy_threshold=2))
    for _ in range(3):
        _run(hw._probe(b))
    assert hw.healthy["a"] is True


def test_health_recovers_after_one_success():
    b = _FakeBackend("a")
    b.script([False, False, True])
    breakers = {"a": CircuitBreaker(failures_threshold=10)}
    hw = HealthWatcher([b], breakers,
                      config=HealthCheckConfig(unhealthy_threshold=2, healthy_threshold=1))
    _run(hw._probe(b))
    _run(hw._probe(b))
    assert hw.healthy["a"] is False
    _run(hw._probe(b))
    assert hw.healthy["a"] is True   # one success is enough to recover


# ---- BackendPool probe-bypass safety net ----------------------------

from backend_proxy.backends.pool import BackendPool  # noqa: E402
from backend_proxy.config.schema import RoutingConfig  # noqa: E402


def test_pool_pick_bypasses_probe_when_breaker_closed():
    """If the probe says everyone is unhealthy but no breaker has tripped,
    pick() must NOT raise — give one backend a chance. This is what prevents
    a flapping probe from locking out real users."""
    a = _make_backend("a")
    b = _make_backend("b")
    pool = BackendPool([a, b], RoutingConfig())
    # Force probe state to "everyone unhealthy" without tripping breakers
    pool._health.healthy["a"] = False
    pool._health.healthy["b"] = False
    chosen = pool.pick(model=None)
    assert chosen.name in ("a", "b")


def test_pool_pick_raises_when_all_breakers_open():
    a = _make_backend("a")
    pool = BackendPool([a], RoutingConfig())
    pool._health.healthy["a"] = False
    # Trip the breaker
    for _ in range(pool.routing.circuit_breaker.failures):
        pool._breakers["a"].record_failure()
    with pytest.raises(Exception, match="no healthy backend"):
        pool.pick(model=None)
