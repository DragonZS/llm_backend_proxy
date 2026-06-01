"""Prometheus metrics — singleton registry exposed at /metrics.

Lazy-initialised: importing this module does not create the metrics objects.
That keeps tests using ``create_app`` fresh on each invocation safe.
"""

from __future__ import annotations

from typing import Optional

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    _AVAILABLE = True
except Exception:  # noqa: BLE001
    _AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"


class Metrics:
    """Bundled metrics. We re-create the registry per app instance so tests are
    isolated; in production there's a single app per process anyway."""

    def __init__(self) -> None:
        if not _AVAILABLE:
            self.enabled = False
            return
        self.enabled = True
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "backend_proxy_requests_total",
            "Total proxied requests",
            labelnames=("agent", "backend", "route", "status"),
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "backend_proxy_request_duration_seconds",
            "Per-request duration",
            labelnames=("agent", "backend", "route"),
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
            registry=self.registry,
        )
        self.inflight = Gauge(
            "backend_proxy_inflight",
            "In-flight requests per backend",
            labelnames=("backend",),
            registry=self.registry,
        )
        self.upstream_errors = Counter(
            "backend_proxy_upstream_errors_total",
            "Upstream errors raised by a backend",
            labelnames=("backend", "code"),
            registry=self.registry,
        )

    def render(self) -> tuple[bytes, str]:
        if not self.enabled:
            return b"# prometheus_client not installed\n", "text/plain"
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
