"""Per-request context propagated through the adapter pipeline.

Kept small and immutable-ish — adapters mutate `extras` but should not rewrite
core fields. ``RequestContext`` is built by FastAPI dependencies and travels
along the call chain so adapters can attach trace ids, see the chosen backend,
record timings, etc.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RequestContext:
    trace_id: str
    started_at: float
    method: str
    path: str
    client_headers: Dict[str, str] = field(default_factory=dict)
    agent_name: Optional[str] = None
    backend_name: Optional[str] = None
    model: Optional[str] = None
    stream: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, *, method: str, path: str, headers: Dict[str, str]) -> "RequestContext":
        return cls(
            trace_id=uuid.uuid4().hex[:16],
            started_at=time.perf_counter(),
            method=method,
            path=path,
            client_headers=headers,
        )

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0
