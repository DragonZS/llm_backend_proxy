"""Pytest fixtures shared across the suite."""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from typing import Iterator

import pytest
import uvicorn
from fastapi import FastAPI


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ThreadedServer:
    """Run a uvicorn server in a thread for integration tests. Avoids needing
    `httpx.AsyncClient(transport=...)` plumbing for full end-to-end coverage."""

    def __init__(self, app: FastAPI, host: str = "127.0.0.1", port: int | None = None) -> None:
        self.host = host
        self.port = port or free_port()
        self._cfg = uvicorn.Config(app, host=host, port=self.port, log_level="warning",
                                   loop="asyncio", lifespan="on")
        self._server = uvicorn.Server(self._cfg)
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        # Wait for startup
        import time
        for _ in range(100):
            if self._server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("uvicorn failed to start in time")

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)


@contextmanager
def serve(app: FastAPI) -> Iterator[_ThreadedServer]:
    s = _ThreadedServer(app)
    s.start()
    try:
        yield s
    finally:
        s.stop()


@pytest.fixture
def serve_app():
    return serve
