"""Logging configured for both stdlib and FastAPI/Uvicorn loggers.

M0 ships a plain-text formatter; M4 will swap in structlog + JSON for log
aggregation. The public surface (``configure_logging`` + ``get_logger``) is
stable so M4 can do that swap without touching call sites.
"""

from __future__ import annotations

import logging
import sys


_LEVELS = {
    "trace": logging.DEBUG,
    "debug": logging.DEBUG,
    "info":  logging.INFO,
    "warn":  logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def configure_logging(level: str = "info") -> None:
    lvl = _LEVELS.get(level.lower(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(lvl)
    # Keep uvicorn access logs on the same handler.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        log = logging.getLogger(name)
        log.handlers[:] = [handler]
        log.setLevel(lvl)
        log.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
