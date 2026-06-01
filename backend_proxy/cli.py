"""CLI entry point: ``backend-proxy run --config ...``."""

from __future__ import annotations

import argparse
import sys

from .app import create_app
from .config import load_config


def _cmd_run(args: argparse.Namespace) -> int:
    import uvicorn

    cfg = load_config(args.config)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    workers = args.workers or cfg.server.workers

    # Pass config_path via env so ``--workers >1`` (which re-imports the app
    # in each worker) can rebuild the same configuration.
    if args.config:
        import os
        os.environ.setdefault("BACKEND_PROXY_CONFIG", args.config)

    app_factory = "backend_proxy.cli:_app_factory"

    extras = {}
    try:
        import uvloop  # noqa: F401
        extras["loop"] = "uvloop"
    except ImportError:
        pass
    try:
        import httptools  # noqa: F401
        extras["http"] = "httptools"
    except ImportError:
        pass

    uvicorn.run(
        app_factory,
        factory=True,
        host=host,
        port=port,
        workers=workers,
        log_level=cfg.server.log_level,
        access_log=True,
        **extras,
    )
    return 0


def _app_factory():
    """Importable factory used by uvicorn workers."""
    import os
    import warnings
    from .app import create_app  # local import to keep ``run`` fast
    cfg_path = os.environ.get("BACKEND_PROXY_CONFIG")
    if cfg_path is None and "VLLM_PROXY_CONFIG" in os.environ:
        warnings.warn(
            "VLLM_PROXY_CONFIG is deprecated; use BACKEND_PROXY_CONFIG instead",
            DeprecationWarning, stacklevel=2,
        )
        cfg_path = os.environ["VLLM_PROXY_CONFIG"]
    return create_app(config_path=cfg_path)


def _cmd_check(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(f"OK: loaded config with {len(cfg.backends)} backend(s)")
    for b in cfg.backends:
        auth = b.auth.scheme if b.auth else "none"
        print(f"  - {b.name:20s} [{b.type:14s} auth={auth:14s}] {b.base_url}  models={b.models}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="backend-proxy")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the proxy server")
    run.add_argument("--config", "-c", default=None, help="path to YAML config")
    run.add_argument("--host", default=None)
    run.add_argument("--port", type=int, default=None)
    run.add_argument("--workers", type=int, default=None)
    run.set_defaults(func=_cmd_run)

    check = sub.add_parser("check-config", help="validate a YAML config")
    check.add_argument("--config", "-c", required=True)
    check.set_defaults(func=_cmd_check)

    args = p.parse_args(argv)
    return args.func(args)


def main_legacy(argv: list[str] | None = None) -> int:
    """Deprecated entry point kept under the ``vllm-proxy`` console script.
    Forwards to :func:`main` after emitting a one-line notice on stderr."""
    print(
        "warning: 'vllm-proxy' is deprecated; use 'backend-proxy' instead",
        file=sys.stderr,
    )
    return main(argv)


if __name__ == "__main__":
    sys.exit(main())
