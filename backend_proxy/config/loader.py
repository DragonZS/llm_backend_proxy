"""Load YAML config, expand $VAR / ${VAR} from the environment, apply env overrides."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .schema import ProxyConfig

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _alias_env(new_name: str, old_name: str) -> Optional[str]:
    """Return os.environ[new_name] if set; else fall back to old_name with a
    DeprecationWarning so `VLLM_PROXY_*` env vars keep working for one release."""
    if new_name in os.environ:
        return os.environ[new_name]
    if old_name in os.environ:
        warnings.warn(
            f"{old_name} is deprecated; use {new_name} instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return os.environ[old_name]
    return None


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} / ${VAR:-default} / $VAR inside strings.

    The ``${VAR:-default}`` form follows bash semantics: an unset *or empty*
    ``VAR`` falls back to ``default``. Treating an exported-but-empty env var
    as "missing" matters when shell wrappers do ``UPSTREAM="${UPSTREAM:-}"``
    and re-export it to children — without this, the child sees ``""`` and
    silently overrides the YAML default with an empty string.
    """
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            name = m.group(1) or m.group(3)
            default = m.group(2)
            val = os.environ.get(name)
            if default is not None:
                # ${VAR:-default}: empty-string is "missing" (bash :- semantics)
                return val if val else default
            # ${VAR} / $VAR: pass through whatever is set, or "" if unset
            return val if val is not None else ""
        return _ENV_RE.sub(repl, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Honour a small set of environment variables for backwards compatibility
    with the legacy `start-proxy.sh`:

      UPSTREAM             -> backends[0].base_url (only if a single backend is
                              declared; otherwise ignored, since "which backend?"
                              would be ambiguous).
      PORT                 -> server.port
      BACKEND_PROXY_HOST   -> server.host  (alias: VLLM_PROXY_HOST)
    """
    server = data.setdefault("server", {})
    if "PORT" in os.environ:
        server["port"] = int(os.environ["PORT"])
    host = _alias_env("BACKEND_PROXY_HOST", "VLLM_PROXY_HOST")
    if host:
        server["host"] = host

    upstream = os.environ.get("UPSTREAM")
    if upstream:  # empty string is treated as "not set" (matches bash ${UPSTREAM:-})
        backends = data.get("backends") or []
        if len(backends) == 1:
            backends[0]["base_url"] = upstream.rstrip("/")
        elif len(backends) == 0:
            data["backends"] = [{
                "name": "default",
                "base_url": upstream.rstrip("/"),
            }]
        # If multiple backends are configured, ignore UPSTREAM silently — it
        # would be ambiguous which one to override.
    return data


def load_config(path: Optional[str | os.PathLike[str]] = None) -> ProxyConfig:
    """Load + validate a YAML config file. ``path=None`` produces a config that
    is built purely from env vars (useful for the simplest "UPSTREAM=… run" flow)."""
    if path is None:
        raw: dict[str, Any] = {}
    else:
        text = Path(path).read_text(encoding="utf-8")
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"config root must be a mapping, got {type(loaded).__name__}")
        raw = dict(loaded)

    raw = _expand_env(raw)
    raw = _apply_env_overrides(raw)
    return ProxyConfig.model_validate(raw)
