"""Unit test: YAML loader + env override + $VAR expansion."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from backend_proxy.config import load_config


def write(tmp: Path, text: str) -> Path:
    p = tmp / "c.yaml"
    p.write_text(textwrap.dedent(text))
    return p


def test_minimal_yaml(tmp_path: Path):
    p = write(tmp_path, """
        backends:
          - name: a
            base_url: http://localhost:8080/
    """)
    cfg = load_config(p)
    assert cfg.server.port == 9099
    assert cfg.backends[0].base_url == "http://localhost:8080"  # trailing slash stripped


def test_env_var_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MY_UPSTREAM", "http://example:9000")
    p = write(tmp_path, """
        backends:
          - name: a
            base_url: ${MY_UPSTREAM}
    """)
    cfg = load_config(p)
    assert cfg.backends[0].base_url == "http://example:9000"


def test_env_var_default_when_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    p = write(tmp_path, """
        backends:
          - name: a
            base_url: ${MISSING_VAR:-http://fallback:1}
    """)
    cfg = load_config(p)
    assert cfg.backends[0].base_url == "http://fallback:1"


def test_env_var_default_when_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """${VAR:-default} must fall back when VAR is exported but empty (bash :-).

    Regression: shell wrappers like start-proxy.sh do
    ``UPSTREAM="${UPSTREAM:-}"`` and re-export it, so children see ``""``.
    Python's ``os.environ.get(name, default)`` treats that as a present value
    and would otherwise stomp the YAML default with an empty string.
    """
    monkeypatch.setenv("EMPTY_VAR", "")
    p = write(tmp_path, """
        backends:
          - name: a
            base_url: ${EMPTY_VAR:-http://fallback:2}
    """)
    cfg = load_config(p)
    assert cfg.backends[0].base_url == "http://fallback:2"


def test_legacy_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("UPSTREAM", "http://override:7000")
    monkeypatch.setenv("PORT", "12345")
    p = write(tmp_path, """
        backends:
          - name: a
            base_url: http://localhost:8080
    """)
    cfg = load_config(p)
    assert cfg.backends[0].base_url == "http://override:7000"
    assert cfg.server.port == 12345


def test_duplicate_backend_names_rejected(tmp_path: Path):
    p = write(tmp_path, """
        backends:
          - name: a
            base_url: http://localhost:8080
          - name: a
            base_url: http://localhost:8081
    """)
    with pytest.raises(Exception):
        load_config(p)


def test_no_backend_rejected(tmp_path: Path):
    p = write(tmp_path, "{}")
    with pytest.raises(Exception):
        load_config(p)
