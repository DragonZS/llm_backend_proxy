"""Smoke test: every shipped example config must parse cleanly."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend_proxy.config import load_config

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.parametrize("name", ["config.minimal.yaml",
                                  "config.mixed.yaml",
                                  "config.example.yaml"])
def test_example_config_loads(monkeypatch, name):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("TGI_KEY", "test")
    monkeypatch.setenv("SGLANG_KEY", "test")
    monkeypatch.setenv("LOCAL_VLLM_KEY", "test")
    cfg = load_config(CONFIGS_DIR / name)
    assert len(cfg.backends) >= 1
    # Every backend must have a resolved auth block (model_validator promotes
    # legacy api_key fields).
    for b in cfg.backends:
        assert b.auth is not None
