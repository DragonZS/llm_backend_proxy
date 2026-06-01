"""Integration: POST /admin/reload swaps in a new pool from a fresh YAML."""

from __future__ import annotations

import textwrap
from pathlib import Path

import httpx

from backend_proxy.app import create_app

from tests.conftest import serve
from tests.fake_vllm import make_fake_vllm


def _write_cfg(path: Path, upstream: str, name: str = "A") -> None:
    path.write_text(textwrap.dedent(f"""
        server:
          host: 0.0.0.0
          log_level: warning
        backends:
          - name: {name}
            base_url: {upstream}
    """))


def test_admin_reload_swaps_backends(tmp_path: Path):
    a = make_fake_vllm(name="A", models=["a-model"])
    b = make_fake_vllm(name="B", models=["b-model"])
    with serve(a) as sa, serve(b) as sb:
        cfg_path = tmp_path / "c.yaml"
        _write_cfg(cfg_path, sa.base_url, name="A")

        proxy_app = create_app(config_path=str(cfg_path))
        with serve(proxy_app) as proxy_srv:
            # Initially: only "a-model" is reachable
            ids = sorted(m["id"] for m in
                         httpx.get(proxy_srv.base_url + "/v1/models").json()["data"])
            assert ids == ["a-model"]

            # Rewrite YAML to point to backend B; reload
            _write_cfg(cfg_path, sb.base_url, name="B")
            r = httpx.post(proxy_srv.base_url + "/admin/reload")
            assert r.status_code == 200, r.text
            assert r.json()["backends"] == ["B"]

            # After reload: now sees b-model
            import time
            time.sleep(0.3)  # allow drain
            ids = sorted(m["id"] for m in
                         httpx.get(proxy_srv.base_url + "/v1/models").json()["data"])
            assert ids == ["b-model"]


def test_admin_reload_token_required(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BACKEND_PROXY_ADMIN_TOKEN", "secret")
    a = make_fake_vllm(name="A", models=["x"])
    with serve(a) as sa:
        cfg_path = tmp_path / "c.yaml"
        _write_cfg(cfg_path, sa.base_url, name="A")
        proxy_app = create_app(config_path=str(cfg_path))
        with serve(proxy_app) as proxy_srv:
            r = httpx.post(proxy_srv.base_url + "/admin/reload")
            assert r.status_code == 401
            r = httpx.post(proxy_srv.base_url + "/admin/reload",
                           headers={"Authorization": "Bearer secret"})
            assert r.status_code == 200
