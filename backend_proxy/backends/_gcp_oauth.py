"""Google OAuth2 access-token cache.

Self-contained helper so Vertex backends don't each carry their own token
refresh code. Three credential sources are supported:

* ``credentials_file`` — path to a Service Account JSON file (recommended).
* ``credentials_inline`` — the same JSON inlined into YAML.
* ``application_default`` — fall back to ADC
  (``gcloud auth application-default login``, GCE metadata server, etc.).

The token is cached in-memory; we proactively refresh ~60 s before expiry to
avoid races on long-running streams.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

log = logging.getLogger("backend_proxy.gcp_oauth")


_CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


@dataclass
class _CachedToken:
    value: str
    expires_at: float


class GcpAccessTokenProvider:
    """Lazy, refresh-on-expiry OAuth2 access-token provider.

    ``opts`` is an arbitrary dict from a backend's auth config; honoured keys:

      credentials_file:   "/etc/sa.json"
      credentials_inline: { ... full SA JSON ... }
      scope:              defaults to cloud-platform
      target_audience:    optional; produces an ID token instead of access token

    Falls back to Application Default Credentials when neither file nor
    inline JSON is provided.
    """

    def __init__(self, opts: Optional[Dict[str, Any]] = None) -> None:
        opts = opts or {}
        self._creds_file = opts.get("credentials_file")
        self._creds_inline = opts.get("credentials_inline")
        self._scope = opts.get("scope", _CLOUD_SCOPE)
        self._target_audience = opts.get("target_audience")
        self._cache: Optional[_CachedToken] = None
        self._lock = threading.Lock()
        self._refresh_lock = asyncio.Lock()

    async def get(self) -> str:
        """Return a valid access token, refreshing if needed."""
        now = time.time()
        with self._lock:
            cached = self._cache
        if cached is not None and cached.expires_at > now + 60:
            return cached.value
        # Refresh under an asyncio lock so concurrent callers don't all hit
        # the refresh path simultaneously.
        async with self._refresh_lock:
            with self._lock:
                cached = self._cache
            if cached is not None and cached.expires_at > now + 60:
                return cached.value
            value, expires_at = await asyncio.to_thread(self._refresh_blocking)
            with self._lock:
                self._cache = _CachedToken(value=value, expires_at=expires_at)
            return value

    # The actual refresh runs google-auth's blocking client; do it off the loop.
    def _refresh_blocking(self) -> tuple[str, float]:
        try:
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account
            import google.auth as ga
        except ImportError as e:  # pragma: no cover - exercised only without google-auth
            raise RuntimeError(
                "Vertex backends require 'google-auth'; "
                "pip install google-auth"
            ) from e

        creds: Any
        if self._creds_inline:
            info = self._creds_inline
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=[self._scope],
            )
        elif self._creds_file:
            path = os.path.expanduser(self._creds_file)
            creds = service_account.Credentials.from_service_account_file(
                path, scopes=[self._scope],
            )
        else:
            creds, _ = ga.default(scopes=[self._scope])

        creds.refresh(Request())
        # google-auth stores expiry as a naive UTC datetime
        if creds.expiry is None:
            expires_at = time.time() + 3600
        else:
            expires_at = creds.expiry.timestamp()
        log.debug("refreshed gcp access token, expires in %.0fs",
                  expires_at - time.time())
        return creds.token, expires_at
