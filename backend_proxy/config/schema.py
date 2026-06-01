"""Pydantic models describing the proxy configuration.

The schema is intentionally permissive in this milestone: only the fields needed
for M0 (single-backend pass-through) are required. Later milestones add routing,
adapters, agents, etc. — those fields are declared up-front so YAML written
today stays valid tomorrow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class RetryConfig(BaseModel):
    """Automatic retry for transient upstream failures.

    * ``max_retries``          — maximum number of retry attempts (0 = disabled).
    * ``initial_backoff``      — seconds to wait before the first retry.
    * ``max_backoff``          — cap on backoff duration.
    * ``backoff_multiplier``   — multiplier applied to backoff after each attempt.
    * ``retryable_status_codes`` — upstream HTTP status codes that trigger retry.
    """
    max_retries: int = 2
    initial_backoff: float = 0.5
    max_backoff: float = 8.0
    backoff_multiplier: float = 2.0
    retryable_status_codes: List[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )


class StreamTimeoutConfig(BaseModel):
    """Timeouts for streaming requests.

    * ``idle_timeout_s``     — abort if no chunk received from upstream for this
      many seconds. 0 = disabled.
    * ``absolute_timeout_s`` — abort if total stream duration exceeds this many
      seconds. 0 = disabled (falls back to ``server.request_timeout``).
    """
    idle_timeout_s: float = 0.0
    absolute_timeout_s: float = 0.0


class ConnectConfig(BaseModel):
    """httpx connection pool tuning.

    * ``connect_timeout``      — TCP connect timeout in seconds.
    * ``keepalive_expiry``     — how long idle keepalive connections are retained.
    * ``pool_max_connections``  — override for httpx max_connections.
      0 = use existing formula (max_concurrency * 2, min 64).
    * ``pool_max_keepalive``   — override for httpx max_keepalive_connections.
      0 = use existing formula (max_concurrency, min 32).
    """
    connect_timeout: float = 10.0
    keepalive_expiry: float = 60.0
    pool_max_connections: int = 0
    pool_max_keepalive: int = 0


class HeartbeatConfig(BaseModel):
    """SSE keepalive injected into long streams while the upstream is silent.

    * ``enabled``    — feature toggle.
    * ``interval_s`` — emit a comment frame after this many seconds with no
      upstream chunk. Browsers / CLIs typically need < 60s to keep their
      connection open.
    * ``payload``    — the comment body. SSE comments start with ``:`` and
      are ignored by clients.
    """
    enabled: bool = False
    interval_s: float = 15.0
    payload: str = "hb"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 9099
    workers: int = 1
    log_level: str = "info"
    request_timeout: float = 600.0
    drain_window_s: float = 2.0
    stream_timeout: StreamTimeoutConfig = Field(default_factory=StreamTimeoutConfig)
    cors_allow_origins: List[str] = Field(default_factory=lambda: ["*"])
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)


class CircuitBreakerConfig(BaseModel):
    failures: int = 5
    cooldown_s: float = 30.0


class HealthCheckConfig(BaseModel):
    """Background liveness probe tuning.

    Default values are chosen so that a single transient failure (a closed
    keepalive socket, a 50ms hiccup during heavy generation) does NOT flip
    the backend to unhealthy. The probe must fail ``unhealthy_threshold``
    times *consecutively* before the backend is removed from the candidate
    pool — and a single success brings it straight back.
    """
    enabled: bool = True
    interval_s: float = 5.0
    # Independent timeout: must NOT inherit the 600s inference timeout, or a
    # stuck probe blocks the whole watcher.
    probe_timeout_s: float = 2.0
    # Consecutive failures before flipping healthy -> unhealthy.
    unhealthy_threshold: int = 2
    # Consecutive successes before flipping unhealthy -> healthy. 1 = recover
    # as fast as possible; raise if you see flapping.
    healthy_threshold: int = 1


class HealthCheckConfig(BaseModel):
    """Background liveness probe tuning.

    Default values are chosen so that a single transient failure (a closed
    keepalive socket, a 50ms hiccup during heavy generation) does NOT flip
    the backend to unhealthy. The probe must fail ``unhealthy_threshold``
    times *consecutively* before the backend is removed from the candidate
    pool — and a single success brings it straight back.
    """
    enabled: bool = True
    interval_s: float = 5.0
    # Independent timeout: must NOT inherit the 600s inference timeout, or a
    # stuck probe blocks the whole watcher.
    probe_timeout_s: float = 2.0
    # Consecutive failures before flipping healthy -> unhealthy.
    unhealthy_threshold: int = 2
    # Consecutive successes before flipping unhealthy -> healthy. 1 = recover
    # as fast as possible; raise if you see flapping.
    healthy_threshold: int = 1


class RateLimitConfig(BaseModel):
    """Token-bucket rate limiter applied per backend.

    * ``rps``    — sustained refill rate (tokens / second).
    * ``burst``  — bucket capacity (max tokens kept).
    * ``mode``   — what to do when the bucket is empty:
        ``wait``   (default): block until a token is available.
        ``reject`` : return 429 immediately.

    Set ``rps: 0`` (the default) to disable.
    """
    rps: float = 0.0
    burst: float = 0.0
    mode: Literal["wait", "reject"] = "wait"

    @model_validator(mode="after")
    def _normalise(self) -> "RateLimitConfig":
        if self.rps and not self.burst:
            self.burst = max(1.0, self.rps)
        return self


# Removed duplicate definitions below (kept HeartbeatConfig only at top).


class AuthConfig(BaseModel):
    """How to authenticate to a backend.

    ``scheme`` selects the header style; ``api_key`` is the credential value.
    The legacy ``BackendConfig.api_key`` field is honoured for back-compat and
    will be promoted to ``auth.api_key`` automatically when set.

    * ``bearer``         — ``Authorization: Bearer <key>`` (vLLM, OpenAI, ...)
    * ``api_key_header`` — custom header (``header_name``: ``<key>``)
    * ``x_api_key``      — ``x-api-key: <key>``  (Anthropic-style)
    * ``passthrough``    — forward the client's own ``Authorization`` header
    * ``none``           — no auth (Ollama default, local vLLM without keys)

    ``options`` is a free-form bag the backend may consult (e.g. Vertex's
    ``credentials_file`` / ``credentials_inline`` / ``scope``).
    """
    scheme: Literal[
        "bearer", "api_key_header", "x_api_key", "passthrough", "none"
    ] = "none"
    api_key: Optional[str] = None
    header_name: Optional[str] = None  # only used when scheme == api_key_header
    options: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "AuthConfig":
        if self.scheme == "api_key_header" and not self.header_name:
            raise ValueError("auth.header_name is required when scheme=api_key_header")
        if self.scheme in ("bearer", "api_key_header", "x_api_key") and not self.api_key:
            # Allow declaring the scheme without a key (e.g. picked up later by
            # env interpolation that resolves to "") — empty key skips auth.
            pass
        return self


class BackendConfig(BaseModel):
    name: str
    # Backend kind. Default ``vllm`` keeps every existing config valid.
    type: Literal[
        "vllm",
        "ollama",          # native /api/* protocol
        "ollama-openai",   # Ollama's /v1/* OpenAI-compatible endpoint
        "openai",          # api.openai.com (or any OpenAI-compatible cloud)
        "anthropic",       # api.anthropic.com (native /v1/messages)
        "vertex-claude",   # Claude on Google Vertex AI (Anthropic protocol)
        "vertex-gemini",   # Gemini on Google Vertex AI (native generateContent)
        "tgi",             # HuggingFace text-generation-inference
        "llamacpp",        # llama.cpp's server.cpp (--api)
        "sglang",          # SGLang server
    ] = "vllm"
    base_url: str
    # Legacy short-hand for `auth: {scheme: bearer, api_key: ...}`. Kept so
    # existing configs continue to work without the ``auth:`` block.
    api_key: Optional[str] = None
    auth: Optional[AuthConfig] = None
    models: List[str] = Field(default_factory=list)
    weight: int = 1
    max_concurrency: int = 64
    timeout: float = 600.0
    # When unset, each backend type uses its preset (e.g. vllm -> /health,
    # ollama -> /, openai -> /v1/models). See backends/factory.py.
    health_path: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    connect: ConnectConfig = Field(default_factory=ConnectConfig)
    retry: Optional[RetryConfig] = None  # None = use routing.retry

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @model_validator(mode="after")
    def _normalise_auth(self) -> "BackendConfig":
        # Promote legacy api_key -> auth.bearer when no explicit auth block.
        if self.auth is None:
            if self.api_key:
                self.auth = AuthConfig(scheme="bearer", api_key=self.api_key)
            else:
                self.auth = AuthConfig(scheme="none")
        return self


class RoutingConfig(BaseModel):
    strategy: Literal[
        "round_robin", "least_busy", "model_affinity", "sticky"
    ] = "model_affinity"
    fallback: Literal["round_robin", "least_busy", "none"] = "round_robin"
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)


class AdapterConfig(BaseModel):
    """Free-form adapter knobs. Each adapter implementation reads its own key."""

    enabled: bool = True
    options: Dict[str, Any] = Field(default_factory=dict)


class AgentMatcher(BaseModel):
    header: Optional[str] = None
    contains: Optional[str] = None
    path_prefix: Optional[str] = None
    query_param: Optional[str] = None
    default: bool = False


class AgentProfileConfig(BaseModel):
    name: str
    match: AgentMatcher = Field(default_factory=AgentMatcher)
    request_adapters: List[str] = Field(default_factory=list)
    stream_adapters: List[str] = Field(default_factory=list)
    response_adapters: List[str] = Field(default_factory=list)
    backend_selector: Optional[str] = None
    model_aliases: Dict[str, str] = Field(default_factory=dict)


class ProxyConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    backends: List[BackendConfig] = Field(default_factory=list)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    adapters: Dict[str, AdapterConfig] = Field(default_factory=dict)
    agents: List[AgentProfileConfig] = Field(default_factory=list)

    @field_validator("backends")
    @classmethod
    def _unique_backend_names(cls, v: List[BackendConfig]) -> List[BackendConfig]:
        names = [b.name for b in v]
        if len(names) != len(set(names)):
            raise ValueError(f"backend names must be unique, got: {names}")
        return v

    @model_validator(mode="after")
    def _at_least_one_backend(self) -> "ProxyConfig":
        if not self.backends:
            raise ValueError("at least one backend must be configured")
        return self
