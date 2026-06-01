# backend-proxy

FastAPI-based proxy in front of one or many LLM backends. Adapts requests for
multiple CLI coding agents (Codex, Claude Code, OpenCode, ...) and unifies
multiple backend types (vLLM, Ollama, TGI, llama.cpp, SGLang, OpenAI cloud)
behind a single OpenAI-compatible API.

Status: feature-complete through R5. 100 tests, all green.

## Quickstart

```bash
pip install -e '.[perf,metrics,dev]'
backend-proxy check-config -c configs/config.minimal.yaml
backend-proxy run         -c configs/config.minimal.yaml
```

Probe:

```bash
curl -s localhost:9099/healthz
curl -s localhost:9099/readyz
curl -s localhost:9099/v1/models
```

## Backend type matrix

| `type:`         | Class                  | Default `health_path` | Notes                                                  |
|-----------------|------------------------|-----------------------|--------------------------------------------------------|
| `vllm`          | `VLLMBackend`          | `/health`             | Default. OpenAI-compatible.                            |
| `ollama`        | `OllamaNativeBackend`  | `/`                   | Translates `/v1/chat/completions` ↔ `/api/chat` (NDJSON). Models from `/api/tags`. |
| `ollama-openai` | `OllamaOpenAIBackend`  | `/`                   | Uses Ollama's built-in `/v1/*` endpoints (Ollama 0.1.31+). No translation. |
| `openai`        | `OpenAICloudBackend`   | `/v1/models`          | api.openai.com (or any OpenAI-shaped cloud).           |
| `anthropic`     | `AnthropicBackend`     | `/v1/models`          | api.anthropic.com. `/v1/messages` passthrough; `/v1/chat/completions` auto-translated to/from Messages API. |
| `vertex-claude` | `VertexClaudeBackend`  | (token mint)          | Claude on Vertex AI. OAuth2 service-account auto-refresh; URL `/v1/projects/.../publishers/anthropic/models/X:streamRawPredict`. |
| `vertex-gemini` | `VertexGeminiBackend`  | (token mint)          | Gemini on Vertex AI. OpenAI Chat ↔ `generateContent` adapters. |
| `tgi`           | `TGIBackend`           | `/health`             | HuggingFace text-generation-inference.                 |
| `llamacpp`      | `LlamaCppBackend`      | `/health`             | llama.cpp server (`--api`).                            |
| `sglang`        | `SGLangBackend`        | `/health`             | SGLang server.                                         |

Multiple instances of the same `type` are fine — each is an independent
backend with its own auth, concurrency cap, rate limit, and health state.

## Vertex AI auth (service account)

```yaml
- name: vertex-claude
  type: vertex-claude
  base_url: https://us-central1-aiplatform.googleapis.com
  auth:
    scheme: none                           # token comes from credentials, not from auth.api_key
    options:
      project: my-gcp-project
      location: us-central1
      credentials_file: /etc/gcp/sa.json   # OR
      # credentials_inline: { ... full SA JSON ... }
      # (omit both → fall back to Application Default Credentials)
  models: [claude-3-5-sonnet@20240620]
```

Tokens are minted via `google.oauth2.service_account` and refreshed in-memory
~60 s before expiry. No restart needed when keys rotate.

## Per-backend rate limit (token bucket)

```yaml
backends:
  - name: openai-cloud
    type: openai
    rate_limit:
      rps: 5            # sustained tokens per second
      burst: 10         # bucket capacity
      mode: wait        # 'wait' (default) or 'reject' (immediate 429)
```

Each backend gets its own bucket. ``mode: reject`` returns HTTP 429 with
``error.type=rate_limited``; ``mode: wait`` blocks the request asynchronously.

## SSE heartbeat (long streams)

```yaml
server:
  heartbeat:
    enabled: true
    interval_s: 15      # emit ': hb\n\n' if upstream silent for >15s
    payload: hb
```

Comment frames keep idle proxies / load balancers from dropping the
connection during long Chain-of-Thought generations. Clients ignore them.

## Per-backend auth

Each backend declares its own credentials. The client does **not** carry the
upstream key. Schemes (`auth.scheme`):

| `scheme:`       | What it sends                                           |
|-----------------|---------------------------------------------------------|
| `bearer`        | `Authorization: Bearer <api_key>`                       |
| `x_api_key`     | `x-api-key: <api_key>`  (Anthropic-style)               |
| `api_key_header`| `<header_name>: <api_key>`                              |
| `passthrough`   | Forwards the *client's* `Authorization` (legacy mode)   |
| `none`          | No auth headers (local Ollama / unprotected vLLM)       |

The legacy `api_key: ...` shorthand is auto-promoted to
`auth: { scheme: bearer, api_key: ... }`.

## Routing

```yaml
routing:
  strategy: model_affinity      # route by `model` field if a backend declares it
  fallback: round_robin         # otherwise rotate over healthy backends
  circuit_breaker:
    failures: 5
    cooldown_s: 30
```

Strategies: `round_robin`, `least_busy`, `model_affinity`, `sticky`, `none`.

## Multi-agent

Agent profiles select an adapter pipeline based on incoming request shape
(path prefix, header substring, query param, default). Built-in profiles:

| Agent          | Match                       | Pipeline                                                    |
|----------------|-----------------------------|-------------------------------------------------------------|
| `codex`        | `/v1/responses`             | `responses_to_chat` → upstream → `chat_to_responses(_stream)` |
| `claude-code`  | `/v1/messages`              | `anthropic_to_chat` → upstream → `chat_to_anthropic(_stream)` |
| `opencode`     | (default)                   | passthrough                                                 |

Override or extend via the `agents:` section in YAML.

## Examples

* `configs/config.minimal.yaml` — single vLLM, env-driven.
* `configs/config.mixed.yaml`   — vLLM + Ollama + OpenAI cloud, separate keys.
* `configs/config.example.yaml` — every backend type, every auth scheme.

## Operations

* `scripts/start-proxy.sh` — idempotent starter, probes `/healthz`.
* `scripts/backend-proxy.service` — systemd unit (`systemctl reload` → SIGHUP → live config reload).
* `Dockerfile` — slim image with HEALTHCHECK.
* `POST /admin/reload` — hot-reload the YAML; optional `BACKEND_PROXY_ADMIN_TOKEN` gate.
* `GET /metrics` — Prometheus.

## Layout

```
backend_proxy/
├── adapters/      # responses↔chat, anthropic↔chat, model_alias, ...
├── agents/        # AgentResolver + builtins
├── api/           # FastAPI routers (responses / messages / models / admin / passthrough)
├── backends/      # Backend ABC, OpenAICompatBackend (+ presets), OllamaNativeBackend, BackendPool
├── config/        # pydantic schema + YAML loader + hot-reload
├── core/          # logging, metrics, middleware, errors, RequestContext
├── streaming/     # SSE parse/serialise
└── utils/         # ids
```

## Migration notes (vllm-proxy → backend-proxy)

* Console script `vllm-proxy` still works — emits a deprecation notice and
  forwards to `backend-proxy`.
* Env vars `VLLM_PROXY_CONFIG`, `VLLM_PROXY_HOST`, `VLLM_PROXY_ADMIN_TOKEN`
  remain honoured with a `DeprecationWarning`. Use the `BACKEND_PROXY_*`
  spellings going forward.
* Existing single-backend YAML keeps working unchanged: `type:` defaults to
  `vllm`, and `api_key:` is auto-promoted to `auth: { scheme: bearer, ... }`.
