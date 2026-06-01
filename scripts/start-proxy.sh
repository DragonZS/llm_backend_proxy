#!/usr/bin/env bash
# Idempotently start backend-proxy in the background.
# Probes /healthz on $PORT; if something is already serving the proxy there,
# does nothing. Mirrors the legacy proxy_v1/start-proxy.sh contract.
#
# Env overrides:
#   BACKEND_PROXY_PORT        (default 9099)
#   BACKEND_PROXY_CONFIG      (path to YAML; required unless UPSTREAM is set)
#   UPSTREAM                  (legacy: single-backend URL; used iff BACKEND_PROXY_CONFIG unset)
#   BACKEND_PROXY_LOG         (default /var/log/backend-proxy.log)
#   BACKEND_PROXY_AUTO_NUDGE  (default 1 — streaming inline rescue ON; set to 0 to disable)
#                             When 1, the proxy detects "model talks but
#                             doesn't act" patterns mid-stream and
#                             transparently re-issues with a nudge,
#                             splicing the second response into the same
#                             SSE connection. Token-by-token streaming is
#                             preserved on the happy path. Set to 0 if
#                             you want strict pass-through behaviour or
#                             are diagnosing model output without proxy
#                             intervention.
#   PYTHON                    (default: first python3 on $PATH)

set -e

PORT="${BACKEND_PROXY_PORT:-9099}"
LOG="${BACKEND_PROXY_LOG:-/var/log/backend-proxy.log}"
PY="${PYTHON:-$(command -v python3)}"

# Inline streaming rescue is the recommended default for Codex + non-OpenAI
# models. Set BACKEND_PROXY_AUTO_NUDGE=0 explicitly to disable.
AUTO_NUDGE="${BACKEND_PROXY_AUTO_NUDGE:-1}"

if [[ -z "${BACKEND_PROXY_CONFIG:-}" && -z "${UPSTREAM:-}" ]]; then
  echo "[backend-proxy] error: set BACKEND_PROXY_CONFIG=<path-to-yaml> or UPSTREAM=<url>" >&2
  exit 2
fi

port_in_use() {
  "$PY" - "$PORT" <<'PYEOF' >/dev/null 2>&1
import socket, sys
s = socket.socket()
s.settimeout(0.4)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
    sys.exit(0)
except Exception:
    sys.exit(1)
finally:
    s.close()
PYEOF
}

proxy_healthy() {
  "$PY" - "$PORT" <<'PYEOF' >/dev/null 2>&1
import sys, urllib.request
try:
    r = urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/healthz", timeout=2)
    sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PYEOF
}

if proxy_healthy; then
  echo "[backend-proxy] already healthy on :${PORT}, nothing to do"
  exit 0
fi
if port_in_use; then
  echo "[backend-proxy] something is on :${PORT} but /healthz is not OK — refusing to start" >&2
  echo "             stop the other process or set BACKEND_PROXY_PORT to a free port" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG")"

# Anchor the import path on the repo root so `python -m backend_proxy` works
# even when the package isn't pip-installed and the script is invoked from
# an arbitrary cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARGS=("-m" "backend_proxy" "run" "--port" "$PORT")
if [[ -n "${BACKEND_PROXY_CONFIG:-}" ]]; then
  ARGS+=("--config" "$BACKEND_PROXY_CONFIG")
fi

# UPSTREAM is honoured by config.loader.apply_env_overrides when present.
# Only pass it through if non-empty — exporting UPSTREAM="" would otherwise
# stomp the YAML's ${UPSTREAM:-default} fallback with an empty string.
ENV_PAIRS=("PORT=$PORT")
if [[ -n "${BACKEND_PROXY_CONFIG:-}" ]]; then
  ENV_PAIRS+=("BACKEND_PROXY_CONFIG=$BACKEND_PROXY_CONFIG")
fi
if [[ -n "${UPSTREAM:-}" ]]; then
  ENV_PAIRS+=("UPSTREAM=$UPSTREAM")
fi
# Forward the nudge toggle. "0" disables; anything else (default "1") enables.
if [[ -n "$AUTO_NUDGE" ]]; then
  ENV_PAIRS+=("BACKEND_PROXY_AUTO_NUDGE=$AUTO_NUDGE")
fi
# Make the source tree importable when the package isn't pip-installed.
if [[ -z "${PYTHONPATH:-}" ]]; then
  ENV_PAIRS+=("PYTHONPATH=$REPO_ROOT")
else
  ENV_PAIRS+=("PYTHONPATH=$REPO_ROOT:$PYTHONPATH")
fi

env "${ENV_PAIRS[@]}" \
  setsid "$PY" "${ARGS[@]}" >>"$LOG" 2>&1 < /dev/null &
disown

for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  sleep 0.3
  if proxy_healthy; then
    if [[ "$AUTO_NUDGE" == "0" ]]; then
      nudge_label="off"
    else
      nudge_label="on (BACKEND_PROXY_AUTO_NUDGE=$AUTO_NUDGE)"
    fi
    echo "[backend-proxy] started on :${PORT} (config=${BACKEND_PROXY_CONFIG:-<env>}, log=$LOG, auto-nudge=$nudge_label)"
    exit 0
  fi
done

echo "[backend-proxy] FAILED to start; see $LOG" >&2
tail -20 "$LOG" >&2 || true
exit 1
