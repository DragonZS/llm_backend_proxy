#!/usr/bin/env bash
# Idempotently control backend-proxy in the background.
#
# Subcommands:
#   start    (default) — start if not already healthy on $PORT.
#   stop              — SIGTERM the running proxy, then SIGKILL if it lingers.
#   restart           — stop (if running) then start.
#   status            — print whether the proxy is healthy / a stale PID exists.
#
# Probes /healthz on $PORT; if something is already serving the proxy there,
# `start` does nothing. Mirrors the legacy proxy_v1/start-proxy.sh contract.
#
# Env overrides:
#   BACKEND_PROXY_PORT          (default 9099)
#   BACKEND_PROXY_CONFIG        (path to YAML; required for `start` unless UPSTREAM is set)
#   UPSTREAM                    (legacy: single-backend URL; used iff BACKEND_PROXY_CONFIG unset)
#   BACKEND_PROXY_LOG           (default /var/log/backend-proxy.log)
#   BACKEND_PROXY_PID           (default /var/run/backend-proxy.pid)
#   BACKEND_PROXY_STOP_TIMEOUT  (seconds to wait for graceful exit, default 10)
#   BACKEND_PROXY_AUTO_NUDGE    (default 1 — streaming inline rescue ON; set to 0 to disable)
#                               When 1, the proxy detects "model talks but
#                               doesn't act" patterns mid-stream and
#                               transparently issues a second upstream call
#                               with a nudge, splicing it into the same SSE
#                               connection. Token-by-token streaming is
#                               preserved; a brief pause appears only when a
#                               rescue actually fires. Set to 0 if you need
#                               strict pass-through semantics or want to
#                               diagnose model behaviour without proxy
#                               intervention.
#   PYTHON                      (default: first python3 on $PATH)

set -e

PORT="${BACKEND_PROXY_PORT:-9099}"
LOG="${BACKEND_PROXY_LOG:-/var/log/backend-proxy.log}"
PID_FILE="${BACKEND_PROXY_PID:-/var/run/backend-proxy.pid}"
STOP_TIMEOUT="${BACKEND_PROXY_STOP_TIMEOUT:-10}"
PY="${PYTHON:-$(command -v python3)}"

# Inline streaming rescue is the recommended default for Codex + non-OpenAI
# models (Qwen, etc.): without it ~20% of agent turns end with the model
# announcing an action it didn't take. The proxy splices a nudged second
# upstream call into the same SSE stream, transparently to the client.
# Explicitly set BACKEND_PROXY_AUTO_NUDGE=0 to disable.
AUTO_NUDGE="${BACKEND_PROXY_AUTO_NUDGE:-1}"

CMD="${1:-start}"

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

read_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  echo "$pid"
}

do_start() {
  if [[ -z "${BACKEND_PROXY_CONFIG:-}" && -z "${UPSTREAM:-}" ]]; then
    echo "[backend-proxy] error: set BACKEND_PROXY_CONFIG=<path-to-yaml> or UPSTREAM=<url>" >&2
    exit 2
  fi

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
  mkdir -p "$(dirname "$PID_FILE")" 2>/dev/null || true

  ARGS=("-m" "backend_proxy" "run" "--port" "$PORT")
  if [[ -n "${BACKEND_PROXY_CONFIG:-}" ]]; then
    ARGS+=("--config" "$BACKEND_PROXY_CONFIG")
  fi

  # The package may not be installed; let `python -m backend_proxy` resolve
  # it from the source tree by anchoring PYTHONPATH on the repo root.
  # ${BASH_SOURCE[0]} = .../backend_proxy/scripts/control-proxy.sh
  # repo root         = .../backend_proxy
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local repo_root
  repo_root="$(cd "$script_dir/.." && pwd)"

  # The package may not be installed; let `python -m backend_proxy` resolve
  # it from the source tree by anchoring PYTHONPATH on the repo root.
  # ${BASH_SOURCE[0]} = .../backend_proxy/scripts/control-proxy.sh
  # repo root         = .../backend_proxy
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local repo_root
  repo_root="$(cd "$script_dir/.." && pwd)"

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
  # Forward the nudge toggle to the child unless explicitly empty. "0"
  # disables the inline streaming rescue; anything non-empty other than
  # "0" enables it. Default chosen above is "1".
  if [[ -n "$AUTO_NUDGE" ]]; then
    ENV_PAIRS+=("BACKEND_PROXY_AUTO_NUDGE=$AUTO_NUDGE")
  fi
  # Make the source tree importable when the package isn't pip-installed.
  if [[ -z "${PYTHONPATH:-}" ]]; then
    ENV_PAIRS+=("PYTHONPATH=$repo_root")
  else
    ENV_PAIRS+=("PYTHONPATH=$repo_root:$PYTHONPATH")
  fi

  env "${ENV_PAIRS[@]}" \
    setsid "$PY" "${ARGS[@]}" >>"$LOG" 2>&1 < /dev/null &
  local child=$!
  disown || true

  # Persist the PID so `stop`/`restart` can find it. setsid leaves $! as the
  # session leader, which is exactly what we want to signal later.
  if ! echo "$child" > "$PID_FILE" 2>/dev/null; then
    echo "[backend-proxy] warning: could not write PID file $PID_FILE" >&2
  fi

  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    sleep 0.3
    if proxy_healthy; then
      local nudge_label
      if [[ "$AUTO_NUDGE" == "0" ]]; then
        nudge_label="off"
      else
        nudge_label="on (BACKEND_PROXY_AUTO_NUDGE=$AUTO_NUDGE)"
      fi
      echo "[backend-proxy] started on :${PORT} (pid=$child, config=${BACKEND_PROXY_CONFIG:-<env>}, log=$LOG, auto-nudge=$nudge_label)"
      exit 0
    fi
  done

  echo "[backend-proxy] FAILED to start; see $LOG" >&2
  tail -20 "$LOG" >&2 || true
  exit 1
}

do_stop() {
  local pid=""
  pid="$(read_pid || true)"

  if [[ -z "$pid" ]]; then
    if proxy_healthy; then
      echo "[backend-proxy] healthy on :${PORT} but no PID file at $PID_FILE — refusing to guess" >&2
      echo "             kill the process manually, or set BACKEND_PROXY_PID to its pidfile" >&2
      exit 1
    fi
    echo "[backend-proxy] not running"
    rm -f "$PID_FILE" 2>/dev/null || true
    return 0
  fi

  echo "[backend-proxy] stopping pid=$pid ..."
  # setsid made the child a session leader; signal the whole group so any
  # forked workers go down with it. Fall back to the PID if the group send
  # is rejected (e.g. process already exiting).
  kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true

  local waited=0
  while (( waited < STOP_TIMEOUT * 10 )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE" 2>/dev/null || true
      echo "[backend-proxy] stopped"
      return 0
    fi
    sleep 0.1
    waited=$((waited + 1))
  done

  echo "[backend-proxy] pid=$pid did not exit within ${STOP_TIMEOUT}s, sending SIGKILL" >&2
  kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  sleep 0.3
  rm -f "$PID_FILE" 2>/dev/null || true
  if kill -0 "$pid" 2>/dev/null; then
    echo "[backend-proxy] FAILED to stop pid=$pid" >&2
    exit 1
  fi
  echo "[backend-proxy] killed"
}

do_status() {
  local pid=""
  pid="$(read_pid || true)"
  if proxy_healthy; then
    echo "[backend-proxy] healthy on :${PORT}${pid:+ (pid=$pid)}"
    exit 0
  fi
  if [[ -n "$pid" ]]; then
    echo "[backend-proxy] pid=$pid alive but /healthz not OK on :${PORT}"
    exit 1
  fi
  if port_in_use; then
    echo "[backend-proxy] something on :${PORT} but not the proxy"
    exit 1
  fi
  echo "[backend-proxy] not running"
  exit 3
}

case "$CMD" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop || true; do_start ;;
  status)  do_status ;;
  -h|--help|help)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    echo "usage: $(basename "$0") {start|stop|restart|status}" >&2
    exit 2
    ;;
esac
