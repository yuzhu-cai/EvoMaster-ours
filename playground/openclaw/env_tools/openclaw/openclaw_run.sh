#!/bin/bash
# OpenClaw run script for swe-agents StepTron adaptor.
#
# Runs the embedded OpenClaw agent (no Gateway) against the local FastAPI proxy:
# - FastAPI proxy logs request/response pairs to /tmp/fastapi_logs/*.jsonl
# - Proxy forwards requests to the configured upstream without compatibility shims
#
# Args:
#   $1 api_key
#   $2 model_id
#   $3 workdir (default: /testbed)
#   $4 timeout_sec (optional)

set -euo pipefail

# Some containers do not define HOME; guard for nounset mode.
export HOME="${HOME:-/tmp}"

# NOTE: This script runs in SessionRouter containers; source bashrc best-effort
# with nounset disabled to avoid failures from non-interactive configs.
if [[ -f "$HOME/.bashrc" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$HOME/.bashrc" || true
  set -u
fi

export IS_SANDBOX=1

# Avoid proxying localhost traffic.
if [[ -z "${NO_PROXY:-}" && -z "${no_proxy:-}" ]]; then
  export NO_PROXY="127.0.0.1,localhost"
  export no_proxy="127.0.0.1,localhost"
fi

wait_for_fastapi_log_flush() {
  local exclude_regex="${1:-}"
  local latest_jsonl=""
  for _ in $(seq 1 10); do
    if [[ -n "$exclude_regex" ]]; then
      latest_jsonl="$(ls -1t /tmp/fastapi_logs/*.jsonl 2>/dev/null | grep -Ev "$exclude_regex" | head -n 1 || true)"
    else
      latest_jsonl="$(ls -1t /tmp/fastapi_logs/*.jsonl 2>/dev/null | head -n 1 || true)"
    fi
    if [[ -n "$latest_jsonl" && -s "$latest_jsonl" ]]; then
      return 0
    fi
    sleep 0.5
  done
  return 0
}

if [[ ! -f /tmp/fastapi.ini ]]; then
  echo "fastapi.ini not found" >&2
  exit 61
fi

ports=$(grep 'port = ' /tmp/fastapi.ini | awk -F '= ' '{print $2}' || true)
port=$(echo "$ports" | tail -n 1 | xargs || true)
pids=$(grep 'pid = ' /tmp/fastapi.ini | awk -F '= ' '{print $2}' || true)

api_key="${1:-}"
model_id="${2:-}"
workdir="${3:-/testbed}"
timeout_sec="${4:-}"
openclaw_model_api="${SWE_OPENCLAW_MODEL_API:-${OPENCLAW_MODEL_API:-openai-responses}}"
openclaw_model_api="$(echo "${openclaw_model_api}" | tr '[:upper:]' '[:lower:]')"
case "$openclaw_model_api" in
  openai-responses|openai-completions|anthropic-messages)
    ;;
  *)
    openclaw_model_api="openai-responses"
    ;;
esac
openclaw_provider="${SWE_OPENCLAW_PROVIDER:-}"
openclaw_provider="$(echo "${openclaw_provider}" | tr '[:upper:]' '[:lower:]')"
if [[ -z "$openclaw_provider" ]]; then
  if [[ "$openclaw_model_api" == "anthropic-messages" ]]; then
    openclaw_provider="anthropic"
  else
    openclaw_provider="openai"
  fi
fi
if [[ "$openclaw_model_api" == "anthropic-messages" ]]; then
  openclaw_provider="anthropic"
fi
if [[ "$openclaw_provider" != "openai" && "$openclaw_provider" != "anthropic" ]]; then
  openclaw_provider="openai"
fi
export SWE_OPENCLAW_MODEL_API="$openclaw_model_api"
export SWE_OPENCLAW_PROVIDER="$openclaw_provider"

if [[ -z "$api_key" || -z "$model_id" ]]; then
  echo "missing api_key or model_id" >&2
  exit 62
fi

if [[ -z "$port" ]]; then
  echo "fastapi port not found" >&2
  exit 63
fi

prompt_file="/tmp/OPENCLAW.md"
if [[ ! -f "$prompt_file" ]]; then
  echo "prompt file not found: ${prompt_file}" >&2
  exit 65
fi

mkdir -p /tmp/fastapi_logs
mkdir -p "$workdir"
cd "$workdir"

OPENCLAW_BIN="/tmp/openclaw_wrapper"
if [[ ! -x "$OPENCLAW_BIN" ]]; then
  echo "openclaw executable not found: ${OPENCLAW_BIN}. Only the fixed /tmp runtime upload is allowed." >&2
  exit 64
fi

if [[ -z "${OPENCLAW_SESSION_ID:-}" ]]; then
  echo "OPENCLAW_SESSION_ID not set" >&2
  exit 67
fi

proxy_port_file="/tmp/openclaw_proxy_port.txt"
proxy_pid_file="/tmp/openclaw_proxy_pid.txt"
rm -f "$proxy_port_file" "$proxy_pid_file" || true

if [[ "$openclaw_model_api" == "anthropic-messages" ]]; then
  # Native Anthropic Messages path: send requests directly to FastAPI /v1/messages.
  base_url="http://127.0.0.1:${port}"
else
  if [[ ! -f /tmp/openclaw_models_proxy.py ]]; then
    echo "openclaw models proxy script missing: /tmp/openclaw_models_proxy.py" >&2
    exit 66
  fi

  upstream_base="http://127.0.0.1:${port}"
  nohup python3 /tmp/openclaw_models_proxy.py \
    --upstream "$upstream_base" \
    --model-id "$model_id" \
    --port-file "$proxy_port_file" \
    --pid-file "$proxy_pid_file" \
    --state-file /tmp/openclaw_chat_state.json \
    --listen-host 127.0.0.1 \
    --listen-port 0 \
    > /tmp/fastapi_logs/openclaw_models_proxy.log 2>&1 & disown

  proxy_port=""
  for _ in $(seq 1 300); do
    if [[ -f "$proxy_port_file" ]]; then
      proxy_port="$(cat "$proxy_port_file" 2>/dev/null | head -n 1 | xargs || true)"
      break
    fi
    sleep 0.1
  done

  if [[ -z "$proxy_port" ]]; then
    echo "failed to start openclaw models proxy (no port file)" >&2
    tail -n 200 /tmp/fastapi_logs/openclaw_models_proxy.log >&2 || true
    exit 67
  fi
  # Use 127.0.0.1 instead of localhost to avoid IPv6 ::1 resolution issues.
  base_url="http://127.0.0.1:${proxy_port}/v1"
fi

# Isolate OpenClaw mutable state (sessions, models.json, caches).
OPENCLAW_STATE_DIR="/tmp/openclaw_state"
OPENCLAW_CONFIG_PATH="${OPENCLAW_STATE_DIR}/openclaw.json"
mkdir -p "$OPENCLAW_STATE_DIR"
export OPENCLAW_API_KEY="${OPENCLAW_API_KEY:-$api_key}"

python3 - <<PY
import json
import os

workdir = os.environ.get("OPENCLAW_WORKDIR", "${workdir}")
base_url = os.environ.get("OPENCLAW_BASE_URL", "${base_url}")
model_id = os.environ.get("OPENCLAW_MODEL_ID", "${model_id}")
api_key = os.environ.get("OPENCLAW_API_KEY", "")
model_api = os.environ.get("SWE_OPENCLAW_MODEL_API", os.environ.get("OPENCLAW_MODEL_API", "openai-responses"))
model_api = (model_api or "").strip() or "openai-responses"
if model_api not in {"openai-responses", "openai-completions", "anthropic-messages"}:
  model_api = "openai-responses"
provider_id = (os.environ.get("SWE_OPENCLAW_PROVIDER", "") or "").strip().lower()
if provider_id not in {"openai", "anthropic"}:
  provider_id = "anthropic" if model_api == "anthropic-messages" else "openai"
if model_api == "anthropic-messages":
  provider_id = "anthropic"
context_window_raw = os.environ.get("OPENCLAW_CONTEXT_WINDOW", "262144")
max_tokens_raw = os.environ.get("OPENCLAW_MAX_TOKENS", "16384")
try:
  context_window = max(1024, int(str(context_window_raw).strip()))
except Exception:
  context_window = 262144
try:
  max_tokens = max(1, int(str(max_tokens_raw).strip()))
except Exception:
  max_tokens = 16384

primary_provider = "anthropic" if provider_id == "anthropic" else "openai"
model_key = f"{primary_provider}/{model_id}"
provider_cfg = {
  "baseUrl": base_url,
  "api": model_api,
  "models": [
    {
      "id": model_id,
      "name": model_id,
      "api": model_api,
      "reasoning": True,
      "input": ["text"],
      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
      "contextWindow": context_window,
      "maxTokens": max_tokens,
    }
  ],
}
if provider_id == "anthropic":
  provider_cfg["apiKey"] = api_key

cfg = {
  "agents": {
    "defaults": {
      # Use the StepTron workspace as agent cwd.
      "workspace": workdir,
      # Avoid writing BOOTSTRAP.md/TOOLS.md/etc into the task workspace by default.
      "skipBootstrap": True,
      # Always specify provider to avoid deprecated implicit behavior.
      "model": {"primary": model_key},
      # OpenClaw extra params resolve from agents.defaults.models.<provider/model>.params.
      # This path is required for OpenAI providers to propagate maxTokens into request payload.
      "models": {
        model_key: {
          "params": {
            "maxTokens": max_tokens,
          }
        }
      },
    }
  },
  "models": {
    "mode": "replace",
    "providers": {
      # Route model traffic through local FastAPI (or models-proxy, depending on mode).
      provider_id: provider_cfg
    },
  },
  "tools": {
    # Provide the standard coding tool surface.
    "profile": "coding",
    # Disable interactive exec approvals; StepTron runs must be non-interactive.
    "exec": {"security": "full", "ask": "off", "host": "sandbox"},
  },
}

path = "${OPENCLAW_CONFIG_PATH}"
with open(path, "w", encoding="utf-8") as f:
  json.dump(cfg, f, indent=2, ensure_ascii=False)
  f.write("\\n")
PY

export OPENCLAW_STATE_DIR="$OPENCLAW_STATE_DIR"
export OPENCLAW_CONFIG_PATH="$OPENCLAW_CONFIG_PATH"

# Use standard provider auth headers; FastAPI forwards them unchanged.
export OPENAI_API_KEY="${OPENAI_API_KEY:-$api_key}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$api_key}"

# OpenClaw can load a lot of integrations; skip channel registration for speed.
export OPENCLAW_SKIP_CHANNELS="${OPENCLAW_SKIP_CHANNELS:-1}"
export CLAWDBOT_SKIP_CHANNELS="${CLAWDBOT_SKIP_CHANNELS:-1}"

OPENCLAW_MESSAGE="$(cat "$prompt_file")"

thinking_level="${OPENCLAW_THINKING:-${SWE_OPENCLAW_THINKING:-high}}"
thinking_level="${thinking_level,,}"
case "$thinking_level" in
  off|minimal|low|medium|high)
    ;;
  *)
    echo "invalid OPENCLAW_THINKING=${thinking_level}, fallback to high" >&2
    thinking_level="high"
    ;;
esac

args=(
  agent
  --local
  --agent main
  --session-id "${OPENCLAW_SESSION_ID}"
  --message "${OPENCLAW_MESSAGE}"
  --thinking "${thinking_level}"
  --json
)
if [[ -n "$timeout_sec" ]]; then
  args+=(--timeout "$timeout_sec")
fi

set +e
"$OPENCLAW_BIN" "${args[@]}"
cmd_rc=$?
set -e

# FastAPI writes request/response records asynchronously; allow a short
# grace period before killing proxy processes to avoid truncated logs.
wait_for_fastapi_log_flush "responses_state\\.jsonl"

if [[ -f "$proxy_pid_file" ]]; then
  proxy_pid="$(cat "$proxy_pid_file" 2>/dev/null | head -n 1 | xargs || true)"
  if [[ -n "$proxy_pid" ]]; then
    kill -9 "$proxy_pid" || :
  fi
fi

if [[ -n "$pids" ]]; then
  for pi in $pids; do
    kill -9 "$pi" || :
  done
fi

if [[ $cmd_rc -ne 0 ]]; then
  exit "$cmd_rc"
fi
