#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

SPLIT="${1:-debug}"
CONFIG="${2:-configs/paperbench_codedev_agent/solve.yaml}"
RUN_ROOT="${PAPERBENCH_CODEDEV_RUN_ROOT:-runs/evomaster4paperbench}"
RUN_DIR="${3:-${RUN_ROOT}/generation/full/paperbench_codedev_${SPLIT}_$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_WORKERS="${4:-1}"
MODEL="${PAPERBENCH_CODEDEV_MODEL:-ksyun/gpt-5.4}"
ENV_FILE="${PAPERBENCH_CODEDEV_ENV:-$ROOT_DIR/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

export GPT_CHAT_MODEL="$MODEL"
export EVOMASTER_LLM_MIN_INTERVAL_SECONDS="${EVOMASTER_LLM_MIN_INTERVAL_SECONDS:-6}"
export EVOMASTER_LLM_RATE_LIMIT_JITTER_SECONDS="${EVOMASTER_LLM_RATE_LIMIT_JITTER_SECONDS:-1.0}"
export EVOMASTER_LLM_429_COOLDOWN_SECONDS="${EVOMASTER_LLM_429_COOLDOWN_SECONDS:-45}"
export EVOMASTER_LLM_RATE_LIMIT_KEY="${EVOMASTER_LLM_RATE_LIMIT_KEY:-paperbench-${MODEL}}"

mkdir -p "$RUN_DIR"
TASK_FILE="$RUN_DIR/tasks_${SPLIT}.json"
python playground/paperbench_codedev_agent/scripts/make_tasks.py \
  --split "$SPLIT" \
  --output "$TASK_FILE"

finish_marker() {
  local status="$1"
  python - "$RUN_DIR" "$SPLIT" "$status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
payload = {
    "split": sys.argv[2],
    "status": sys.argv[3],
    "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
(run_dir / "EVOMASTER_RUN_FINISHED.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
}

trap 'finish_marker interrupted; exit 130' INT TERM HUP
set +e
python run.py \
  --agent paperbench_codedev_agent \
  --config "$CONFIG" \
  --task-file "$TASK_FILE" \
  --parallel \
  --max-workers "$MAX_WORKERS" \
  --run-dir "$RUN_DIR"
rc=$?
set -e
finish_marker "$([[ "$rc" == 0 ]] && echo completed || echo failed)"
exit "$rc"
