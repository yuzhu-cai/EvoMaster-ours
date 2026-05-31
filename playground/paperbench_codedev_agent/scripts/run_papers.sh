#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PAPERS="${1:?comma-separated paper ids or path to a one-id-per-line file}"
CONFIG="${2:-configs/paperbench_codedev_agent/competitive.yaml}"
RUN_DIR="${3:-runs/paperbench_codedev_papers_$(date -u +%Y%m%dT%H%M%SZ)}"
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
TASK_FILE="$RUN_DIR/tasks_selected.json"
if [[ -f "$PAPERS" ]]; then
  python playground/paperbench_codedev_agent/scripts/make_tasks.py \
    --papers-file "$PAPERS" \
    --output "$TASK_FILE"
else
  IFS=',' read -r -a PAPER_ARRAY <<< "$PAPERS"
  args=()
  for paper_id in "${PAPER_ARRAY[@]}"; do
    [[ -n "$paper_id" ]] && args+=(--paper-id "$paper_id")
  done
  python playground/paperbench_codedev_agent/scripts/make_tasks.py \
    "${args[@]}" \
    --output "$TASK_FILE"
fi

finish_marker() {
  local status="$1"
  python - "$RUN_DIR" "$status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
payload = {
    "split": "selected",
    "status": sys.argv[2],
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
