#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

SPLIT="${1:-debug}"
CONFIG="${2:-configs/paperbench_codedev_agent/solve.yaml}"
RUN_DIR="${3:-runs/paperbench_codedev_${SPLIT}_$(date -u +%Y%m%dT%H%M%SZ)}"
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

mkdir -p "$RUN_DIR"
TASK_FILE="$RUN_DIR/tasks_${SPLIT}.json"
python playground/paperbench_codedev_agent/scripts/make_tasks.py \
  --split "$SPLIT" \
  --output "$TASK_FILE"

python run.py \
  --agent paperbench_codedev_agent \
  --config "$CONFIG" \
  --task-file "$TASK_FILE" \
  --parallel \
  --max-workers "$MAX_WORKERS" \
  --run-dir "$RUN_DIR"

