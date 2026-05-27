#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PAPER_ID="${1:-rice}"
CONFIG="${2:-configs/paperbench_codedev_agent/config.yaml}"
RUN_DIR="${3:-runs/paperbench_codedev_${PAPER_ID}_$(date -u +%Y%m%dT%H%M%SZ)}"
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
TASK_FILE="$RUN_DIR/tasks_${PAPER_ID}.json"
python - "$TASK_FILE" "$PAPER_ID" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
paper_id = sys.argv[2]
paperbench_root = Path("/data/yuzhu/Devs/third_party/frontier-evals/project/paperbench")
paper_dir = paperbench_root / "data" / "papers" / paper_id
if not paper_dir.is_dir():
    raise SystemExit(f"paper directory not found: {paper_dir}")
tasks = [{
    "id": paper_id,
    "description": json.dumps({
        "paper_id": paper_id,
        "paper_dir": str(paper_dir),
        "paperbench_root": str(paperbench_root),
        "description": f"Reproduce PaperBench Code-Dev paper {paper_id}.",
    }),
}]
path.write_text(json.dumps(tasks, indent=2) + "\n")
PY

python run.py \
  --agent paperbench_codedev_agent \
  --config "$CONFIG" \
  --task-file "$TASK_FILE" \
  --run-dir "$RUN_DIR"
