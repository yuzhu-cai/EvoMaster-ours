#!/usr/bin/env bash
set -euo pipefail

# One-shot FrontierScience pipeline:
# 1) run_test_jsonl.py
# 2) merge_solution_jsonl.py
# 3) eval.py
#
# Usage:
#   bash playground/frontierscience/scripts/run_full_pipeline.sh
#   bash playground/frontierscience/scripts/run_full_pipeline.sh --jsonl /path/to/test.jsonl --workers 4
#
# Optional environment variables:
#   PYTHON_BIN            Python executable (default: python3)
#   AGENT_NAME            run.py --agent value (default: frontierscience)
#   CONFIG_PATH           Optional config path for run_test_jsonl.py --config
#   RUN_WORKERS           Worker count for run_test_jsonl.py (default: 1)
#   RUN_LINES             Optional line selector, e.g. "2,5,8-10"
#   RUN_LIMIT             Optional max rows
#   RUN_START_LINE        Optional start line (default: 1)
#   RUN_STOP_ON_ERROR     If "1", pass --stop-on-error
#
#   EVAL_WORKERS          Worker count for eval.py (default: 4)
#   EVAL_MODEL            eval.py --model
#   EVAL_BASE_URL         eval.py --base-url
#   EVAL_API_KEY          eval.py --api-key
#   EVAL_MAX_TOKENS       eval.py --max-tokens
#   EVAL_TIMEOUT          eval.py --timeout
#   EVAL_RETRIES          eval.py --retries
#   EVAL_PASS_THRESHOLD   eval.py --pass-threshold (default: 7.0)
#   EVAL_REASONING_EFFORT eval.py --reasoning-effort (default: high)
#   EVAL_KEEP_REASONING   If "1", pass --keep-reasoning

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
AGENT_NAME="${AGENT_NAME:-frontierscience}"
RUN_WORKERS="${RUN_WORKERS:-10}"
RUN_START_LINE="${RUN_START_LINE:-1}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
EVAL_PASS_THRESHOLD="${EVAL_PASS_THRESHOLD:-7.0}"
EVAL_REASONING_EFFORT="${EVAL_REASONING_EFFORT:-high}"

JSONL_PATH="${PROJECT_ROOT}/playground/frontierscience/test/test.jsonl"
BASE_RUN_DIR="${PROJECT_ROOT}/runs/frontierscience_jsonl_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jsonl)
      JSONL_PATH="$2"
      shift 2
      ;;
    --base-run-dir)
      BASE_RUN_DIR="$2"
      shift 2
      ;;
    --workers)
      RUN_WORKERS="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      echo "Use --help for usage." >&2
      exit 1
      ;;
  esac
done

RUN_SCRIPT="${SCRIPT_DIR}/run_test_jsonl.py"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_solution_jsonl.py"
EVAL_SCRIPT="${SCRIPT_DIR}/eval.py"

for p in "$RUN_SCRIPT" "$MERGE_SCRIPT" "$EVAL_SCRIPT"; do
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] Script not found: $p" >&2
    exit 1
  fi
done

if [[ ! -f "$JSONL_PATH" ]]; then
  echo "[ERROR] JSONL not found: $JSONL_PATH" >&2
  exit 1
fi

mkdir -p "$BASE_RUN_DIR"
SOLUTION_JSONL="${BASE_RUN_DIR}/solution.jsonl"
SCORED_JSONL="${BASE_RUN_DIR}/solution_scored.jsonl"

echo "============================================================"
echo "[PIPELINE] project_root: $PROJECT_ROOT"
echo "[PIPELINE] jsonl:        $JSONL_PATH"
echo "[PIPELINE] run_dir:      $BASE_RUN_DIR"
echo "============================================================"

run_cmd=(
  "$PYTHON_BIN" "$RUN_SCRIPT"
  --jsonl "$JSONL_PATH"
  --agent "$AGENT_NAME"
  --base-run-dir "$BASE_RUN_DIR"
  --workers "$RUN_WORKERS"
  --start-line "$RUN_START_LINE"
)
if [[ -n "${CONFIG_PATH:-}" ]]; then
  run_cmd+=(--config "$CONFIG_PATH")
fi
if [[ -n "${RUN_LINES:-}" ]]; then
  run_cmd+=(--lines "$RUN_LINES")
fi
if [[ -n "${RUN_LIMIT:-}" ]]; then
  run_cmd+=(--limit "$RUN_LIMIT")
fi
if [[ "${RUN_STOP_ON_ERROR:-0}" == "1" ]]; then
  run_cmd+=(--stop-on-error)
fi

echo "[STEP 1/3] run_test_jsonl.py"
"${run_cmd[@]}"

echo "[STEP 2/3] merge_solution_jsonl.py"
merge_cmd=(
  "$PYTHON_BIN" "$MERGE_SCRIPT"
  --jsonl "$JSONL_PATH"
  --runs-dir "$BASE_RUN_DIR"
  --output "$SOLUTION_JSONL"
)
"${merge_cmd[@]}"

echo "[STEP 3/3] eval.py"
eval_cmd=(
  "$PYTHON_BIN" "$EVAL_SCRIPT"
  --input "$SOLUTION_JSONL"
  --output "$SCORED_JSONL"
  --workers "$EVAL_WORKERS"
  --pass-threshold "$EVAL_PASS_THRESHOLD"
  --reasoning-effort "$EVAL_REASONING_EFFORT"
)
if [[ -n "${EVAL_MODEL:-}" ]]; then
  eval_cmd+=(--model "$EVAL_MODEL")
fi
if [[ -n "${EVAL_BASE_URL:-}" ]]; then
  eval_cmd+=(--base-url "$EVAL_BASE_URL")
fi
if [[ -n "${EVAL_API_KEY:-}" ]]; then
  eval_cmd+=(--api-key "$EVAL_API_KEY")
fi
if [[ -n "${EVAL_MAX_TOKENS:-}" ]]; then
  eval_cmd+=(--max-tokens "$EVAL_MAX_TOKENS")
fi
if [[ -n "${EVAL_TIMEOUT:-}" ]]; then
  eval_cmd+=(--timeout "$EVAL_TIMEOUT")
fi
if [[ -n "${EVAL_RETRIES:-}" ]]; then
  eval_cmd+=(--retries "$EVAL_RETRIES")
fi
if [[ "${EVAL_KEEP_REASONING:-0}" == "1" ]]; then
  eval_cmd+=(--keep-reasoning)
fi

"${eval_cmd[@]}"

echo "============================================================"
echo "[DONE] Pipeline finished."
echo "[DONE] Run dir:         $BASE_RUN_DIR"
echo "[DONE] Merged solution: $SOLUTION_JSONL"
echo "[DONE] Scored output:   $SCORED_JSONL"
echo "============================================================"
