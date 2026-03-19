#!/usr/bin/env bash
set -euo pipefail

source /data/conda/ourconda_bashrc
conda activate evomaster_fs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_NAME="${RUN_NAME:-DeepSeek-v3.2_4}"
AGENT_NAME="${AGENT_NAME:-frontierscience}"
JSONL_PATH="${JSONL_PATH:-playground/frontierscience/test/test.jsonl}"
CONFIG_PATH="${CONFIG_PATH:-configs/frontierscience/config.yaml}"
RUN_WORKERS="${RUN_WORKERS:-20}"
EVAL_WORKERS="${EVAL_WORKERS:-20}"
EVAL_REPEAT_COUNT="${EVAL_REPEAT_COUNT:-5}"
EVAL_MODEL="${EVAL_MODEL:-Vendor2/GPT-5}"
EVAL_BASE_URL="${EVAL_BASE_URL:-https://api.gpugeek.com/v1}"
EVAL_API_KEY="${EVAL_API_KEY:-009njbjuxu4q9001000degiwa1xfovvi008wbdqe}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-128000}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-600}"
EVAL_RETRIES="${EVAL_RETRIES:-2}"
EVAL_PASS_THRESHOLD="${EVAL_PASS_THRESHOLD:-7.0}"
EVAL_REASONING_EFFORT="${EVAL_REASONING_EFFORT:-high}"

BASE_RUN_DIR="runs/${RUN_NAME}"
EVAL_DIR="${BASE_RUN_DIR}/eval"
EVAL_REPEAT_DIR="${BASE_RUN_DIR}/eval1"
LOG_DIR="${BASE_RUN_DIR}/logs"
SUMMARY_LOG="${LOG_DIR}/workflow.log"

mkdir -p "${BASE_RUN_DIR}" "${EVAL_DIR}" "${EVAL_REPEAT_DIR}" "${LOG_DIR}"

if [[ -z "${EVAL_API_KEY}" ]]; then
  echo "[ERROR] Missing EVAL_API_KEY/JUDGE_MODEL_APIKEY/OPENAI_API_KEY" >&2
  exit 1
fi

run_eval_batch() {
  local input_path="$1"
  local prefix="$2"

  for ((i=1; i<=EVAL_REPEAT_COUNT; i++)); do
    local output_path="${EVAL_REPEAT_DIR}/${prefix}${i}.jsonl"
    echo "[EVAL] ${input_path} -> ${output_path} (${i}/${EVAL_REPEAT_COUNT})" | tee -a "${SUMMARY_LOG}"
    python playground/frontierscience/scripts/eval.py \
      --input "${input_path}" \
      --output "${output_path}" \
      --workers "${EVAL_WORKERS}" \
      --model "${EVAL_MODEL}" \
      --base-url "${EVAL_BASE_URL}" \
      --api-key "${EVAL_API_KEY}" \
      --max-tokens "${EVAL_MAX_TOKENS}" \
      --timeout "${EVAL_TIMEOUT}" \
      --retries "${EVAL_RETRIES}" \
      --pass-threshold "${EVAL_PASS_THRESHOLD}" \
      --reasoning-effort "${EVAL_REASONING_EFFORT}" \
      2>&1 | tee -a "${SUMMARY_LOG}"
  done
}

{
  echo "============================================================"
  echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
  echo "[INFO] RUN_NAME=${RUN_NAME}"
  echo "[INFO] JSONL_PATH=${JSONL_PATH}"
  echo "[INFO] BASE_RUN_DIR=${BASE_RUN_DIR}"
  echo "[INFO] EVAL_REPEAT_COUNT=${EVAL_REPEAT_COUNT}"
  echo "============================================================"
} | tee "${SUMMARY_LOG}"

echo "[STEP 1/4] run" | tee -a "${SUMMARY_LOG}"
python playground/frontierscience/scripts/run.py \
  --jsonl "${JSONL_PATH}" \
  --agent "${AGENT_NAME}" \
  --config "${CONFIG_PATH}" \
  --base-run-dir "${BASE_RUN_DIR}" \
  --workers "${RUN_WORKERS}" \
  2>&1 | tee -a "${SUMMARY_LOG}"

echo "[STEP 2/4] merge" | tee -a "${SUMMARY_LOG}"
python playground/frontierscience/scripts/merge.py \
  --jsonl "${JSONL_PATH}" \
  --runs-dir "${BASE_RUN_DIR}" \
  --output-dir "${EVAL_DIR}" \
  2>&1 | tee -a "${SUMMARY_LOG}"

echo "[STEP 3/4] eval x${EVAL_REPEAT_COUNT}" | tee -a "${SUMMARY_LOG}"
run_eval_batch "${EVAL_DIR}/solution.jsonl" "solution_scored"
run_eval_batch "${EVAL_DIR}/solution_refined.jsonl" "solution_refined_scored"

echo "[STEP 4/4] summarize" | tee -a "${SUMMARY_LOG}"
python playground/frontierscience/scripts/summarize.py \
  --runs-dir "${BASE_RUN_DIR}" \
  --eval-repeat-dir "${EVAL_REPEAT_DIR}" \
  2>&1 | tee -a "${SUMMARY_LOG}"

echo "[DONE] workflow finished: ${BASE_RUN_DIR}" | tee -a "${SUMMARY_LOG}"
