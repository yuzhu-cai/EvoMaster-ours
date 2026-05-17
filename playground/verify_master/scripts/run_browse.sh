#!/bin/bash
# VerifyMaster evaluation workflow script
#
# Usage:
#   IDS="0-9" RUN_NAME="verify_test" ./playground/verify_master/scripts/run_browse.sh

set -e

IDS="${IDS:-0,1,3-9}"
RUN_NAME="${RUN_NAME:-verify_master_test3}"
RUN_WORKERS="${RUN_WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
DATA_JSON="${DATA_JSON:-playground/verify_master/test/browsecomp_decrypted.json}"

RUN_DIR="runs/${RUN_NAME}"
RESULTS_DIR="${RUN_DIR}/results"
LOG_FILE="${RUN_DIR}/workflow.log"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

mkdir -p "${RUN_DIR}" "${RESULTS_DIR}"

echo "========================================" | tee -a "${LOG_FILE}"
echo "VerifyMaster Evaluation Workflow" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "RUN_NAME:     ${RUN_NAME}" | tee -a "${LOG_FILE}"
echo "IDS:          ${IDS}" | tee -a "${LOG_FILE}"
echo "RUN_WORKERS:  ${RUN_WORKERS}" | tee -a "${LOG_FILE}"
echo "EVAL_WORKERS: ${EVAL_WORKERS}" | tee -a "${LOG_FILE}"
echo "DATA_JSON:    ${DATA_JSON}" | tee -a "${LOG_FILE}"
echo "RUN_DIR:      ${RUN_DIR}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "[Step 1/4] Running batch inference..." | tee -a "${LOG_FILE}"
python "${PROJECT_ROOT}/playground/verify_master/scripts/run_batch.py" \
    --json "${DATA_JSON}" \
    --lines "${IDS}" \
    --run-dir "${RUN_DIR}" \
    --workers "${RUN_WORKERS}"
echo "[Step 1/4] Done" | tee -a "${LOG_FILE}"

# echo "" | tee -a "${LOG_FILE}"
# echo "[Step 2/4] Merging results..." | tee -a "${LOG_FILE}"
# python "${PROJECT_ROOT}/playground/verify_master/scripts/merge.py" \
#     --json "${DATA_JSON}" \
#     --run-dir "${RUN_DIR}" \
#     --output-dir "${RESULTS_DIR}"
# echo "[Step 2/4] Done" | tee -a "${LOG_FILE}"

# echo "" | tee -a "${LOG_FILE}"
# echo "[Step 3/4] Evaluating with LLM..." | tee -a "${LOG_FILE}"
# python "${PROJECT_ROOT}/playground/verify_master/scripts/eval.py" \
#     --input "${RESULTS_DIR}/merge.jsonl" \
#     --output "${RESULTS_DIR}/eval.jsonl" \
#     --workers "${EVAL_WORKERS}" \
#     --model "Vendor2/GPT-5.4"
# echo "[Step 3/4] Done" | tee -a "${LOG_FILE}"

# echo "" | tee -a "${LOG_FILE}"
# echo "[Step 4/4] Summarizing..." | tee -a "${LOG_FILE}"
# python "${PROJECT_ROOT}/playground/verify_master/scripts/summarize.py" \
#     --jsonl "${RESULTS_DIR}/eval.jsonl" \
#     --result "${RESULTS_DIR}/results.json"
# echo "[Step 4/4] Done" | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "Workflow complete!" | tee -a "${LOG_FILE}"
echo "Results: ${RESULTS_DIR}/results.json" | tee -a "${LOG_FILE}"
echo "Log:     ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
