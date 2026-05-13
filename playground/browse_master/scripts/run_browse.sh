#!/bin/bash
# BrowseMaster evaluation workflow script
#
# Usage:
#   IDS="0-9" RUN_NAME="test_run" ./playground/browse_master/scripts/run_browse.sh
#
# Required environment variables (or edit defaults below):
#   IDS          - ID ranges to evaluate, e.g., "0", "0-9", "0,5,10"
#   RUN_NAME     - Name of this run (used for directory naming)
#   RUN_WORKERS  - Parallel workers for run_batch.py (default: 4)
#   EVAL_WORKERS - Parallel workers for eval.py (default: 4)
#   DATA_JSON    - Path to dataset JSON file

set -e

# Configuration defaults
IDS="${IDS:-0-1266}"
RUN_NAME="${RUN_NAME:-browse_dsv4pro}"
RUN_WORKERS="${RUN_WORKERS:-30}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
DATA_JSON="${DATA_JSON:-playground/browse_master/test/browsecomp_decrypted.json}"

RUN_DIR="runs/${RUN_NAME}"
RESULTS_DIR="${RUN_DIR}/results"
LOG_FILE="${RUN_DIR}/workflow.log"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

mkdir -p "${RUN_DIR}" "${RESULTS_DIR}"

echo "========================================" | tee -a "${LOG_FILE}"
echo "BrowseMaster Evaluation Workflow" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "RUN_NAME:     ${RUN_NAME}" | tee -a "${LOG_FILE}"
echo "IDS:          ${IDS}" | tee -a "${LOG_FILE}"
echo "RUN_WORKERS:  ${RUN_WORKERS}" | tee -a "${LOG_FILE}"
echo "EVAL_WORKERS: ${EVAL_WORKERS}" | tee -a "${LOG_FILE}"
echo "DATA_JSON:    ${DATA_JSON}" | tee -a "${LOG_FILE}"
echo "RUN_DIR:      ${RUN_DIR}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# Step 1: Run batch
echo "" | tee -a "${LOG_FILE}"
echo "[Step 1/4] Running batch inference..." | tee -a "${LOG_FILE}"
python "${PROJECT_ROOT}/playground/browse_master/scripts/run_batch.py" \
    --json "${DATA_JSON}" \
    --lines "${IDS}" \
    --run-dir "${RUN_DIR}" \
    --workers "${RUN_WORKERS}"
echo "[Step 1/4] Done" | tee -a "${LOG_FILE}"

# Extract final answers into task_xxxx/solution.txt files.
echo "" | tee -a "${LOG_FILE}"
echo "[Post-run] Extracting final solutions..." | tee -a "${LOG_FILE}"
python "${PROJECT_ROOT}/playground/browse_master/scripts/extract_browse_solutions.py" \
    "${RUN_DIR}"
echo "[Post-run] Done" | tee -a "${LOG_FILE}"

# Step 2: Merge results
# echo "" | tee -a "${LOG_FILE}"
# echo "[Step 2/4] Merging results..." | tee -a "${LOG_FILE}"
# python "${PROJECT_ROOT}/playground/browse_master/scripts/merge.py" \
#     --json "${DATA_JSON}" \
#     --run-dir "${RUN_DIR}" \
#     --output-dir "${RESULTS_DIR}"
# echo "[Step 2/4] Done" | tee -a "${LOG_FILE}"

# # Step 3: Evaluate
# echo "" | tee -a "${LOG_FILE}"
# echo "[Step 3/4] Evaluating with LLM..." | tee -a "${LOG_FILE}"
# python "${PROJECT_ROOT}/playground/browse_master/scripts/eval.py" \
#     --input "${RESULTS_DIR}/merge.jsonl" \
#     --output "${RESULTS_DIR}/eval.jsonl" \
#     --workers "${EVAL_WORKERS}"\
#     --model "Vendor2/GPT-5.4"
# echo "[Step 3/4] Done" | tee -a "${LOG_FILE}"

# # Step 4: Summarize
# echo "" | tee -a "${LOG_FILE}"
# echo "[Step 4/4] Summarizing..." | tee -a "${LOG_FILE}"
# python "${PROJECT_ROOT}/playground/browse_master/scripts/summarize.py" \
#     --jsonl "${RESULTS_DIR}/eval.jsonl" \
#     --result "${RESULTS_DIR}/results.json"
# echo "[Step 4/4] Done" | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "Workflow complete!" | tee -a "${LOG_FILE}"
echo "Results: ${RESULTS_DIR}/results.json" | tee -a "${LOG_FILE}"
echo "Log:     ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
