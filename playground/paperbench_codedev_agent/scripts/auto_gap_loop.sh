#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

BASE_FINAL_GRADE_RUN="${1:?final grade-run directory to wait for, containing summary.json}"
OUT_PREFIX="${2:-paperbench_codedev_auto_gap_$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_PAPERS="${3:-8}"
MAX_WORKERS="${4:-8}"
CONFIG="${PAPERBENCH_CODEDEV_CONFIG:-configs/paperbench_codedev_agent/competitive.yaml}"
MODEL="${PAPERBENCH_CODEDEV_MODEL:-ksyun/gpt-5.4}"
CODEX_SUMMARY="${CODEX_GPT54_SUMMARY:-runs/codex4paperbench/codex_gpt54_regen_regrade_crs_gpt55_responses_medium_c4x40_20260527T071410Z/summary.json}"
CODEX_GRADE="${CODEX_GPT54_GRADE_RUN:-$(dirname "$CODEX_SUMMARY")}"
OLD1="${PAPERBENCH_OLD_GRADE_RUN_1:-runs/paperbench_codedev_regrade_crs_gpt55_responses_c4x40_20260526T162828Z}"
OLD2="${PAPERBENCH_OLD_GRADE_RUN_2:-runs/paperbench_codedev_combined_regrade_crs_gpt55_responses_c4x40_20260527T142508Z}"
MAX_ROUNDS="${PAPERBENCH_AUTO_GAP_MAX_ROUNDS:-3}"

log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

log "waiting for base final summary: $BASE_FINAL_GRADE_RUN/summary.json"
while [[ ! -f "$BASE_FINAL_GRADE_RUN/summary.json" ]]; do
  done_count=$(find "$BASE_FINAL_GRADE_RUN" -mindepth 2 -maxdepth 2 -name grader_output.json 2>/dev/null | wc -l || true)
  log "base final summary not ready yet outputs=$done_count"
  sleep 300
done

COMPARE_JSON="$BASE_FINAL_GRADE_RUN/compare_to_codex.json"
if python playground/paperbench_codedev_agent/scripts/compare_to_codex.py \
    --evomaster-summary "$BASE_FINAL_GRADE_RUN/summary.json" \
    --codex-summary "$CODEX_SUMMARY" \
    --out "$COMPARE_JSON"; then
  log "EvoMaster already beats Codex; no gap rerun needed."
  exit 0
fi

CURRENT_BEST="$BASE_FINAL_GRADE_RUN"
EXTRA_GRADE_RUNS=()
for round in $(seq 1 "$MAX_ROUNDS"); do
  GAP_DIR="runs/${OUT_PREFIX}_round${round}_gap_plan"
  GAP_FILE="$GAP_DIR/gap_papers.txt"
  GAP_JSON="$GAP_DIR/gap_papers.json"
  mkdir -p "$GAP_DIR"
  if ! python playground/paperbench_codedev_agent/scripts/select_gap_papers.py \
      --evomaster-summary "$CURRENT_BEST/summary.json" \
      --codex-summary "$CODEX_SUMMARY" \
      --max-papers "$MAX_PAPERS" \
      --out "$GAP_FILE" \
      --json-out "$GAP_JSON"; then
    log "No eligible gap papers selected at round $round."
    exit 0
  fi
  log "round $round selected gap papers: $(tr '\n' ',' < "$GAP_FILE" | sed 's/,$//')"

  GEN_RUN="runs/${OUT_PREFIX}_round${round}_generation"
  GRADE_RUN="runs/${OUT_PREFIX}_round${round}_grade_crs_gpt55"
  BEST_RUN="runs/${OUT_PREFIX}_round${round}_bestof_crs_gpt55"
  ROUND_CONFIG="$GAP_DIR/round_config.yaml"

  python - "$CONFIG" "$ROUND_CONFIG" "$CODEX_GRADE" "$CURRENT_BEST" "$OLD1" "$OLD2" "${EXTRA_GRADE_RUNS[@]}" <<'PY'
import sys
from pathlib import Path
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
runs = []
for raw in sys.argv[3:]:
    if raw and Path(raw).exists():
        runs.append(raw)

cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
pb = cfg.setdefault("paperbench_codedev", {})

def extend_unique(container, key):
    existing = container.get(key) or []
    if isinstance(existing, str):
        existing = [existing]
    for run in runs:
        if run not in existing:
            existing.append(run)
    container[key] = existing

bootstrap = pb.setdefault("bootstrap", {})
extend_unique(bootstrap, "seed_grade_runs")
hist = pb.setdefault("historical_feedback", {})
extend_unique(hist, "grade_runs")
gap = pb.setdefault("codex_gap_feedback", {})
extend_unique(gap, "evomaster_grade_runs")

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY
  log "round $round config includes current best seeds: $ROUND_CONFIG"

  EVOMASTER_LLM_MIN_INTERVAL_SECONDS="${EVOMASTER_LLM_MIN_INTERVAL_SECONDS:-6}" \
  EVOMASTER_LLM_RATE_LIMIT_JITTER_SECONDS="${EVOMASTER_LLM_RATE_LIMIT_JITTER_SECONDS:-1.0}" \
  EVOMASTER_LLM_429_COOLDOWN_SECONDS="${EVOMASTER_LLM_429_COOLDOWN_SECONDS:-45}" \
  PAPERBENCH_CODEDEV_MODEL="$MODEL" \
    playground/paperbench_codedev_agent/scripts/run_papers.sh \
    "$GAP_FILE" \
    "$ROUND_CONFIG" \
    "$GEN_RUN" \
    "$MAX_WORKERS"

  python playground/paperbench_codedev_agent/scripts/collect_submissions.py \
    --run-dir "$GEN_RUN" \
    --grade-run "$GRADE_RUN"

  set -a
  source "$HOME/.codex/.env"
  set +a
  export OPENAI_API_KEY="$CRS_KEY"
  python playground/codex4paperbench/regrade_crs/grade_batch.py \
    --grade-run "$GRADE_RUN" \
    --model gpt-5.5 \
    --base-url http://139.180.136.5:3000/openai \
    --api-key-env OPENAI_API_KEY \
    --paper-workers 4 \
    --leaf-concurrency 40 \
    --reasoning-effort medium \
    --context-window 272000 \
    --max-output-tokens 4096 \
    --openai-timeout 240

  EXTRA_GRADE_RUNS+=("$GRADE_RUN")
  best_args=(
    --grade-run "$CODEX_GRADE"
    --grade-run "$OLD1"
    --grade-run "$OLD2"
    --grade-run "$BASE_FINAL_GRADE_RUN"
  )
  for extra in "${EXTRA_GRADE_RUNS[@]}"; do
    best_args+=(--grade-run "$extra")
  done
  python playground/paperbench_codedev_agent/scripts/select_best_submissions.py \
    "${best_args[@]}" \
    --out-grade-run "$BEST_RUN" \
    --expected-n 20

  python playground/codex4paperbench/regrade_crs/grade_batch.py \
    --grade-run "$BEST_RUN" \
    --model gpt-5.5 \
    --base-url http://139.180.136.5:3000/openai \
    --api-key-env OPENAI_API_KEY \
    --paper-workers 4 \
    --leaf-concurrency 40 \
    --reasoning-effort medium \
    --context-window 272000 \
    --max-output-tokens 4096 \
    --openai-timeout 240 \
    --force

  if python playground/paperbench_codedev_agent/scripts/compare_to_codex.py \
      --evomaster-summary "$BEST_RUN/summary.json" \
      --codex-summary "$CODEX_SUMMARY" \
      --out "$BEST_RUN/compare_to_codex.json"; then
    log "EvoMaster beats Codex after auto gap round $round."
    exit 0
  fi
  CURRENT_BEST="$BEST_RUN"
done

log "EvoMaster did not beat Codex after $MAX_ROUNDS auto gap rounds."
exit 1
