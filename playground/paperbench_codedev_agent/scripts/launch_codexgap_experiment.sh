#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

STAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONFIG="${PAPERBENCH_CODEDEV_CONFIG:-configs/paperbench_codedev_agent/competitive.yaml}"
MODEL="${PAPERBENCH_CODEDEV_MODEL:-ksyun/gpt-5.4}"
FULL_WORKERS="${PAPERBENCH_FULL_WORKERS:-8}"
TARGET_WORKERS="${PAPERBENCH_TARGET_WORKERS:-4}"
AUTO_GAP_ROUNDS="${PAPERBENCH_AUTO_GAP_MAX_ROUNDS:-3}"
TARGET_PAPERS="${PAPERBENCH_TARGET_PAPERS:-all-in-one,lbcs,ftrl,bam,bbox,fre,mechanistic-understanding,pinn}"
LOG_DIR="runs/paperbench_codedev_launch_logs"
mkdir -p "$LOG_DIR"

FULL_RUN="runs/paperbench_codedev_all_competitive_codexgap_throttled_ksyun_gpt54_rep_${STAMP}"
FULL_GRADE="runs/paperbench_codedev_all_competitive_codexgap_throttled_ksyun_gpt54_rep_crs_gpt55_${STAMP}"
TARGET_RUN="runs/paperbench_codedev_targeted_codexgap_throttled_ksyun_gpt54_${STAMP}"
TARGET_GRADE="runs/paperbench_codedev_targeted_codexgap_throttled_ksyun_gpt54_crs_gpt55_${STAMP}"
BEST_GRADE="runs/paperbench_codedev_codexgap_rep_plus_target_bestof_crs_gpt55_${STAMP}"
AUTO_PREFIX="paperbench_codedev_auto_gap_after_codexgap_rep_${STAMP}"
CODEX_SUMMARY="${CODEX_GPT54_SUMMARY:-runs/codex4paperbench/codex_gpt54_regen_regrade_crs_gpt55_responses_medium_c4x40_20260527T071410Z/summary.json}"
CODEX_GRADE="${CODEX_GPT54_GRADE_RUN:-$(dirname "$CODEX_SUMMARY")}"
OLD1="${PAPERBENCH_OLD_GRADE_RUN_1:-runs/paperbench_codedev_regrade_crs_gpt55_responses_c4x40_20260526T162828Z}"
OLD2="${PAPERBENCH_OLD_GRADE_RUN_2:-runs/paperbench_codedev_combined_regrade_crs_gpt55_responses_c4x40_20260527T142508Z}"

write_generation_cmd() {
  local out_log="$1" mode="$2" run_dir="$3" workers="$4" papers="$5"
  cat > "${out_log}.cmd.sh" <<EOF_CMD
#!/usr/bin/env bash
set -euo pipefail
cd $ROOT_DIR
EVOMASTER_LLM_MIN_INTERVAL_SECONDS="\${EVOMASTER_LLM_MIN_INTERVAL_SECONDS:-8}" \\
EVOMASTER_LLM_RATE_LIMIT_JITTER_SECONDS="\${EVOMASTER_LLM_RATE_LIMIT_JITTER_SECONDS:-2}" \\
EVOMASTER_LLM_429_COOLDOWN_SECONDS="\${EVOMASTER_LLM_429_COOLDOWN_SECONDS:-60}" \\
PAPERBENCH_CODEDEV_MODEL="$MODEL" \\
EOF_CMD
  if [[ "$mode" == "all" ]]; then
    cat >> "${out_log}.cmd.sh" <<EOF_CMD
  playground/paperbench_codedev_agent/scripts/run_split.sh all "$CONFIG" "$run_dir" "$workers"
EOF_CMD
  else
    cat >> "${out_log}.cmd.sh" <<EOF_CMD
  playground/paperbench_codedev_agent/scripts/run_papers.sh "$papers" "$CONFIG" "$run_dir" "$workers"
EOF_CMD
  fi
  chmod +x "${out_log}.cmd.sh"
}

write_grade_watcher() {
  local watch_log="$1" gen_run="$2" grade_run="$3" expected="$4" label="$5"
  cat > "${watch_log}.sh" <<EOF_WATCH
#!/usr/bin/env bash
set -euo pipefail
cd $ROOT_DIR
GEN_RUN="$gen_run"
GRADE_RUN="$grade_run"
EXPECTED="$expected"
log(){ printf '[%s] %s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\$*"; }
count_complete(){ find "\$GEN_RUN/workspaces" -path '*/.git' -prune -o -path '*/artifacts/EVOMASTER_COMPLETE.json' -type f -print 2>/dev/null | wc -l || true; }
count_tars(){ find "\$GEN_RUN/workspaces" -path '*/.git' -prune -o -path '*/artifacts/submission.tar.gz' -type f -print 2>/dev/null | wc -l || true; }
log "watching $label generation run: \$GEN_RUN"
while [[ ! -f "\$GEN_RUN/EVOMASTER_RUN_FINISHED.json" ]]; do
  log "generation still running markers=\$(count_complete) tars=\$(count_tars)"
  sleep 300
done
log "generation finished: \$(cat "\$GEN_RUN/EVOMASTER_RUN_FINISHED.json")"
python playground/paperbench_codedev_agent/scripts/collect_submissions.py --run-dir "\$GEN_RUN" --grade-run "\$GRADE_RUN"
rows=\$(python - "\$GRADE_RUN/manifest.json" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))))
PY
)
log "collected manifest rows=\$rows"
if [[ "\$rows" -ne "\$EXPECTED" ]]; then
  log "manifest row count is not \$EXPECTED; aborting grade"
  exit 2
fi
set -a
source "\$HOME/.codex/.env"
set +a
export OPENAI_API_KEY="\$CRS_KEY"
python playground/codex4paperbench/regrade_crs/grade_batch.py \\
  --grade-run "\$GRADE_RUN" \\
  --model gpt-5.5 \\
  --base-url http://139.180.136.5:3000/openai \\
  --api-key-env OPENAI_API_KEY \\
  --paper-workers 4 \\
  --leaf-concurrency 40 \\
  --reasoning-effort medium \\
  --context-window 272000 \\
  --max-output-tokens 4096 \\
  --openai-timeout 240
log "$label grade finished"
python - "\$GRADE_RUN/summary.json" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
print(json.dumps({'n': s.get('n'), 'mean_score': s.get('mean_score')}, indent=2))
PY
EOF_WATCH
  chmod +x "${watch_log}.sh"
}

FULL_LOG="$LOG_DIR/$(basename "$FULL_RUN").log"
TARGET_LOG="$LOG_DIR/$(basename "$TARGET_RUN").log"
FULL_WATCH_LOG="$LOG_DIR/$(basename "$FULL_GRADE").watch.log"
TARGET_WATCH_LOG="$LOG_DIR/$(basename "$TARGET_GRADE").watch.log"
BEST_WATCH_LOG="$LOG_DIR/$(basename "$BEST_GRADE").watch.log"
AUTO_LOG="$LOG_DIR/${AUTO_PREFIX}.log"

write_generation_cmd "$FULL_LOG" all "$FULL_RUN" "$FULL_WORKERS" ""
write_generation_cmd "$TARGET_LOG" selected "$TARGET_RUN" "$TARGET_WORKERS" "$TARGET_PAPERS"
write_grade_watcher "$FULL_WATCH_LOG" "$FULL_RUN" "$FULL_GRADE" 20 full
write_grade_watcher "$TARGET_WATCH_LOG" "$TARGET_RUN" "$TARGET_GRADE" 8 targeted

cat > "${BEST_WATCH_LOG}.sh" <<EOF_BEST
#!/usr/bin/env bash
set -euo pipefail
cd $ROOT_DIR
FULL_GRADE="$FULL_GRADE"
TARGET_GRADE="$TARGET_GRADE"
BEST_GRADE="$BEST_GRADE"
OLD1="$OLD1"
OLD2="$OLD2"
CODEX_GRADE="$CODEX_GRADE"
CODEX_SUMMARY="$CODEX_SUMMARY"
log(){ printf '[%s] %s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\$*"; }
for grade in "\$FULL_GRADE" "\$TARGET_GRADE"; do
  log "waiting for grade summary: \$grade/summary.json"
  while [[ ! -f "\$grade/summary.json" ]]; do
    outputs=\$(find "\$grade" -mindepth 2 -maxdepth 2 -name grader_output.json 2>/dev/null | wc -l || true)
    log "grade not done yet: \$grade outputs=\$outputs"
    sleep 300
  done
done
python playground/paperbench_codedev_agent/scripts/select_best_submissions.py \\
  --grade-run "\$CODEX_GRADE" \\
  --grade-run "\$OLD1" \\
  --grade-run "\$OLD2" \\
  --grade-run "\$FULL_GRADE" \\
  --grade-run "\$TARGET_GRADE" \\
  --out-grade-run "\$BEST_GRADE" \\
  --expected-n 20
set -a
source "\$HOME/.codex/.env"
set +a
export OPENAI_API_KEY="\$CRS_KEY"
python playground/codex4paperbench/regrade_crs/grade_batch.py \\
  --grade-run "\$BEST_GRADE" \\
  --model gpt-5.5 \\
  --base-url http://139.180.136.5:3000/openai \\
  --api-key-env OPENAI_API_KEY \\
  --paper-workers 4 \\
  --leaf-concurrency 40 \\
  --reasoning-effort medium \\
  --context-window 272000 \\
  --max-output-tokens 4096 \\
  --openai-timeout 240 \\
  --force
python playground/paperbench_codedev_agent/scripts/compare_to_codex.py \\
  --evomaster-summary "\$BEST_GRADE/summary.json" \\
  --codex-summary "\$CODEX_SUMMARY" \\
  --expected-n 20 \\
  --out "\$BEST_GRADE/compare_to_codex.json"
EOF_BEST
chmod +x "${BEST_WATCH_LOG}.sh"

cat > "${AUTO_LOG}.cmd.sh" <<EOF_AUTO
#!/usr/bin/env bash
set -euo pipefail
cd $ROOT_DIR
EVOMASTER_LLM_MIN_INTERVAL_SECONDS="\${EVOMASTER_LLM_MIN_INTERVAL_SECONDS:-8}" \\
EVOMASTER_LLM_RATE_LIMIT_JITTER_SECONDS="\${EVOMASTER_LLM_RATE_LIMIT_JITTER_SECONDS:-2}" \\
EVOMASTER_LLM_429_COOLDOWN_SECONDS="\${EVOMASTER_LLM_429_COOLDOWN_SECONDS:-60}" \\
PAPERBENCH_AUTO_GAP_MAX_ROUNDS="$AUTO_GAP_ROUNDS" \\
CODEX_GPT54_GRADE_RUN="$CODEX_GRADE" \\
  playground/paperbench_codedev_agent/scripts/auto_gap_loop.sh "$BEST_GRADE" "$AUTO_PREFIX" 8 4
EOF_AUTO
chmod +x "${AUTO_LOG}.cmd.sh"

start_tmux() {
  local session="$1" cmd_file="$2" log_file="$3"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    return 1
  fi
  tmux new-session -d -s "$session" "bash '$cmd_file' > '$log_file' 2>&1"
}

start_tmux "pb_cgap_full_${STAMP}" "${FULL_LOG}.cmd.sh" "$FULL_LOG"
start_tmux "pb_cgap_full_grade_${STAMP}" "${FULL_WATCH_LOG}.sh" "$FULL_WATCH_LOG"
start_tmux "pb_cgap_tgt_${STAMP}" "${TARGET_LOG}.cmd.sh" "$TARGET_LOG"
start_tmux "pb_cgap_tgt_grade_${STAMP}" "${TARGET_WATCH_LOG}.sh" "$TARGET_WATCH_LOG"
start_tmux "pb_cgap_best_${STAMP}" "${BEST_WATCH_LOG}.sh" "$BEST_WATCH_LOG"
start_tmux "pb_cgap_auto_${STAMP}" "${AUTO_LOG}.cmd.sh" "$AUTO_LOG"

echo "$FULL_RUN" > "${FULL_LOG}.run_dir"
echo "$TARGET_RUN" > "${TARGET_LOG}.run_dir"
echo "$FULL_GRADE" > "${FULL_WATCH_LOG}.grade_run"
echo "$TARGET_GRADE" > "${TARGET_WATCH_LOG}.grade_run"
echo "$BEST_GRADE" > "${BEST_WATCH_LOG}.grade_run"
echo "$BEST_GRADE" > "${AUTO_LOG}.base_final_grade_run"

cat <<EOF_SUMMARY
STAMP=$STAMP
CODEX_GRADE=$CODEX_GRADE
FULL_RUN=$FULL_RUN
FULL_GRADE=$FULL_GRADE
TARGET_RUN=$TARGET_RUN
TARGET_GRADE=$TARGET_GRADE
BEST_GRADE=$BEST_GRADE
AUTO_PREFIX=$AUTO_PREFIX
EOF_SUMMARY
