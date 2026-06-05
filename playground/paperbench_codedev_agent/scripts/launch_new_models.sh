#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

STAMP="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
shift || true
REQUESTED_MODELS=("$@")
RUN_ROOT="${PAPERBENCH_CODEDEV_RUN_ROOT:-runs/evomaster4paperbench}"
GEN_WORKERS="${PAPERBENCH_NEW_MODEL_WORKERS:-20}"
EXPECTED_PAPERS="${PAPERBENCH_EXPECTED_PAPERS:-20}"
GRADE_PAPER_WORKERS="${PAPERBENCH_CRS_PAPER_WORKERS:-4}"
GRADE_LEAF_CONCURRENCY="${PAPERBENCH_CRS_LEAF_CONCURRENCY:-40}"
LOG_DIR="${RUN_ROOT}/launch-logs"

mkdir -p "$LOG_DIR"

write_generation_cmd() {
  local label="$1" model="$2" config="$3" gen_run="$4" out_log="$5"
  cat > "${out_log}.cmd.sh" <<EOF_CMD
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT_DIR"
PAPERBENCH_CODEDEV_RUN_ROOT="$RUN_ROOT" \\
PAPERBENCH_CODEDEV_MODEL="$model" \\
EVOMASTER_LLM_RATE_LIMIT_KEY="paperbench-${label}" \\
  playground/paperbench_codedev_agent/scripts/run_split.sh \\
    all \\
    "$config" \\
    "$gen_run" \\
    "$GEN_WORKERS"
EOF_CMD
  chmod +x "${out_log}.cmd.sh"
}

write_grade_watcher() {
  local label="$1" gen_run="$2" grade_run="$3" out_log="$4"
  cat > "${out_log}.watch.sh" <<EOF_WATCH
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT_DIR"
LABEL="$label"
GEN_RUN="$gen_run"
GRADE_RUN="$grade_run"
EXPECTED="$EXPECTED_PAPERS"
log(){ printf '[%s] [%s] %s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\$LABEL" "\$*"; }
count_complete(){ find "\$GEN_RUN/workspaces" -path '*/.git' -prune -o -path '*/artifacts/EVOMASTER_COMPLETE.json' -type f -print 2>/dev/null | wc -l || true; }
count_tars(){ find "\$GEN_RUN/workspaces" -path '*/.git' -prune -o -path '*/artifacts/submission.tar.gz' -type f -print 2>/dev/null | wc -l || true; }

log "watching generation run: \$GEN_RUN"
while [[ ! -f "\$GEN_RUN/EVOMASTER_RUN_FINISHED.json" ]]; do
  log "generation still running complete_markers=\$(count_complete) submission_tars=\$(count_tars)"
  sleep 300
done

log "generation finished: \$(cat "\$GEN_RUN/EVOMASTER_RUN_FINISHED.json")"
python playground/paperbench_codedev_agent/scripts/collect_submissions.py \\
  --run-dir "\$GEN_RUN" \\
  --allow-incomplete \\
  --grade-run "\$GRADE_RUN"

rows=\$(python - "\$GRADE_RUN/manifest.json" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
)
log "collected manifest rows=\$rows"
if [[ "\$rows" -ne "\$EXPECTED" ]]; then
  log "manifest row count is not \$EXPECTED; aborting final CRS grade"
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
  --paper-workers "$GRADE_PAPER_WORKERS" \\
  --leaf-concurrency "$GRADE_LEAF_CONCURRENCY" \\
  --reasoning-effort medium \\
  --context-window 272000 \\
  --max-output-tokens 4096 \\
  --openai-timeout 240

log "final CRS grade finished"
python - "\$GRADE_RUN/summary.json" <<'PY'
import json
import sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({"n": s.get("n"), "mean_score": s.get("mean_score")}, indent=2))
PY
EOF_WATCH
  chmod +x "${out_log}.watch.sh"
}

start_tmux() {
  local session="$1" cmd_file="$2" log_file="$3"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    return 1
  fi
  tmux new-session -d -s "$session" "bash '$cmd_file' > '$log_file' 2>&1"
}

launch_one() {
  local short="$1" label="$2" slug="$3" model="$4" config="$5"
  if [[ "${#REQUESTED_MODELS[@]}" -gt 0 ]]; then
    local requested=0
    local item
    for item in "${REQUESTED_MODELS[@]}"; do
      if [[ "$item" == "$short" || "$item" == "$label" || "$item" == "$slug" ]]; then
        requested=1
        break
      fi
    done
    if [[ "$requested" -eq 0 ]]; then
      return 0
    fi
  fi

  local gen_run="${RUN_ROOT}/generation/full/paperbench_codedev_all_${slug}_solve_c${GEN_WORKERS}_${STAMP}"
  local grade_run="${RUN_ROOT}/grades/final/paperbench_codedev_all_${slug}_solve_c${GEN_WORKERS}_${STAMP}_crs_gpt55"
  local gen_log="${LOG_DIR}/$(basename "$gen_run").log"
  local grade_log="${LOG_DIR}/$(basename "$grade_run").watch.log"

  if [[ -e "$gen_run" || -e "$grade_run" ]]; then
    echo "refusing to reuse existing run path for $label" >&2
    echo "GEN_RUN=$gen_run" >&2
    echo "GRADE_RUN=$grade_run" >&2
    return 1
  fi

  write_generation_cmd "$label" "$model" "$config" "$gen_run" "$gen_log"
  write_grade_watcher "$label" "$gen_run" "$grade_run" "$grade_log"
  start_tmux "pb_nm_${short}_${STAMP}" "${gen_log}.cmd.sh" "$gen_log"
  start_tmux "pb_nm_${short}_grade_${STAMP}" "${grade_log}.watch.sh" "$grade_log"
  echo "$gen_run" > "${gen_log}.run_dir"
  echo "$grade_run" > "${grade_log}.grade_run"

  cat <<EOF_ONE
[$label]
GEN_RUN=$gen_run
GRADE_RUN=$grade_run
GEN_LOG=$gen_log
GRADE_LOG=$grade_log
GEN_SESSION=pb_nm_${short}_${STAMP}
GRADE_SESSION=pb_nm_${short}_grade_${STAMP}
EOF_ONE
}

launch_one \
  "dsv4p" \
  "sjtu-deepseek-v4-pro" \
  "sjtu_deepseek_v4_pro" \
  "sjtu/deepseek-v4-pro" \
  "configs/paperbench_codedev_agent/sjtu-deepseek-v4-pro.yaml"

launch_one \
  "q35" \
  "local-qwen3_5-35b-a3b" \
  "local_qwen3_5_35b_a3b" \
  "Qwen/Qwen3.5-35B-A3B" \
  "configs/paperbench_codedev_agent/local-qwen3_5-35b-a3b.yaml"

launch_one \
  "q35ss" \
  "local-qwen3_5-35b-a3b-science-seeker-0507-ep2" \
  "local_qwen3_5_35b_a3b_science_seeker_0507_ep2" \
  "qwen35-iter140-sft" \
  "configs/paperbench_codedev_agent/local-qwen3_5-35b-a3b-science-seeker-0507-ep2.yaml"

launch_one \
  "q35v2" \
  "local-qwen3_5-35b-a3b-chem-phys-bio-0525" \
  "local_qwen3_5_35b_a3b_chem_phys_bio_0525" \
  "qwen35-trainedv2-sft" \
  "configs/paperbench_codedev_agent/local-qwen3_5-35b-a3b-chem-phys-bio-0525.yaml"

cat <<EOF_SUMMARY
STAMP=$STAMP
RUN_ROOT=$RUN_ROOT
GEN_WORKERS=$GEN_WORKERS
FINAL_CRS=paper-workers:${GRADE_PAPER_WORKERS},leaf-concurrency:${GRADE_LEAF_CONCURRENCY},reasoning-effort:medium
REQUESTED_MODELS=${REQUESTED_MODELS[*]:-all}
EOF_SUMMARY
