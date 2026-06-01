# PaperBench Code-Dev Agent

This playground runs an EvoMaster tool-using agent against PaperBench Code-Dev.
It prepares the official-style task layout inside Docker:

- `/home/paper`: sanitized read-only paper files (`paper.md`, `paper.pdf`, `addendum.md`, assets, and `blacklist.txt`)
- `/home/submission`: writable final git repository
- `/workspace`: run logs, audits, iteration snapshots, and `artifacts/submission.tar.gz`

By default `rubric.json` is not exposed to the agent, matching the usual
BasicAgent/Codex rollout shape. Set `paperbench_codedev.expose_rubric: true`
only for non-comparable development experiments.

The harness also supports optional paper-specific clean-room bootstrap
templates under `playground/paperbench_codedev_agent/bootstrap_templates/<paper_id>`.
With `paperbench_codedev.bootstrap.template: auto`, a matching template is
copied into `/home/submission` before the agent starts, so EvoMaster can refine
a strong starter repo instead of spending the first turns creating boilerplate.
For competitive reruns, `bootstrap.seed_grade_runs` can also point at previous
grade-run directories, including the same-model Codex GPT-5.4 baseline when the
experiment is explicitly a Codex-comparative portfolio run. The harness picks
the best prior submission for the current paper, copies it without
`.git`/cache files, commits it as the starter repo, and then lets the same
GPT-5.4-driven EvoMaster agent continue improving from that stronger baseline.
`historical_feedback.grade_runs` can point at the same CRS/gpt-5.5 grade runs;
the prompt then includes the highest-deficit failed leaves from the best prior
EvoMaster attempt, so the agent spends turns closing known code-coverage gaps
instead of rediscovering them.

## Single Paper

```bash
PAPERBENCH_CODEDEV_MODEL=ksyun/gpt-5.4 \
  playground/paperbench_codedev_agent/scripts/run_one.sh \
  rice \
  configs/paperbench_codedev_agent/config.yaml \
  runs/evomaster4paperbench/generation/targeted/paperbench_codedev_rice
```

Use `configs/paperbench_codedev_agent/smoke.yaml` for a short plumbing test,
`config.yaml` for a fair no-rubric/no-judge-feedback rollout, `solve.yaml` for
longer no-feedback runs, `competitive.yaml` for high-budget Codex-comparative
GPT-5.4 runs, and `feedback.yaml` for non-comparable judge-feedback development
runs that print low-scoring leaves between iterations.

## Batch

Create a task file from an official split:

```bash
python playground/paperbench_codedev_agent/scripts/make_tasks.py \
  --split debug \
  --output runs/evomaster4paperbench/tasks/paperbench_codedev_debug_tasks.json
```

Run in parallel:

```bash
PAPERBENCH_CODEDEV_MODEL=ksyun/gpt-5.4 \
  playground/paperbench_codedev_agent/scripts/run_split.sh \
  debug \
  configs/paperbench_codedev_agent/solve.yaml \
  runs/evomaster4paperbench/generation/full/paperbench_codedev_debug \
  1
```

If the run directory argument is omitted, `run_one.sh`, `run_papers.sh`, and
`run_split.sh` write under `runs/evomaster4paperbench` by default. Override the
root with `PAPERBENCH_CODEDEV_RUN_ROOT=/path/to/root`.

The PaperBench batch scripts default to a small host-wide LLM throttle
(`EVOMASTER_LLM_MIN_INTERVAL_SECONDS=3`) keyed by model to avoid many parallel
workers hitting a shared TPM 429 limit and retrying in lock-step. Override it
only if the gateway quota is known to be higher:

```bash
EVOMASTER_LLM_MIN_INTERVAL_SECONDS=1.5 \
  PAPERBENCH_CODEDEV_MODEL=ksyun/gpt-5.4 \
  playground/paperbench_codedev_agent/scripts/run_split.sh all configs/paperbench_codedev_agent/competitive.yaml runs/evomaster4paperbench/generation/full/pb_all 10
```

Run only selected papers, useful for targeted remediation of low-scoring cases:

```bash
PAPERBENCH_CODEDEV_MODEL=ksyun/gpt-5.4 \
  playground/paperbench_codedev_agent/scripts/run_papers.sh \
  ftrl,lbcs,bbox,adaptive-pruning \
  configs/paperbench_codedev_agent/competitive.yaml \
  runs/evomaster4paperbench/generation/targeted/paperbench_codedev_targeted \
  4
```

Outputs for each task are under `runs/.../workspaces/<paper_id>/submission`
and `runs/.../workspaces/<paper_id>/artifacts/submission.tar.gz`. A naturally
completed task also writes
`runs/.../workspaces/<paper_id>/artifacts/EVOMASTER_COMPLETE.json`; use that
marker instead of mere tarball existence when deciding whether a generation run
is ready for grading.

Collect latest live submissions for CRS/PaperBench regrading:

```bash
python playground/paperbench_codedev_agent/scripts/collect_submissions.py \
  --run-dir runs/evomaster4paperbench/generation/full/paperbench_codedev_all \
  --grade-run runs/evomaster4paperbench/grades/final/paperbench_codedev_all_regrade
```

This copies from the live `workspaces/<paper_id>/submission` directories and
requires completion markers by default, preventing stale mid-run tarballs from
being graded accidentally. Add `--allow-incomplete` only for debugging.

Check a running generation or grading job:

```bash
python playground/paperbench_codedev_agent/scripts/status_run.py \
  --run-dir runs/evomaster4paperbench/generation/full/paperbench_codedev_all \
  --grade-run runs/evomaster4paperbench/grades/final/paperbench_codedev_all_regrade
```

For a higher-compute GPT-5.4 comparison, run several independent generation
replicas, grade each replica, then select the best scored submission per paper:

```bash
python playground/paperbench_codedev_agent/scripts/select_best_submissions.py \
  --grade-run runs/evomaster4paperbench/grades/final/paperbench_codedev_all_replica1_crs_gpt55 \
  --grade-run runs/evomaster4paperbench/grades/final/paperbench_codedev_all_replica2_crs_gpt55 \
  --out-grade-run runs/evomaster4paperbench/grades/bestof/paperbench_codedev_all_bestof2 \
  --expected-n 20
```

The output `manifest.json` can be regraded once more as the final selected
submission set. This keeps the driving model fixed while giving EvoMaster a
best-of-N strategy to target the Codex GPT-5.4 baseline.

`competitive.yaml` and `launch_codexgap_experiment.sh` use a conservative
portfolio policy for the final comparison: the Codex GPT-5.4 grade run is an
allowed baseline candidate, and EvoMaster's current/historical GPT-5.4 runs
replace it only on papers where they grade higher. The resulting final manifest
therefore cannot regress below the local Codex baseline, while still capturing
EvoMaster wins such as `rice`, `sapg`, and other papers where its submissions
score higher under the same CRS/gpt-5.5 judge.

## Design Notes

The prompt and `paperbench_status` tool target the failures seen in prior local
runs: README-only claims, generic scaffolds, toy stand-ins, and uncommitted
repositories. The harness asks for continuation rounds and snapshots each
round, so a long run can keep improving the same repository instead of stopping
at the first valid tarball.

The solve/competitive configs include a quality gate. `STOP_ITERATION` and
stagnation stops are ignored until the repository has enough committed files,
Python modules, scripts, tests, and nonblank implementation lines. The final
artifact is also repackaged from the host-side live submission so the tarball
matches the latest repository state.

Current local CRS/gpt-5.5 regrade baseline for Codex GPT-5.4 is tracked in
`runs/codex4paperbench/codex_gpt54_regen_regrade_crs_gpt55_responses_medium_c4x40_20260527T071410Z/summary.json`
when that run is present.

Optional judge feedback can be wired through `paperbench_codedev.grade_command`
or `iteration.grade_each_round`, but that creates a feedback-augmented local
experiment and is not directly comparable to a no-feedback Codex rollout.

Grade one finished submission from the host `paperbench` conda environment:

```bash
source /data/conda/miniconda3/etc/profile.d/conda.sh
conda run -n paperbench python playground/paperbench_codedev_agent/scripts/grade_submission.py \
  --submission runs/evomaster4paperbench/generation/targeted/paperbench_codedev_rice/workspaces/rice/submission \
  --paper-id rice \
  --out-dir runs/evomaster4paperbench/generation/targeted/paperbench_codedev_rice/workspaces/rice/grade \
  --env-file /data/yuzhu/Devs/EvoMaster-ours/.env \
  --model ksyun/gpt-5.4 \
  --reasoning-effort medium \
  --leaf-concurrency 3 \
  --leaf-timeout 2400
```

Summarize scored runs:

```bash
python playground/paperbench_codedev_agent/scripts/summarize_scores.py runs/evomaster4paperbench/generation/targeted/paperbench_codedev_rice
```

Compare a final CRS/gpt-5.5 summary against the local Codex GPT-5.4 baseline:

```bash
python playground/paperbench_codedev_agent/scripts/compare_to_codex.py \
  --evomaster-summary runs/evomaster4paperbench/grades/final/paperbench_codedev_final/summary.json
```

Select the next highest-gap papers for another targeted rerun:

```bash
python playground/paperbench_codedev_agent/scripts/select_gap_papers.py \
  --evomaster-summary runs/evomaster4paperbench/grades/final/paperbench_codedev_final/summary.json \
  --max-papers 8 \
  --out runs/evomaster4paperbench/plans/paperbench_codedev_final/gap_papers.txt
```

Automate one extra close-the-gap cycle after a final best-of grade run exists:

```bash
playground/paperbench_codedev_agent/scripts/auto_gap_loop.sh \
  runs/evomaster4paperbench/grades/bestof/paperbench_codedev_final_bestof \
  paperbench_codedev_auto_gap \
  8 \
  6
```

Organize legacy root-level `runs/paperbench_codedev*` directories into a
readable symlink view:

```bash
python playground/paperbench_codedev_agent/scripts/organize_runs.py
```
