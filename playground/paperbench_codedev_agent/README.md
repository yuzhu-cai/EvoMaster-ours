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

## Single Paper

```bash
PAPERBENCH_CODEDEV_MODEL=ksyun/gpt-5.4 \
  playground/paperbench_codedev_agent/scripts/run_one.sh \
  rice \
  configs/paperbench_codedev_agent/config.yaml \
  runs/paperbench_codedev_rice
```

Use `configs/paperbench_codedev_agent/smoke.yaml` for a short plumbing test,
`config.yaml` for a fair no-rubric/no-judge-feedback rollout, `solve.yaml` for
longer no-feedback runs, and `feedback.yaml` for non-comparable judge-feedback
development runs that print low-scoring leaves between iterations.

## Batch

Create a task file from an official split:

```bash
python playground/paperbench_codedev_agent/scripts/make_tasks.py \
  --split debug \
  --output runs/paperbench_codedev_debug_tasks.json
```

Run in parallel:

```bash
PAPERBENCH_CODEDEV_MODEL=ksyun/gpt-5.4 \
  playground/paperbench_codedev_agent/scripts/run_split.sh \
  debug \
  configs/paperbench_codedev_agent/solve.yaml \
  runs/paperbench_codedev_debug \
  1
```

Outputs for each task are under `runs/.../workspaces/<paper_id>/submission`
and `runs/.../workspaces/<paper_id>/artifacts/submission.tar.gz`.

## Design Notes

The prompt and `paperbench_status` tool target the failures seen in prior local
runs: README-only claims, generic scaffolds, toy stand-ins, and uncommitted
repositories. The harness asks for continuation rounds and snapshots each
round, so a long run can keep improving the same repository instead of stopping
at the first valid tarball.

Local Codex GPT-5.4 Code-Dev baseline to beat:

- all-paper mean: `0.7513799403271421`
- `rice`: `0.7429248366013071`

Optional judge feedback can be wired through `paperbench_codedev.grade_command`
or `iteration.grade_each_round`, but that creates a feedback-augmented local
experiment and is not directly comparable to a no-feedback Codex rollout.

Grade one finished submission from the host `paperbench` conda environment:

```bash
source /data/conda/miniconda3/etc/profile.d/conda.sh
conda run -n paperbench python playground/paperbench_codedev_agent/scripts/grade_submission.py \
  --submission runs/paperbench_codedev_rice/workspaces/rice/submission \
  --paper-id rice \
  --out-dir runs/paperbench_codedev_rice/workspaces/rice/grade \
  --env-file /data/yuzhu/Devs/EvoMaster-ours/.env \
  --model ksyun/gpt-5.4 \
  --reasoning-effort medium \
  --leaf-concurrency 3 \
  --leaf-timeout 2400
```

Summarize scored runs:

```bash
python playground/paperbench_codedev_agent/scripts/summarize_scores.py runs/paperbench_codedev_rice
```
