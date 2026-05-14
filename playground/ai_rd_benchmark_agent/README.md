# AI R&D Benchmark Agent

`ai_rd_benchmark_agent` is a thin single-agent benchmark harness for MLE-Bench,
PaperBench, and PostTrainBench. The agent runs inside Docker, keeps a persistent
`/workspace`, and uses tools to inspect, implement, run, debug, and validate.
The `solve.yaml` profile also enables long-horizon continuation: after the
agent finishes a checkpoint, the harness can append another user message to the
same dialog and workspace so the agent keeps improving instead of stopping at
the first valid artifact.

Default environment:

- Docker image: `mlmaster-worker:latest`
- Conda env activated for every command: `evomaster_yuzhu`
- Host conda mount: `/data/conda/miniconda3 -> /data/conda/miniconda3:ro`
- Network: `host` (proxy defaults to host port `5890`)
- GPU binding: one GPU per container from `gpu_devices: ["0", "1", "2", "3"]`
- Benchmark inputs: dynamically mounted read-only from the task spec

Run an MLE-Bench task spec:

```bash
python run.py \
  --agent ai_rd_benchmark_agent \
  --config configs/ai_rd_benchmark_agent/solve.yaml \
  --task configs/ai_rd_benchmark_agent/examples/mlebench_task.yaml
```

Task specs are YAML/JSON files. Important fields:

- `benchmark`: `mlebench`, `paperbench`, or `posttrainbench`
- MLE-Bench input: `input_dir`, `public_dir`, or `competition_dir`
- PaperBench input: `paper_dir`, `input_dir`, or `dataset_dir`
- PostTrainBench input: `task_dir`, `input_dir`, or `benchmark_dir`
- `extra_mounts`: optional list of `{host, target, read_only}` mounts
- `description`: benchmark-specific instructions for the agent

Required artifacts:

- MLE-Bench: `/workspace/submission.csv`
- PaperBench: `/workspace/reproduce.sh`
- PostTrainBench: `/workspace/final_model`

Long-horizon iteration:

- Configure under `benchmark_agent.iteration`.
- `enabled: true` keeps the same agent dialog and Docker workspace alive across
  multiple continuation rounds.
- Each round asks for one or more new, atomic improvement experiments and writes
  snapshots under `workspaces/<task_id>/iterations/round_XX/`.
- Round summaries include structured experiment-memory state from
  `logs/experiments.jsonl` and `artifacts/best_meta.json`; the harness can
  treat rounds without a new `experiment_tracker(action="record", ...)` entry
  or without best-score progress as stagnant.
- The agent is instructed to maintain `/workspace/artifacts/best_submission.csv`
  and `/workspace/artifacts/best_solution.py`; when present, the harness
  promotes that best artifact back to `/workspace/submission.csv` before final
  grading.
- The agent can create `/workspace/artifacts/STOP_ITERATION` after the minimum
  number of rounds if the remaining ideas are exhausted.
- `benchmark_agent.iteration.max_stagnant_rounds` optionally mirrors
  `ml_master`'s max-improve-failure idea: after `min_rounds`, stop only after
  the configured number of continuation rounds fail to improve the
  tool-maintained best score.

Experiment memory:

- `experiment_tracker` is enabled alongside `benchmark_status`.
- `experiment_tracker(action="record", ...)` appends structured history to
  `/workspace/logs/experiments.jsonl`, appends the human ledger at
  `/workspace/logs/iteration_ledger.md`, and compares the new score against the
  current best.
- On artifact-backed improvement, the tool copies the submitted artifact/code to
  `/workspace/artifacts/best_submission.csv` and
  `/workspace/artifacts/best_solution.py`.
- Score-only probes without `submission_path` are still logged as evidence, but
  they are not promoted as the best record. This keeps `best_meta.json`,
  `best_submission.csv`, and the final `/workspace/submission.csv` aligned.
- `experiment_tracker(action="best", ...)` lets continuation rounds recover the
  current best score and recent experiments before choosing the next branch.
- `experiment_tracker(action="plan", candidates=[...],
  candidate_families={...}, selected_candidate=...)` records a lightweight
  branch backlog in `/workspace/logs/experiment_backlog.*` so rounds leave an
  explicit reasoning trail instead of making ad hoc edits. Optional
  score/cost/risk estimates may be included, but the tool does not schedule
  experiments for the agent.
- Recorded experiment outcomes are folded into
  `/workspace/logs/branch_search_state.json`, which tracks branch/model-family
  visits, improvements, non-improvements, and simple stagnation guidance.
- After each round the harness writes `/workspace/logs/strategy_memory.md`,
  a compact long-horizon brief with best state, score trajectory, negative
  evidence, and next-round rules. This mirrors the reusable memory ideas from
  `playground/ml_master` and knowledge-promotion summaries in
  `playground/ml_master_2`.
