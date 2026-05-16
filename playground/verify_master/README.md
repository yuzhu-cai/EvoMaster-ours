# VerifyMaster Report

## Goal

`playground/verify_master` is a verification-centric BrowseComp playground. In this round I did not try to improve benchmark accuracy; I only completed the missing wiring so this playground can actually use the local tools and run with its own config/scripts.

## What I connected

### 1. Playground and experiment wiring

- Fixed `playground/verify_master/core/playground.py` so it now behaves like a normal EvoMaster playground.
- Added default config resolution to `configs/verify_master/config.yaml`.
- Added BrowseComp dataset loading plus `dataset:<id>` / `id:<id>` parsing support.
- Wired the three agents as `planner_agent`, `executor_agent`, and `verifier_agent`.
- Hooked `web_fetch.set_llm(...)` to the executor LLM so `web_fetch` can use the model-backed extraction path.
- Fixed `playground/verify_master/core/exp.py` so the multi-agent workflow is runnable:
  - planner proposes `<task>` or `<answer>`
  - executor runs the assigned subtask
  - verifier scores local evidence and the final answer
  - final answer is logged as `Agent final answer:` for downstream scripts

### 2. Tool access

The tools under `playground/verify_master/tools` are now exposed through config and agent setup.

- `planner`: no tools; text-only role
- `executor`: `think`, `finish`, `google_search`, `web_fetch`
- `verifier`: no tools; text-only role

This keeps the browsing capability concentrated in the executor instead of letting every role call tools.

### 3. Prompt alignment

The prompt files were adjusted so they match the current runtime contract.

Main fixes:
- Replaced the previous unresolved placeholders like `{subtask}`, `{research_context}`, `{verification_type}` with a simpler `{description}`-based interface that the current agent runtime always fills.
- Updated the executor prompt to explicitly name and explain the available tools:
  - `think`
  - `google_search`
  - `web_fetch`
- Added a concrete execution workflow so the executor knows it should search first and only then write an evidence report.
- Kept planner/verifier prompts text-only so they do not accidentally depend on unavailable tools.

### 4. Config and script completion

Added:
- `configs/verify_master/config.yaml`
- `configs/verify_master/config_gpt.yaml`

Updated:
- `playground/verify_master/scripts/run_batch.py`
- `playground/verify_master/scripts/run_browse.sh`

These now point to `verify_master` instead of `browse_master`, and they use the local dataset under `playground/verify_master/test/browsecomp_decrypted.json`.

## Current structure

- `playground/verify_master/core/playground.py`: playground entry, dataset parsing, agent setup
- `playground/verify_master/core/exp.py`: planner-executor-verifier workflow
- `playground/verify_master/tools/google_search.py`: search tool
- `playground/verify_master/tools/web_fetch.py`: fetch/extraction tool
- `playground/verify_master/prompts/*.txt`: role prompts
- `configs/verify_master/*.yaml`: runnable configs
- `playground/verify_master/scripts/*.py`: batch/eval helpers

## How to run

Single task:

```bash
python run.py --agent verify_master --config configs/verify_master/config_gpt.yaml --task "dataset:0"
```

Batch run:

```bash
IDS="0-9" RUN_NAME="verify_master_smoke" ./playground/verify_master/scripts/run_browse.sh
```

## What this change intentionally did not do

- No benchmark-oriented optimization
- No prompt tuning for higher accuracy
- No architecture redesign
- No new tools beyond the existing `google_search` and `web_fetch`

This round is only about making `verify_master` complete and runnable with the existing tool stack.

## Validation done

I ran Python syntax checks for the modified playground/config wiring files.

## Likely next step

Once you confirm this wiring is okay, the next natural step is a small smoke run on a few BrowseComp tasks to verify:
- the executor really calls `google_search` / `web_fetch`
- the verifier returns structured PASS/FAIL tags reliably
- the scripts can extract final answers from logs end-to-end

nohup playground/verify_master/scripts/run_browse.sh > output_veri.log 2>&1 &