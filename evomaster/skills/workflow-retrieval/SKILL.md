---
name: workflow-retrieval
description: Retrieve and adapt predefined scientific workflows (eg, cs_kernel, free_fall) for Clarifier planning. Use when a task needs stage-wise decomposition aligned with known methodology templates.
license: Apache-2.0
---

# Workflow Retrieval Skill

Use this skill when you need a high-quality stage template before decomposing a task into subtasks.

## When To Use

- The user asks for a physics/scientific workflow, plan, or subtask decomposition.
- You want to align planning with known methodology templates instead of drafting from scratch.

## Actions

- `use_skill` with `action="run_script"` and `script_name="retrieve_workflow.py"`:
  - `script_args="--query '<task text>'"`
  - Returns ranked workflow candidates and the best match in JSON.

- `use_skill` with `action="get_reference"`:
  - `reference_name="workflow/cs_kernel.yaml"` or `"workflow/free_fall.yaml"`
  - Returns the raw workflow template.

## Output Usage

- Prefer the `best_match` workflow as decomposition scaffold.
- Reuse its `Goal` and `Stages` structure, then adapt wording to the user task.
- If no candidate is strong, fall back to generic decomposition but keep schema constraints.
