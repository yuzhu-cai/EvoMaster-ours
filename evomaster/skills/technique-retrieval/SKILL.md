---
name: technique_retrieval
description: Retrieve technique templates from playground/phy_master/LANDAU/technique for physics method selection. Use when Theoretician needs concrete methodology candidates (eg, asymptotic expansion, sanity checks) before solving a node.
license: Apache-2.0
---

# Technique Retrieval Skill

Retrieve technique definitions under `playground/phy_master/LANDAU/technique`.

## When To Use

- You need candidate methods before implementing a node.
- You need a reusable technique checklist or quality gate from LANDau technique files.

## Actions

- Search techniques by query:
  - `use_skill` with `action="run_script"`
  - `script_name="technique_search.py"`
  - Example: `script_args="--query \"LaMET expansion for large Pz\" --top_k 5"`

## Output Notes

- Returns JSON with `results`.
- Each result includes `skill_id`, `category`, `path`, `score`, and `preview`.
