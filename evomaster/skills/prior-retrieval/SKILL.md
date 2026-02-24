---
name: prior-retrieval
description: Retrieve chunks from the PHY prior knowledge base with RAG. Use when a task needs domain prior evidence from playground/phy_master/LANDAU/prior.
license: Apache-2.0
---

# Prior Retrieval Skill

This is a retrieval-only skill for querying PHY prior knowledge.

Prior knowledge base build/update is done outside task execution via:
`python playground/phy_master/core/prior_analysis.py`

## When To Use

- You need semantic retrieval over existing prior chunks.
- You want evidence snippets from the PHY prior library for current reasoning.

## Actions

- Search prior knowledge:
  - `use_skill` with `action="run_script"`
  - `script_name="prior_search.py"`
  - Example `script_args="--query \"critical level views violate which condition\" --top_k 5 --threshold 1.5"`

## Output Notes

- `prior_search.py` returns JSON with `results`.
- Each item includes `chunk_id`, `distance`, `content`, and full `node` payload.
