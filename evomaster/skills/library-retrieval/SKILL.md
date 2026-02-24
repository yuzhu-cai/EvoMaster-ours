---
name: library-retrieval
description: Retrieve high-relevance paper/library chunks from a local HTTP service (`POST /chunks`). Use when Clarifier/Supervisor/Theoretician need literature evidence for planning, instruction refinement, or solving.
license: Apache-2.0
---

# Library Retrieval Skill

Call a local retrieval service and return ranked literature chunks in JSON.

## Service Contract

- Endpoint: `POST http://127.0.0.1:30388/chunks`
- Request JSON:
  - `{"query":"..."}`
- Response JSON:
  - `{"query": QUERY, "chunks": [chunk_0, chunk_1, ...]}`

## When To Use

- Clarifier needs related literature context before decomposition.
- Supervisor needs evidence to refine current node description.
- Theoretician needs references to guide method choice and validation.

## Actions

- Run retrieval:
  - `use_skill` with `action="run_script"`
  - `script_name="library_search.py"`
  - Example: `script_args="--query \"LaMET matching for quasi-TMD\" --top_k 5"`

## Output Notes

- Script returns JSON with:
  - `query`
  - `service_url`
  - `count`
  - `chunks`
- If the service is unavailable, returns JSON with `error` and `status`.
