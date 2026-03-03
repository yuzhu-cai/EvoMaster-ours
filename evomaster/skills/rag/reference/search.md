## `scripts/search.py`: Vector Search and Content Retrieval

`search.py` provides a general-purpose **vector search plus optional content retrieval** capability, suitable for any vector store that has a FAISS index built.

### Feature Overview

- **Vector search**: Perform similarity search based on `faiss.index` and `nodes.jsonl` under `vec_dir`.
- **Optional content retrieval**: If you provide `nodes_data` (JSON/JSONL), the script can use retrieved `node_id`s to look up original content.
- **Multiple embedding backends**:
  - Local Transformer models (HuggingFace or local paths).
  - OpenAI Embedding API (e.g., `text-embedding-3-large`).
- **Generic schema design**:
  - No fixed business fields; everything is configurable via parameters such as `node_id_key`, `content_path`, etc.

The core implementation lives in the `RAGSearcher` class, which handles:

- Loading FAISS indexes.
- Text encoding (via the shared `create_embedder`).
- Similarity search (`search_similar` / `search_by_text`).
- Accessing content (`get_knowledge` / `get_knowledge_by_path` / `get_node_data`).

### Required Inputs and File Layout

The minimal requirement is a vector directory `vec_dir` containing at least:

- `faiss.index`: FAISS index file.
- `nodes.jsonl`: One JSON per line with fields that identify each node (by default `node_id`).

Optional:

- `embeddings.npy`: Precomputed embedding matrix (mainly for debugging; not required).
- `nodes_data.json`: Node detail file; its structure is completely defined by your application.

The script does not constrain the schema of `nodes_data` as long as it can be accessed via `node_id` or another agreed-upon ID key.

### Usage Pattern (Conceptual)

In real projects, `search.py` is typically invoked by the host system through a “skill script runner” rather than directly from the terminal. This section explains which parameters the script cares about when called, and provides **recommended CLI patterns** so that upstream embedding configuration (e.g. OpenAI vs. local models) is respected rather than silently falling back to defaults.

#### Basic Search (Vector Results Only)

At minimum, calls should provide:

- A vector-store directory `vec_dir`.
- A query text `query`.
- Optional `top_k` (number of returned results) and `threshold` (distance cutoff).

When you **already have an external configuration that decides the embedding model**, you should forward that information explicitly via `--model`, `--embedding_type` and (optionally) `--embedding_dimensions`.

Common parameters:

- `--vec_dir`: Vector-store directory (required).
- `--query`: Query text (required).
- `--top_k`: Number of results to return, default `5`.
- `--threshold`: Optional distance threshold; results worse than this are filtered out.
- `--output`: `text` or `json`, default `text`.

#### Search and Retrieve Original Content

If you have a node-details file (e.g., `nodes_data.json` or other JSON/JSONL), you can also provide:

- `nodes_data`: Path to the node-details JSON file.
- `content_path`: A dot path used to extract the specific field to return from each node.

Typical parameters:

- `--nodes_data`: Path to the node-details JSON file.
- `--content_path`: Dot path used to pull a specific field from each node, e.g.:
  - `content.text`
  - `content.code`
  - `meta.task_name`
  - `data_knowledge`
  - `model_knowledge`

If `--content_path` is not provided, the script tries a list of common candidate fields (such as `content.text`, `text`, etc.). If none of those exist, the entire node object is returned.

#### Using OpenAI Embeddings

To use OpenAI embeddings (such as `text-embedding-3-large`), set `embedding_type` to `openai` and provide the model name, dimensions, and credentials via parameters or environment variables.

You can provide the API key / base URL via:

- CLI parameters: `--embedding_api_key`, `--embedding_base_url`.
- Environment variables: `OPENAI_EMBEDDING_API_KEY` or `OPENAI_API_KEY`, and `OPENAI_EMBEDDING_BASE_URL` or `OPENAI_BASE_URL`.

**Recommended pattern (for agents / host systems)**  
If your upstream config exposes:

- `model` (e.g. `text-embedding-3-large`);
- `embedding_type` (e.g. `openai`);
- `embedding_dimensions` (e.g. `3072`);

then a typical call looks like:

```bash
python evomaster/skills/rag/scripts/search.py \
  --vec_dir "{vec_dir}" \
  --nodes_data "{nodes_data}" \
  --query "{query}" \
  --top_k {top_k} \
  --threshold {threshold} \
  --model "{model}" \
  --embedding_type "{embedding_type}" \
  --embedding_dimensions {embedding_dimensions}
```

This ensures that:

- When `embedding_type` is `"openai"`, the script uses the exact OpenAI embedding model selected by your higher-level config (and not its own local default such as `all-mpnet-base-v2`).
- When `embedding_type` is `"local"`, the script uses the correct local Transformer model path or name.

### Key Parameters at a Glance

- **Vector-store related**
  - `--vec_dir`: Directory of the vector store (required).
  - `--nodes_data`: Node-details JSON file (optional).
  - `--node_id_key`: Field name in `nodes.jsonl` used as the node ID, default `node_id`. If missing, the script tries `task_name` or falls back to the line index.
- **Search-related**
  - `--query`: Query text.
  - `--top_k`: Number of top results to return.
  - `--threshold`: Distance threshold; results beyond this value are filtered out.
- **Content-retrieval related**
  - `--content_path`: Dot path for extracting content from `nodes_data`.
  - If `content_path` is omitted, the script automatically tries a set of common field names.
- **Output-related**
  - `--output`: `text` or `json`.
- **Embedding-related**
  - `--model`: Local model path, HuggingFace model name, or OpenAI model name.
  - `--embedding_type`: `auto` / `local` / `openai`.
  - `--embedding_api_key` / `--embedding_base_url`: For OpenAI.
  - `--embedding_dimensions`: For `text-embedding-3-*` models, allows configuring the embedding dimension.

### Design Principles for Generality

- No assumptions about business-specific schemas:
  - Field names in `nodes.jsonl` and `nodes_data` are fully defined by the user.
  - Adaptation is done via `node_id_key` and `content_path`.
- No hard dependency on any specific model or provider:
  - Supports arbitrary HuggingFace/local Transformer models.
  - Optionally integrates with the OpenAI Embedding API via parameters.
- Easy to migrate across RAG projects:
  - A vector store only needs a FAISS index plus a corresponding list of node IDs.
  - Node-detail files can be any JSON structure.

### FAISS Index Types and Distance Metrics (Optional Background)

`search.py` does not restrict which FAISS index type or distance metric you use, but understanding them helps tune `top_k` and `threshold`:

- **Common index types**:
  - `IndexFlatL2`: Exact L2 search, simple implementation, suitable for small to medium datasets.
  - `IndexIVFFlat`: Inverted file + approximate search, requires index training, suitable for large-scale stores.
  - `IndexHNSW`: Graph-based approximate search, fast queries, good for high-dimensional vectors.
- **Common distance metrics**:
  - L2 (Euclidean): Smaller values mean more similar.
  - Inner Product: With normalized vectors, larger values mean more similar.
  - Cosine distance/similarity: Equivalent to inner product after normalization.

When using a distance threshold (`--threshold`), you need to tune it based on the distance type and your data distribution. A common approach is to inspect a sample of distances first and then select a reasonable cutoff.

### Performance Tuning and Troubleshooting

- **Performance**:
  - With local models, you can choose smaller/faster models (e.g., `all-MiniLM-L6-v2`) to improve speed.
  - For large-scale vector stores, consider IVF- or HNSW-based index structures to accelerate search.
  - For further speedups, you can move FAISS indexes to GPU in your own code (this script stays neutral and does not enforce GPU usage).
- **Memory**:
  - Load `nodes_data` only when needed to avoid loading excessively large JSON files into memory.
  - If you hit GPU memory limits, force CPU usage or reduce batch size.
- **Quality**:
  - Ensure that indexing and querying use **the same embedding model and dimension**.
  - If retrieval quality is poor:
    - Check whether the raw data is clean and has enough context.
    - Check whether the chosen model (dimension, semantic capability) is appropriate.
    - Adjust `top_k` and `threshold` and inspect the distance distribution.
- **Paths and files**:
  - Make sure `vec_dir` at minimum contains `faiss.index` and `nodes.jsonl`.
  - If you provide `nodes_data`, ensure its keys line up with the ID field in `nodes.jsonl` (or with the fallback strategy).

