# Prior Library Layout

Default prior base directory:
`playground/phy_master/LANDAU/prior`

Expected generated files for retrieval:

- `knowledge/chunks.jsonl`
- `knowledge/nodes_data.json`
- `index/vectorstore/faiss.index`
- `index/vectorstore/embeddings.npy`
- `index/vectorstore/nodes.jsonl`

Build/update these files with:

```bash
python playground/phy_master/core/prior_analysis.py --source_dir playground/phy_master/LANDAU/prior/source
```

Or ingest one file:

```bash
python playground/phy_master/core/prior_analysis.py --input /path/to/file.pdf
```

The generated vectorstore is compatible with `evomaster/skills/rag/scripts/search.py`.
