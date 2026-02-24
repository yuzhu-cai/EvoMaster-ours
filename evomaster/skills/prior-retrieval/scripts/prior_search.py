#!/usr/bin/env python3
"""Search PHY prior knowledge base with existing RAG searcher."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _load_rag_searcher():
    script_path = PROJECT_ROOT / "evomaster" / "skills" / "rag" / "scripts" / "search.py"
    spec = importlib.util.spec_from_file_location("rag_search_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load RAG search module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RAGSearcher


def _default_prior_base() -> Path:
    return Path(__file__).resolve().parents[4] / "playground" / "phy_master" / "LANDAU" / "prior"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search PHY prior knowledge")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--base_dir", default=str(_default_prior_base()), help="Prior base dir")
    parser.add_argument(
        "--model",
        default="evomaster/skills/rag/local_models/all-mpnet-base-v2",
        help="Embedding model path/name",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    prior_base = Path(args.base_dir)
    vec_dir = prior_base / "index" / "vectorstore"
    nodes_data = prior_base / "knowledge" / "nodes_data.json"

    RAGSearcher = _load_rag_searcher()
    searcher = RAGSearcher(
        vec_dir=str(vec_dir),
        model_name=args.model,
        nodes_data_json=str(nodes_data),
        device=args.device,
    )
    hits = searcher.search_by_text(args.query, top_k=args.top_k, distance_threshold=args.threshold)

    results = []
    for node_id, distance in hits:
        results.append(
            {
                "chunk_id": node_id,
                "distance": float(distance),
                "content": searcher.get_knowledge(node_id),
                "node": searcher.get_node_data(node_id),
            }
        )

    print(
        json.dumps(
            {
                "query": args.query,
                "top_k": args.top_k,
                "threshold": args.threshold,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
