#!/usr/bin/env python3
"""Search PHY prior knowledge base with local FAISS + embedding model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _default_prior_base() -> Path:
    root = Path(__file__).resolve().parents[4]
    candidates = [
        root / "prior",
        root / "playground" / "phy_master" / "landau" / "prior",
        root / "playground" / "phy_master" / "LANDAU" / "prior",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _load_id_map(base_dir: Path) -> dict[int, str]:
    id_map_jsonl = base_dir / "index" / "id_map.jsonl"
    if id_map_jsonl.exists():
        mapping: dict[int, str] = {}
        for line in id_map_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            mapping[int(obj["index_id"])] = str(obj["chunk_id"])
        return mapping

    nodes_jsonl = base_dir / "index" / "vectorstore" / "nodes.jsonl"
    if nodes_jsonl.exists():
        mapping = {}
        for idx, line in enumerate(nodes_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines()):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            mapping[idx] = str(obj.get("node_id", ""))
        return {k: v for k, v in mapping.items() if v}

    raise FileNotFoundError(f"id mapping file not found under {base_dir / 'index'}")


def _load_knowledge(base_dir: Path) -> dict[str, dict[str, Any]]:
    chunks_jsonl = base_dir / "knowledge" / "chunks.jsonl"
    if chunks_jsonl.exists():
        kb: dict[str, dict[str, Any]] = {}
        for line in chunks_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = str(obj.get("chunk_id") or obj.get("id") or "")
            if cid:
                kb[cid] = obj
        if kb:
            return kb

    nodes_data_json = base_dir / "knowledge" / "nodes_data.json"
    if nodes_data_json.exists():
        raw = json.loads(nodes_data_json.read_text(encoding="utf-8"))
        kb = {}
        if isinstance(raw, dict):
            for cid, node in raw.items():
                if not isinstance(node, dict):
                    continue
                content = node.get("content", {})
                text = ""
                if isinstance(content, dict):
                    text = str(content.get("text") or content.get("knowledge") or "")
                elif isinstance(content, str):
                    text = content
                kb[str(cid)] = {
                    "chunk_id": str(cid),
                    "text": text,
                    "citation": node.get("citation", ""),
                    "locator": node.get("locator", {}),
                    "source": node.get("source", {}),
                    "keywords": node.get("keywords", []),
                    "prev_chunk_id": node.get("prev_chunk_id"),
                    "next_chunk_id": node.get("next_chunk_id"),
                }
        if kb:
            return kb

    raise FileNotFoundError(f"knowledge file not found under {base_dir / 'knowledge'}")


def _find_faiss_index(base_dir: Path) -> Path:
    candidates = [
        base_dir / "index" / "index.faiss",
        base_dir / "index" / "vectorstore" / "faiss.index",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"FAISS index not found under {base_dir / 'index'}")


def _build_node(chunk_id: str, item: dict[str, Any]) -> dict[str, Any]:
    text = str(item.get("text") or item.get("content") or item.get("knowledge") or "")
    return {
        "node_id": chunk_id,
        "content": {"text": text, "knowledge": text},
        "source": item.get("source", {}),
        "locator": item.get("locator", {}),
        "citation": item.get("citation", ""),
        "keywords": item.get("keywords", []),
        "prev_chunk_id": item.get("prev_chunk_id"),
        "next_chunk_id": item.get("next_chunk_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search PHY prior knowledge")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1.5, help="L2 distance threshold")
    parser.add_argument("--base_dir", default=str(_default_prior_base()), help="Prior base dir")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name/path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    import faiss
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    base_dir = Path(args.base_dir).resolve()
    faiss_path = _find_faiss_index(base_dir)
    id_map = _load_id_map(base_dir)
    kb = _load_knowledge(base_dir)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = SentenceTransformer(args.model, device=device)
    index = faiss.read_index(str(faiss_path))

    query_vec = model.encode([args.query], normalize_embeddings=True).astype("float32")
    distances, indices = index.search(query_vec, max(1, args.top_k))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if int(idx) < 0:
            continue
        if float(dist) > args.threshold:
            continue
        chunk_id = id_map.get(int(idx))
        if not chunk_id:
            continue
        item = kb.get(chunk_id)
        if not item:
            continue
        text = str(item.get("text") or item.get("content") or item.get("knowledge") or "")
        results.append(
            {
                "chunk_id": chunk_id,
                "distance": float(np.float32(dist)),
                "content": text,
                "node": _build_node(chunk_id, item),
            }
        )

    print(
        json.dumps(
            {
                "query": args.query,
                "top_k": args.top_k,
                "threshold": args.threshold,
                "engine": "rag_faiss",
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
