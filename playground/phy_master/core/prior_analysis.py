"""Prior knowledge ingestion and indexing for PHY Master.

This module parses documents into structured chunks, persists them into
`playground/phy_master/LANDAU/prior/knowledge`, and builds a FAISS vectorstore
compatible with `evomaster/skills/rag/scripts/search.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_EXTS = {".pdf", ".md", ".txt"}


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "evomaster").exists():
            return parent
    return Path.cwd()


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip().lower())
    return value.strip("_") or "doc"


@dataclass
class PriorPaths:
    base_dir: Path
    source_dir: Path
    out_dir: Path
    knowledge_dir: Path
    index_dir: Path
    chunks_jsonl: Path
    nodes_data_json: Path
    nodes_jsonl: Path
    embeddings_npy: Path
    faiss_index: Path

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> "PriorPaths":
        source_dir = base_dir / "source"
        out_dir = base_dir / "out"
        knowledge_dir = base_dir / "knowledge"
        index_dir = base_dir / "index"
        vectorstore_dir = index_dir / "vectorstore"
        return cls(
            base_dir=base_dir,
            source_dir=source_dir,
            out_dir=out_dir,
            knowledge_dir=knowledge_dir,
            index_dir=index_dir,
            chunks_jsonl=knowledge_dir / "chunks.jsonl",
            nodes_data_json=knowledge_dir / "nodes_data.json",
            nodes_jsonl=vectorstore_dir / "nodes.jsonl",
            embeddings_npy=vectorstore_dir / "embeddings.npy",
            faiss_index=vectorstore_dir / "faiss.index",
        )


class PriorAnalyzer:
    """Parse input files and maintain prior knowledge vectorstore."""

    def __init__(
        self,
        base_dir: str | Path | None = None,
        model_name: str = "evomaster/skills/rag/local_models/all-mpnet-base-v2",
        device: str | None = None,
        chunk_chars: int = 1000,
        overlap_chars: int = 180,
    ) -> None:
        project_root = _find_project_root()
        if base_dir is None:
            base_dir = project_root / "playground" / "phy_master" / "LANDAU" / "prior"
        self.paths = PriorPaths.from_base_dir(Path(base_dir).resolve())

        self.model_name = model_name
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "prior_analysis requires `torch` and `transformers`. "
                "Please install dependencies first."
            ) from e

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_chars = max(200, int(chunk_chars))
        self.overlap_chars = max(0, int(overlap_chars))

        self._ensure_dirs()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _ensure_dirs(self) -> None:
        for d in [
            self.paths.base_dir,
            self.paths.source_dir,
            self.paths.out_dir,
            self.paths.knowledge_dir,
            self.paths.index_dir,
            self.paths.nodes_jsonl.parent,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        if not self.paths.chunks_jsonl.exists():
            self.paths.chunks_jsonl.write_text("", encoding="utf-8")
        if not self.paths.nodes_data_json.exists():
            self.paths.nodes_data_json.write_text("{}\n", encoding="utf-8")

    def ingest(self, input_path: str | Path) -> dict[str, Any]:
        path = Path(input_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported file type: {path.suffix}. Supported: {sorted(SUPPORTED_EXTS)}")

        chunks = self._parse_file(path)
        if not chunks:
            return {
                "status": "ok",
                "input": str(path),
                "added": 0,
                "total": self._count_chunks(),
                "message": "No valid chunks extracted.",
            }

        added, total = self._append_chunks(chunks)
        if added > 0:
            self.build_index()

        return {
            "status": "ok",
            "input": str(path),
            "added": added,
            "total": total,
            "knowledge_jsonl": str(self.paths.chunks_jsonl),
            "vectorstore_dir": str(self.paths.nodes_jsonl.parent),
            "nodes_data": str(self.paths.nodes_data_json),
        }

    def ingest_source_dir(self, source_dir: str | Path | None = None) -> dict[str, Any]:
        src = Path(source_dir).expanduser().resolve() if source_dir else self.paths.source_dir
        if not src.exists() or not src.is_dir():
            raise FileNotFoundError(f"Source directory not found: {src}")

        files = sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
        added_total = 0
        processed = 0
        skipped = 0

        for p in files:
            processed += 1
            chunks = self._parse_file(p)
            if not chunks:
                skipped += 1
                continue
            added, _ = self._append_chunks(chunks)
            added_total += added

        if added_total > 0:
            index_info = self.build_index()
        else:
            index_info = {"status": "ok", "indexed": 0, "message": "No new chunks, skipped index rebuild."}

        return {
            "status": "ok",
            "source_dir": str(src),
            "processed_files": processed,
            "skipped_files": skipped,
            "added_chunks": added_total,
            "total_chunks": self._count_chunks(),
            "knowledge_jsonl": str(self.paths.chunks_jsonl),
            "vectorstore_dir": str(self.paths.nodes_jsonl.parent),
            "nodes_data": str(self.paths.nodes_data_json),
            "index": index_info,
        }

    def build_index(self) -> dict[str, Any]:
        try:
            import faiss
            import numpy as np
        except Exception as e:
            raise RuntimeError(
                "Building prior index requires `faiss` and `numpy`."
            ) from e

        chunks = self._load_chunks()
        if not chunks:
            return {"status": "ok", "indexed": 0, "message": "No chunks to index."}

        texts = [c.get("text", "") for c in chunks]
        embeddings = self._encode_texts(texts)
        if embeddings.size == 0:
            return {"status": "ok", "indexed": 0, "message": "No embeddings generated."}

        np.save(self.paths.embeddings_npy, embeddings)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype("float32"))
        faiss.write_index(index, str(self.paths.faiss_index))

        with self.paths.nodes_jsonl.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({"node_id": c["chunk_id"]}, ensure_ascii=False) + "\n")

        nodes_data: dict[str, Any] = {}
        for c in chunks:
            cid = c["chunk_id"]
            nodes_data[cid] = {
                "node_id": cid,
                "content": {
                    "text": c.get("text", ""),
                    "knowledge": c.get("text", ""),
                },
                "source": c.get("source", {}),
                "locator": c.get("locator", {}),
                "citation": c.get("citation", ""),
                "keywords": c.get("keywords", []),
                "prev_chunk_id": c.get("prev_chunk_id"),
                "next_chunk_id": c.get("next_chunk_id"),
            }

        with self.paths.nodes_data_json.open("w", encoding="utf-8") as f:
            json.dump(nodes_data, f, ensure_ascii=False, indent=2)

        return {
            "status": "ok",
            "indexed": len(chunks),
            "faiss_index": str(self.paths.faiss_index),
            "nodes_jsonl": str(self.paths.nodes_jsonl),
            "nodes_data_json": str(self.paths.nodes_data_json),
        }

    def _encode_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        import numpy as np

        torch = self._torch
        vectors: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                h = outputs.last_hidden_state
                attn = inputs["attention_mask"].unsqueeze(-1)
                emb = (h * attn).sum(dim=1) / attn.sum(dim=1)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            vectors.append(emb.cpu().numpy())
        return np.concatenate(vectors, axis=0).astype("float32") if vectors else np.array([], dtype="float32")

    def _parse_file(self, path: Path) -> list[dict[str, Any]]:
        ext = path.suffix.lower()
        if ext in {".md", ".txt"}:
            full_text = path.read_text(encoding="utf-8", errors="ignore")
        elif ext == ".pdf":
            full_text = self._read_pdf_text(path)
        else:
            return []

        segments = self._split_text(full_text)
        if not segments:
            return []

        base = _slugify(path.stem)
        year_match = re.search(r"(19|20)\d{2}", path.name)
        year = int(year_match.group(0)) if year_match else 2024

        chunks: list[dict[str, Any]] = []
        for idx, seg in enumerate(segments, start=1):
            digest = hashlib.sha1(f"{path.name}:{seg}".encode("utf-8")).hexdigest()[:10]
            chunk_id = f"{base}:{idx:04d}:{digest}"
            chunk = {
                "chunk_id": chunk_id,
                "text": seg,
                "source": {
                    "source_id": path.name,
                    "title": path.stem,
                    "authors": ["Unknown Author"],
                    "year": year,
                    "edition": "Original",
                },
                "locator": {
                    "chapter": "0",
                    "section": "0",
                    "page_start": 0,
                    "page_end": 0,
                    "equation_id": "",
                },
                "citation": f"{path.stem} ({year})",
                "keywords": self._extract_keywords(seg),
                "prev_chunk_id": None,
                "next_chunk_id": None,
            }
            chunks.append(chunk)

        for i in range(len(chunks)):
            if i > 0:
                chunks[i]["prev_chunk_id"] = chunks[i - 1]["chunk_id"]
            if i < len(chunks) - 1:
                chunks[i]["next_chunk_id"] = chunks[i + 1]["chunk_id"]
        return chunks

    def _read_pdf_text(self, path: Path) -> str:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Reading PDF requires `pypdf`. Install it or provide md/txt input."
            ) from e

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)

    def _split_text(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []

        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            para = re.sub(r"\s+", " ", para)
            if len(current) + len(para) + 1 <= self.chunk_chars:
                current = f"{current} {para}".strip()
            else:
                if current:
                    chunks.append(current)
                if len(para) <= self.chunk_chars:
                    current = para
                else:
                    step = max(1, self.chunk_chars - self.overlap_chars)
                    for i in range(0, len(para), step):
                        part = para[i : i + self.chunk_chars].strip()
                        if part:
                            chunks.append(part)
                    current = ""
        if current:
            chunks.append(current)
        return chunks

    def _extract_keywords(self, text: str, top_k: int = 6) -> list[str]:
        words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text.lower())
        stop = {
            "that", "this", "with", "from", "into", "were", "have", "then", "than",
            "when", "where", "which", "while", "their", "there", "about", "using",
            "based", "analysis", "model", "results", "method", "equation", "figure",
        }
        freq: dict[str, int] = {}
        for w in words:
            if w in stop:
                continue
            freq[w] = freq.get(w, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in ranked[:top_k]]

    def _load_chunks(self) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        if not self.paths.chunks_jsonl.exists():
            return chunks
        with self.paths.chunks_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("chunk_id"):
                    chunks.append(obj)
        return chunks

    def _append_chunks(self, new_chunks: list[dict[str, Any]]) -> tuple[int, int]:
        existing = self._load_chunks()
        seen = {c.get("chunk_id") for c in existing}
        uniq = [c for c in new_chunks if c.get("chunk_id") not in seen]

        if uniq:
            with self.paths.chunks_jsonl.open("a", encoding="utf-8") as f:
                for c in uniq:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")

        return len(uniq), len(existing) + len(uniq)

    def _count_chunks(self) -> int:
        return len(self._load_chunks())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/update PHY prior knowledge base for RAG")
    parser.add_argument("--input", default=None, help="Path to source file (.pdf/.md/.txt)")
    parser.add_argument(
        "--source_dir",
        default=None,
        help="Batch ingest all supported files from this directory (default: prior/source)",
    )
    parser.add_argument(
        "--rebuild_index_only",
        action="store_true",
        help="Rebuild FAISS index and nodes_data from existing knowledge/chunks.jsonl only.",
    )
    parser.add_argument("--base_dir", default=None, help="Prior base dir (default: playground/phy_master/LANDAU/prior)")
    parser.add_argument(
        "--model",
        default="evomaster/skills/rag/local_models/all-mpnet-base-v2",
        help="Embedding model path/name",
    )
    parser.add_argument("--chunk_chars", type=int, default=1000)
    parser.add_argument("--overlap_chars", type=int, default=180)
    args = parser.parse_args()

    analyzer = PriorAnalyzer(
        base_dir=args.base_dir,
        model_name=args.model,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
    )

    if args.rebuild_index_only:
        result = analyzer.build_index()
    elif args.input:
        result = analyzer.ingest(args.input)
    else:
        result = analyzer.ingest_source_dir(args.source_dir)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
