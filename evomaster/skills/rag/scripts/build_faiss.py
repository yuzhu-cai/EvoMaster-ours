#!/usr/bin/env python3
"""Build FAISS index from embeddings.npy

从 vec_dir 下的 embeddings.npy 生成 faiss.index，供 search.py 在 --use_faiss 时使用。
向量会先做 L2 归一化，再写入 IndexFlatIP（内积即余弦相似度）。
"""

import logging
import sys
from pathlib import Path

import numpy as np

# 与 search.py 一致的路径解析
from search import _find_project_root, _resolve_path

logger = logging.getLogger(__name__)


def build_faiss_index(vec_dir: str | Path, project_root: Path | None = None) -> Path:
    """从 vec_dir/embeddings.npy 生成 vec_dir/faiss.index。

    Args:
        vec_dir: 向量目录（含 embeddings.npy）
        project_root: 项目根目录，用于解析 evomaster/ 相对路径；None 时自动查找。

    Returns:
        写入的 faiss.index 路径。

    Raises:
        FileNotFoundError: embeddings.npy 不存在
        ImportError: 未安装 faiss
        RuntimeError: 向量维度/格式异常
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "Building faiss.index requires the faiss package. "
            "Install with: pip install faiss-cpu  (or faiss-gpu for GPU)"
        ) from None

    vec_dir = Path(vec_dir)
    if not vec_dir.is_absolute() and str(vec_dir).replace("\\", "/").startswith("evomaster/"):
        root = project_root or _find_project_root()
        vec_dir = _resolve_path(str(vec_dir), root)

    vec_dir = vec_dir.resolve()
    emb_path = vec_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    emb = np.load(emb_path)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    if emb.ndim != 2:
        raise RuntimeError(f"Expected 2D array (n_vectors, dim), got shape {emb.shape}")

    # 归一化，便于用内积表示余弦相似度
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb = emb.astype(np.float32) / norms

    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    out_path = vec_dir / "faiss.index"
    faiss.write_index(index, str(out_path))
    logger.info(f"Built faiss.index from {emb_path} (shape {emb.shape}) -> {out_path}")
    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build faiss.index from embeddings.npy in a vector store directory."
    )
    parser.add_argument(
        "--vec_dir",
        required=True,
        help="Vector store directory containing embeddings.npy (evomaster/ prefix resolved relative to project root)",
    )
    args = parser.parse_args()

    project_root = _find_project_root()
    vec_dir_resolved = str(_resolve_path(args.vec_dir, project_root))

    try:
        build_faiss_index(vec_dir_resolved, project_root=project_root)
        print(f"Done. faiss.index written to {vec_dir_resolved}/faiss.index")
    except (FileNotFoundError, ImportError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
