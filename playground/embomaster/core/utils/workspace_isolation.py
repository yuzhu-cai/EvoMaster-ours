"""Workspace isolation manager.

Implements per-round isolated codebase preparation with large directory mount strategy:
- small files/dirs are copied
- large data/assets dirs are represented as placeholders and mounted in K8S
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

LARGE_DIR_KEYWORDS = ["assets", "data", "ckpt", "checkpoint"]
DEFAULT_SIZE_THRESHOLD_MB = 30
DEFAULT_SIZE_THRESHOLD_BYTES = DEFAULT_SIZE_THRESHOLD_MB * 1024 * 1024
COPY_EXCLUDE_PATTERNS = ["*.ckpt", "*.pth", "*.pt", "*.safetensors", "eval_result", "run_results"]


@dataclass
class WorkspaceCodebaseInfo:
    """Isolated workspace codebase metadata."""

    path: Path
    large_dirs: list[dict] = field(default_factory=list)
    source_type: str = "unknown"
    parent_workspace_id: str | None = None


def get_dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.iterdir():
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
            elif entry.is_dir() and not entry.is_symlink():
                total += get_dir_size(entry)
    except (PermissionError, OSError):
        pass
    return total


def _contains_keyword(name: str) -> bool:
    name_lower = name.lower()
    return any(kw in name_lower for kw in LARGE_DIR_KEYWORDS)


def _should_exclude_file(name: str) -> bool:
    name_lower = name.lower()
    for pattern in COPY_EXCLUDE_PATTERNS:
        if pattern.startswith("*."):
            ext = pattern[1:]
            if name_lower.endswith(ext):
                return True
        elif name_lower == pattern.lower():
            return True
    return False


def _should_exclude_dir(name: str) -> bool:
    return name.lower() in ["eval_result", "__pycache__", ".git", "run_results"]


def _find_large_dirs_recursive(
    src_dir: Path,
    dst_dir: Path,
    rel_prefix: str,
    size_threshold: int,
    large_dirs: list[dict],
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    for entry in src_dir.iterdir():
        src_path = entry
        dst_path = dst_dir / entry.name
        rel_path = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name

        if entry.is_symlink():
            link_target = entry.readlink()
            if not dst_path.exists():
                dst_path.symlink_to(link_target)
            continue

        if entry.is_file():
            if _should_exclude_file(entry.name):
                continue
            shutil.copy2(src_path, dst_path)
            continue

        if not entry.is_dir():
            continue
        if _should_exclude_dir(entry.name):
            continue

        dir_size = get_dir_size(src_path)
        dir_size_mb = dir_size / 1024 / 1024
        if dir_size > size_threshold and _contains_keyword(entry.name):
            dst_path.mkdir(parents=True, exist_ok=True)
            large_dirs.append(
                {"src": str(src_path), "rel": rel_path, "size_mb": round(dir_size_mb, 1)}
            )
            logger.info("Large dir mount candidate: %s (%.1f MB)", rel_path, dir_size_mb)
            continue

        if dir_size > size_threshold:
            _find_large_dirs_recursive(
                src_dir=src_path,
                dst_dir=dst_path,
                rel_prefix=rel_path,
                size_threshold=size_threshold,
                large_dirs=large_dirs,
            )
            continue

        def _ignore_func(_d: str, files: list[str]) -> list[str]:
            return [f for f in files if _should_exclude_file(f) or _should_exclude_dir(f)]

        shutil.copytree(src_path, dst_path, symlinks=True, ignore=_ignore_func)


def smart_copy_codebase(
    src: Path,
    dst: Path,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD_BYTES,
) -> list[dict]:
    large_dirs: list[dict] = []
    dst.mkdir(parents=True, exist_ok=True)
    _find_large_dirs_recursive(
        src_dir=src,
        dst_dir=dst,
        rel_prefix="",
        size_threshold=size_threshold,
        large_dirs=large_dirs,
    )
    return large_dirs


def _copy_from_parent_filtered(
    src_dir: Path,
    dst_dir: Path,
    parent_large_dirs: list[dict],
) -> list[dict]:
    dst_dir.mkdir(parents=True, exist_ok=True)

    def _copy_recursive(src: Path, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        for entry in src.iterdir():
            src_path = entry
            dst_path = dst / entry.name
            if entry.is_symlink():
                link_target = entry.readlink()
                if not dst_path.exists():
                    dst_path.symlink_to(link_target)
            elif entry.is_file():
                if _should_exclude_file(entry.name):
                    continue
                shutil.copy2(src_path, dst_path)
            elif entry.is_dir():
                if _should_exclude_dir(entry.name):
                    continue
                _copy_recursive(src_path, dst_path)

    _copy_recursive(src_dir, dst_dir)

    inherited: list[dict] = []
    for item in parent_large_dirs:
        rel_path = str(item.get("rel", ""))
        if not rel_path:
            continue
        target_dir = dst_dir / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)
        inherited.append(item.copy())
    return inherited


def load_large_dirs(codebase_dir: Path) -> list[dict]:
    file_path = codebase_dir / "large_dirs.json"
    if file_path.exists():
        try:
            with file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load %s: %s", file_path, e)
    return []


def save_large_dirs(codebase_dir: Path, large_dirs: list[dict]) -> None:
    if not large_dirs:
        return
    file_path = codebase_dir / "large_dirs.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(large_dirs, f, indent=2, ensure_ascii=False)


def get_workspace_codebase_path(session_dir: Path, workspace_id: str) -> Path:
    short_id = workspace_id[:8]
    return session_dir / "round_workspaces" / short_id / f"codebase_{short_id}"


def get_parent_workspace_codebase_path(
    session_dir: Path, parent_workspace_id: str | None
) -> Path | None:
    if not parent_workspace_id:
        return None
    short_id = parent_workspace_id[:8]
    parent_workspace_dir = session_dir / "round_workspaces" / short_id
    if not parent_workspace_dir.exists():
        return None

    symlink_path = parent_workspace_dir / "codebase"
    if symlink_path.exists() or symlink_path.is_symlink():
        resolved = symlink_path.resolve()
        if resolved.exists():
            return resolved

    candidates = [
        p for p in parent_workspace_dir.iterdir() if p.is_dir() and p.name.startswith("codebase_")
    ]
    if candidates:
        return candidates[0].resolve()
    return None


def _create_codebase_symlink(symlink_path: Path, target_path: Path) -> None:
    try:
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(target_path.name)
    except OSError as e:
        logger.warning("Failed to create codebase symlink %s -> %s: %s", symlink_path, target_path, e)


def prepare_workspace_codebase(
    session_dir: Path,
    workspace_id: str,
    source_codebase_dir: Path | None = None,
    parent_workspace_id: str | None = None,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD_BYTES,
) -> WorkspaceCodebaseInfo:
    short_id = workspace_id[:8]
    workspace_dir = session_dir / "round_workspaces" / short_id
    workspace_dir.mkdir(parents=True, exist_ok=True)

    dest_codebase = workspace_dir / f"codebase_{short_id}"
    symlink_path = workspace_dir / "codebase"

    existing: Path | None = None
    if symlink_path.exists() or symlink_path.is_symlink():
        existing = symlink_path.resolve()
    elif dest_codebase.exists():
        existing = dest_codebase

    if existing and existing.exists():
        return WorkspaceCodebaseInfo(
            path=existing,
            large_dirs=load_large_dirs(existing),
            source_type="existing",
            parent_workspace_id=parent_workspace_id,
        )

    parent_codebase = get_parent_workspace_codebase_path(session_dir, parent_workspace_id)
    if parent_codebase and parent_codebase.exists():
        parent_large_dirs = load_large_dirs(parent_codebase)
        inherited_large_dirs = _copy_from_parent_filtered(parent_codebase, dest_codebase, parent_large_dirs)
        save_large_dirs(dest_codebase, inherited_large_dirs)
        _create_codebase_symlink(symlink_path, dest_codebase)
        return WorkspaceCodebaseInfo(
            path=dest_codebase,
            large_dirs=inherited_large_dirs,
            source_type="parent",
            parent_workspace_id=parent_workspace_id,
        )

    if source_codebase_dir and source_codebase_dir.exists():
        large_dirs = smart_copy_codebase(source_codebase_dir, dest_codebase, size_threshold=size_threshold)
        save_large_dirs(dest_codebase, large_dirs)
        _create_codebase_symlink(symlink_path, dest_codebase)
        return WorkspaceCodebaseInfo(
            path=dest_codebase,
            large_dirs=large_dirs,
            source_type="original",
            parent_workspace_id=parent_workspace_id,
        )

    dest_codebase.mkdir(parents=True, exist_ok=True)
    _create_codebase_symlink(symlink_path, dest_codebase)
    return WorkspaceCodebaseInfo(
        path=dest_codebase,
        large_dirs=[],
        source_type="empty",
        parent_workspace_id=parent_workspace_id,
    )


def cleanup_eval_result(codebase_dir: Path) -> None:
    for dirname in ["eval_result", "run_results"]:
        target = codebase_dir / dirname
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
