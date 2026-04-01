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
from typing import Any, Callable

logger = logging.getLogger(__name__)

LARGE_DIR_KEYWORDS = ["assets", "data", "ckpt", "checkpoint"]
VENV_DIR_PREFIX = ".venv"
DEFAULT_SIZE_THRESHOLD_MB = 30
DEFAULT_SIZE_THRESHOLD_BYTES = DEFAULT_SIZE_THRESHOLD_MB * 1024 * 1024
COPY_EXCLUDE_PATTERNS = [
    "*.ckpt",
    "*.pth",
    "*.pt",
    "*.safetensors",
    "*.pyc",
    "*.pyo",
    "eval_result",
    "run_results",
    ".embomaster_copy_plan.json",
]
COPY_EXCLUDE_DIR_NAMES = {
    "checkpoints",
    "eval_result",
    "__pycache__",
    ".git",
    "logs",
    "run_results",
    "venv",
    "env",
    ".conda",
    ".mamba",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "__pypackages__",
    "wandb",
}
COPY_EXCLUDE_REL_PREFIXES = [
    "XDG_CACHE_HOME/uv/archive-v0",
    "policy/TinyVLA/model_param",
    "policy/TinyVLA/src/nvidia-curobo",
    "policy/ACT/act_ckpt",
]
LOCAL_WORKSPACE_DIR_PREFIXES = [
    "policy/ACT/act_ckpt",
]
UV_EPHEMERAL_SEGMENT_PREFIXES = ("git-v", "sdists-v", "simple-v", ".tmp")
COPY_PLAN_CACHE_FILENAME = ".embomaster_copy_plan.json"
COPY_PLAN_VERSION = 7
WORKSPACE_OUTPUT_DIR_NAMES = ("eval_result", "run_results", "checkpoints")
CopyPlanProgressCallback = Callable[[int, int, int, str], None]


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


def _is_venv_dir_name(name: str) -> bool:
    name_lower = name.lower()
    return name_lower == VENV_DIR_PREFIX or name_lower.startswith(f"{VENV_DIR_PREFIX}_")


def _normalize_rel_path(rel_path: str) -> str:
    return rel_path.replace("\\", "/").strip().strip("/")


def _workspace_dir_name(workspace_id: str) -> str:
    workspace_id_norm = _normalize_rel_path(workspace_id)
    if not workspace_id_norm:
        return "workspace"
    return workspace_id_norm.replace("/", "-")


def _is_excluded_rel_path(rel_path: str) -> bool:
    rel_norm = _normalize_rel_path(rel_path)
    if not rel_norm:
        return False
    uv_prefix = "XDG_CACHE_HOME/uv/"
    if rel_norm.startswith(uv_prefix):
        seg = rel_norm[len(uv_prefix) :].split("/", 1)[0]
        if seg and any(seg.startswith(prefix) for prefix in UV_EPHEMERAL_SEGMENT_PREFIXES):
            return True
    for prefix in COPY_EXCLUDE_REL_PREFIXES:
        prefix_norm = _normalize_rel_path(prefix)
        if rel_norm == prefix_norm or rel_norm.startswith(f"{prefix_norm}/"):
            return True
    return False


def _should_exclude_file(name: str, rel_path: str | None = None) -> bool:
    if rel_path and _is_excluded_rel_path(rel_path):
        return True
    name_lower = name.lower()
    for pattern in COPY_EXCLUDE_PATTERNS:
        if pattern.startswith("*."):
            ext = pattern[1:]
            if name_lower.endswith(ext):
                return True
        elif name_lower == pattern.lower():
            return True
    return False


def _should_exclude_dir(name: str, rel_path: str | None = None) -> bool:
    if _is_venv_dir_name(name):
        return False
    if name.lower() in COPY_EXCLUDE_DIR_NAMES:
        return True
    if rel_path and _is_excluded_rel_path(rel_path):
        return True
    return False


def _should_exclude_dir_rel_path(rel_path: str) -> bool:
    rel_norm = _normalize_rel_path(rel_path)
    if not rel_norm:
        return False
    return _should_exclude_dir(Path(rel_norm).name, rel_path=rel_norm)


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
            if _should_exclude_file(entry.name, rel_path=rel_path):
                continue
            shutil.copy2(src_path, dst_path)
            continue

        if not entry.is_dir():
            continue
        if _should_exclude_dir(entry.name, rel_path=rel_path):
            continue

        dir_size = get_dir_size(src_path)
        dir_size_mb = dir_size / 1024 / 1024
        if _is_venv_dir_name(entry.name) or (dir_size > size_threshold and _contains_keyword(entry.name)):
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

        subtree_root_resolved = src_path.resolve()

        def _ignore_func(_d: str, files: list[str]) -> list[str]:
            ignored: list[str] = []
            current_dir = Path(_d)
            try:
                suffix_rel = current_dir.resolve().relative_to(subtree_root_resolved).as_posix()
            except Exception:
                suffix_rel = ""
            suffix_rel = "" if suffix_rel == "." else suffix_rel
            current_rel = f"{rel_path}/{suffix_rel}" if suffix_rel else rel_path
            for name in files:
                item_rel = f"{current_rel}/{name}" if current_rel else name
                if _should_exclude_file(name, rel_path=item_rel) or _should_exclude_dir(
                    name, rel_path=item_rel
                ):
                    ignored.append(name)
            return ignored

        shutil.copytree(src_path, dst_path, symlinks=True, ignore=_ignore_func)


def _collect_large_dirs_recursive(
    src_dir: Path,
    rel_prefix: str,
    size_threshold: int,
    large_dirs: list[dict],
    progress_total: int = 0,
    progress_state: dict[str, int] | None = None,
    progress_callback: CopyPlanProgressCallback | None = None,
) -> None:
    try:
        entries = list(src_dir.iterdir())
    except (PermissionError, OSError) as e:
        logger.warning("Skip unreadable directory while building copy plan: %s (%s)", src_dir, e)
        return

    for entry in entries:
        rel_path = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name
        src_path = entry

        try:
            is_file = entry.is_file()
            is_dir = entry.is_dir()
        except (PermissionError, OSError) as e:
            logger.warning("Skip unreadable path while building copy plan: %s (%s)", entry, e)
            continue

        if is_file:
            if _should_exclude_file(entry.name, rel_path=rel_path):
                continue
            continue

        if not is_dir:
            continue
        if _should_exclude_dir(entry.name, rel_path=rel_path):
            continue
        if progress_state is not None:
            progress_state["visited"] = progress_state.get("visited", 0) + 1
            if progress_callback:
                progress_callback(
                    int(progress_state["visited"]),
                    int(progress_total),
                    len(large_dirs),
                    rel_path,
                )

        dir_size = get_dir_size(src_path)
        dir_size_mb = dir_size / 1024 / 1024
        if _is_venv_dir_name(entry.name) or (dir_size > size_threshold and _contains_keyword(entry.name)):
            large_dirs.append(
                {"src": str(src_path.resolve()), "rel": rel_path, "size_mb": round(dir_size_mb, 1)}
            )
            if progress_state is not None and progress_callback:
                progress_callback(
                    int(progress_state.get("visited", 0)),
                    int(progress_total),
                    len(large_dirs),
                    rel_path,
                )
            continue

        if dir_size > size_threshold:
            _collect_large_dirs_recursive(
                src_dir=src_path,
                rel_prefix=rel_path,
                size_threshold=size_threshold,
                large_dirs=large_dirs,
                progress_total=progress_total,
                progress_state=progress_state,
                progress_callback=progress_callback,
            )


def _count_scannable_dirs(src_dir: Path, rel_prefix: str = "") -> int:
    total = 0
    try:
        entries = list(src_dir.iterdir())
    except (PermissionError, OSError) as e:
        logger.warning("Skip unreadable directory while counting copy plan entries: %s (%s)", src_dir, e)
        return 0

    for entry in entries:
        rel_path = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name
        try:
            is_dir = entry.is_dir()
        except (PermissionError, OSError) as e:
            logger.warning("Skip unreadable path while counting copy plan entries: %s (%s)", entry, e)
            continue
        if not is_dir:
            continue
        if _should_exclude_dir(entry.name, rel_path=rel_path):
            continue
        total += 1
        total += _count_scannable_dirs(entry, rel_prefix=rel_path)
    return total


def _scan_large_dirs(
    src: Path,
    size_threshold: int,
    progress_callback: CopyPlanProgressCallback | None = None,
) -> list[dict]:
    large_dirs: list[dict] = []
    progress_total = _count_scannable_dirs(src) if progress_callback else 0
    progress_state = {"visited": 0} if progress_callback else None
    _collect_large_dirs_recursive(
        src_dir=src,
        rel_prefix="",
        size_threshold=size_threshold,
        large_dirs=large_dirs,
        progress_total=progress_total,
        progress_state=progress_state,
        progress_callback=progress_callback,
    )
    if progress_callback:
        progress_callback(progress_total, progress_total, len(large_dirs), "")
    return large_dirs


def _normalize_large_dirs(large_dirs: list[dict], src_root: Path) -> list[dict]:
    normalized: list[dict] = []
    seen: set[str] = set()
    for item in large_dirs:
        if not isinstance(item, dict):
            continue
        rel_raw = str(item.get("rel", "")).strip().strip("/")
        if not rel_raw or rel_raw in seen:
            continue
        if _is_excluded_rel_path(rel_raw) or _should_exclude_dir_rel_path(rel_raw):
            continue
        src_raw = str(item.get("src", "")).strip()
        src_path = Path(src_raw).expanduser() if src_raw else (src_root / rel_raw)
        src_resolved = src_path.resolve()
        if not src_resolved.exists() or not src_resolved.is_dir():
            continue
        size_mb = item.get("size_mb")
        try:
            size_val = float(size_mb) if size_mb is not None else None
        except (TypeError, ValueError):
            size_val = None
        normalized_item: dict[str, Any] = {"src": str(src_resolved), "rel": rel_raw}
        if size_val is not None:
            normalized_item["size_mb"] = round(size_val, 1)
        normalized.append(normalized_item)
        seen.add(rel_raw)
    return normalized


def _load_copy_plan(
    cache_file: Path,
    src: Path,
    size_threshold: int,
) -> list[dict] | None:
    if not cache_file.exists():
        return None
    try:
        with cache_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load copy plan cache %s: %s", cache_file, e)
        return None

    if not isinstance(payload, dict):
        return None
    version = int(payload.get("version", -1))
    source_root = str(payload.get("source_root", "")).strip()
    threshold = int(payload.get("size_threshold", -1))
    if version != COPY_PLAN_VERSION:
        return None
    if source_root != str(src.resolve()):
        return None
    if threshold != size_threshold:
        return None

    large_dirs = payload.get("large_dirs", [])
    if not isinstance(large_dirs, list):
        return None
    return _normalize_large_dirs(large_dirs, src_root=src)


def _save_copy_plan(
    cache_file: Path,
    src: Path,
    size_threshold: int,
    large_dirs: list[dict],
) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": COPY_PLAN_VERSION,
        "source_root": str(src.resolve()),
        "size_threshold": int(size_threshold),
        "large_dirs": large_dirs,
    }
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _has_large_dir_descendant(rel_path: str, large_rel_set: set[str]) -> bool:
    prefix = f"{rel_path}/"
    for rel in large_rel_set:
        if rel.startswith(prefix):
            return True
    return False


def _copytree_with_filters(src_root: Path, src_path: Path, dst_path: Path) -> None:
    src_root_resolved = src_root.resolve()

    def _ignore_func(_d: str, files: list[str]) -> list[str]:
        ignored: list[str] = []
        current_dir = Path(_d)
        try:
            current_rel = current_dir.resolve().relative_to(src_root_resolved).as_posix()
        except Exception:
            current_rel = ""
        current_rel = "" if current_rel == "." else current_rel

        for name in files:
            rel_path = f"{current_rel}/{name}" if current_rel else name
            if _should_exclude_file(name, rel_path=rel_path) or _should_exclude_dir(
                name, rel_path=rel_path
            ):
                ignored.append(name)
        return ignored

    shutil.copytree(src_path, dst_path, symlinks=False, ignore=_ignore_func)


def _copy_from_plan(src: Path, dst: Path, large_dirs: list[dict]) -> None:
    large_rel_set: set[str] = {
        rel
        for rel in (
            str(item.get("rel", "")).strip().strip("/")
            for item in large_dirs
            if isinstance(item, dict) and str(item.get("rel", "")).strip()
        )
        if rel and not _should_exclude_dir_rel_path(rel)
    }
    large_rel_set = {rel for rel in large_rel_set if rel}

    for rel in sorted(large_rel_set):
        (dst / rel).mkdir(parents=True, exist_ok=True)

    def _copy_recursive(src_dir: Path, dst_dir: Path, rel_prefix: str) -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for entry in src_dir.iterdir():
            rel_path = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name
            dst_path = dst_dir / entry.name

            if entry.is_file():
                if _should_exclude_file(entry.name, rel_path=rel_path):
                    continue
                shutil.copy2(entry, dst_path, follow_symlinks=True)
                continue

            if not entry.is_dir():
                continue
            if _should_exclude_dir(entry.name, rel_path=rel_path):
                continue
            if rel_path in large_rel_set:
                dst_path.mkdir(parents=True, exist_ok=True)
                continue

            if _has_large_dir_descendant(rel_path, large_rel_set):
                _copy_recursive(entry, dst_path, rel_path)
                continue

            _copytree_with_filters(src_root=src, src_path=entry, dst_path=dst_path)

    _copy_recursive(src, dst, "")


def build_copy_plan_cache(
    src: Path,
    cache_file: Path,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD_BYTES,
    progress_callback: CopyPlanProgressCallback | None = None,
) -> list[dict]:
    large_dirs = _scan_large_dirs(
        src,
        size_threshold=size_threshold,
        progress_callback=progress_callback,
    )
    large_dirs = _normalize_large_dirs(large_dirs, src_root=src)
    _save_copy_plan(cache_file, src=src, size_threshold=size_threshold, large_dirs=large_dirs)
    return large_dirs


def smart_copy_codebase(
    src: Path,
    dst: Path,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD_BYTES,
    copy_plan_cache_file: Path | None = None,
    use_copy_plan_cache: bool = True,
    force_rebuild_copy_plan: bool = False,
) -> list[dict]:
    if use_copy_plan_cache:
        cache_file = copy_plan_cache_file or (src / COPY_PLAN_CACHE_FILENAME)
        large_dirs: list[dict] | None = None
        if not force_rebuild_copy_plan:
            large_dirs = _load_copy_plan(
                cache_file=cache_file,
                src=src,
                size_threshold=size_threshold,
            )
        if large_dirs is None:
            large_dirs = build_copy_plan_cache(
                src=src,
                cache_file=cache_file,
                size_threshold=size_threshold,
            )
            logger.info("Copy plan cache refreshed: %s", cache_file)
        else:
            logger.info("Copy plan cache hit: %s", cache_file)

        dst.mkdir(parents=True, exist_ok=True)
        _copy_from_plan(src=src, dst=dst, large_dirs=large_dirs)
        return large_dirs

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

    def _copy_recursive(src: Path, dst: Path, rel_prefix: str = "") -> None:
        dst.mkdir(parents=True, exist_ok=True)
        for entry in src.iterdir():
            src_path = entry
            dst_path = dst / entry.name
            rel_path = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name
            if entry.is_symlink():
                link_target = entry.readlink()
                if not dst_path.exists():
                    dst_path.symlink_to(link_target)
            elif entry.is_file():
                if _should_exclude_file(entry.name, rel_path=rel_path):
                    continue
                shutil.copy2(src_path, dst_path)
            elif entry.is_dir():
                if _should_exclude_dir(entry.name, rel_path=rel_path):
                    continue
                _copy_recursive(src_path, dst_path, rel_prefix=rel_path)

    _copy_recursive(src_dir, dst_dir)

    inherited: list[dict] = []
    for item in parent_large_dirs:
        rel_path = str(item.get("rel", ""))
        if not rel_path:
            continue
        if _is_excluded_rel_path(rel_path) or _should_exclude_dir_rel_path(rel_path):
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
                payload = json.load(f)
            if isinstance(payload, list):
                return _normalize_large_dirs(payload, src_root=codebase_dir)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load %s: %s", file_path, e)
    return []


def save_large_dirs(codebase_dir: Path, large_dirs: list[dict]) -> None:
    if not large_dirs:
        return
    file_path = codebase_dir / "large_dirs.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(large_dirs, f, indent=2, ensure_ascii=False)


def ensure_local_workspace_dirs(codebase_dir: Path) -> list[str]:
    created_dirs: list[str] = []
    for rel_path in LOCAL_WORKSPACE_DIR_PREFIXES:
        rel_norm = _normalize_rel_path(rel_path)
        if not rel_norm:
            continue
        target_dir = codebase_dir / rel_norm
        if target_dir.exists() and not target_dir.is_dir():
            logger.warning("Skip local dir creation because non-dir exists: %s", target_dir)
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        created_dirs.append(rel_norm)
    return created_dirs


def get_workspace_codebase_path(session_dir: Path, workspace_id: str) -> Path:
    workspace_dir_name = _workspace_dir_name(workspace_id)
    return session_dir / "round_workspaces" / workspace_dir_name / f"codebase_{workspace_dir_name}"


def get_parent_workspace_codebase_path(
    session_dir: Path, parent_workspace_id: str | None
) -> Path | None:
    if not parent_workspace_id:
        return None

    preferred_dir = _workspace_dir_name(parent_workspace_id)
    legacy_dir = parent_workspace_id[:8]
    candidates: list[str] = []
    for item in (preferred_dir, legacy_dir):
        if item and item not in candidates:
            candidates.append(item)

    for workspace_dir_name in candidates:
        parent_workspace_dir = session_dir / "round_workspaces" / workspace_dir_name
        if not parent_workspace_dir.exists():
            continue

        symlink_path = parent_workspace_dir / "codebase"
        if symlink_path.exists() or symlink_path.is_symlink():
            resolved = symlink_path.resolve()
            if resolved.exists():
                return resolved

        codebase_candidates = [
            p
            for p in parent_workspace_dir.iterdir()
            if p.is_dir() and p.name.startswith("codebase_")
        ]
        if codebase_candidates:
            return codebase_candidates[0].resolve()
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
    bootstrap_codebase_dir: Path | None = None,
    parent_workspace_id: str | None = None,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD_BYTES,
    copy_plan_cache_file: Path | None = None,
    use_copy_plan_cache: bool = True,
    force_rebuild_copy_plan: bool = False,
) -> WorkspaceCodebaseInfo:
    workspace_dir_name = _workspace_dir_name(workspace_id)
    workspace_dir = session_dir / "round_workspaces" / workspace_dir_name
    workspace_dir.mkdir(parents=True, exist_ok=True)

    dest_codebase = workspace_dir / f"codebase_{workspace_dir_name}"
    symlink_path = workspace_dir / "codebase"

    existing: Path | None = None
    if symlink_path.exists() or symlink_path.is_symlink():
        existing = symlink_path.resolve()
    elif dest_codebase.exists():
        existing = dest_codebase

    if existing and existing.exists():
        ensure_local_workspace_dirs(existing)
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
        ensure_local_workspace_dirs(dest_codebase)
        return WorkspaceCodebaseInfo(
            path=dest_codebase,
            large_dirs=inherited_large_dirs,
            source_type="parent",
            parent_workspace_id=parent_workspace_id,
        )

    if bootstrap_codebase_dir and bootstrap_codebase_dir.exists():
        bootstrap_large_dirs = load_large_dirs(bootstrap_codebase_dir)
        inherited_large_dirs = _copy_from_parent_filtered(
            bootstrap_codebase_dir,
            dest_codebase,
            bootstrap_large_dirs,
        )
        save_large_dirs(dest_codebase, inherited_large_dirs)
        _create_codebase_symlink(symlink_path, dest_codebase)
        ensure_local_workspace_dirs(dest_codebase)
        return WorkspaceCodebaseInfo(
            path=dest_codebase,
            large_dirs=inherited_large_dirs,
            source_type="bootstrap",
            parent_workspace_id=parent_workspace_id,
        )

    if source_codebase_dir and source_codebase_dir.exists():
        large_dirs = smart_copy_codebase(
            source_codebase_dir,
            dest_codebase,
            size_threshold=size_threshold,
            copy_plan_cache_file=copy_plan_cache_file,
            use_copy_plan_cache=use_copy_plan_cache,
            force_rebuild_copy_plan=force_rebuild_copy_plan,
        )
        save_large_dirs(dest_codebase, large_dirs)
        _create_codebase_symlink(symlink_path, dest_codebase)
        ensure_local_workspace_dirs(dest_codebase)
        return WorkspaceCodebaseInfo(
            path=dest_codebase,
            large_dirs=large_dirs,
            source_type="original",
            parent_workspace_id=parent_workspace_id,
        )

    dest_codebase.mkdir(parents=True, exist_ok=True)
    _create_codebase_symlink(symlink_path, dest_codebase)
    ensure_local_workspace_dirs(dest_codebase)
    return WorkspaceCodebaseInfo(
        path=dest_codebase,
        large_dirs=[],
        source_type="empty",
        parent_workspace_id=parent_workspace_id,
    )


def _collect_workspace_output_dirs(codebase_dir: Path, dirname: str) -> list[Path]:
    if not codebase_dir.exists():
        return []

    matches: list[Path] = []
    root_target = codebase_dir / dirname
    if root_target.exists():
        matches.append(root_target)

    for path in sorted(codebase_dir.rglob(dirname), key=lambda p: len(p.parts), reverse=True):
        if path == root_target:
            continue
        if path.exists():
            matches.append(path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in matches:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def cleanup_eval_result(codebase_dir: Path) -> dict[str, str]:
    status: dict[str, str] = {}
    for dirname in WORKSPACE_OUTPUT_DIR_NAMES:
        targets = _collect_workspace_output_dirs(codebase_dir, dirname)
        if not targets:
            status[dirname] = "missing"
            continue

        failed = False
        removed_count = 0
        for target in sorted(targets, key=lambda p: len(p.parts), reverse=True):
            try:
                if target.is_symlink():
                    target.unlink()
                else:
                    shutil.rmtree(target)
                removed_count += 1
            except OSError as e:
                failed = True
                logger.warning("Failed to clean up workspace output %s: %s", target, e)

        status[dirname] = "failed" if failed else "removed"
        if removed_count:
            logger.info(
                "Workspace cleanup removed %d '%s' director%s under %s",
                removed_count,
                dirname,
                "y" if removed_count == 1 else "ies",
                codebase_dir,
            )
    return status
