"""OpenClaw-private support helpers.

These helpers intentionally live under `impl/openclaw` so the integration can
be merged onto `origin/main` without carrying shared utils drift.
"""

from __future__ import annotations

from contextlib import suppress
import base64
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import shlex
import tarfile
import time
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

from loguru import logger
from loguru import logger as default_logger

from swe_agents.core.types import SWEInstance
from swe_agents.utils.session_router_client import SessionRouterClient
from swe_agents.utils.swe_env import extract_patch


STEP2_MOUNT_ROOTS: tuple[Path, ...] = (
    Path("/mnt/step2_alignment_jfs"),
    Path("/mnt/step2-alignment-jfs"),
)
DEFAULT_PACKAGE_OWNERS: tuple[str, ...] = ("youwang", "wuxin")
FASTAPI_BUNDLE_CANDIDATES = (
    "fastapi-proxy-v14-3.10-bin-v2.tar.gz",
    "fastapi-proxy-v13-3.10-bin-v2.tar.gz",
)
DEBUG_CAPTURE_ENV = "SWE_FASTAPI_CAPTURE_RAW_LOGS"
DEBUG_CAPTURE_INSTANCE_KEYS = (
    "swe_fastapi_capture_raw_logs",
    "fastapi_capture_raw_logs",
)
DEFAULT_FASTAPI_LOG_DIR = "/tmp/fastapi_logs"


def _openclaw_tools_dir() -> Path:
    return Path(__file__).resolve().parent / "env_tools" / "openclaw"


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def pick_existing_file(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
        except Exception:
            continue
    return None


def step2_package_dirs(
    package_name: str,
    *,
    owners: Sequence[str] = DEFAULT_PACKAGE_OWNERS,
    mount_roots: Sequence[Path] = STEP2_MOUNT_ROOTS,
) -> list[Path]:
    paths: list[Path] = []
    for owner in owners:
        for mount_root in mount_roots:
            # Some packages live under `<owner>/<name>/packages`, while newer
            # bundles are published directly under `<owner>/packages`.
            if package_name:
                paths.append(mount_root / owner / package_name / "packages")
            paths.append(mount_root / owner / "packages")
    return _dedupe_paths(paths)


def step2_package_candidates(
    package_name: str,
    file_names: Sequence[str],
    *,
    owners: Sequence[str] = DEFAULT_PACKAGE_OWNERS,
    mount_roots: Sequence[Path] = STEP2_MOUNT_ROOTS,
) -> list[Path]:
    candidates: list[Path] = []
    for package_dir in step2_package_dirs(
        package_name,
        owners=owners,
        mount_roots=mount_roots,
    ):
        candidates.extend(package_dir / name for name in file_names)
    return _dedupe_paths(candidates)


@dataclass(frozen=True)
class FastAPIBootstrapAssets:
    utils_dir: Path
    bin_dir: Path
    server_src: Path
    init_src: Path
    bundle_src: Path

    def bundle_upload_files(self) -> list[tuple[Path, str, bool]]:
        return [(self.bundle_src, f"/tmp/{self.bundle_src.name}", False)]


def resolve_fastapi_bundle_path(*, consumer_name: str) -> Path:
    tools_dir = _openclaw_tools_dir()
    bin_dir = tools_dir / "bin"
    bundle_path = pick_existing_file(
        [
            *(bin_dir / name for name in FASTAPI_BUNDLE_CANDIDATES),
            *step2_package_candidates("fastapi", FASTAPI_BUNDLE_CANDIDATES),
        ]
    )
    if bundle_path is None:
        raise RuntimeError(
            f"missing {consumer_name} FastAPI tarball in {bin_dir} "
            f"(candidates={FASTAPI_BUNDLE_CANDIDATES})"
        )
    return bundle_path


def resolve_fastapi_bootstrap_assets(*, consumer_name: str) -> FastAPIBootstrapAssets:
    tools_dir = _openclaw_tools_dir()
    server_src = tools_dir / "fastapi_server.py"
    init_src = tools_dir / "fastapi_init.sh"

    for required in (server_src, init_src):
        if not required.exists():
            raise RuntimeError(f"missing {consumer_name} FastAPI asset: {required}")

    bundle_src = resolve_fastapi_bundle_path(consumer_name=consumer_name)
    return FastAPIBootstrapAssets(
        utils_dir=tools_dir,
        bin_dir=tools_dir / "bin",
        server_src=server_src,
        init_src=init_src,
        bundle_src=bundle_src,
    )


def _is_http_endpoint(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def build_fastapi_proxy_env(
    *,
    model_endpoint: str | None = None,
    endpoint: str | None = None,
    include_stepcast_endpoint: bool = False,
) -> dict[str, str]:
    env: dict[str, str] = {}
    if (
        include_stepcast_endpoint
        and model_endpoint == "stepcast"
        and isinstance(endpoint, str)
        and endpoint.strip()
    ):
        resolved = endpoint.strip()
        if _is_http_endpoint(resolved):
            env["STEPTRON_VLLM_ENDPOINT"] = resolved
            env["SWE_STEPCAST_BASE_URL"] = resolved
    return env


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _capture_enabled(instance: dict[str, Any] | None = None) -> bool:
    if isinstance(instance, dict):
        for key in DEBUG_CAPTURE_INSTANCE_KEYS:
            if key in instance:
                return _is_truthy(instance.get(key))
    return _is_truthy(os.getenv(DEBUG_CAPTURE_ENV))


def _build_placeholder_tar_base64(note: str) -> str:
    buf = io.BytesIO()
    payload = (note or "fastapi raw capture returned empty payload").encode(
        "utf-8", errors="replace"
    )
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="fastapi_logs_capture_info.txt")
        info.size = len(payload)
        info.mtime = int(time.time())
        tf.addfile(info, io.BytesIO(payload))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def maybe_attach_fastapi_logs_tar_base64(
    *,
    meta: dict[str, Any],
    session_client: Any,
    instance: dict[str, Any] | None = None,
    logger=default_logger,
    log_dir: str = DEFAULT_FASTAPI_LOG_DIR,
) -> None:
    if not _capture_enabled(instance):
        return

    normalized_log_dir = log_dir.rstrip("/") or "/tmp/fastapi_logs"
    parent_dir = os.path.dirname(normalized_log_dir) or "/"
    base_name = os.path.basename(normalized_log_dir)
    remote_tar = f"/tmp/fastapi_logs_{session_client.session_id}.tar.gz"
    pack_cmd = (
        f"test -d {shlex.quote(normalized_log_dir)} && "
        f"tar -C {shlex.quote(parent_dir)} -czf {shlex.quote(remote_tar)} "
        f"{shlex.quote(base_name)}"
    )

    fallback_note = ""
    try:
        ret = session_client.exec_bash(pack_cmd, timeout=120)
        if isinstance(ret, dict) and ret.get("error"):
            fallback_note = str(ret.get("error"))
            logger.warning(f"fastapi raw capture tar failed: {fallback_note}")
        else:
            payload = session_client.download(remote_tar, timeout=120)
            if payload:
                meta["fastapi_logs_tar_base64"] = base64.b64encode(payload).decode("ascii")
                return
            fallback_note = "fastapi raw capture: empty tar payload"
            logger.warning(fallback_note)
    except Exception as exc:
        fallback_note = str(exc)
        logger.warning(f"fastapi raw capture failed: {fallback_note}")
    finally:
        with suppress(Exception):
            session_client.exec_bash(f"rm -f {shlex.quote(remote_tar)}", timeout=30)

    meta["fastapi_logs_tar_base64"] = _build_placeholder_tar_base64(fallback_note)
    if fallback_note:
        meta["fastapi_logs_capture_note"] = fallback_note[:1000]


def _is_swe_instance(instance: SWEInstance) -> bool:
    repo = instance.get("repo")
    base_commit = instance.get("base_commit")
    return (
        isinstance(repo, str)
        and bool(repo.strip())
        and isinstance(base_commit, str)
        and bool(base_commit.strip())
    )


def _remove_binary_diffs(patch_text: str) -> str:
    lines = patch_text.split("\n")
    cleaned_lines: list[str] = []
    block: list[str] = []
    is_binary_block = False

    for line in lines:
        if line.startswith("diff --git "):
            if block and not is_binary_block:
                cleaned_lines.extend(block)
            block = [line]
            is_binary_block = False
        elif "Binary files" in line:
            is_binary_block = True
            block.append(line)
        else:
            block.append(line)

    if block and not is_binary_block:
        cleaned_lines.extend(block)
    return "\n".join(cleaned_lines)


def _decode_patch(patch_bytes: bytes) -> str:
    if not patch_bytes:
        return ""
    try:
        return patch_bytes.decode()
    except UnicodeDecodeError:
        return patch_bytes.decode("utf-8", errors="ignore")


def _exec_and_download_patch(
    sr_client: SessionRouterClient,
    commands: list[str],
    patch_path: str,
    *,
    max_retry: int = 5,
) -> bytes:
    retry_times = 0
    while retry_times < max_retry:
        try:
            retry_times += 1
            sr_client.exec_stateful_commands(commands)
            return sr_client.download(patch_path)
        except Exception:
            logger.error(f"Failed to extract patch with command: {commands[0]}")
            if retry_times >= max_retry:
                return b""
            time.sleep(3)
    return b""


def _workspace_patch_baseline_file(workspace_root: str) -> str:
    digest = hashlib.sha256(workspace_root.encode("utf-8")).hexdigest()
    return f"/tmp/swe_agents_workspace_patch_base_{digest}.txt"


def extract_patch_from_workspace(
    sr_client: SessionRouterClient,
    workspace_root: str,
) -> str:
    root = workspace_root.strip()
    if not root:
        return ""

    quoted_root = shlex.quote(root)
    baseline_path = _workspace_patch_baseline_file(root)
    quoted_baseline_path = shlex.quote(baseline_path)
    patch_path = "/tmp/workspace_patch.diff"
    quoted_patch_path = shlex.quote(patch_path)
    commands = [
        f"""if [ -d {quoted_root}/.git ]; then
            cd {quoted_root}
            git config --global core.pager ""
            git add -A
            if [ -s {quoted_baseline_path} ]; then
                base=$(cat {quoted_baseline_path})
                git diff --no-color --cached "$base" > {quoted_patch_path} 2>/dev/null || git diff --no-color --cached > {quoted_patch_path}
            else
                git diff --no-color --cached > {quoted_patch_path} 2>/dev/null || git diff --no-color --cached > {quoted_patch_path}
            fi
        else
            : > {quoted_patch_path}
        fi"""
    ]
    patch_bytes = _exec_and_download_patch(sr_client, commands, patch_path)
    return _remove_binary_diffs(_decode_patch(patch_bytes))


def prepare_patch_workspace(
    sr_client: SessionRouterClient,
    workspace_root: str,
) -> None:
    root = workspace_root.strip()
    if not root:
        return

    quoted_root = shlex.quote(root)
    baseline_path = _workspace_patch_baseline_file(root)
    quoted_baseline_path = shlex.quote(baseline_path)
    commands = [
        f"""if [ ! -d {quoted_root} ]; then
            mkdir -p {quoted_root} >/dev/null 2>&1 || true
        fi
        if [ -d {quoted_root} ]; then
            cd {quoted_root}
            if [ ! -d .git ]; then
                git init >/dev/null 2>&1
            fi
            git config user.email "swe-agents@stepfun.com"
            git config user.name "swe-agents"
            git config --global core.pager ""
            git add -A
            git commit --allow-empty --no-verify -m "SWE-Agents: workspace patch baseline" >/dev/null 2>&1 || true
            if git rev-parse --verify HEAD >/dev/null 2>&1; then
                git rev-parse HEAD > {quoted_baseline_path}
            else
                : > {quoted_baseline_path}
            fi
        else
            : > {quoted_baseline_path}
        fi"""
    ]
    _exec_and_download_patch(sr_client, commands, baseline_path)


def prepare_patch_workspace_auto(
    sr_client: SessionRouterClient,
    instance: SWEInstance,
    *,
    workspace_root: str | None = None,
) -> None:
    if _is_swe_instance(instance):
        return

    root = workspace_root
    if root is None:
        raw_root = instance.get("workspace_root")
        root = raw_root.strip() if isinstance(raw_root, str) else ""
    if not isinstance(root, str) or not root.strip():
        return
    prepare_patch_workspace(sr_client, root)


def extract_patch_auto(
    sr_client: SessionRouterClient,
    instance: SWEInstance,
    *,
    workspace_root: str | None = None,
) -> str:
    if _is_swe_instance(instance):
        patch = extract_patch(sr_client, instance)
        if patch.strip():
            return patch

    root = workspace_root
    if root is None:
        raw_root = instance.get("workspace_root")
        root = raw_root.strip() if isinstance(raw_root, str) else ""
    if not isinstance(root, str) or not root.strip():
        return ""
    return extract_patch_from_workspace(sr_client, root)
