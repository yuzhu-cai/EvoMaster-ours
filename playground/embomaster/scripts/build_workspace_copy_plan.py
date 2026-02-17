#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from playground.embomaster.core.utils.workspace_isolation import (
    DEFAULT_SIZE_THRESHOLD_BYTES,
    build_copy_plan_cache,
)


class _TerminalProgressBar:
    def __init__(self, width: int = 32) -> None:
        self.width = max(10, int(width))
        self.enabled = bool(sys.stdout.isatty())
        self._last_render_ts = 0.0

    def update(self, visited: int, total: int, large_count: int, current_rel: str) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if visited < total and now - self._last_render_ts < 0.1:
            return
        self._last_render_ts = now

        pct = (visited / total) if total > 0 else 1.0
        filled = min(self.width, int(round(pct * self.width)))
        bar = "#" * filled + "-" * (self.width - filled)
        tail = f" | current: {current_rel}" if current_rel else ""
        line = (
            f"\rscan [{bar}] {visited}/{total} "
            f"({pct * 100:5.1f}%) | large_dirs: {large_count}{tail}"
        )
        print(line, end="", flush=True)

    def close(self) -> None:
        if self.enabled:
            print("", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and cache EmboMaster workspace copy plan")
    parser.add_argument("src", type=Path, help="Source codebase directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output cache file path (default: <src>/.embomaster_copy_plan.json)",
    )
    parser.add_argument(
        "--size-threshold-mb",
        type=int,
        default=DEFAULT_SIZE_THRESHOLD_BYTES // (1024 * 1024),
        help="Large directory threshold in MB",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bar",
    )
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    if not src.exists() or not src.is_dir():
        raise SystemExit(f"source directory not found: {src}")

    output = args.output.expanduser().resolve() if args.output else (src / ".embomaster_copy_plan.json")
    size_threshold = int(args.size_threshold_mb) * 1024 * 1024

    progress = _TerminalProgressBar()
    callback = None if args.no_progress else progress.update
    large_dirs = build_copy_plan_cache(
        src=src,
        cache_file=output,
        size_threshold=size_threshold,
        progress_callback=callback,
    )
    progress.close()
    print(f"copy plan saved: {output}")
    print(f"large dirs: {len(large_dirs)}")
    for item in large_dirs:
        rel = str(item.get("rel", ""))
        size_mb = item.get("size_mb", "?")
        print(f"- {rel} ({size_mb} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
