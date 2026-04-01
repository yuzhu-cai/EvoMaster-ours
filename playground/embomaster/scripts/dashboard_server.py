#!/usr/bin/env python3
"""Start the EmboMaster dashboard server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Web dashboard for EmboMaster experiments")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "workspaces",
        help="Workspace root, one run dir, or one trajectory file",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8766, help="Bind port")
    parser.add_argument("--preview-chars", type=int, default=220, help="Step preview max chars")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        from playground.embomaster.dashboard import create_app
        app = create_app(source_path=args.source, preview_chars=max(80, args.preview_chars))
    except ModuleNotFoundError as exc:
        if exc.name != "flask":
            raise
        from playground.embomaster.dashboard.http_server import serve_dashboard

        serve_dashboard(
            source_path=args.source,
            host=args.host,
            port=args.port,
            preview_chars=max(80, args.preview_chars),
        )
        return 0

    print(f"[embomaster-dashboard] serving on http://{args.host}:{args.port}")
    print(f"[embomaster-dashboard] source: {Path(args.source).expanduser().resolve()}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
