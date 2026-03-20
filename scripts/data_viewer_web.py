#!/usr/bin/env python3
"""Unified web viewer for project data sources."""

try:
    from data_viewer_web.app import main
except ModuleNotFoundError as exc:
    if exc.name == "flask":
        raise SystemExit("Missing dependency: flask. Install it before starting the viewer.") from exc
    raise

if __name__ == "__main__":
    raise SystemExit(main())
