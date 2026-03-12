#!/usr/bin/env python3
"""Memory Viewer Web Dashboard

Usage:
    python scripts/memory_viewer_web.py                          # default DB + port
    python scripts/memory_viewer_web.py --db path/to/memories.db # custom DB
    python scripts/memory_viewer_web.py --port 8766 --host 0.0.0.0
"""

from memory_viewer_web.app import main

if __name__ == "__main__":
    raise SystemExit(main())
