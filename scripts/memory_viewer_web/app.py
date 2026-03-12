"""Memory Viewer Web Dashboard — Flask backend

Usage:
    python -m scripts.memory_viewer_web.app --db data/memory/memories.db --port 8766
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, request, send_from_directory

logger = logging.getLogger("memory_viewer_web")

DEFAULT_DB = Path(__file__).parent.parent.parent / "data" / "memory" / "memories.db"

CATEGORY_LABELS = {
    "preference": "偏好",
    "fact": "事实",
    "decision": "决策",
    "entity": "实体",
    "other": "其他",
}

SOURCE_LABELS = {
    "auto": "自动提取",
    "manual": "主动保存",
    "compaction": "压缩提取",
}

ALLOWED_SORT_COLS = {"updated_at", "created_at", "importance", "access_count"}


def _ts_to_str(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _format_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "content": row["content"],
        "category": row["category"],
        "category_label": CATEGORY_LABELS.get(row["category"], row["category"]),
        "importance": row["importance"],
        "source": row["source"],
        "source_label": SOURCE_LABELS.get(row["source"], row["source"]),
        "created_at": row["created_at"],
        "created_at_str": _ts_to_str(row["created_at"]),
        "updated_at": row["updated_at"],
        "updated_at_str": _ts_to_str(row["updated_at"]),
        "access_count": row["access_count"],
    }


def create_app(db_path: Path) -> Flask:
    app = Flask(
        __name__,
        static_folder="static",
        static_url_path="/static",
    )

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "db_path": str(db_path)})

    @app.get("/api/overview")
    def api_overview():
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        users = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM memories"
        ).fetchone()[0]

        cat_rows = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM memories "
            "GROUP BY category ORDER BY cnt DESC"
        ).fetchall()
        categories = {r["category"]: r["cnt"] for r in cat_rows}

        src_rows = conn.execute(
            "SELECT source, COUNT(*) as cnt FROM memories "
            "GROUP BY source ORDER BY cnt DESC"
        ).fetchall()
        sources = {r["source"]: r["cnt"] for r in src_rows}

        return jsonify({
            "total_memories": total,
            "total_users": users,
            "categories": categories,
            "sources": sources,
            "db_path": str(db_path),
        })

    @app.get("/api/users")
    def api_users():
        rows = conn.execute(
            "SELECT user_id, COUNT(*) as cnt FROM memories "
            "GROUP BY user_id ORDER BY cnt DESC"
        ).fetchall()
        users = []
        for r in rows:
            uid = r["user_id"]
            users.append({
                "user_id": uid,
                "count": r["cnt"],
                "short_id": uid[-8:] if len(uid) > 8 else uid,
            })
        return jsonify({"users": users})

    @app.get("/api/memories")
    def api_memories():
        user_id = request.args.get("user_id")
        category = request.args.get("category")
        search = request.args.get("search")
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(200, max(1, int(request.args.get("per_page", 50))))
        sort = request.args.get("sort", "updated_at")
        order = request.args.get("order", "desc")

        if sort not in ALLOWED_SORT_COLS:
            sort = "updated_at"
        if order not in ("asc", "desc"):
            order = "desc"

        conditions: list[str] = []
        params: list[Any] = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if category:
            conditions.append("category = ?")
            params.append(category)
        if search:
            conditions.append("content LIKE ?")
            params.append(f"%{search}%")

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        total = conn.execute(
            f"SELECT COUNT(*) FROM memories {where}", params
        ).fetchone()[0]

        offset = (page - 1) * per_page
        rows = conn.execute(
            f"SELECT * FROM memories {where} ORDER BY {sort} {order} "
            f"LIMIT ? OFFSET ?",
            params + [per_page, offset],
        ).fetchall()

        return jsonify({
            "memories": [_format_row(r) for r in rows],
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": max(1, (total + per_page - 1) // per_page),
            },
            "filters": {
                "user_id": user_id,
                "category": category,
                "search": search,
            },
        })

    @app.delete("/api/memories/<memory_id>")
    def api_delete_memory(memory_id: str):
        row = conn.execute(
            "SELECT rowid FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return jsonify({
                "ok": False, "error": "memory not found", "id": memory_id,
            }), 404

        # Clean FTS index first (same pattern as store.py)
        try:
            conn.execute(
                "DELETE FROM memories_fts WHERE rowid = ?", (row["rowid"],)
            )
        except sqlite3.OperationalError:
            pass  # FTS table may not exist

        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        return jsonify({"ok": True, "id": memory_id})

    return app


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("memory_viewer_web")
    p.add_argument(
        "--db", type=str, default=str(DEFAULT_DB),
        help="SQLite database path (default: data/memory/memories.db)",
    )
    p.add_argument("--host", default="127.0.0.1", help="bind host")
    p.add_argument("--port", default=8766, type=int, help="bind port")
    p.add_argument("--debug", action="store_true", help="flask debug mode")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    db_path = Path(args.db).resolve()
    if not db_path.exists():
        raise SystemExit(f"Database file not found: {db_path}")

    app = create_app(db_path)
    logger.info("memory_viewer_web serving db=%s", db_path)
    logger.info("open: http://%s:%s/", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
