"""SQLite data viewer web dashboard.

Usage:
    python -m scripts.data_viewer_web.app --port 8766
    python scripts/data_viewer_web.py
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, request, send_from_directory

logger = logging.getLogger("data_viewer_web")

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MEMORY_DB = ROOT / "data" / "memory" / "memories.db"
DEFAULT_SCHEDULER_DB = ROOT / "data" / "scheduler" / "schedules.db"
DEFAULT_SOURCE = "memory"

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
JOB_STATUS_LABELS = {
    "active": "启用",
    "paused": "暂停",
    "done": "完成",
    "failed": "失败",
}
SCHEDULE_TYPE_LABELS = {
    "at": "延时",
    "cron": "Cron",
    "interval": "间隔",
}
RUN_STATUS_LABELS = {
    "success": "成功",
    "failed": "失败",
    "running": "运行中",
}


@dataclass(frozen=True)
class SourceConfig:
    name: str
    title: str
    db_path: Path


SOURCES = {
    "memory": SourceConfig("memory", "Memory", DEFAULT_MEMORY_DB),
    "scheduler": SourceConfig("scheduler", "Scheduler", DEFAULT_SCHEDULER_DB),
}


def _ts_to_str(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _safe_query(conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    try:
        return conn.execute(query, params).fetchall()
    except sqlite3.OperationalError:
        return []


def _safe_count(conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> int:
    try:
        row = conn.execute(query, params).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0]) if row and row[0] is not None else 0


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        logger.warning("failed to enable WAL for %s", db_path)
    return conn


def _memory_overview(conn: sqlite3.Connection) -> dict[str, Any]:
    total = _safe_count(conn, "SELECT COUNT(*) FROM memories")
    users = _safe_count(conn, "SELECT COUNT(DISTINCT user_id) FROM memories")
    categories = {
        row["category"]: row["cnt"]
        for row in _safe_query(conn, "SELECT category, COUNT(*) AS cnt FROM memories GROUP BY category ORDER BY cnt DESC")
    }
    sources = {
        row["source"]: row["cnt"]
        for row in _safe_query(conn, "SELECT source, COUNT(*) AS cnt FROM memories GROUP BY source ORDER BY cnt DESC")
    }
    return {
        "cards": [
            {"label": "Memories", "value": total},
            {"label": "Users", "value": users},
            {"label": "Categories", "value": len(categories)},
            {"label": "Sources", "value": len(sources)},
        ],
        "charts": [
            {"title": "Category Distribution", "data": categories, "labels": CATEGORY_LABELS},
            {"title": "Source Distribution", "data": sources, "labels": SOURCE_LABELS},
        ],
    }


def _memory_facets(conn: sqlite3.Connection) -> dict[str, Any]:
    users = []
    for row in _safe_query(conn, "SELECT user_id, COUNT(*) AS cnt FROM memories GROUP BY user_id ORDER BY cnt DESC LIMIT 200"):
        uid = row["user_id"]
        users.append({"value": uid, "label": f"{uid[-8:] if len(uid) > 8 else uid} ({row['cnt']})"})
    categories = []
    for row in _safe_query(conn, "SELECT category, COUNT(*) AS cnt FROM memories GROUP BY category ORDER BY cnt DESC"):
        categories.append({"value": row["category"], "label": f"{CATEGORY_LABELS.get(row['category'], row['category'])} ({row['cnt']})"})
    return {
        "filters": [
            {"key": "user_id", "label": "用户", "all_label": "所有用户", "options": users},
            {"key": "category", "label": "分类", "all_label": "所有分类", "options": categories},
        ],
        "views": [{"value": "records", "label": "Memories"}],
        "search_placeholder": "搜索 memory 内容...",
        "item_label": "memories",
    }


def _format_memory_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "primary_text": row["content"],
        "secondary_text": row["user_id"],
        "badges": [
            {"text": CATEGORY_LABELS.get(row["category"], row["category"]), "kind": f"category {row['category']}"},
            {"text": SOURCE_LABELS.get(row["source"], row["source"]), "kind": "source"},
            {"text": f"importance {row['importance']:.1f}", "kind": "metric"},
            {"text": f"access {row['access_count']}", "kind": "metric"},
        ],
        "meta": [
            {"label": "User", "value": row["user_id"]},
            {"label": "Created", "value": _ts_to_str(row["created_at"])},
            {"label": "Updated", "value": _ts_to_str(row["updated_at"])},
        ],
    }


def _memory_records(conn: sqlite3.Connection) -> dict[str, Any]:
    user_id = request.args.get("user_id")
    category = request.args.get("category")
    search = request.args.get("search")
    page = max(1, _safe_int(request.args.get("page"), 1))
    per_page = min(200, max(1, _safe_int(request.args.get("per_page"), 50)))
    sort = request.args.get("sort", "updated_at")
    order = request.args.get("order", "desc").lower()
    if sort not in {"updated_at", "created_at", "importance", "access_count"}:
        sort = "updated_at"
    if order not in {"asc", "desc"}:
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
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    total = _safe_count(conn, f"SELECT COUNT(*) FROM memories {where}", tuple(params))
    rows = _safe_query(conn, f"SELECT * FROM memories {where} ORDER BY {sort} {order} LIMIT ? OFFSET ?", tuple(params + [per_page, (page - 1) * per_page]))
    return {
        "items": [_format_memory_row(row) for row in rows],
        "pagination": {"page": page, "per_page": per_page, "total": total, "total_pages": max(1, (total + per_page - 1) // per_page)},
    }


def _delete_memory(conn: sqlite3.Connection, item_id: str):
    row = conn.execute("SELECT rowid FROM memories WHERE id = ?", (item_id,)).fetchone()
    if not row:
        return jsonify({"ok": False, "error": "memory not found", "id": item_id}), 404
    try:
        conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (row["rowid"],))
    except sqlite3.OperationalError:
        pass
    conn.execute("DELETE FROM memories WHERE id = ?", (item_id,))
    conn.commit()
    return jsonify({"ok": True, "id": item_id})


def _scheduler_overview(conn: sqlite3.Connection) -> dict[str, Any]:
    jobs_total = _safe_count(conn, "SELECT COUNT(*) FROM scheduled_jobs")
    active_jobs = _safe_count(conn, "SELECT COUNT(*) FROM scheduled_jobs WHERE status = 'active'")
    runs_total = _safe_count(conn, "SELECT COUNT(*) FROM scheduled_runs")
    failed_runs = _safe_count(conn, "SELECT COUNT(*) FROM scheduled_runs WHERE success = 0")
    status_counts = {row["status"]: row["cnt"] for row in _safe_query(conn, "SELECT status, COUNT(*) AS cnt FROM scheduled_jobs GROUP BY status ORDER BY cnt DESC")}
    schedule_counts = {row["schedule_type"]: row["cnt"] for row in _safe_query(conn, "SELECT schedule_type, COUNT(*) AS cnt FROM scheduled_jobs GROUP BY schedule_type ORDER BY cnt DESC")}
    run_counts = {("running" if row["success"] is None else ("success" if row["success"] else "failed")): row["cnt"] for row in _safe_query(conn, "SELECT success, COUNT(*) AS cnt FROM scheduled_runs GROUP BY success ORDER BY cnt DESC")}
    return {
        "cards": [
            {"label": "Jobs", "value": jobs_total},
            {"label": "Active Jobs", "value": active_jobs},
            {"label": "Runs", "value": runs_total},
            {"label": "Failed Runs", "value": failed_runs},
        ],
        "charts": [
            {"title": "Job Status", "data": status_counts, "labels": JOB_STATUS_LABELS},
            {"title": "Schedule Type", "data": schedule_counts, "labels": SCHEDULE_TYPE_LABELS},
            {"title": "Run Status", "data": run_counts, "labels": RUN_STATUS_LABELS},
        ],
    }


def _scheduler_facets(conn: sqlite3.Connection) -> dict[str, Any]:
    statuses = []
    for row in _safe_query(conn, "SELECT status, COUNT(*) AS cnt FROM scheduled_jobs GROUP BY status ORDER BY cnt DESC"):
        statuses.append({"value": row["status"], "label": f"{JOB_STATUS_LABELS.get(row['status'], row['status'])} ({row['cnt']})"})
    schedule_types = []
    for row in _safe_query(conn, "SELECT schedule_type, COUNT(*) AS cnt FROM scheduled_jobs GROUP BY schedule_type ORDER BY cnt DESC"):
        schedule_types.append({"value": row["schedule_type"], "label": f"{SCHEDULE_TYPE_LABELS.get(row['schedule_type'], row['schedule_type'])} ({row['cnt']})"})
    job_options = []
    for row in _safe_query(conn, "SELECT job_id, task_description FROM scheduled_jobs ORDER BY created_at DESC LIMIT 200"):
        title = row["task_description"] or row["job_id"]
        short_id = row["job_id"][:12]
        job_options.append({"value": row["job_id"], "label": f"{short_id} · {title}"})
    return {
        "filters_by_view": {
            "jobs": [
                {"key": "status", "label": "状态", "all_label": "所有状态", "options": statuses},
                {"key": "schedule_type", "label": "调度类型", "all_label": "所有类型", "options": schedule_types},
            ],
            "runs": [
                {"key": "job_id", "label": "任务", "all_label": "所有任务", "options": job_options},
            ],
        },
        "views": [
            {"value": "jobs", "label": "Scheduler · 任务"},
            {"value": "runs", "label": "Scheduler · 执行记录"},
        ],
        "search_placeholder": "搜索任务、chat、creator、结果...",
        "item_label": "records",
    }


def _format_scheduler_job(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["job_id"],
        "primary_text": row["task_description"],
        "secondary_text": f"{row['schedule_type']} {row['schedule_expr']}",
        "badges": [
            {"text": JOB_STATUS_LABELS.get(row["status"], row["status"]), "kind": f"status {row['status']}"},
            {"text": SCHEDULE_TYPE_LABELS.get(row["schedule_type"], row["schedule_type"]), "kind": "schedule"},
            {"text": f"runs {row['run_count']}", "kind": "metric"},
            {"text": row["agent_name"], "kind": "agent"},
        ],
        "meta": [
            {"label": "Chat", "value": row["chat_id"]},
            {"label": "Creator", "value": row["creator_id"]},
            {"label": "Timezone", "value": row["timezone"]},
            {"label": "Next", "value": _ts_to_str(row["next_run_at"])},
            {"label": "Last", "value": _ts_to_str(row["last_run_at"])},
            {"label": "Error", "value": row["last_error"] or "-"},
        ],
    }


def _format_scheduler_run(row: sqlite3.Row) -> dict[str, Any]:
    status = "running" if row["finished_at"] is None and row["success"] is None else ("success" if row["success"] else "failed")
    return {
        "id": row["run_id"],
        "primary_text": row["result"] or row["error"] or "(无结果内容)",
        "secondary_text": row["job_id"],
        "badges": [
            {"text": RUN_STATUS_LABELS.get(status, status), "kind": f"run-status {status}"},
            {"text": row["error_kind"] or "-", "kind": "metric"},
        ],
        "meta": [
            {"label": "Job", "value": row["job_id"]},
            {"label": "Started", "value": _ts_to_str(row["started_at"])},
            {"label": "Finished", "value": _ts_to_str(row["finished_at"])},
            {"label": "Error", "value": row["error"] or "-"},
        ],
    }


def _scheduler_records(conn: sqlite3.Connection) -> dict[str, Any]:
    view = request.args.get("view", "jobs")
    search = request.args.get("search")
    page = max(1, _safe_int(request.args.get("page"), 1))
    per_page = min(200, max(1, _safe_int(request.args.get("per_page"), 50)))
    if view == "runs":
        job_id = request.args.get("job_id")
        only_failed = request.args.get("only_failed") == "1"
        conditions: list[str] = []
        params: list[Any] = []
        if job_id:
            conditions.append("job_id = ?")
            params.append(job_id)
        if only_failed:
            conditions.append("success = 0")
        if search:
            conditions.append("(result LIKE ? OR error LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        total = _safe_count(conn, f"SELECT COUNT(*) FROM scheduled_runs {where}", tuple(params))
        rows = _safe_query(conn, f"SELECT * FROM scheduled_runs {where} ORDER BY started_at DESC LIMIT ? OFFSET ?", tuple(params + [per_page, (page - 1) * per_page]))
        return {"items": [_format_scheduler_run(row) for row in rows], "pagination": {"page": page, "per_page": per_page, "total": total, "total_pages": max(1, (total + per_page - 1) // per_page)}}
    status = request.args.get("status")
    schedule_type = request.args.get("schedule_type")
    conditions = []
    params = []
    if status:
        conditions.append("status = ?")
        params.append(status)
    if schedule_type:
        conditions.append("schedule_type = ?")
        params.append(schedule_type)
    if search:
        conditions.append("(task_description LIKE ? OR chat_id LIKE ? OR creator_id LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    total = _safe_count(conn, f"SELECT COUNT(*) FROM scheduled_jobs {where}", tuple(params))
    rows = _safe_query(conn, f"SELECT * FROM scheduled_jobs {where} ORDER BY next_run_at ASC LIMIT ? OFFSET ?", tuple(params + [per_page, (page - 1) * per_page]))
    return {"items": [_format_scheduler_job(row) for row in rows], "pagination": {"page": page, "per_page": per_page, "total": total, "total_pages": max(1, (total + per_page - 1) // per_page)}}


def _resolve_source(source: str | None) -> str:
    source_key = (source or DEFAULT_SOURCE).lower()
    if source_key not in SOURCES:
        valid = ", ".join(sorted(SOURCES))
        raise SystemExit(f"Unknown source '{source}'. Valid values: {valid}")
    return source_key


def create_app(initial_source: str = DEFAULT_SOURCE) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    source_connections = {name: _connect(config.db_path.resolve()) for name, config in SOURCES.items() if config.db_path.exists()}
    selected_source = _resolve_source(initial_source)

    def get_source_name() -> str:
        requested = request.args.get("source") or request.headers.get("X-Data-Source") or selected_source
        return _resolve_source(requested)

    def get_conn() -> tuple[str, sqlite3.Connection, SourceConfig]:
        source_name = get_source_name()
        config = SOURCES[source_name]
        conn = source_connections.get(source_name)
        if conn is None:
            raise FileNotFoundError(f"Database file not found: {config.db_path.resolve()}")
        return source_name, conn, config

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/health")
    def health():
        source_name, _, config = get_conn()
        return jsonify({"ok": True, "source": source_name, "db_path": str(config.db_path.resolve())})

    @app.get("/api/config")
    def api_config():
        available_sources = []
        for name, config in SOURCES.items():
            available_sources.append({
                "value": name,
                "label": config.title,
                "available": config.db_path.exists(),
                "db_path": str(config.db_path.resolve()),
            })
        source_name, _, config = get_conn()
        return jsonify({
            "title": "Data Viewer",
            "source": source_name,
            "db_path": str(config.db_path.resolve()),
            "sources": available_sources,
        })

    @app.get("/api/overview")
    def api_overview():
        source_name, conn, config = get_conn()
        payload = _scheduler_overview(conn) if source_name == "scheduler" else _memory_overview(conn)
        payload.update({"source": source_name, "db_path": str(config.db_path.resolve())})
        return jsonify(payload)

    @app.get("/api/facets")
    def api_facets():
        source_name, conn, _ = get_conn()
        payload = _scheduler_facets(conn) if source_name == "scheduler" else _memory_facets(conn)
        payload.update({"source": source_name})
        return jsonify(payload)

    @app.get("/api/records")
    def api_records():
        source_name, conn, _ = get_conn()
        payload = _scheduler_records(conn) if source_name == "scheduler" else _memory_records(conn)
        payload.update({"source": source_name})
        return jsonify(payload)

    @app.delete("/api/records/<item_id>")
    def api_delete_record(item_id: str):
        source_name, conn, _ = get_conn()
        if source_name != "memory":
            return jsonify({"ok": False, "error": "current source is read-only"}), 405
        return _delete_memory(conn, item_id)

    return app


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("data_viewer_web")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="default selected source: memory | scheduler")
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", default=8766, type=int, help="bind port")
    parser.add_argument("--debug", action="store_true", help="flask debug mode")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    app = create_app(args.source)
    logger.info("data_viewer_web serving default_source=%s", args.source)
    logger.info("open: http://%s:%s/", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
