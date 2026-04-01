"""Standard-library HTTP server fallback for the EmboMaster dashboard."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .backend import DashboardManager


PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = PACKAGE_ROOT / "templates" / "index.html"
STATIC_ROOT = PACKAGE_ROOT / "static"


def _render_index_html() -> str:
    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = html.replace("{{ url_for('static', filename='dashboard.css') }}", "/static/dashboard.css")
    html = html.replace("{{ url_for('static', filename='dashboard.js') }}", "/static/dashboard.js")
    return html


class DashboardRequestHandler(BaseHTTPRequestHandler):
    manager: DashboardManager
    index_html: str

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, content: str, content_type: str = "text/html; charset=utf-8") -> None:
        body = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _resolve_store(self, query: dict[str, list[str]]):
        run_id = str(query.get("run_id", [""])[0] or "").strip() or None
        actual_run_id, store = self.manager.get_store(run_id)
        if actual_run_id is None or store is None:
            self._send_json({"error": "no run available"}, code=404)
            return None, None, None
        meta = self.manager.get_run_meta(actual_run_id) or {"run_id": actual_run_id, "label": actual_run_id}
        return actual_run_id, store, meta

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/":
            self._send_text(self.index_html)
            return

        if parsed.path.startswith("/static/"):
            rel = parsed.path[len("/static/") :]
            file_path = (STATIC_ROOT / rel).resolve()
            try:
                file_path.relative_to(STATIC_ROOT)
            except ValueError:
                self._send_json({"error": "invalid static path"}, code=400)
                return
            if not file_path.exists() or not file_path.is_file():
                self._send_json({"error": "static file not found"}, code=404)
                return
            content_type = "text/plain; charset=utf-8"
            if file_path.suffix == ".css":
                content_type = "text/css; charset=utf-8"
            elif file_path.suffix == ".js":
                content_type = "application/javascript; charset=utf-8"
            self._send_bytes(file_path.read_bytes(), content_type)
            return

        if parsed.path == "/api/runs":
            refresh = str(query.get("refresh", ["0"])[0]).lower() in {"1", "true", "yes"}
            self._send_json(self.manager.list_runs(force=refresh))
            return

        actual_run_id, store, meta = self._resolve_store(query)
        if actual_run_id is None or store is None or meta is None:
            return

        if parsed.path == "/api/overview":
            payload = store.get_overview()
            payload["run_id"] = actual_run_id
            payload["run"] = meta
            self._send_json(payload)
            return

        if parsed.path == "/api/rounds":
            self._send_json({"run_id": actual_run_id, "run": meta, "rounds": store.get_rounds()})
            return

        if parsed.path == "/api/route":
            payload = store.get_route()
            payload["run_id"] = actual_run_id
            payload["run"] = meta
            self._send_json(payload)
            return

        if parsed.path == "/api/stream":
            round_index_raw = str(query.get("round_index", [""])[0]).strip()
            cursor = int(query.get("cursor", ["0"])[0] or 0)
            limit = int(query.get("limit", ["200"])[0] or 200)
            payload = store.get_stream(
                round_index=int(round_index_raw) if round_index_raw and round_index_raw != "all" else None,
                cursor=cursor,
                limit=limit,
            )
            payload["run_id"] = actual_run_id
            payload["run"] = meta
            self._send_json(payload)
            return

        if parsed.path == "/api/entry":
            index = int(query.get("index", ["-1"])[0] or -1)
            entry = store.get_entry(index)
            if entry is None:
                self._send_json({"error": "entry not found", "index": index}, code=404)
                return
            self._send_json({"run_id": actual_run_id, "run": meta, "entry": entry})
            return

        if parsed.path == "/api/round_detail":
            round_index = int(query.get("round_index", ["-1"])[0] or -1)
            round_item = store.get_round_detail(round_index)
            if round_item is None:
                self._send_json({"error": "round not found", "round_index": round_index}, code=404)
                return
            self._send_json({"run_id": actual_run_id, "run": meta, "round": round_item})
            return

        if parsed.path == "/api/pods":
            refresh = str(query.get("refresh", ["0"])[0]).lower() in {"1", "true", "yes"}
            tail = int(query.get("tail", ["300"])[0] or 300)
            payload = store.get_pod_payload(refresh=refresh, tail=tail)
            payload["run_id"] = actual_run_id
            payload["run"] = meta
            self._send_json(payload)
            return

        if parsed.path == "/api/pod_logs":
            pod_name = str(query.get("pod_name", [""])[0] or "")
            job_name = str(query.get("job_name", [""])[0] or "")
            namespace = str(query.get("namespace", ["default"])[0] or "default")
            round_index_raw = str(query.get("round_index", [""])[0]).strip()
            tail = int(query.get("tail", ["300"])[0] or 300)
            refresh = str(query.get("refresh", ["0"])[0]).lower() in {"1", "true", "yes"}
            payload = store.get_pod_logs(
                pod_name=pod_name,
                namespace=namespace,
                tail=tail,
                job_name=job_name,
                round_index=int(round_index_raw) if round_index_raw else None,
                refresh=refresh,
            )
            payload["run_id"] = actual_run_id
            payload["run"] = meta
            self._send_json(payload)
            return

        if parsed.path == "/api/artifacts":
            round_index = int(query.get("round_index", ["-1"])[0] or -1)
            payload = store.get_artifacts(round_index)
            payload["run_id"] = actual_run_id
            payload["run"] = meta
            self._send_json(payload)
            return

        if parsed.path == "/api/artifact_preview":
            round_index = int(query.get("round_index", ["-1"])[0] or -1)
            absolute_path = str(query.get("absolute_path", [""])[0] or "")
            max_bytes = int(query.get("max_bytes", ["32768"])[0] or 32768)
            payload = store.preview_artifact(
                round_index=round_index,
                absolute_path=absolute_path,
                max_bytes=max_bytes,
            )
            payload["run_id"] = actual_run_id
            payload["run"] = meta
            self._send_json(payload)
            return

        self._send_json({"error": "not found", "path": parsed.path}, code=404)


def serve_dashboard(
    source_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8766,
    preview_chars: int = 220,
) -> None:
    handler_cls = DashboardRequestHandler
    handler_cls.manager = DashboardManager(
        source_path=Path(source_path),
        preview_chars=max(80, int(preview_chars)),
    )
    handler_cls.index_html = _render_index_html()
    server = ThreadingHTTPServer((host, port), handler_cls)
    print(f"[embomaster-dashboard] serving on http://{host}:{port}")
    print(f"[embomaster-dashboard] source: {Path(source_path).expanduser().resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
