"""Flask application for the EmboMaster dashboard."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .backend import DashboardManager


def create_app(source_path: str | Path, preview_chars: int = 220) -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.json.ensure_ascii = False
    app.config["EMBOMASTER_MANAGER"] = DashboardManager(
        source_path=Path(source_path),
        preview_chars=max(80, int(preview_chars)),
    )

    @app.after_request
    def add_no_store_headers(response):  # type: ignore[no-untyped-def]
        response.headers["Cache-Control"] = "no-store"
        return response

    def manager() -> DashboardManager:
        return app.config["EMBOMASTER_MANAGER"]

    def resolve_store():
        run_id = str(request.args.get("run_id", "") or "").strip() or None
        actual_run_id, store = manager().get_store(run_id)
        if actual_run_id is None or store is None:
            return None, None, jsonify({"error": "no run available"}), 404
        meta = manager().get_run_meta(actual_run_id) or {"run_id": actual_run_id, "label": actual_run_id}
        return actual_run_id, store, meta, None

    @app.get("/")
    def index():  # type: ignore[no-untyped-def]
        return render_template("index.html")

    @app.get("/api/runs")
    def api_runs():  # type: ignore[no-untyped-def]
        refresh = str(request.args.get("refresh", "0")).lower() in {"1", "true", "yes"}
        return jsonify(manager().list_runs(force=refresh))

    @app.get("/api/overview")
    def api_overview():  # type: ignore[no-untyped-def]
        actual_run_id, store, meta, error = resolve_store()
        if error is not None:
            return error
        payload = store.get_overview()
        payload["run_id"] = actual_run_id
        payload["run"] = meta
        return jsonify(payload)

    @app.get("/api/rounds")
    def api_rounds():  # type: ignore[no-untyped-def]
        actual_run_id, store, meta, error = resolve_store()
        if error is not None:
            return error
        return jsonify({"run_id": actual_run_id, "run": meta, "rounds": store.get_rounds()})

    @app.get("/api/route")
    def api_route():  # type: ignore[no-untyped-def]
        actual_run_id, store, meta, error = resolve_store()
        if error is not None:
            return error
        payload = store.get_route()
        payload["run_id"] = actual_run_id
        payload["run"] = meta
        return jsonify(payload)

    @app.get("/api/stream")
    def api_stream():  # type: ignore[no-untyped-def]
        actual_run_id, store, meta, error = resolve_store()
        if error is not None:
            return error
        round_index = request.args.get("round_index")
        cursor = int(request.args.get("cursor", 0) or 0)
        limit = int(request.args.get("limit", 200) or 200)
        payload = store.get_stream(
            round_index=int(round_index) if round_index not in (None, "", "all") else None,
            cursor=cursor,
            limit=limit,
        )
        payload["run_id"] = actual_run_id
        payload["run"] = meta
        return jsonify(payload)

    @app.get("/api/entry")
    def api_entry():  # type: ignore[no-untyped-def]
        actual_run_id, store, meta, error = resolve_store()
        if error is not None:
            return error
        index = int(request.args.get("index", -1) or -1)
        entry = store.get_entry(index)
        if entry is None:
            return jsonify({"error": "entry not found", "index": index}), 404
        return jsonify({"run_id": actual_run_id, "run": meta, "entry": entry})

    @app.get("/api/round_detail")
    def api_round_detail():  # type: ignore[no-untyped-def]
        actual_run_id, store, meta, error = resolve_store()
        if error is not None:
            return error
        round_index = int(request.args.get("round_index", -1) or -1)
        payload = store.get_round_detail(round_index)
        if payload is None:
            return jsonify({"error": "round not found", "round_index": round_index}), 404
        return jsonify({"run_id": actual_run_id, "run": meta, "round": payload})

    @app.get("/api/pods")
    def api_pods():  # type: ignore[no-untyped-def]
        actual_run_id, store, meta, error = resolve_store()
        if error is not None:
            return error
        refresh = str(request.args.get("refresh", "0")).lower() in {"1", "true", "yes"}
        tail = int(request.args.get("tail", 300) or 300)
        payload = store.get_pod_payload(refresh=refresh, tail=tail)
        payload["run_id"] = actual_run_id
        payload["run"] = meta
        return jsonify(payload)

    @app.get("/api/pod_logs")
    def api_pod_logs():  # type: ignore[no-untyped-def]
        actual_run_id, store, meta, error = resolve_store()
        if error is not None:
            return error
        pod_name = str(request.args.get("pod_name", "") or "")
        namespace = str(request.args.get("namespace", "default") or "default")
        job_name = str(request.args.get("job_name", "") or "")
        round_index_raw = request.args.get("round_index")
        tail = int(request.args.get("tail", 300) or 300)
        refresh = str(request.args.get("refresh", "0")).lower() in {"1", "true", "yes"}
        payload = store.get_pod_logs(
            pod_name=pod_name,
            namespace=namespace,
            tail=tail,
            job_name=job_name,
            round_index=int(round_index_raw) if round_index_raw not in (None, "") else None,
            refresh=refresh,
        )
        payload["run_id"] = actual_run_id
        payload["run"] = meta
        return jsonify(payload)

    @app.get("/api/artifacts")
    def api_artifacts():  # type: ignore[no-untyped-def]
        actual_run_id, store, meta, error = resolve_store()
        if error is not None:
            return error
        round_index = int(request.args.get("round_index", -1) or -1)
        payload = store.get_artifacts(round_index)
        payload["run_id"] = actual_run_id
        payload["run"] = meta
        return jsonify(payload)

    @app.get("/api/artifact_preview")
    def api_artifact_preview():  # type: ignore[no-untyped-def]
        actual_run_id, store, meta, error = resolve_store()
        if error is not None:
            return error
        round_index = int(request.args.get("round_index", -1) or -1)
        absolute_path = str(request.args.get("absolute_path", "") or "")
        max_bytes = int(request.args.get("max_bytes", 32768) or 32768)
        payload = store.preview_artifact(
            round_index=round_index,
            absolute_path=absolute_path,
            max_bytes=max_bytes,
        )
        payload["run_id"] = actual_run_id
        payload["run"] = meta
        return jsonify(payload)

    return app
