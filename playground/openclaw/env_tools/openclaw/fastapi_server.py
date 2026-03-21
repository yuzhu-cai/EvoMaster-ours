import argparse
import configparser
from contextlib import asynccontextmanager, closing
import gzip
import json
import logging
import os
from pathlib import Path
import socket
import time
from typing import Any
import zlib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from starlette.background import BackgroundTask
import uvicorn


LOG_DIR = Path("/tmp/fastapi_logs")
LOG_FILE: Path | None = None
LOG_BODY_LIMIT = max(0, int(os.getenv("SWE_FASTAPI_LOG_BODY_LIMIT", str(10 * 1024 * 1024))))

HOP_BY_HOP_HEADERS = {
    "accept-encoding",
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
SENSITIVE_HEADERS = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "x-api-key",
}


def _strip_v1_suffix(url: str) -> str:
    url = (url or "").strip()
    while url.endswith("/"):
        url = url[:-1]
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def _resolve_upstream_base(model_endpoint: str) -> str:
    endpoint = (model_endpoint or "").strip()
    lowered = endpoint.lower()
    if endpoint.startswith(("http://", "https://")):
        return _strip_v1_suffix(endpoint)
    if lowered == "model_proxy":
        return "https://models-proxy.stepfun-inc.com"
    if lowered == "agent_router":
        return "https://agentrouter.org"
    if lowered == "step_api":
        return "https://api.stepfun.com"
    if lowered in {"", "stepcast"}:
        pinned = os.getenv("STEPTRON_VLLM_ENDPOINT") or os.getenv("SWE_STEPCAST_BASE_URL")
        if pinned:
            return _strip_v1_suffix(pinned)
        return "http://stepcast-router:9200"
    raise ValueError(f"unsupported model endpoint: {model_endpoint}")


def _get_available_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _write_server_ini(port: int) -> None:
    config = configparser.ConfigParser()
    config["Server"] = {
        "pid": str(os.getpid()),
        "port": str(port),
    }
    with open("/tmp/fastapi.ini", "w", encoding="utf-8") as handle:
        config.write(handle)


def _filtered_headers(headers: httpx.Headers | Any) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        sanitized[lowered] = "<redacted>" if lowered in SENSITIVE_HEADERS else value
    return sanitized


def _truncate_body(body: bytes) -> tuple[bytes, bool]:
    if LOG_BODY_LIMIT <= 0 or len(body) <= LOG_BODY_LIMIT:
        return body, False
    return body[:LOG_BODY_LIMIT], True


def _decode_text(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _decode_logged_body(body: bytes, content_encoding: str) -> bytes:
    if not body:
        return body

    normalized = (content_encoding or "").strip().lower()
    if not normalized:
        return body

    try:
        if "gzip" in normalized or "x-gzip" in normalized:
            return gzip.decompress(body)
        if "deflate" in normalized:
            return zlib.decompress(body)
    except Exception:
        return body
    return body


def _parse_body(body: bytes, content_type: str, *, truncated: bool) -> Any:
    if not body:
        return None

    raw = body
    text = _decode_text(body)
    content_type = (content_type or "").lower()
    wants_json = "json" in content_type or text[:1] in {"{", "["}
    if wants_json:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = text
    elif (
        content_type.startswith("text/")
        or "xml" in content_type
        or "javascript" in content_type
        or "event-stream" in content_type
    ):
        parsed = text
    else:
        parsed = {
            "type": "binary",
            "bytes": len(raw),
            "preview": text,
        }

    if truncated:
        return {
            "truncated": True,
            "body": parsed,
            "logged_bytes": len(body),
        }
    return parsed


def _append_log(entry: dict[str, Any]) -> None:
    global LOG_FILE
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_FILE is None:
        LOG_FILE = LOG_DIR / f"fastapi_{int(time.time() * 1000)}.jsonl"
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_app(model_endpoint: str, port: int) -> FastAPI:
    upstream_base = _resolve_upstream_base(model_endpoint)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_name = f"fastapi_{int(time.time() * 1000)}.jsonl"
        app.state.log_file = LOG_DIR / log_name
        global LOG_FILE
        LOG_FILE = app.state.log_file
        app.state.upstream_base = upstream_base
        app.state.http_client = httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(connect=30.0, read=None, write=60.0, pool=60.0),
        )
        _write_server_ini(port)
        try:
            yield
        finally:
            await app.state.http_client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.listen_port = port

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy(request: Request, path: str):
        del path  # routed through request.url.path

        request_headers = _filtered_headers(request.headers)
        logged_request_headers = _sanitize_headers(request_headers)
        request_body = await request.body()
        request_body_for_log, request_truncated = _truncate_body(request_body)
        request_content_type = request.headers.get("content-type", "")

        upstream_url = f"{app.state.upstream_base}{request.url.path}"
        if request.url.query:
            upstream_url = f"{upstream_url}?{request.url.query}"

        log_entry: dict[str, Any] = {
            "ts": time.time(),
            "method": request.method,
            "path": request.url.path,
            "query": request.url.query,
            "endpoint": app.state.upstream_base,
            "header": logged_request_headers,
            "headers": logged_request_headers,
            "request": _parse_body(
                request_body_for_log,
                request_content_type,
                truncated=request_truncated,
            ),
            "request_body": _parse_body(
                request_body_for_log,
                request_content_type,
                truncated=request_truncated,
            ),
        }

        try:
            upstream_request = app.state.http_client.build_request(
                request.method,
                upstream_url,
                headers=request_headers,
                content=request_body,
            )
            upstream_response = await app.state.http_client.send(
                upstream_request,
                stream=True,
            )
        except httpx.HTTPError as exc:
            log_entry["status_code"] = 502
            log_entry["error"] = str(exc)
            _append_log(log_entry)
            return JSONResponse(
                status_code=502,
                content={
                    "status": 502,
                    "message": str(exc),
                },
            )

        response_headers = _filtered_headers(upstream_response.headers)
        logged_response_headers = _sanitize_headers(response_headers)
        log_entry["status_code"] = upstream_response.status_code
        log_entry["response_headers"] = logged_response_headers

        response_chunks: list[bytes] = []
        response_bytes_logged = 0
        response_content_type = upstream_response.headers.get("content-type", "")

        async def body_iter():
            nonlocal response_bytes_logged
            try:
                async for chunk in upstream_response.aiter_raw():
                    if LOG_BODY_LIMIT > 0 and response_bytes_logged < LOG_BODY_LIMIT:
                        remaining = LOG_BODY_LIMIT - response_bytes_logged
                        response_chunks.append(chunk[:remaining])
                        response_bytes_logged += min(len(chunk), remaining)
                    yield chunk
            finally:
                combined = b"".join(response_chunks)
                truncated = LOG_BODY_LIMIT > 0 and response_bytes_logged >= LOG_BODY_LIMIT
                logged_body = _decode_logged_body(
                    combined,
                    upstream_response.headers.get("content-encoding", ""),
                )
                parsed_response = _parse_body(
                    logged_body,
                    response_content_type,
                    truncated=truncated,
                )
                log_entry["response"] = parsed_response
                log_entry["response_body"] = parsed_response
                _append_log(log_entry)
                await upstream_response.aclose()

        return StreamingResponse(
            body_iter(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            background=BackgroundTask(lambda: None),
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-endpoint", required=True)
    args = parser.parse_args()

    port = _get_available_port()
    app = _build_app(args.model_endpoint, port)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
