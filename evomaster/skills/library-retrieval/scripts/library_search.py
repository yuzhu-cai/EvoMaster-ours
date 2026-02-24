#!/usr/bin/env python3
"""Search literature chunks from local retrieval HTTP service."""

from __future__ import annotations

import argparse
import json
import sys
from urllib import error, request


def post_chunks(service_url: str, query: str, timeout: float) -> dict:
    payload = {"query": query}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        service_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve literature chunks via HTTP service")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--url", default="http://127.0.0.1:30388/chunks", help="Service URL")
    parser.add_argument("--top_k", type=int, default=0, help="Optional client-side truncate (>0 enabled)")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    try:
        result = post_chunks(args.url, args.query, args.timeout)
        chunks = result.get("chunks", [])
        if not isinstance(chunks, list):
            chunks = [chunks]
        if args.top_k and args.top_k > 0:
            chunks = chunks[: args.top_k]

        out = {
            "query": result.get("query", args.query),
            "service_url": args.url,
            "count": len(chunks),
            "chunks": chunks,
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    except error.HTTPError as e:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "service_url": args.url,
                    "error": f"HTTPError: {e.code}",
                    "status": e.code,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        sys.exit(1)
    except Exception as e:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "service_url": args.url,
                    "error": str(e),
                    "status": None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
