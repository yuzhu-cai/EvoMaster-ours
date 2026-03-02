#!/usr/bin/env python3
"""Minimal video API call test for OpenAI-compatible VLM endpoints.

Usage:
  python evomaster/test/test_video_api.py /path/to/video.mp4 \
    --prompt "Describe the video content" \
    --base-url "http://127.0.0.1:30030/v1" \
    --model "Qwen/Qwen3-VL-7B-Instruct"
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path
from typing import Any

from openai import OpenAI


def encode_video_as_data_uri(video_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(video_path))
    if not mime_type:
        mime_type = "video/mp4"
    with video_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_text_from_response(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            item_type = getattr(item, "type", "")
            text = getattr(item, "text", "")
            if item_type == "text" and isinstance(text, str):
                parts.append(text)
        return "\n".join(parts).strip()
    return str(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call VLM video API with a local video file.")
    parser.add_argument("--video_path", type=str, default="/data/zixing/code/EvoMaster-ours/playground/embomaster/workspaces/embomaster_config_robotwin_adjust_bottle_dsv32_e2e_10s_20260218_131240/workspaces/task_0/codebase/eval_result/put_object_cabinet/DP/demo_clean_1000/demo_clean_1000/2026-01-01 07:00:41/episode4.mp4", help="Path to local video file, e.g. ./demo.mp4")
    parser.add_argument("--prompt", type=str, default="Describe the video content.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30030/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL235B-A22B-Instruct")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    video_path = Path(args.video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Not a file: {video_path}")

    video_data_uri = encode_video_as_data_uri(video_path)

    client = OpenAI(api_key=args.api_key or "EMPTY", base_url=args.base_url)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_data_uri}},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )

    content = response.choices[0].message.content
    print(extract_text_from_response(content))


if __name__ == "__main__":
    main()
