"""Video descriptor tool for EmboMaster playground.

Describe robot-operation videos and provide actionable feedback with a VLM.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession


class VideoDescriptorToolParams(BaseToolParams):
    """Describe an eval video and provide improvement feedback for the next round.

    Before calling this tool, first locate the target video under:
    `workspaces/task_0/codebase/eval_result`.
    """

    name: ClassVar[str] = "video-descriptor"

    video_path: str = Field(
        description=(
            "Path to the video file. Locate it under "
            "`workspaces/task_0/codebase/eval_result` first, then pass either an absolute path "
            "or a workspace-relative path (for example: `eval_result/xxx.mp4`)."
        )
    )
    prompt: str = Field(description="Evaluation prompt for the robot-operation video.")
    model: str = Field(
        default="",
        description="Optional model override. Empty means using tool default.",
    )
    input_mode: str = Field(
        default="",
        description="Optional input mode override. One of: auto, video, frames. Empty means using tool default.",
    )


class VideoDescriptorTool(BaseTool):
    """Tool wrapper for robot-operation video understanding."""

    name: ClassVar[str] = "video-descriptor"
    params_class: ClassVar[type[BaseToolParams]] = VideoDescriptorToolParams

    def __init__(
        self,
        api_key: str,
        base_url: str | None = "http://127.0.0.1:30030/v1",
        model: str = "Qwen/Qwen3-VL-235B-A22B-Instruct",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 120,
        max_frames: int = 12,
        input_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.base_url = self._normalize_base_url(base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_frames = max(1, int(max_frames))
        self.input_mode = self._normalize_input_mode(input_mode)

    def execute(self, session: "BaseSession", args_json: str) -> tuple[str, dict[str, Any]]:
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {str(e)}", {"error": str(e)}

        assert isinstance(params, VideoDescriptorToolParams)

        workspace = Path(getattr(session.config, "workspace_path", ".")).resolve()
        requested_path = Path(params.video_path)
        video_path = requested_path if requested_path.is_absolute() else (workspace / requested_path)
        video_path = video_path.resolve()

        if not video_path.exists():
            msg = (
                f"[video-descriptor] Video not found: {video_path}. "
                "Please first locate the video in workspaces/task_0/codebase/eval_result."
            )
            return msg, {"tool": self.name, "error": "video_not_found", "video_path": str(video_path)}
        if not video_path.is_file():
            msg = f"[video-descriptor] Not a file: {video_path}"
            return msg, {"tool": self.name, "error": "not_a_file", "video_path": str(video_path)}

        model = params.model.strip() if params.model.strip() else self.model
        input_mode = self._resolve_input_mode(params.input_mode)

        try:
            output_text, payload_info = self._query_vlm(
                model=model,
                prompt=params.prompt,
                video_path=video_path,
                input_mode=input_mode,
            )
        except Exception as e:
            msg = f"[video-descriptor] Model call failed: {str(e)}"
            return msg, {
                "tool": self.name,
                "error": "model_call_failed",
                "detail": str(e),
                "input_mode": input_mode,
            }

        observation = (
            f"[video-descriptor] success model={model} "
            f"video={video_path} input_mode={payload_info.get('input_mode', input_mode)}\n{output_text}"
        )
        info = {
            "tool": self.name,
            "model": model,
            "video_path": str(video_path),
            "video_size_bytes": video_path.stat().st_size,
            **payload_info,
        }
        return observation, info

    def _query_vlm(self, model: str, prompt: str, video_path: Path, input_mode: str) -> tuple[str, dict[str, Any]]:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai package not installed. Install with: pip install openai") from e

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if not client_kwargs["api_key"]:
            client_kwargs["api_key"] = "EMPTY"
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)

        user_content, payload_info = self._build_user_content(video_path, prompt, input_mode)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a robotics evaluation assistant. "
                        "Given sampled frames from one robot-operation video, describe behavior and give practical feedback."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

        content = response.choices[0].message.content
        if isinstance(content, str):
            return content.strip() or "[video-descriptor] Empty model response.", payload_info
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        if isinstance(text, str) and text.strip():
                            text_parts.append(text.strip())
                    continue
                item_type = getattr(item, "type", "")
                text = getattr(item, "text", "")
                if item_type == "text" and isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
            if text_parts:
                return "\n".join(text_parts), payload_info
        return "[video-descriptor] Empty model response.", payload_info

    def _build_user_content(self, video_path: Path, prompt: str, input_mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt_text = (
            f"{prompt.strip()}\n\n"
            "Please analyze this robot-operation video and respond in Chinese with:\n"
            "1) 视频内容概述\n"
            "2) 关键问题诊断\n"
            "3) 可执行改进建议（至少3条）"
        )
        if input_mode == "video":
            video_data_url = self._encode_file_as_data_url(video_path, default_mime_type="video/mp4")
            return [
                {"type": "video_url", "video_url": {"url": video_data_url}},
                {"type": "text", "text": prompt_text},
            ], {"input_mode": "video", "frame_count": 0}

        frame_paths = self._extract_frames(video_path, max_frames=self.max_frames)
        image_parts: list[dict[str, Any]] = []
        for frame_path in frame_paths:
            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._encode_file_as_data_url(frame_path, default_mime_type="image/jpeg")},
                }
            )
        image_parts.append({"type": "text", "text": prompt_text})
        return image_parts, {"input_mode": "frames", "frame_count": len(frame_paths)}

    def _extract_frames(self, video_path: Path, max_frames: int) -> list[Path]:
        ffmpeg = self._require_binary("ffmpeg")
        ffprobe = self._require_binary("ffprobe")
        duration = self._probe_duration_seconds(ffprobe, video_path)
        if duration <= 0:
            raise RuntimeError(f"Could not determine video duration for frame extraction: {video_path}")

        with tempfile.TemporaryDirectory(prefix="video_descriptor_frames_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            frame_paths: list[Path] = []
            step = duration / float(max_frames + 1)
            for idx in range(max_frames):
                timestamp = max(0.0, min(duration, step * (idx + 1)))
                frame_path = tmp_dir / f"frame_{idx:02d}.jpg"
                cmd = [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    str(frame_path),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg frame extraction failed at t={timestamp:.3f}s: {proc.stderr.strip() or proc.stdout.strip()}"
                    )
                if frame_path.exists() and frame_path.stat().st_size > 0:
                    final_path = video_path.parent / f".video_descriptor_{video_path.stem}_{idx:02d}.jpg"
                    final_path.write_bytes(frame_path.read_bytes())
                    frame_paths.append(final_path)
            if not frame_paths:
                raise RuntimeError(f"No frames extracted from video: {video_path}")
            return frame_paths

    def _probe_duration_seconds(self, ffprobe: str, video_path: Path) -> float:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {proc.stderr.strip() or proc.stdout.strip()}")
        payload = json.loads(proc.stdout or "{}")
        duration_raw = payload.get("format", {}).get("duration")
        if duration_raw in (None, ""):
            return 0.0
        return max(0.0, float(duration_raw))

    def _encode_file_as_data_url(self, path: Path, default_mime_type: str) -> str:
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = default_mime_type
        with path.open("rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _normalize_base_url(self, base_url: str | None) -> str | None:
        if not base_url:
            return None
        url = str(base_url).strip()
        if not url:
            return None
        if not (url.startswith("http://") or url.startswith("https://")):
            url = f"http://{url}"
        if "/v1" not in url.rstrip("/").split("?")[0]:
            url = f"{url.rstrip('/')}/v1"
        return url

    def _normalize_input_mode(self, input_mode: str | None) -> str:
        mode = (input_mode or "auto").strip().lower()
        if mode not in {"auto", "video", "frames"}:
            return "auto"
        return mode

    def _resolve_input_mode(self, requested_mode: str | None) -> str:
        mode = self._normalize_input_mode(requested_mode or self.input_mode)
        if mode != "auto":
            return mode
        return "frames" if self._uses_remote_openai_protocol() else "video"

    def _uses_remote_openai_protocol(self) -> bool:
        if not self.base_url:
            return False
        normalized = self.base_url.lower()
        return not any(host in normalized for host in ("127.0.0.1", "localhost", "0.0.0.0"))

    def _require_binary(self, binary_name: str) -> str:
        from shutil import which

        binary = which(binary_name)
        if not binary:
            raise RuntimeError(
                f"{binary_name} is required for video frame sampling mode, but it was not found in PATH"
            )
        return binary
