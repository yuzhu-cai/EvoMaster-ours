"""Video descriptor tool for EmboMaster playground.

Describe robot-operation videos and provide actionable feedback with a VLM.
"""

from __future__ import annotations

import base64
import mimetypes
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
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.base_url = self._normalize_base_url(base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

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

        try:
            video_data_url = self._encode_video_as_data_url(video_path)
        except Exception as e:
            msg = f"[video-descriptor] Failed to encode video: {str(e)}"
            return msg, {"tool": self.name, "error": "video_encode_failed", "detail": str(e)}

        try:
            output_text = self._query_vlm(
                model=model,
                prompt=params.prompt,
                video_data_url=video_data_url,
            )
        except Exception as e:
            msg = f"[video-descriptor] Model call failed: {str(e)}"
            return msg, {"tool": self.name, "error": "model_call_failed", "detail": str(e)}

        observation = (
            f"[video-descriptor] success model={model} "
            f"video={video_path}\n{output_text}"
        )
        info = {
            "tool": self.name,
            "model": model,
            "video_path": str(video_path),
            "video_size_bytes": video_path.stat().st_size,
        }
        return observation, info

    def _encode_video_as_data_url(self, video_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(video_path))
        if not mime_type:
            mime_type = "video/mp4"
        with video_path.open("rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _query_vlm(self, model: str, prompt: str, video_data_url: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai package not installed. Install with: pip install openai") from e

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if not client_kwargs["api_key"]:
            # Local OpenAI-compatible gateways often ignore auth; keep a placeholder for SDK compatibility.
            client_kwargs["api_key"] = "EMPTY"
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)

        user_content: list[dict[str, Any]] = [
            {"type": "video_url", "video_url": {"url": video_data_url}},
            {
                "type": "text",
                "text": (
                    f"{prompt.strip()}\n\n"
                    "Please analyze this robot-operation video and respond in Chinese with:\n"
                    "1) 视频内容概述\n"
                    "2) 关键问题诊断\n"
                    "3) 可执行改进建议（至少3条）"
                ),
            }
        ]

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
            return content.strip() or "[video-descriptor] Empty model response."
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
                return "\n".join(text_parts)
        return "[video-descriptor] Empty model response."

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
