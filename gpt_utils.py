import json
import traceback
from typing import Callable, Dict, Optional
from openai import OpenAI

from utils.save_utils import MarkdownWriter

# OpenAI API配置
base_url = "http://123.129.219.111:3000/v1"
api_key = "sk-hnHVnUfqxBsPjGmC7EYAu5TWp9m31YqUJlPga0zulaqWRPTA"


def call_model_wo_tools(
    system_prompt: str,
    user_prompt: str,
    model_name: str = "gpt-5",
    markdown_writer: Optional[MarkdownWriter] = None,
    agent_label: Optional[str] = None,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    final_content = (completion.choices[0].message.content or "").strip()
    if markdown_writer:
        markdown_writer.log_message(agent_label or "agent Response", final_content)
    return final_content


def call_model(
    system_prompt: str,
    user_prompt: str,
    tools: list | None = None,
    tool_functions: Optional[Dict[str, Callable]] = None,
    model_name: str = "gpt-5",
    max_tool_calls: int = 20,
    markdown_writer: Optional[MarkdownWriter] = None,
    agent_label: Optional[str] = None,
) -> str:
    """
    调用GPT模型并处理工具调用，统一记录 Markdown：
    - Agent 每次回复：## <agent_label or Agent Response>
    - Tool 调用：# Tool Call - <name> + JSON 参数
    - Tool 结果：# Tool Result - <name> + 文本/JSON
    """
    tools = tools or []
    tool_functions = tool_functions or {}

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for _ in range(max_tool_calls):
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools if tools else None,
        )

        msg = completion.choices[0].message
        messages.append(msg)

        content = msg.content or ""
        if markdown_writer and content:
            markdown_writer.log_message(agent_label or "Agent Response", content)

        tool_call_list = msg.tool_calls or []
        if not tool_call_list:
            if completion.choices[0].finish_reason == "stop":
                return content.strip()
            continue

        for tool_call in tool_call_list:
            raw_args = tool_call.function.arguments or "{}"
            try:
                call_args = json.loads(raw_args)
            except Exception:
                call_args = {"_raw": raw_args}

            if markdown_writer:
                markdown_writer.log_tool_call(tool_call.function.name, call_args)

            try:
                fn = tool_functions.get(tool_call.function.name)
                if fn is None:
                    result = f"[tool:{tool_call.function.name}] not implemented"
                else:
                    result = fn(**call_args) if isinstance(call_args, dict) else fn(call_args)
            except Exception:
                result = traceback.format_exc()

            if markdown_writer:
                markdown_writer.log_tool_result(tool_call.function.name, result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

    # fallback: last agent message with content
    for m in reversed(messages):
        if m.get("role") == "agent" and m.get("content"):
            return str(m["content"]).strip()
    return ""
