"""Openclaw Tool Bridge — 管理 Node.js bridge 子进程与 Openclaw 插件工具的通信

通过 stdin/stdout JSON-RPC 协议与 Node.js bridge 子进程通信，
加载 Openclaw 插件并执行其注册的工具。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OpenclawBridge:
    """管理 Node.js bridge 子进程生命周期和工具执行"""

    def __init__(self, skills_ts_dir: Path):
        """初始化 OpenclawBridge

        Args:
            skills_ts_dir: skills_ts 目录路径（包含 bridge/ 和 plugins/）
        """
        self.skills_ts_dir = skills_ts_dir
        self.process: subprocess.Popen | None = None
        self._tools_info: dict[str, dict[str, Any]] = {}
        self._request_id = 0
        self._lock = threading.Lock()
        self._stderr_thread: threading.Thread | None = None

    def start(self, plugins: list[str]) -> None:
        """启动 bridge 子进程，发送 init，接收工具列表

        Args:
            plugins: 要加载的插件名称列表（如 ["feishu"]）
        """
        env = {**os.environ}
        self.process = subprocess.Popen(
            ["npx", "tsx", "bridge/server.ts"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.skills_ts_dir),
            env=env,
            bufsize=0,
        )

        # 启动 stderr 读取线程（日志输出）
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self._stderr_thread.start()

        # 发送 init 消息
        response = self._send_and_recv(
            "init", {"plugins": plugins}
        )

        if "error" in response:
            raise RuntimeError(
                f"Bridge init failed: {response['error'].get('message', 'unknown error')}"
            )

        # 解析工具信息
        tools_list = response.get("result", {}).get("tools", [])
        self._tools_info = {t["name"]: t for t in tools_list}

        logger.info(
            "Openclaw bridge started with %d tools: %s",
            len(self._tools_info),
            ", ".join(self._tools_info.keys()),
        )

    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """执行工具，返回文本结果

        Args:
            tool_name: 工具名称（如 "feishu_doc"）
            args: 工具参数

        Returns:
            工具执行结果文本
        """
        response = self._send_and_recv(
            "execute", {"tool_name": tool_name, "args": args}
        )

        if "error" in response:
            return json.dumps(
                {"error": response["error"].get("message", "unknown error")}
            )

        content = response.get("result", {}).get("content", [])
        if content:
            return content[0].get("text", "")
        return ""

    def get_tools_info(self) -> dict[str, dict[str, Any]]:
        """返回所有工具的 name/description/parameters

        Returns:
            工具名称到工具信息的映射
        """
        return self._tools_info

    def stop(self) -> None:
        """关闭 bridge 子进程"""
        if self.process and self.process.poll() is None:
            try:
                self._send({"id": self._next_id(), "method": "shutdown"})
                self.process.wait(timeout=5)
            except Exception:
                logger.warning("Bridge shutdown timed out, killing process")
                self.process.kill()
                self.process.wait(timeout=3)
            finally:
                self.process = None

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send_and_recv(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """发送请求并等待响应

        Args:
            method: JSON-RPC 方法名
            params: 请求参数

        Returns:
            响应字典
        """
        with self._lock:
            req_id = self._next_id()
            msg = {"id": req_id, "method": method, "params": params}
            self._send(msg)
            return self._recv()

    def _send(self, msg: dict[str, Any]) -> None:
        """发送 JSON 行到 stdin"""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Bridge process not running")
        line = json.dumps(msg) + "\n"
        self.process.stdin.write(line.encode("utf-8"))
        self.process.stdin.flush()

    def _recv(self) -> dict[str, Any]:
        """从 stdout 读取 JSON 行"""
        if not self.process or not self.process.stdout:
            raise RuntimeError("Bridge process not running")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Bridge process terminated unexpectedly")
        return json.loads(line.decode("utf-8"))

    def _read_stderr(self) -> None:
        """持续读取 stderr 并输出到日志"""
        if not self.process or not self.process.stderr:
            return
        try:
            for line in self.process.stderr:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug("[bridge] %s", text)
        except Exception:
            pass
