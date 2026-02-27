"""EvoMaster 本地 Session 实现

在本地直接执行命令，无需容器。
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from evomaster.env.local import LocalEnv, LocalEnvConfig

from .base import BaseSession, SessionConfig


class LocalSessionConfig(SessionConfig):
    """本地 Session 配置"""
    encoding: str = Field(default="utf-8", description="文件编码")
    symlinks: dict[str, str] = Field(
        default_factory=dict,
        description="软链接配置，格式：{源目录路径: 工作空间内的目标路径}"
    )
    config_dir: str | None = Field(
        default=None,
        description="配置文件所在目录，用于解析 symlinks 中的相对路径"
    )
    gpu_devices: str | list[str] | None = Field(
        default=None,
        description="GPU 设备，如 '2' 或 ['0', '1']，None 表示不使用 GPU 限制"
    )
    cpu_devices: str | list[int] | None = Field(
        default=None,
        description="CPU 设备，如 '0-15' 或 [0, 1, 2, 3]，None 表示不使用 CPU 限制"
    )


class LocalSession(BaseSession):
    """本地 Session 实现
    
    在本地直接执行 bash 命令，无需容器。
    内部使用 LocalEnv 来完成底层操作。
    """

    def __init__(self, config: LocalSessionConfig | None = None):
        super().__init__(config)
        self.config: LocalSessionConfig = config or LocalSessionConfig()
        # 创建 LocalEnv 实例
        env_config = LocalEnvConfig(session_config=self.config)
        self._env = LocalEnv(env_config)
        
    def open(self) -> None:
        """打开本地会话"""
        if self._is_open:
            self.logger.warning("Session already open")
            return
        
        # 使用 LocalEnv 来设置环境
        if not self._env.is_ready:
            self._env.setup()
        
        self._is_open = True
        self.logger.info("Local session opened")

    def close(self) -> None:
        """关闭本地会话"""
        if not self._is_open:
            return
        
        # 使用 LocalEnv 来清理环境
        if self._env.is_ready:
            self._env.teardown()
        
        self._is_open = False
        self.logger.info("Session closed")

    def exec_bash(
        self,
        command: str,
        timeout: int | None = None,
        is_input: bool = False,
    ) -> dict[str, Any]:
        """执行 bash 命令
        
        提供本地命令执行能力。
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        timeout = timeout or self.config.timeout
        command = command.strip()
        
        # 本地环境不支持 is_input 模式
        if is_input:
            return {
                "stdout": "ERROR: Local session does not support is_input mode.",
                "stderr": "",
                "exit_code": 1,
            }
        
        # 使用 LocalEnv 执行命令
        result = self._env.local_exec(command, timeout=timeout)
        
        # 获取工作目录
        workspace = self.config.workspace_path
        
        # 构建结果
        return {
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "exit_code": result.get("exit_code", -1),
            "working_dir": workspace,
            "output": result.get("output", ""),
        }

    def upload(self, local_path: str, remote_path: str) -> None:
        """上传文件到本地环境"""
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        self._env.upload_file(local_path, remote_path)

    def read_file(self, remote_path: str, encoding: str = "utf-8") -> str:
        """读取远程文件内容（文本）"""
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.read_file_content(remote_path, encoding)
    
    def write_file(self, remote_path: str, content: str, encoding: str = "utf-8") -> None:
        """写入内容到远程文件"""
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        self._env.write_file_content(remote_path, content, encoding)
    
    def download(self, remote_path: str, timeout: int | None = None) -> bytes:
        """从本地环境下载文件"""
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.download_file(remote_path, timeout)
    
    def path_exists(self, remote_path: str) -> bool:
        """检查远程路径是否存在"""
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.path_exists(remote_path)
    
    def is_file(self, remote_path: str) -> bool:
        """检查远程路径是否是文件"""
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.is_file(remote_path)
    
    def is_directory(self, remote_path: str) -> bool:
        """检查远程路径是否是目录"""
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.is_directory(remote_path)
