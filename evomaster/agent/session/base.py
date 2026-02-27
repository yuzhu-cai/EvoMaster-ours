"""EvoMaster Session 基类

Session 是 Agent 与集群 Env 交互的介质，提供命令执行、文件操作等基础能力。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class SessionConfig(BaseModel):
    """Session 基础配置"""
    timeout: int = Field(default=300, description="默认执行超时时间（秒）")
    workspace_path: str = Field(default="/workspace", description="工作空间路径")


class BaseSession(ABC):
    """Session 抽象基类
    
    定义 Agent 与环境交互的标准接口：
    - 命令执行
    - 文件上传/下载
    - 会话生命周期管理
    """

    def __init__(self, config: SessionConfig | None = None):
        self.config = config or SessionConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """会话是否已打开"""
        return self._is_open

    @abstractmethod
    def open(self) -> None:
        """打开会话，建立与环境的连接"""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭会话，释放资源"""
        pass

    @abstractmethod
    def exec_bash(
        self,
        command: str,
        timeout: int | None = None,
        is_input: bool = False,
    ) -> dict[str, Any]:
        """执行 Bash 命令
        
        Args:
            command: 要执行的命令
            timeout: 超时时间（秒），None 使用默认值
            is_input: 是否是向正在运行的进程发送输入
            
        Returns:
            执行结果字典，包含：
            - stdout: 标准输出
            - stderr: 标准错误
            - exit_code: 退出码
            - working_dir: 当前工作目录
            - 其他环境信息
        """
        pass

    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> None:
        """上传文件到远程环境
        
        Args:
            local_path: 本地文件路径
            remote_path: 远程文件路径
        """
        pass

    @abstractmethod
    def download(self, remote_path: str, timeout: int | None = None) -> bytes:
        """从远程环境下载文件
        
        Args:
            remote_path: 远程文件路径
            timeout: 超时时间
            
        Returns:
            文件内容（字节）
        """
        pass

    def read_file(self, remote_path: str, encoding: str = "utf-8") -> str:
        """读取远程文件内容（文本）
        
        Args:
            remote_path: 远程文件路径
            encoding: 文件编码
            
        Returns:
            文件内容（字符串）
        """
        content = self.download(remote_path)
        return content.decode(encoding)

    def write_file(self, remote_path: str, content: str, encoding: str = "utf-8") -> None:
        """写入内容到远程文件
        
        Args:
            remote_path: 远程文件路径
            content: 文件内容
            encoding: 文件编码
        """
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(content.encode(encoding))
            temp_path = f.name
        
        try:
            self.upload(temp_path, remote_path)
        finally:
            os.unlink(temp_path)

    def path_exists(self, remote_path: str) -> bool:
        """检查远程路径是否存在
        
        Args:
            remote_path: 远程路径
            
        Returns:
            是否存在
        """
        result = self.exec_bash(f'test -e "{remote_path}" && echo "exists" || echo "not_exists"')
        stdout = result.get("stdout", "").strip()
        # 精确匹配，避免误判（如 "exists" in "not_exists"）
        return stdout == "exists"

    def is_file(self, remote_path: str) -> bool:
        """检查远程路径是否是文件
        
        Args:
            remote_path: 远程路径
            
        Returns:
            是否是文件
        """
        result = self.exec_bash(f'test -f "{remote_path}" && echo "file" || echo "not_file"')
        stdout = result.get("stdout", "").strip()
        # 精确匹配，避免误判（如 "file" in "not_file"）
        return stdout == "file"

    def is_directory(self, remote_path: str) -> bool:
        """检查远程路径是否是目录
        
        Args:
            remote_path: 远程路径
            
        Returns:
            是否是目录
        """
        result = self.exec_bash(f'test -d "{remote_path}" && echo "dir" || echo "not_dir"')
        stdout = result.get("stdout", "").strip()
        # 精确匹配，避免误判（如 "dir" in "not_dir"）
        return stdout == "dir"

    def __enter__(self) -> BaseSession:
        """上下文管理器入口"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.close()

