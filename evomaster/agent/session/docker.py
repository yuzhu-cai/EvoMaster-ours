"""EvoMaster Docker Session 实现

基于 Docker 容器的 Session 实现，提供隔离的执行环境。
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import Field

from evomaster.env.docker import DockerEnv, DockerEnvConfig, PS1_PATTERN, BashMetadata

from .base import BaseSession, SessionConfig


class DockerSessionConfig(SessionConfig):
    """Docker Session 配置"""
    image: str = Field(default="python:3.11-slim", description="Docker 镜像")
    container_name: str | None = Field(default=None, description="容器名称，None 自动生成")
    working_dir: str = Field(default="/workspace", description="工作目录")
    memory_limit: str = Field(default="4g", description="内存限制")
    cpu_limit: float = Field(default=2.0, description="CPU 限制")
    gpu_devices: str | list[str] | None = Field(default=None, description="GPU 设备，如 'all' 或 ['0', '1']，None 表示不使用 GPU")
    network_mode: str = Field(default="bridge", description="网络模式")
    volumes: dict[str, str] = Field(default_factory=dict, description="挂载卷 {host_path: container_path}")
    env_vars: dict[str, str] = Field(default_factory=dict, description="环境变量")
    auto_remove: bool = Field(default=True, description="容器结束后自动删除")
    use_existing_container: str | None = Field(default=None, description="使用已存在的容器名称，如果设置则不会创建新容器")


class DockerSession(BaseSession):
    """Docker Session 实现
    
    使用 Docker 容器提供隔离的执行环境。
    内部使用 DockerEnv 来完成底层操作。
    """

    def __init__(self, config: DockerSessionConfig | None = None):
        super().__init__(config)
        self.config: DockerSessionConfig = config or DockerSessionConfig()
        # 创建 DockerEnv 实例
        env_config = DockerEnvConfig(session_config=self.config)
        self._env = DockerEnv(env_config)
        # 会话状态管理
        self._last_ps1_count: int = 0
        self._prev_command_status: Literal["completed", "timeout"] = "completed"
        self._prev_command_output: str = ""
        
    def open(self) -> None:
        """启动 Docker 容器"""
        if self._is_open:
            self.logger.warning("Session already open")
            return
        
        # 使用 DockerEnv 来设置环境
        if not self._env.is_ready:
            self._env.setup()
        
        # 获取初始 PS1 计数
        logs = self._env.get_tmux_logs()
        matches = list(PS1_PATTERN.finditer(logs))
        self._last_ps1_count = len(matches)
        
        self._is_open = True
        self.logger.info("Docker session opened")

    def close(self) -> None:
        """关闭会话
        
        如果 auto_remove=True，会停止并删除容器。
        如果 auto_remove=False，只标记会话为关闭状态，容器继续运行以便复用。
        """
        if not self._is_open:
            return
        
        # 使用 DockerEnv 来清理环境
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
        """通过 tmux 执行 bash 命令
        
        提供持久化的 bash 环境，支持环境变量、工作目录等状态保持。
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        timeout = timeout or self.config.timeout
        command = command.strip()
        
        # 处理输入模式
        if is_input:
            if self._prev_command_status == "completed":
                if command == "":
                    return {
                        "stdout": "ERROR: No previous running command to retrieve logs from.",
                        "stderr": "",
                        "exit_code": 1,
                    }
                else:
                    return {
                        "stdout": "ERROR: No previous running command to interact with.",
                        "stderr": "",
                        "exit_code": 1,
                    }
            
            # 发送控制信号或输入
            if command.startswith("C-") and len(command) == 3:
                self._env.tmux_send_keys(command, enter=False)
            elif command == "":
                pass  # 只获取日志
            else:
                self._env.tmux_send_keys(command, enter=True)
        else:
            # 正常命令执行
            if self._prev_command_status != "completed" and command != "":
                return {
                    "stdout": f"[Previous command is still running. Use is_input=true to interact.]",
                    "stderr": "",
                    "exit_code": 1,
                }
            
            if command != "":
                self._env.tmux_send_keys(command, enter=True)
        
        # 等待命令完成
        start_time = time.time()
        poll_interval = 0.5
        self._prev_command_status = "timeout"
        
        while time.time() - start_time < timeout:
            logs = self._env.get_tmux_logs()
            matches = list(PS1_PATTERN.finditer(logs))
            ps1_count = len(matches)
            
            if ps1_count > self._last_ps1_count:
                # 命令完成
                self._prev_command_status = "completed"
                break
            
            time.sleep(poll_interval)
        
        # 解析输出
        logs = self._env.get_tmux_logs()
        matches = list(PS1_PATTERN.finditer(logs))
        ps1_count = len(matches)
        
        output = ""
        exit_code = -1
        working_dir = ""
        
        if ps1_count > self._last_ps1_count:
            # 提取最后一个命令的输出
            if self._last_ps1_count > 0:
                prev_match = matches[self._last_ps1_count - 1]
                curr_match = matches[ps1_count - 1]
                output = logs[prev_match.end():curr_match.start()]
            else:
                curr_match = matches[ps1_count - 1]
                output = logs[:curr_match.start()]
            
            # 解析元数据
            try:
                metadata = BashMetadata.from_json(matches[-1].group(1))
                exit_code = metadata.exit_code
                working_dir = metadata.working_dir
            except Exception:
                pass
            
            self._last_ps1_count = ps1_count
        else:
            # 超时，获取部分输出
            if self._last_ps1_count > 0 and matches:
                prev_match = matches[self._last_ps1_count - 1]
                output = logs[prev_match.end():]
        
        # 清理输出
        output = output.strip()
        if command and output.startswith(command):
            output = output[len(command):].strip()
        
        # 构建结果
        result = {
            "stdout": output,
            "stderr": "",
            "exit_code": exit_code,
            "working_dir": working_dir,
            "output": output,
        }
        
        if self._prev_command_status == "timeout":
            result["stdout"] += f"\n[Command timed out after {timeout}s]"
            result["exit_code"] = -1
        
        return result

    def upload(self, local_path: str, remote_path: str) -> None:
        """上传文件到容器
        
        如果目标路径在挂载的卷中，直接在宿主机复制文件。
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        self._env.upload_file(local_path, remote_path)

    def read_file(self, remote_path: str, encoding: str = "utf-8") -> str:
        """读取远程文件内容（文本）
        
        如果路径在挂载的卷中，直接在宿主机读取。
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.read_file_content(remote_path, encoding)
    
    def write_file(self, remote_path: str, content: str, encoding: str = "utf-8") -> None:
        """写入内容到远程文件
        
        如果路径在挂载的卷中，直接在宿主机写入。
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        self._env.write_file_content(remote_path, content, encoding)
    
    def download(self, remote_path: str, timeout: int | None = None) -> bytes:
        """从容器下载文件
        
        如果路径在挂载的卷中，直接在宿主机读取。
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.download_file(remote_path, timeout)
    
    def path_exists(self, remote_path: str) -> bool:
        """检查远程路径是否存在
        
        如果路径在挂载的卷中，直接在宿主机检查。
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.path_exists(remote_path)
    
    def is_file(self, remote_path: str) -> bool:
        """检查远程路径是否是文件
        
        如果路径在挂载的卷中，直接在宿主机检查。
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.is_file(remote_path)
    
    def is_directory(self, remote_path: str) -> bool:
        """检查远程路径是否是目录
        
        如果路径在挂载的卷中，直接在宿主机检查。
        """
        if not self._is_open:
            raise RuntimeError("Session not open")
        
        return self._env.is_directory(remote_path)
