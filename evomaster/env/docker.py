"""Docker 环境实现

提供 Docker 容器的底层操作接口。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from .base import BaseEnv, EnvConfig
from evomaster.agent.session.base import SessionConfig


class DockerEnvConfig(EnvConfig):
    """Docker 环境配置"""
    session_config: SessionConfig = Field(
        ...,
        description="Session 配置"
    )


# PS1 Prompt 配置，用于解析 bash 输出
PS1_BEGIN = "\n===PS1JSONBEGIN===\n"
PS1_END = "\n===PS1JSONEND===\n"
PS1_PATTERN = re.compile(
    f"{PS1_BEGIN.strip()}(.*?){PS1_END.strip()}",
    re.DOTALL | re.MULTILINE,
)


class BashMetadata:
    """Bash 执行元数据"""
    
    def __init__(
        self,
        exit_code: int = -1,
        working_dir: str = "",
        pid: int = -1,
    ):
        self.exit_code = exit_code
        self.working_dir = working_dir
        self.pid = pid

    @classmethod
    def to_ps1_prompt(cls) -> str:
        """生成 PS1 提示符配置"""
        prompt = "===PS1JSONBEGIN==="
        json_str = json.dumps({
            "pid": "$!",
            "exit_code": "$?",
            "working_dir": r"$(pwd)",
        }, indent=2)
        prompt += json_str.replace('"', r'\"')
        prompt += "===PS1JSONEND===\n"
        return prompt

    @classmethod
    def from_json(cls, json_str: str) -> BashMetadata:
        """从 JSON 解析元数据"""
        try:
            data = json.loads(json_str)
            return cls(
                exit_code=int(data.get("exit_code", -1)),
                working_dir=data.get("working_dir", ""),
                pid=int(data.get("pid", -1)) if data.get("pid") else -1,
            )
        except (json.JSONDecodeError, ValueError):
            return cls()


class DockerEnv(BaseEnv):
    """Docker 环境实现

    提供 Docker 容器的底层操作接口：
    - 容器生命周期管理
    - 命令执行
    - 文件操作
    - tmux 会话管理
    """

    def __init__(self, config: DockerEnvConfig | None = None):
        """初始化 Docker 环境

        Args:
            config: Docker 环境配置
        """
        if config is None:
            raise ValueError("DockerEnv requires DockerEnvConfig with session_config")
        super().__init__(config)
        self.config: DockerEnvConfig = config
        self._container_id: str | None = None
        self._tmux_session: str | None = None
        self._tmux_log_path: str | None = None

    def setup(self) -> None:
        """初始化 Docker 环境"""
        if self._is_ready:
            self.logger.warning("Environment already setup")
            return

        self.logger.info("Setting up Docker environment")
        self._create_or_get_container()
        self._setup_tmux()
        self._is_ready = True
        self.logger.info("Docker environment setup complete")

    def teardown(self) -> None:
        """清理 Docker 环境资源"""
        if not self._is_ready:
            return

        self.logger.info("Tearing down Docker environment")

        if self._container_id:
            session_config = self.config.session_config
            if session_config.auto_remove:
                # 自动删除模式：停止并删除容器
                self.logger.info(f"Stopping and removing container: {self._container_id[:12]}")
                try:
                    subprocess.run(
                        ["docker", "stop", self._container_id],
                        capture_output=True,
                        timeout=30,
                    )
                    subprocess.run(
                        ["docker", "rm", "-f", self._container_id],
                        capture_output=True,
                        timeout=30,
                    )
                except Exception as e:
                    self.logger.warning(f"Error stopping/removing container: {e}")
                self._container_id = None
            else:
                # 保留容器模式：只标记关闭，容器继续运行
                self.logger.info(f"Environment closed (container {self._container_id[:12]} kept running for reuse)")

        self._is_ready = False
        self.logger.info("Docker environment teardown complete")

    def get_session(self) -> Any:
        """获取 Session（DockerEnv 不直接提供 Session，由调用方管理）"""
        raise NotImplementedError("DockerEnv does not provide session directly")

    def submit_job(
        self,
        command: str,
        job_type: str = "debug",
        **kwargs: Any,
    ) -> str:
        """提交作业（DockerEnv 不直接支持作业调度）"""
        raise NotImplementedError("DockerEnv does not support job submission")

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """查询作业状态（DockerEnv 不直接支持作业调度）"""
        raise NotImplementedError("DockerEnv does not support job status")

    def cancel_job(self, job_id: str) -> None:
        """取消作业（DockerEnv 不直接支持作业调度）"""
        raise NotImplementedError("DockerEnv does not support job cancellation")

    @property
    def container_id(self) -> str | None:
        """获取容器 ID"""
        return self._container_id

    def _create_or_get_container(self) -> None:
        """创建或获取 Docker 容器"""
        session_config = self.config.session_config

        # 如果容器 ID 已存在（之前打开过但关闭了），检查容器状态
        if self._container_id:
            # 检查容器是否还在运行
            result = subprocess.run(
                ["docker", "ps", "--filter", f"id={self._container_id}", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                # 容器正在运行，直接复用
                self.logger.info(f"Reusing existing running container: {self._container_id[:12]}")
                return
            else:
                # 容器已停止，尝试启动
                self.logger.info(f"Starting existing stopped container: {self._container_id[:12]}")
                try:
                    result = subprocess.run(
                        ["docker", "start", self._container_id],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        # 等待容器完全启动
                        time.sleep(1)
                        return
                    else:
                        self.logger.warning(f"Failed to start container: {result.stderr}")
                        # 继续创建新容器
                        self._container_id = None
                except Exception as e:
                    self.logger.warning(f"Error starting container: {e}")
                    # 继续创建新容器
                    self._container_id = None

        # 如果指定使用已存在的容器
        if session_config.use_existing_container:
            self.logger.info(f"Using existing container: {session_config.use_existing_container}")
            # 检查容器是否存在且运行中
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={session_config.use_existing_container}", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                self._container_id = result.stdout.strip()
                self.logger.info(f"Found running container: {self._container_id[:12]}")
            else:
                # 尝试查找停止的容器
                result = subprocess.run(
                    ["docker", "ps", "-a", "--filter", f"name={session_config.use_existing_container}", "--format", "{{.ID}}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    container_id = result.stdout.strip()
                    # 启动容器
                    subprocess.run(
                        ["docker", "start", container_id],
                        capture_output=True,
                        timeout=30,
                    )
                    self._container_id = container_id
                    self.logger.info(f"Started existing container: {self._container_id[:12]}")
                else:
                    raise RuntimeError(f"Container '{session_config.use_existing_container}' not found")
            return

        self.logger.info(f"Starting Docker container with image: {session_config.image}")

        # 容器名
        container_name = session_config.container_name or f"evomaster-{os.getpid()}-{int(time.time())}"

        # 如果指定了容器名，检查容器是否已存在
        if session_config.container_name:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.ID}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                container_id = result.stdout.strip()
                # 容器已存在，检查是否运行中
                result_running = subprocess.run(
                    ["docker", "ps", "--filter", f"id={container_id}", "--format", "{{.ID}}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result_running.returncode == 0 and result_running.stdout.strip():
                    # 容器正在运行，直接复用
                    self.logger.info(f"Reusing existing running container: {container_id[:12]}")
                    self._container_id = container_id
                    return
                else:
                    # 容器已停止，启动它
                    self.logger.info(f"Starting existing stopped container: {container_id[:12]}")
                    subprocess.run(
                        ["docker", "start", container_id],
                        capture_output=True,
                        timeout=30,
                    )
                    time.sleep(1)
                    self._container_id = container_id
                    return

        # 构建 docker run 命令
        cmd = ["docker", "run", "-d"]
        cmd.extend(["--name", container_name])

        # 资源限制
        cmd.extend(["--memory", session_config.memory_limit])
        cmd.extend(["--cpus", str(session_config.cpu_limit)])

        # GPU 设备
        if session_config.gpu_devices is not None:
            if isinstance(session_config.gpu_devices, str):
                if session_config.gpu_devices.lower() == "all":
                    cmd.extend(["--gpus", "all"])
                else:
                    cmd.extend(["--gpus", f"device={session_config.gpu_devices}"])
            elif isinstance(session_config.gpu_devices, list):
                devices_str = ",".join(session_config.gpu_devices)
                cmd.extend(["--gpus", f"device={devices_str}"])

        # 网络
        cmd.extend(["--network", session_config.network_mode])

        # 工作目录
        cmd.extend(["-w", session_config.working_dir])

        # 挂载卷
        for host_path, container_path in session_config.volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])

        # 环境变量
        for key, value in session_config.env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # 自动删除
        if session_config.auto_remove:
            cmd.append("--rm")

        # 镜像和命令（使用 tail -f 保持容器运行）
        cmd.extend([session_config.image, "tail", "-f", "/dev/null"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to start container: {result.stderr}")

            self._container_id = result.stdout.strip()
            self.logger.info(f"Container started: {self._container_id[:12]}")

            # 初始化工作目录权限，确保可以写入文件
            try:
                self.docker_exec(f"mkdir -p {session_config.working_dir} && chmod 777 {session_config.working_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize workspace directory: {e}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Timeout starting Docker container")
        except Exception as e:
            self.logger.error(f"Failed to start Docker container: {e}")
            raise

    def _setup_tmux(self) -> None:
        """设置 tmux 会话"""
        if not self._container_id:
            raise RuntimeError("Container not started")

        session_name = f"evo-{self._container_id[:8]}"
        log_path = f"/tmp/evo-{self._container_id[:8]}.log"

        self._tmux_session = session_name
        self._tmux_log_path = log_path

        # 清理旧的 tmux 日志文件（容器池复用场景）
        self.docker_exec(f"rm -f {log_path}")

        # 安装 tmux（先检查是否已安装，跳过不必要的 apt-get update）
        check = self.docker_exec("which tmux", timeout=10)
        if check.get("exit_code") != 0:
            self.docker_exec("apt-get update && apt-get install -y tmux || true", timeout=120)

        # 杀掉可能残留的同名 tmux session（容器池复用场景）
        self.docker_exec(f"tmux kill-session -t {session_name} 2>/dev/null || true", timeout=10)

        # 创建 tmux 会话
        self.docker_exec(f"tmux new-session -d -s {session_name} 'bash -i'")

        # 设置管道日志
        self.docker_exec(f"tmux pipe-pane -o -t {session_name} 'cat >> {log_path}'")

        # 设置 PS1 提示符
        ps1 = BashMetadata.to_ps1_prompt()
        init_cmd = f"PROMPT_COMMAND='PS1=\"{ps1}\"'"
        self.tmux_send_keys(init_cmd, enter=True)

        # cd 到 working_dir（如果配置了）
        working_dir = self.config.session_config.working_dir
        if working_dir and working_dir != "/":
            self.tmux_send_keys(f"cd {working_dir}", enter=True)

        # 触发提示符并等待日志 flush
        self.tmux_send_keys("", enter=True)
        time.sleep(2)  # 从 0.5s 增加到 2s，确保 pipe-pane 日志 flush

        self.logger.debug(f"Tmux session {session_name} initialized")

    def docker_exec(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> dict[str, Any]:
        """在容器中执行命令（直接执行，非 tmux）

        Args:
            command: 要执行的命令
            timeout: 超时时间（秒）
            workdir: 工作目录

        Returns:
            执行结果字典，包含：
            - stdout: 标准输出
            - stderr: 标准错误
            - exit_code: 退出码
            - output: stdout + stderr 的组合
        """
        if not self._container_id:
            raise RuntimeError("Container not started")

        timeout = timeout or self.config.session_config.timeout

        cmd = ["docker", "exec"]
        if workdir:
            cmd.extend(["-w", workdir])
        cmd.extend([self._container_id, "bash", "-c", command])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
            )
            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")
            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.returncode,
                "output": stdout + stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "exit_code": -1,
                "output": f"Command timed out after {timeout}s",
            }

    def tmux_send_keys(self, keys: str, enter: bool = False) -> None:
        """向 tmux 发送按键

        Args:
            keys: 要发送的按键
            enter: 是否按回车
        """
        if not self._tmux_session:
            raise RuntimeError("Tmux session not initialized")

        # 转义单引号
        escaped = keys.replace("'", "'\\''")
        cmd = f"tmux send-keys -t {self._tmux_session} '{escaped}'"
        if enter:
            cmd += " C-m"

        self.docker_exec(cmd)

    def get_tmux_logs(self) -> str:
        """获取 tmux 日志

        Returns:
            tmux 日志内容
        """
        if not self._tmux_log_path:
            return ""

        result = self.docker_exec(f"cat {self._tmux_log_path} 2>/dev/null || echo ''")
        return result.get("stdout", "")

    def is_mounted_path(self, container_path: str) -> tuple[bool, str | None]:
        """检查路径是否在挂载的卷中

        Args:
            container_path: 容器内的路径（应该是绝对路径）

        Returns:
            (is_mounted, host_path): 是否在挂载卷中，以及对应的宿主机路径（如果存在）
        """
        session_config = self.config.session_config
        if not session_config.volumes:
            return False, None

        # 规范化容器路径（确保是绝对路径，去除末尾的斜杠）
        container_path = str(Path(container_path).as_posix())
        if not container_path.startswith("/"):
            # 如果不是绝对路径，可能需要先解析，但这里我们假设已经是绝对路径
            # 如果不是，返回 False
            return False, None

        # 检查每个挂载卷
        for host_path, mount_point in session_config.volumes.items():
            # 规范化挂载点路径
            mount_point_norm = str(Path(mount_point).as_posix())

            # 检查容器路径是否以挂载点开头
            # 需要确保是精确匹配（避免 /workspace 匹配 /workspace2）
            if container_path == mount_point_norm:
                # 完全匹配挂载点本身
                return True, str(Path(host_path))
            elif container_path.startswith(mount_point_norm + "/"):
                # 是挂载点的子路径
                # 计算相对路径
                relative_path = container_path[len(mount_point_norm):].lstrip("/")
                # 构建宿主机路径
                host_path_obj = Path(host_path) / relative_path
                return True, str(host_path_obj)

        return False, None

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """上传文件到容器

        如果目标路径在挂载的卷中，直接在宿主机复制文件。

        Args:
            local_path: 本地文件路径
            remote_path: 远程文件路径（容器内路径）
        """
        if not self._container_id:
            raise RuntimeError("Container not started")

        # 检查是否在挂载卷中
        is_mounted, host_path = self.is_mounted_path(remote_path)

        if is_mounted and host_path:
            # 直接在宿主机复制文件
            try:
                import shutil

                # 确保目标目录存在
                host_path_obj = Path(host_path)
                host_path_obj.parent.mkdir(parents=True, exist_ok=True)

                # 复制文件
                shutil.copy2(local_path, host_path)
            except Exception as e:
                raise RuntimeError(f"Failed to upload file {local_path} to host path {host_path}: {e}")
            return

        # 不在挂载卷中，使用 docker cp
        # 确保远程目录存在并设置正确权限
        remote_dir = str(Path(remote_path).parent)
        # 创建目录并设置权限（777 确保所有用户都可以写入）
        self.docker_exec(f"mkdir -p {remote_dir} && chmod 777 {remote_dir}")

        cmd = ["docker", "cp", local_path, f"{self._container_id}:{remote_path}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to upload file: {result.stderr}")

        # 上传后设置文件权限，确保可读写执行
        self.docker_exec(f"chmod 777 {remote_path}")

    def download_file(self, remote_path: str, timeout: int | None = None) -> bytes:
        """从容器下载文件

        如果路径在挂载的卷中，直接在宿主机读取。

        Args:
            remote_path: 远程文件路径（容器内路径）
            timeout: 超时时间

        Returns:
            文件内容（字节）
        """
        if not self._container_id:
            raise RuntimeError("Container not started")

        # 检查是否在挂载卷中
        is_mounted, host_path = self.is_mounted_path(remote_path)

        if is_mounted and host_path:
            # 直接在宿主机读取
            try:
                # 检查是否是目录
                if os.path.isdir(host_path):
                    raise RuntimeError(f"Cannot download directory: {remote_path}. Use exec_bash to list directory contents instead.")

                with open(host_path, "rb") as f:
                    return f.read()
            except FileNotFoundError:
                raise RuntimeError(f"File not found: {remote_path} (host path: {host_path})")
            except Exception as e:
                raise RuntimeError(f"Failed to download file {remote_path} from host: {e}")

        # 不在挂载卷中，使用 docker cp
        # 检查路径是否是目录，docker cp 不能复制目录
        if self.is_directory(remote_path):
            raise RuntimeError(f"Cannot download directory: {remote_path}. Use exec_bash to list directory contents instead.")

        timeout = timeout or 60

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            cmd = ["docker", "cp", f"{self._container_id}:{remote_path}", temp_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

            if result.returncode != 0:
                # 检查错误信息中是否包含目录相关的错误
                error_msg = result.stderr.strip()
                if "cannot copy directory" in error_msg.lower() or "is a directory" in error_msg.lower():
                    raise RuntimeError(f"Cannot download directory: {remote_path}. Use exec_bash to list directory contents instead.")
                raise RuntimeError(f"Failed to download file: {error_msg}")

            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def read_file_content(self, remote_path: str, encoding: str = "utf-8") -> str:
        """读取远程文件内容（文本）

        如果路径在挂载的卷中，直接在宿主机读取。

        Args:
            remote_path: 远程文件路径（容器内路径）
            encoding: 文件编码

        Returns:
            文件内容（字符串）
        """
        is_mounted, host_path = self.is_mounted_path(remote_path)

        if is_mounted and host_path:
            # 直接在宿主机读取
            try:
                with open(host_path, "r", encoding=encoding) as f:
                    return f.read()
            except FileNotFoundError:
                raise RuntimeError(f"File not found: {remote_path} (host path: {host_path})")
            except Exception as e:
                raise RuntimeError(f"Failed to read file {remote_path} from host: {e}")

        # 不在挂载卷中，使用 download_file
        content = self.download_file(remote_path)
        return content.decode(encoding)

    def write_file_content(self, remote_path: str, content: str, encoding: str = "utf-8") -> None:
        """写入内容到远程文件

        如果路径在挂载的卷中，直接在宿主机写入。

        Args:
            remote_path: 远程文件路径（容器内路径）
            content: 文件内容
            encoding: 文件编码
        """
        is_mounted, host_path = self.is_mounted_path(remote_path)

        if is_mounted and host_path:
            # 直接在宿主机写入
            try:
                # 确保目录存在
                host_path_obj = Path(host_path)
                host_path_obj.parent.mkdir(parents=True, exist_ok=True)

                # 写入文件
                with open(host_path, "w", encoding=encoding) as f:
                    f.write(content)
            except Exception as e:
                raise RuntimeError(f"Failed to write file {remote_path} to host: {e}")
            return

        # 不在挂载卷中，使用 upload_file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(content.encode(encoding))
            temp_path = f.name

        try:
            self.upload_file(temp_path, remote_path)
        finally:
            os.unlink(temp_path)

    def path_exists(self, remote_path: str) -> bool:
        """检查远程路径是否存在

        如果路径在挂载的卷中，直接在宿主机检查。

        Args:
            remote_path: 远程路径（容器内路径）

        Returns:
            是否存在
        """
        is_mounted, host_path = self.is_mounted_path(remote_path)

        if is_mounted and host_path:
            return os.path.exists(host_path)

        # 不在挂载卷中，使用 docker exec
        result = self.docker_exec(f'test -e "{remote_path}" && echo "exists" || echo "not_exists"')
        stdout = result.get("stdout", "").strip()
        return stdout == "exists"

    def is_file(self, remote_path: str) -> bool:
        """检查远程路径是否是文件

        如果路径在挂载的卷中，直接在宿主机检查。

        Args:
            remote_path: 远程路径（容器内路径）

        Returns:
            是否是文件
        """
        is_mounted, host_path = self.is_mounted_path(remote_path)

        if is_mounted and host_path:
            return os.path.isfile(host_path)

        # 不在挂载卷中，使用 docker exec
        result = self.docker_exec(f'test -f "{remote_path}" && echo "file" || echo "not_file"')
        stdout = result.get("stdout", "").strip()
        return stdout == "file"

    def is_directory(self, remote_path: str) -> bool:
        """检查远程路径是否是目录

        如果路径在挂载的卷中，直接在宿主机检查。

        Args:
            remote_path: 远程路径（容器内路径）

        Returns:
            是否是目录
        """
        is_mounted, host_path = self.is_mounted_path(remote_path)

        if is_mounted and host_path:
            return os.path.isdir(host_path)

        # 不在挂载卷中，使用 docker exec
        result = self.docker_exec(f'test -d "{remote_path}" && echo "dir" || echo "not_dir"')
        stdout = result.get("stdout", "").strip()
        return stdout == "dir"

