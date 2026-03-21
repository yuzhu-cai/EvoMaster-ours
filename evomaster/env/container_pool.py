"""Docker 容器池

为飞书 Bot 提供预创建的 Docker 容器池，每个用户分配独立容器（一个用户一个容器），
消除冷启动延迟并实现完全隔离。

容器创建时挂载共享父目录，分配给用户时在容器内创建子目录。
用户的容器在 bot 生命周期内持久存在，/new 只重置对话上下文，不释放容器。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ContainerPoolConfig(BaseModel):
    """容器池配置"""

    enabled: bool = Field(default=False, description="是否启用容器池")
    image: str = Field(default="evomaster/base:latest", description="Docker 镜像")
    initial_size: int = Field(default=3, description="启动时预创建的容器数")
    max_size: int = Field(default=10, description="最大容器数")
    memory_limit: str = Field(default="8g", description="每个容器的内存限制")
    cpu_limit: float = Field(default=4.0, description="每个容器的 CPU 限制")
    gpu_devices: Optional[str] = Field(default=None, description="GPU 设备，如 'all'")
    network_mode: str = Field(default="host", description="网络模式")
    env_vars: dict[str, str] = Field(default_factory=dict, description="环境变量")
    container_name_prefix: str = Field(default="evopool", description="容器名前缀")
    cleanup_workspace: bool = Field(default=False, description="释放时是否清理 workspace（注意：workspace_host_path 是用户目录，包含所有 agent 的 run 目录）")


@dataclass
class ContainerInfo:
    """容器池中单个容器的信息"""

    container_id: str
    container_name: str
    status: Literal["idle", "in_use"] = "idle"
    assigned_user_id: Optional[str] = None
    workspace_host_path: Optional[str] = None


class ContainerPool:
    """线程安全的 Docker 容器池

    容器创建时挂载共享父目录 shared_mount_host -> /workspaces/，
    分配给用户时在容器内创建子目录，释放时清理并复用。
    """

    def __init__(self, config: ContainerPoolConfig, shared_mount_host: str):
        self._config = config
        self._shared_mount_host = shared_mount_host
        self._containers: list[ContainerInfo] = []
        self._lock = threading.RLock()
        self._counter = 0  # 容器编号计数器
        self._started = False

    @property
    def config(self) -> ContainerPoolConfig:
        return self._config

    @property
    def shared_mount_host(self) -> str:
        return self._shared_mount_host

    def start(self) -> None:
        """预创建 initial_size 个容器"""
        Path(self._shared_mount_host).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting container pool: image=%s, initial_size=%d, max_size=%d, mount=%s",
            self._config.image,
            self._config.initial_size,
            self._config.max_size,
            self._shared_mount_host,
        )

        # 清理上次进程遗留的同前缀容器，避免名字冲突
        self._cleanup_stale_containers()

        # 并行创建容器，大幅缩短启动时间
        n = self._config.initial_size
        workers = min(n, 16)  # 限制并发数，避免打满 Docker daemon
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._create_container): i for i in range(n)}
            for future in as_completed(futures):
                try:
                    info = future.result()
                    self._containers.append(info)
                except Exception:
                    logger.exception("Failed to pre-create container")

        self._started = True
        logger.info(
            "Container pool started: %d containers ready", len(self._containers)
        )

    def acquire(self, user_id: str, user_workspace_host: str) -> ContainerInfo:
        """获取用户的容器（一个用户一个容器，复用已有容器）

        Args:
            user_id: 用户 ID（飞书 open_id）
            user_workspace_host: 用户 workspace 在宿主机上的绝对路径

        Returns:
            ContainerInfo

        Raises:
            RuntimeError: 无空闲容器且已达最大容器数
        """
        with self._lock:
            # 先查找该用户是否已有容器
            for info in self._containers:
                if info.status == "in_use" and info.assigned_user_id == user_id:
                    logger.info(
                        "Reusing container %s for user %s",
                        info.container_name,
                        user_id,
                    )
                    return info

            # 用户没有容器，分配空闲容器
            for info in self._containers:
                if info.status == "idle":
                    info.status = "in_use"
                    info.assigned_user_id = user_id
                    info.workspace_host_path = user_workspace_host
                    break
            else:
                # 无空闲容器，尝试创建新的
                if len(self._containers) >= self._config.max_size:
                    raise RuntimeError(
                        f"容器池已满（{self._config.max_size}），无法分配新容器"
                    )
                info = self._create_container()
                info.status = "in_use"
                info.assigned_user_id = user_id
                info.workspace_host_path = user_workspace_host
                self._containers.append(info)

        # 在容器内创建 workspace 目录（lock 外执行避免阻塞）
        try:
            relative = Path(user_workspace_host).relative_to(self._shared_mount_host)
            container_workspace = f"/workspaces/{relative.as_posix()}"
            self._docker_exec(
                info.container_id,
                f"mkdir -p {container_workspace} && chmod 777 {container_workspace}",
            )
            logger.info(
                "Acquired container %s for user_id=%s, workspace=%s",
                info.container_name,
                user_id,
                container_workspace,
            )
        except ValueError:
            logger.warning(
                "Workspace %s is not under shared mount %s",
                user_workspace_host,
                self._shared_mount_host,
            )
        except Exception:
            logger.exception(
                "Failed to create workspace dir in container %s",
                info.container_name,
            )

        return info

    def release(self, container_id: str) -> None:
        """释放容器回池"""
        with self._lock:
            info = self._find_container(container_id)
            if info is None:
                logger.warning("Container %s not found in pool", container_id)
                return
            workspace_host = info.workspace_host_path
            info.status = "idle"
            info.assigned_user_id = None
            info.workspace_host_path = None

        # 清理工作（lock 外执行）
        try:
            # 杀掉容器内用户进程
            self._docker_exec(container_id, "pkill -u root -f -9 tmux || true")
            self._docker_exec(container_id, "tmux kill-server 2>/dev/null || true")
            # 清理 tmux 日志文件（避免下次复用时 PS1 计数错乱）
            self._docker_exec(container_id, "rm -f /tmp/evo-*.log")
        except Exception:
            logger.exception("Failed to kill processes in container %s", container_id)

        # 清理 workspace 目录（host-side，因为是 volume mount）
        if self._config.cleanup_workspace and workspace_host:
            try:
                workspace_path = Path(workspace_host)
                if workspace_path.exists():
                    shutil.rmtree(workspace_path)
                    logger.debug("Cleaned up workspace: %s", workspace_host)
            except Exception:
                logger.exception("Failed to cleanup workspace: %s", workspace_host)

        logger.info("Released container %s back to pool", container_id)

    def shutdown(self) -> None:
        """停止并删除所有池化容器"""
        with self._lock:
            containers = list(self._containers)
            self._containers.clear()

        logger.info("Shutting down container pool, removing %d containers", len(containers))
        for info in containers:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", info.container_id],
                    capture_output=True,
                    timeout=30,
                )
                logger.debug("Removed container %s", info.container_name)
            except Exception:
                logger.exception("Failed to remove container %s", info.container_name)

        logger.info("Container pool shut down")

    def get_status(self) -> dict:
        """返回池状态摘要"""
        with self._lock:
            total = len(self._containers)
            idle = sum(1 for c in self._containers if c.status == "idle")
            in_use = sum(1 for c in self._containers if c.status == "in_use")
        return {
            "total": total,
            "idle": idle,
            "in_use": in_use,
            "max_size": self._config.max_size,
        }

    def _cleanup_stale_containers(self) -> None:
        """清理上次进程遗留的同前缀容器，避免启动时名字冲突"""
        prefix = self._config.container_name_prefix
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name=^{prefix}-",
                 "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=30,
            )
        except Exception:
            logger.exception("Failed to list stale containers")
            return

        if result.returncode != 0:
            logger.warning("Failed to list stale containers: %s", result.stderr.strip())
            return

        stale = [n.strip() for n in result.stdout.strip().splitlines() if n.strip()]
        if not stale:
            return

        logger.info(
            "Found %d stale container(s) from previous run: %s", len(stale), stale
        )
        for name in stale:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    capture_output=True, text=True, timeout=30,
                )
                logger.info("Removed stale container: %s", name)
            except Exception:
                logger.exception("Failed to remove stale container: %s", name)

    def _create_container(self) -> ContainerInfo:
        """创建一个新的池化容器（线程安全）"""
        with self._lock:
            self._counter += 1
            name = f"{self._config.container_name_prefix}-{self._counter}"

        cmd = [
            "docker", "run", "-d",
            "--name", name,
            "-v", f"{self._shared_mount_host}:/workspaces",
            "--memory", self._config.memory_limit,
            "--cpus", str(self._config.cpu_limit),
        ]

        if self._config.network_mode:
            cmd.extend(["--network", self._config.network_mode])

        if self._config.gpu_devices:
            cmd.extend(["--gpus", self._config.gpu_devices])

        for key, value in self._config.env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        cmd.extend([self._config.image, "tail", "-f", "/dev/null"])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            if "Conflict" in result.stderr:
                # 名字冲突：强制删除旧容器后重试
                logger.warning(
                    "Container name conflict for %s, removing stale container and retrying",
                    name,
                )
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    capture_output=True, text=True, timeout=30,
                )
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to create container {name} after conflict retry: "
                        f"{result.stderr.strip()}"
                    )
            else:
                raise RuntimeError(
                    f"Failed to create container {name}: {result.stderr.strip()}"
                )

        container_id = result.stdout.strip()[:12]

        # 预装 tmux（避免每次 _setup_tmux 时执行 apt-get update）
        try:
            subprocess.run(
                ["docker", "exec", container_id, "bash", "-c",
                 "which tmux || (apt-get update && apt-get install -y tmux)"],
                capture_output=True, text=True, timeout=120,
            )
        except Exception:
            logger.warning("Failed to pre-install tmux in container %s", name)

        logger.info("Created pool container: %s (%s)", name, container_id)
        return ContainerInfo(container_id=container_id, container_name=name)

    def _find_container(self, container_id: str) -> Optional[ContainerInfo]:
        """在池中查找容器（调用者需持有 lock）"""
        for info in self._containers:
            if info.container_id == container_id:
                return info
        return None

    @staticmethod
    def _docker_exec(container_id: str, command: str) -> str:
        """在容器内执行命令"""
        result = subprocess.run(
            ["docker", "exec", container_id, "bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout
