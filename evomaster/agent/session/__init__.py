"""EvoMaster Agent Session 模块

Session 是 Agent 与集群 Env 交互的介质。
"""

from .base import BaseSession, SessionConfig
from .docker import DockerSession, DockerSessionConfig
from .local import LocalSession, LocalSessionConfig

__all__ = [
    "BaseSession",
    "SessionConfig",
    "DockerSession",
    "DockerSessionConfig",
    "LocalSession",
    "LocalSessionConfig",
]

