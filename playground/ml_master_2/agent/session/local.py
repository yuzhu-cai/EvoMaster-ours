"""ML Master 2 本地 Session 实现

使用 MLMaster2LocalEnv 替代默认的 LocalEnv，
使主工作空间在 split_workspace_for_exp 启用时仍会创建软链接。
"""

from __future__ import annotations

from evomaster.env.local import LocalEnvConfig
from evomaster.agent.session.base import BaseSession
from evomaster.agent.session.local import LocalSession, LocalSessionConfig

from ...env.local import MLMaster2LocalEnv


class MLMaster2LocalSession(LocalSession):
    """ML Master 2 专用本地 Session

    使用 MLMaster2LocalEnv 替代默认的 LocalEnv，
    使主工作空间在 split_workspace_for_exp 启用时仍会创建软链接。
    """

    def __init__(self, config: LocalSessionConfig | None = None):
        BaseSession.__init__(self, config)
        self.config: LocalSessionConfig = config or LocalSessionConfig()
        env_config = LocalEnvConfig(session_config=self.config)
        self._env = MLMaster2LocalEnv(env_config)
