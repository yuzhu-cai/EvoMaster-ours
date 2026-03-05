"""ML Master 2 本地环境实现

继承 evomaster 的 LocalEnv，区别是：当 split_workspace_for_exp 启用时，
主工作空间的软链接创建不跳过（仍会创建）。
"""

from __future__ import annotations

from pathlib import Path

from evomaster.env.local import LocalEnv, LocalEnvConfig


class MLMaster2LocalEnv(LocalEnv):
    """ML Master 2 专用本地环境

    与基类 LocalEnv 的区别：在 setup() 中，即使启用了 split_workspace_for_exp，
    主工作空间的软链接创建也不会跳过，始终会创建。
    """

    def setup(self) -> None:
        """初始化本地环境

        与基类不同：无论 split_workspace_for_exp 是否启用，都会在主工作空间创建软链接。
        """
        if self._is_ready:
            self.logger.warning("Environment already setup")
            return

        self.logger.info("Setting up ML Master 2 local environment")

        # 确保工作目录存在
        workspace = Path(self.config.session_config.workspace_path)
        workspace.mkdir(parents=True, exist_ok=True)

        # 始终创建主工作空间的软链接（不因 split_workspace_for_exp 而跳过）
        session_config = self.config.session_config
        if hasattr(session_config, "symlinks") and session_config.symlinks:
            self._create_symlinks(workspace, session_config.symlinks)
            self.logger.info("主工作空间软链接已创建")
        else:
            self.logger.debug("无 symlinks 配置，跳过软链接创建")

        self._is_ready = True
        self.logger.info("ML Master 2 local environment setup complete")
