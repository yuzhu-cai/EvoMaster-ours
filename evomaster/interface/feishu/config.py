"""Feishu Bot configuration model and loading

Load Feishu Bot configuration from a YAML file, reusing EvoMaster's _substitute_env pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

# Reuse the environment variable substitution from evomaster.config
from evomaster.config import _substitute_env
from evomaster.env.container_pool import ContainerPoolConfig


class SchedulerConfig(BaseModel):
    """定时任务调度器配置"""

    enabled: bool = Field(default=True, description="是否启用定时任务")
    db_path: str = Field(
        default="data/scheduler/schedules.db",
        description="SQLite 数据库路径（相对于 project_root）",
    )
    max_jobs_per_chat: int = Field(default=20, description="每个会话最大活跃任务数")
    max_jobs_total: int = Field(default=200, description="全局最大活跃任务数")
    max_concurrent_runs: int = Field(default=2, description="最大并发执行数")
    default_timezone: str = Field(
        default="Asia/Shanghai", description="默认时区（IANA）"
    )
    max_poll_interval: float = Field(
        default=60.0, description="最大轮询间隔（秒）"
    )
    max_retries: int = Field(default=3, description="任务失败最大重试次数")

    class Config:
        extra = "allow"


class FeishuBotConfig(BaseModel):
    """Feishu Bot configuration."""

    app_id: str = Field(description="Feishu application App ID")
    app_secret: str = Field(description="Feishu application App Secret")
    domain: str = Field(
        default="https://open.feishu.cn",
        description="Feishu API domain",
    )
    connection_mode: str = Field(
        default="websocket",
        description="Connection mode: websocket or webhook",
    )
    default_agent: str = Field(
        default="magiclaw",
        description="Default playground agent name to use",
    )
    default_config_path: Optional[str] = Field(
        default=None,
        description="Default config file path (relative to project_root); if not set, uses configs/{agent}/config.yaml",
    )
    max_concurrent_tasks: int = Field(
        default=4,
        description="Maximum number of concurrent tasks",
    )
    task_timeout: int = Field(
        default=600,
        description="Timeout for a single task in seconds",
    )
    allow_from: List[str] = Field(
        default_factory=list,
        description="List of allowed user open_ids; empty list means all users are allowed",
    )
    doc_folder_token: Optional[str] = Field(
        default=None,
        description="Feishu folder token for storing trajectory documents; empty means root directory of the application",
    )
    max_sessions: int = Field(
        default=100,
        description="最大并发会话数",
    )
    available_agents: Dict[str, str] = Field(
        default_factory=dict,
        description="可用子智能体白名单，格式为 {agent_name: description}，空字典表示不展示任何内置智能体",
    )
    container_pool: Optional[ContainerPoolConfig] = Field(
        default=None,
        description="Docker 容器池配置，None 或 enabled=False 表示不使用容器池",
    )
    scheduler: Optional[SchedulerConfig] = Field(
        default_factory=SchedulerConfig,
        description="定时任务调度器配置",
    )

    class Config:
        extra = "allow"


def load_feishu_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> FeishuBotConfig:
    """Load Feishu Bot configuration.

    Args:
        config_path: Path to the configuration file.
        project_root: Project root directory, used for searching the .env file.

    Returns:
        A FeishuBotConfig instance.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Feishu config not found: {config_path}")

    # Load .env
    if load_dotenv is not None:
        if project_root:
            env_file = Path(project_root) / ".env"
            if env_file.exists():
                load_dotenv(env_file)
            else:
                load_dotenv()
        else:
            load_dotenv()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw = _substitute_env(raw)

    # Extract the feishu section
    feishu_section = raw.get("feishu", raw)

    return FeishuBotConfig(**feishu_section)
