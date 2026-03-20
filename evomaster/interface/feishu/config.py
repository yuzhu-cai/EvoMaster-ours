"""Feishu Bot 配置模型与加载

从 YAML 文件加载飞书 Bot 配置，复用 EvoMaster 的 _substitute_env 模式。
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

# 复用 evomaster.config 中的环境变量替换
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
    """飞书 Bot 配置"""

    app_id: str = Field(description="飞书应用 App ID")
    app_secret: str = Field(description="飞书应用 App Secret")
    domain: str = Field(
        default="https://open.feishu.cn",
        description="飞书 API 域名",
    )
    connection_mode: str = Field(
        default="websocket",
        description="连接模式: websocket 或 webhook",
    )
    default_agent: str = Field(
        default="magiclaw",
        description="默认使用的 playground agent 名称",
    )
    default_config_path: Optional[str] = Field(
        default=None,
        description="默认配置文件路径（相对于 project_root），不设置则使用 configs/{agent}/config.yaml",
    )
    max_concurrent_tasks: int = Field(
        default=4,
        description="最大并发任务数",
    )
    task_timeout: int = Field(
        default=600,
        description="单个任务超时时间（秒）",
    )
    allow_from: List[str] = Field(
        default_factory=list,
        description="允许的用户 open_id 列表，空列表表示允许所有人",
    )
    doc_folder_token: Optional[str] = Field(
        default=None,
        description="飞书文件夹 token，用于存放轨迹文档。为空则创建在应用根目录",
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
    """加载飞书 Bot 配置

    Args:
        config_path: 配置文件路径
        project_root: 项目根目录，用于搜索 .env 文件

    Returns:
        FeishuBotConfig 实例
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Feishu config not found: {config_path}")

    # 加载 .env
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

    # 提取 feishu 段
    feishu_section = raw.get("feishu", raw)

    return FeishuBotConfig(**feishu_section)
