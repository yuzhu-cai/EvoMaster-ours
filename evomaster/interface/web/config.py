"""Web interface configuration model and loader.

Load web interface config from a YAML file, reusing EvoMaster's _substitute_env pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

from evomaster.config import _substitute_env


class WebAppConfig(BaseModel):
    """Web chat interface configuration."""

    host: str = Field(default="127.0.0.1", description="Host to bind the web server to")
    port: int = Field(default=5001, description="Port to bind the web server to")
    default_agent: str = Field(
        default="evoclaw",
        description="Default playground agent name",
    )
    default_config_path: Optional[str] = Field(
        default=None,
        description="Default config file path (relative to project_root). "
        "If unset, uses configs/{agent}/config.yaml",
    )
    max_concurrent_tasks: int = Field(
        default=4,
        description="Maximum number of concurrent tasks",
    )
    task_timeout: int = Field(
        default=600,
        description="Single task timeout in seconds",
    )
    max_sessions: int = Field(
        default=100,
        description="Maximum number of active sessions",
    )
    available_agents: dict[str, str] = Field(
        default_factory=dict,
        description="Map of agent name to description",
    )
    container_pool: Optional[dict] = Field(
        default=None,
        description="Container pool configuration",
    )

    class Config:
        extra = "allow"


def load_web_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> WebAppConfig:
    """Load web interface config from a YAML file.

    Args:
        config_path: Path to the YAML config file.
        project_root: Project root directory, used to locate .env files.

    Returns:
        A validated WebAppConfig instance.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Web config not found: {config_path}")

    # Load .env if python-dotenv is available
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

    # Extract the "web" section (fall back to the entire dict)
    web_section = raw.get("web", raw)

    return WebAppConfig(**web_section)
