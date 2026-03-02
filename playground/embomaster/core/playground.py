"""EmboMaster playground.

Minimal scaffold:
- register custom debug_test tool
- create coding agent
- execute optional K8S experiment runner in Exp layer
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evomaster.core import BasePlayground, register_playground

from .exp import EmboMasterExp
from .services import K8SExperimentRunner
from .tools import DebugTestTool, VideoDescriptorTool


@register_playground("embomaster")
class EmboMasterPlayground(BasePlayground):
    """EmboMaster custom playground."""

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "embomaster"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.coding_agent = None
        self.feedback_agent = None
        self.k8s_runner = None

    def setup(self) -> None:
        self.logger.info("Setting up EmboMaster playground...")

        llm_config_dict = self._setup_llm_config()
        self._llm_config_dict = llm_config_dict
        self._setup_session()
        self._setup_tools()

        self._register_custom_tools()
        self._apply_tool_policy()
        self._create_agents(llm_config_dict)
        self._create_services()
        self._wire_tool_services()

        self.logger.info("EmboMaster playground setup complete")

    def _register_custom_tools(self) -> None:
        debug_cfg = self._config_section("debug_test")
        if debug_cfg.get("enabled", True):
            tool = DebugTestTool(
                default_timeout=int(debug_cfg.get("default_timeout", 120)),
                default_env_init=str(debug_cfg.get("default_env_init", "")),
                k8s_runner=None,
                use_k8s_debug_pod=bool(debug_cfg.get("use_k8s_debug_pod", False)),
                k8s_fallback_to_local=bool(debug_cfg.get("k8s_fallback_to_local", True)),
            )
            self.tools.register(tool)
            self.logger.info("Custom tool registered: %s", tool.name)

        video_cfg = self._config_section("video_descriptor")
        if video_cfg.get("enabled", True):
            base_llm_cfg = self._llm_config_dict if isinstance(self._llm_config_dict, dict) else {}
            api_key = str(video_cfg.get("api_key") or base_llm_cfg.get("api_key") or "").strip()
            base_url = (
                video_cfg.get("base_url")
                or base_llm_cfg.get("base_url")
                or "http://127.0.0.1:30030/v1"
            )
            tool = VideoDescriptorTool(
                api_key=api_key,
                base_url=str(base_url) if base_url else None,
                model=str(
                    video_cfg.get("model", "Qwen/Qwen3-VL-235B-A22B-Instruct")
                ).strip(),
                temperature=float(video_cfg.get("temperature", 0.2)),
                max_tokens=int(video_cfg.get("max_tokens", 1024)),
                timeout=int(video_cfg.get("timeout", 120)),
            )
            self.tools.register(tool)
            self.logger.info("Custom tool registered: %s", tool.name)

    def _apply_tool_policy(self) -> None:
        """Apply tool allow/deny policy from config.

        Config section:
          tool_policy:
            allowed: ["finish", "str_replace_editor", "debug_test", "video-descriptor"]   # optional
            disabled: ["execute_bash"]                                # optional
            disable_execute_bash: true                                # optional shortcut
        """
        policy = self._config_section("tool_policy")
        if not policy or self.tools is None:
            return

        allowed_raw = policy.get("allowed")
        disabled_raw = policy.get("disabled", [])
        disable_execute_bash = bool(policy.get("disable_execute_bash", False))

        allowed: set[str] | None = None
        if isinstance(allowed_raw, list):
            allowed = {str(name).strip() for name in allowed_raw if str(name).strip()}
            # Keep finish available to avoid dead-end agent loops.
            allowed.add("finish")

        disabled = {str(name).strip() for name in disabled_raw if str(name).strip()}
        if disable_execute_bash:
            disabled.add("execute_bash")
        # Protect finish tool.
        disabled.discard("finish")

        if allowed is not None:
            for name in list(self.tools.get_tool_names()):
                if name not in allowed:
                    self.tools.unregister(name)

        for name in disabled:
            self.tools.unregister(name)

        enabled_names = sorted(self.tools.get_tool_names())
        self.logger.info("Tool policy applied. Enabled tools: %s", ", ".join(enabled_names))

    def _create_agents(self, llm_config_dict: dict[str, Any]) -> None:
        agents_config = getattr(self.config, "agents", {})
        if not agents_config or "coding" not in agents_config:
            raise ValueError("Missing `agents.coding` configuration.")

        coding_cfg = agents_config["coding"]
        self.coding_agent = self._create_agent(
            name="coding",
            agent_config=coding_cfg,
            enable_tools=bool(coding_cfg.get("enable_tools", True)),
            llm_config_dict=llm_config_dict,
            skill_registry=None,
        )
        # BasePlayground defaults still use self.agent in some paths.
        self.agent = self.coding_agent
        self.logger.info("Coding Agent created")

        feedback_cfg = agents_config.get("feedback")
        if feedback_cfg:
            self.feedback_agent = self._create_agent(
                name="feedback",
                agent_config=feedback_cfg,
                enable_tools=bool(feedback_cfg.get("enable_tools", False)),
                llm_config_dict=llm_config_dict,
                skill_registry=None,
            )
            self.logger.info("Feedback Agent created")

    def _create_services(self) -> None:
        k8s_cfg = self._config_section("k8s_runner")
        debug_cfg = self._config_section("debug_test")
        need_debug_runner = bool(debug_cfg.get("use_k8s_debug_pod", False))
        if k8s_cfg.get("enabled", False) or need_debug_runner:
            self.k8s_runner = K8SExperimentRunner(self.session, k8s_cfg)
            self.logger.info(
                "K8SExperimentRunner enabled (namespace=%s)",
                k8s_cfg.get("namespace", "default"),
            )

    def _wire_tool_services(self) -> None:
        if self.tools is None:
            return
        tool = self.tools.get_tool("debug_test")
        if not isinstance(tool, DebugTestTool):
            return

        debug_cfg = self._config_section("debug_test")
        tool.configure_k8s_debug(
            k8s_runner=self.k8s_runner,
            use_k8s_debug_pod=bool(debug_cfg.get("use_k8s_debug_pod", False)),
            k8s_fallback_to_local=bool(debug_cfg.get("k8s_fallback_to_local", True)),
        )

    def _create_exp(self):
        exp = EmboMasterExp(
            coding_agent=self.coding_agent,
            feedback_agent=self.feedback_agent,
            config=self.config,
            k8s_runner=self.k8s_runner,
        )
        if self.run_dir:
            exp.set_run_dir(self.run_dir)
        return exp

    def _config_section(self, key: str) -> dict[str, Any]:
        if hasattr(self.config, "model_dump"):
            cfg_dict = self.config.model_dump()
        else:
            cfg_dict = dict(self.config)
        section = cfg_dict.get(key, {})
        if not isinstance(section, dict):
            return {}
        return section
