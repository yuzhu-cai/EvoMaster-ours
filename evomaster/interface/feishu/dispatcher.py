"""任务调度器

将飞书消息分发到线程池，通过 ChatSessionManager 实现多轮对话上下文延续。
"""

from __future__ import annotations

import importlib
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_playgrounds_imported = False


def _ensure_playgrounds_imported(project_root: Path) -> None:
    """确保所有 playground 模块已导入（触发 @register_playground 装饰器）

    复用 run.py:auto_import_playgrounds() 的逻辑。
    """
    global _playgrounds_imported
    if _playgrounds_imported:
        return

    # 确保 project_root 在 sys.path 中
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    playground_dir = project_root / "playground"
    if not playground_dir.exists():
        logger.warning("Playground directory not found: %s", playground_dir)
        _playgrounds_imported = True
        return

    imported_count = 0
    for agent_dir in playground_dir.iterdir():
        if not agent_dir.is_dir() or agent_dir.name.startswith("_"):
            continue

        module_name = f"playground.{agent_dir.name}.core.playground"
        try:
            importlib.import_module(module_name)
            logger.info("Imported playground: %s", module_name)
            imported_count += 1
        except ImportError as e:
            logger.warning("Failed to import %s: %s", module_name, e)
        except Exception as e:
            logger.warning("Error importing %s: %s", module_name, e)

    logger.info("Auto-imported %d playground modules", imported_count)
    _playgrounds_imported = True


def _extract_final_answer(result: dict[str, Any]) -> str:
    """从执行结果中提取最终回答"""
    from evomaster.core import extract_agent_response

    trajectory = result.get("trajectory")
    if not trajectory:
        error = result.get("error", "")
        if error:
            return f"任务执行失败: {error}"
        return f"任务完成，状态: {result.get('status', 'unknown')}"

    answer = extract_agent_response(trajectory)
    if answer:
        return answer

    status = result.get("status", "unknown")
    steps = result.get("steps", 0)
    return f"任务完成（状态: {status}，步骤: {steps}），但未提取到文本回答。"


# 需要多轮会话的子任务 agent（使用独立会话 key）
_SESSION_SUBTASK_AGENTS = {"agent_builder"}

# 完成后需要确认按钮的子任务 agent（Phase 1 结束后显示「确认生成」按钮）
_CONFIRM_SUBTASK_AGENTS = {"agent_builder"}


class TaskDispatcher:
    """任务调度器：通过会话管理实现多轮对话上下文延续"""

    def __init__(
        self,
        project_root: Path,
        default_agent: str = "chat_agent",
        default_config_path: Optional[str] = None,
        max_workers: int = 4,
        task_timeout: int = 600,
        max_sessions: int = 100,
        on_result: Optional[Callable[[str, str, str], None]] = None,
        step_reporter_factory: Optional[Callable[[str, str | None], Any]] = None,
        feishu_app_id: Optional[str] = None,
        feishu_app_secret: Optional[str] = None,
        feishu_domain: str = "https://open.feishu.cn",
        feishu_doc_folder_token: Optional[str] = None,
    ):
        """
        Args:
            project_root: 项目根目录
            default_agent: 默认 agent 名称
            default_config_path: 默认配置文件路径（相对于 project_root）
            max_workers: 最大并发线程数
            task_timeout: 单轮任务超时（秒）
            max_sessions: 最大并发会话数
            on_result: 结果回调 (chat_id, message_id, result_text) -> None
            step_reporter_factory: 创建 FeishuStepReporter 的工厂函数
            feishu_app_id: 飞书 App ID（用于注入飞书特有工具）
            feishu_app_secret: 飞书 App Secret
            feishu_domain: 飞书 API 域名
            feishu_doc_folder_token: 飞书文件夹 token（用于文档写入工具）
        """
        from .session_manager import ChatSessionManager

        self._project_root = project_root
        self._default_agent = default_agent
        self._default_config_path = default_config_path
        self._task_timeout = task_timeout
        self._on_result = on_result
        self._step_reporter_factory = step_reporter_factory
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="feishu-task",
        )
        self._active_tasks: dict[str, Any] = {}
        self._session_manager = ChatSessionManager(max_sessions=max_sessions)

        # 存储飞书凭证（用于动态创建工具）
        self._feishu_app_id = feishu_app_id
        self._feishu_app_secret = feishu_app_secret
        self._feishu_domain = feishu_domain
        self._feishu_doc_folder_token = feishu_doc_folder_token

        # 飞书 Client（用于 patch 卡片等操作）
        self._feishu_client = None
        if feishu_app_id and feishu_app_secret:
            from .client import create_feishu_client
            self._feishu_client = create_feishu_client(
                app_id=feishu_app_id,
                app_secret=feishu_app_secret,
                domain=feishu_domain,
            )

        # 飞书特有工具（所有 agent 共用）
        self._feishu_tools: list = []
        if feishu_app_id and feishu_app_secret:
            from .doc_reader_tool import FeishuDocReadTool

            self._feishu_tools.append(
                FeishuDocReadTool(
                    app_id=feishu_app_id,
                    app_secret=feishu_app_secret,
                    domain=feishu_domain,
                )
            )

        # 确保 _generated 目录存在（agent_builder 生成的 agent 放在这里）
        (project_root / "configs" / "_generated").mkdir(parents=True, exist_ok=True)
        (project_root / "playground" / "_generated").mkdir(parents=True, exist_ok=True)

        # 预加载 playgrounds
        _ensure_playgrounds_imported(project_root)

    def dispatch(
        self,
        chat_id: str,
        message_id: str,
        task_text: str,
        agent_name: Optional[str] = None,
        sender_open_id: Optional[str] = None,
    ) -> None:
        """提交任务到线程池

        特殊命令：
        - /new: 清除当前会话上下文
        - /shutdown: 关闭 bot 进程
        """
        stripped = task_text.strip()

        # /new 命令：清除会话
        if stripped == "/new":
            self._session_manager.remove(chat_id)
            # 同时清除该 chat 的所有会话级子任务会话
            for agent_name in _SESSION_SUBTASK_AGENTS:
                self._session_manager.remove(f"{chat_id}:{agent_name}")
            if self._on_result:
                self._on_result(chat_id, message_id, "新会话已开始，上下文已清除。")
            return

        # /shutdown 命令：关闭 bot
        if stripped == "/shutdown":
            self._handle_shutdown(chat_id, message_id)
            return

        agent = agent_name or self._default_agent
        future = self._executor.submit(
            self._run_task_with_session,
            chat_id,
            message_id,
            task_text,
            agent,
            sender_open_id,
        )
        self._active_tasks[message_id] = future
        future.add_done_callback(lambda f: self._on_task_done(f, chat_id, message_id))

        # 超时守护线程
        def _timeout_guard():
            try:
                future.result(timeout=self._task_timeout)
            except TimeoutError:
                logger.warning(
                    "Task timed out: message_id=%s, timeout=%ds",
                    message_id,
                    self._task_timeout,
                )
                future.cancel()
            except Exception:
                pass

        threading.Thread(
            target=_timeout_guard,
            daemon=True,
            name=f"timeout-{message_id[:8]}",
        ).start()

    def _create_playground(self, agent_name: str):
        """创建 playground 实例（不调用 setup）。"""
        from evomaster.core import get_playground_class

        if agent_name == self._default_agent and self._default_config_path:
            config_path = self._project_root / self._default_config_path
        else:
            config_path = self._project_root / "configs" / agent_name / "config.yaml"
            # Fallback: 检查 _generated 目录（agent_builder 生成的 agent 放在这里）
            if not config_path.exists():
                config_path = self._project_root / "configs" / "_generated" / agent_name / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        playground = get_playground_class(agent_name, config_path=config_path)

        # 创建 run 目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self._project_root / "runs" / f"feishu_{agent_name}_{timestamp}"
        task_id = f"feishu_{agent_name}"
        playground.set_run_dir(run_dir, task_id=task_id)

        return playground

    def _run_task_with_session(
        self,
        chat_id: str,
        message_id: str,
        task_text: str,
        agent_name: str,
        sender_open_id: Optional[str] = None,
    ) -> str:
        """在后台线程中执行任务，复用会话上下文。

        如果 agent_name 与默认 agent 不同，采用子任务模式：
        独立运行指定 agent，结果注入 chat_agent 上下文。
        """
        from evomaster.utils.types import TaskInstance

        # 始终用默认 agent 创建/获取 session
        session = self._session_manager.get_or_create(
            chat_id,
            playground_factory=lambda: self._create_playground(self._default_agent),
        )

        # 同一 chat 串行处理
        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1

            # 创建实时进度报告器
            reporter = None
            on_step = None
            if self._step_reporter_factory:
                try:
                    reporter = self._step_reporter_factory(
                        chat_id, message_id, sender_open_id
                    )
                    reporter.send_initial_card(task_text)
                    on_step = reporter.on_step
                except Exception:
                    logger.exception("Failed to create step reporter")

            try:
                # 子任务模式：/agent 指定了非默认 agent
                if agent_name != self._default_agent:
                    # 会话级子任务：支持多轮对话（如 agent_builder）
                    if agent_name in _SESSION_SUBTASK_AGENTS:
                        answer = self._run_session_subtask(
                            chat_id, agent_name, task_text, on_step, sender_open_id
                        )
                    else:
                        answer = self._run_subtask(agent_name, task_text, on_step)

                    # 将结果注入 chat_agent 的 dialog 作为上下文
                    if session.initialized and session.agent:
                        summary = (
                            f"[子任务结果 - {agent_name}]\n"
                            f"用户请求: {task_text}\n"
                            f"结果: {answer}"
                        )
                        session.agent.add_user_message(summary)

                    if reporter:
                        try:
                            # 确认类 agent：finalize 时添加确认/取消按钮
                            if agent_name in _CONFIRM_SUBTASK_AGENTS:
                                session_key = f"{chat_id}:{agent_name}"
                                # 截断 answer 嵌入按钮 value，回调时用于保留原始卡片内容
                                _answer_for_button = answer[:2000] if answer else ""
                                actions = [
                                    {
                                        "text": "✅ 确认生成",
                                        "type": "primary",
                                        "value": {
                                            "action": "confirm_agent_build",
                                            "session_key": session_key,
                                            "agent_name": agent_name,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                    {
                                        "text": "❌ 取消",
                                        "type": "danger",
                                        "value": {
                                            "action": "cancel_agent_build",
                                            "session_key": session_key,
                                            "agent_name": agent_name,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                ]
                                reporter.finalize("completed", answer, actions=actions)
                            else:
                                reporter.finalize("completed", answer)
                            return None  # 卡片已包含回答
                        except Exception:
                            logger.exception("Failed to finalize step reporter")

                    return answer

                # 正常 chat_agent 流程
                if not session.initialized:
                    # 首次消息：完整 setup + agent.run()
                    logger.info(
                        "First message in session chat_id=%s, running setup",
                        chat_id,
                    )
                    session.playground.setup()
                    session.playground._setup_trajectory_file()
                    session.agent = session.playground.agent

                    # 注入飞书特有工具
                    self._inject_feishu_tools(session.playground)

                    task = TaskInstance(
                        task_id=f"feishu_{message_id}",
                        task_type="chat",
                        description=task_text,
                    )
                    trajectory = session.agent.run(task, on_step=on_step)
                    session.initialized = True
                else:
                    # 后续消息：continue_run()
                    logger.info(
                        "Continuing session chat_id=%s (message #%d)",
                        chat_id,
                        session.message_count,
                    )
                    trajectory = session.agent.continue_run(
                        task_text, on_step=on_step
                    )

                answer = _extract_final_answer(
                    {"trajectory": trajectory, "status": trajectory.status}
                )
                logger.info(
                    "Task completed in session chat_id=%s, status=%s",
                    chat_id,
                    trajectory.status,
                )

                if reporter:
                    try:
                        reporter.finalize("completed", answer)
                        return None  # 卡片已包含回答，不需要额外消息
                    except Exception:
                        logger.exception("Failed to finalize step reporter")

                return answer

            except Exception as e:
                logger.exception("Task failed in session chat_id=%s", chat_id)
                if reporter:
                    try:
                        reporter.finalize("failed")
                    except Exception:
                        logger.exception(
                            "Failed to finalize step reporter on error"
                        )
                return f"任务执行出错: {e}"

    def _run_subtask(
        self, agent_name: str, task_text: str, on_step: Optional[Callable] = None
    ) -> str:
        """独立运行指定 agent 的子任务，不复用会话上下文。"""
        from evomaster.utils.types import TaskInstance

        logger.info("Running subtask with agent=%s", agent_name)
        playground = self._create_playground(agent_name)
        try:
            playground.setup()
            playground._setup_trajectory_file()
            self._inject_feishu_tools(playground)
            agent = playground.agent
            task = TaskInstance(
                task_id=f"subtask_{agent_name}",
                task_type="subtask",
                description=task_text,
            )
            trajectory = agent.run(task, on_step=on_step)
            return _extract_final_answer(
                {"trajectory": trajectory, "status": trajectory.status}
            )
        except Exception as e:
            logger.exception("Subtask failed: agent=%s", agent_name)
            return f"子任务执行出错: {e}"
        finally:
            try:
                playground.cleanup()
            except Exception:
                logger.exception("Subtask cleanup failed")

    def _inject_feishu_tools(self, playground) -> None:
        """将飞书特有工具注入 playground 的所有 agent。"""
        if not self._feishu_tools:
            return
        for agent in playground.agents.values():
            for tool in self._feishu_tools:
                agent.tools.register(tool)

    def _run_session_subtask(
        self,
        chat_id: str,
        agent_name: str,
        task_text: str,
        on_step: Optional[Callable] = None,
        sender_open_id: Optional[str] = None,
    ) -> str:
        """运行会话级子任务：支持多轮对话的独立 agent 会话。

        使用 {chat_id}:{agent_name} 作为会话 key，支持 continue_run()。
        """
        from evomaster.utils.types import TaskInstance

        session_key = f"{chat_id}:{agent_name}"
        session = self._session_manager.get_or_create(
            session_key,
            playground_factory=lambda: self._create_playground(agent_name),
        )

        # 会话级子任务也串行处理
        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1

            try:
                if not session.initialized:
                    logger.info(
                        "First message in session subtask key=%s, agent=%s",
                        session_key, agent_name,
                    )
                    session.playground.setup()
                    session.playground._setup_trajectory_file()
                    session.agent = session.playground.agent

                    self._inject_feishu_tools(session.playground)
                    self._inject_doc_write_tool(session.playground, sender_open_id)

                    task = TaskInstance(
                        task_id=f"session_subtask_{agent_name}",
                        task_type="session_subtask",
                        description=task_text,
                    )
                    trajectory = session.agent.run(task, on_step=on_step)
                    session.initialized = True
                else:
                    logger.info(
                        "Continuing session subtask key=%s (message #%d)",
                        session_key, session.message_count,
                    )
                    trajectory = session.agent.continue_run(
                        task_text, on_step=on_step
                    )

                return _extract_final_answer(
                    {"trajectory": trajectory, "status": trajectory.status}
                )

            except Exception as e:
                logger.exception(
                    "Session subtask failed: key=%s, agent=%s", session_key, agent_name
                )
                return f"会话子任务执行出错: {e}"

    def _inject_doc_write_tool(self, playground, sender_open_id: str | None) -> None:
        """将飞书文档写入工具注入 playground 的所有 agent。"""
        if not self._feishu_app_id or not self._feishu_app_secret:
            return

        from .client import create_feishu_client
        from .document_writer import FeishuDocumentWriter
        from playground.agent_builder.tools.feishu_doc_write import FeishuDocWriteTool

        client = create_feishu_client(
            app_id=self._feishu_app_id,
            app_secret=self._feishu_app_secret,
            domain=self._feishu_domain,
        )
        writer = FeishuDocumentWriter(
            client,
            folder_token=self._feishu_doc_folder_token,
            domain=self._feishu_domain,
        )
        tool = FeishuDocWriteTool(
            document_writer=writer,
            sender_open_id=sender_open_id,
        )

        for agent in playground.agents.values():
            agent.tools.register(tool)

    def dispatch_card_action(
        self,
        chat_id: str,
        session_key: str,
        agent_name: str,
        task_text: str,
        sender_open_id: str | None = None,
        card_message_id: str | None = None,
        original_answer: str = "",
    ) -> None:
        """处理卡片按钮回调，触发会话级子任务的 continue_run。

        Args:
            chat_id: 聊天 ID（用于发送结果）
            session_key: 会话 key（格式 {chat_id}:{agent_name}）
            agent_name: agent 名称
            task_text: 发送给 agent 的文本（如 "确认"）
            sender_open_id: 操作者 open_id
            card_message_id: 触发按钮的卡片消息 ID
            original_answer: Phase 1 的原始回答内容（用于更新卡片时保留）
        """
        message_id = card_message_id or f"card_action_{session_key}"
        future = self._executor.submit(
            self._continue_session_subtask,
            chat_id,
            session_key,
            agent_name,
            task_text,
            sender_open_id,
            card_message_id,
            original_answer,
        )
        self._active_tasks[message_id] = future
        future.add_done_callback(
            lambda f: self._on_task_done(f, chat_id, message_id)
        )

    def _continue_session_subtask(
        self,
        chat_id: str,
        session_key: str,
        agent_name: str,
        task_text: str,
        sender_open_id: str | None = None,
        card_message_id: str | None = None,
        original_answer: str = "",
    ) -> str | None:
        """继续已有的会话级子任务（由卡片按钮触发）。"""
        session = self._session_manager.get(session_key)
        if session is None or not session.initialized:
            logger.warning(
                "No active session for card action: key=%s", session_key
            )
            return f"会话已过期或不存在，请重新发起 /agent {agent_name} 命令。"

        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1

            # 创建进度报告器
            reporter = None
            on_step = None
            if self._step_reporter_factory:
                try:
                    reporter = self._step_reporter_factory(
                        chat_id, card_message_id, sender_open_id
                    )
                    # agent_builder: 延迟发送卡片，等 TODO 解析后一次性发送
                    if agent_name not in _CONFIRM_SUBTASK_AGENTS:
                        reporter.send_initial_card(f"[{agent_name}] {task_text}")
                    on_step = reporter.on_step
                except Exception:
                    logger.exception("Failed to create step reporter for card action")

            try:
                logger.info(
                    "Continuing session subtask via card action: key=%s (message #%d)",
                    session_key, session.message_count,
                )

                # agent_builder 双 agent 模式：Phase 2 使用 builder_agent（全新 run）
                if (
                    agent_name == "agent_builder"
                    and hasattr(session.playground, "agents")
                    and hasattr(session.playground.agents, "builder_agent")
                ):
                    from evomaster.utils.types import TaskInstance

                    # 解析 planner 输出中的 TODO 清单，设置到 reporter
                    todo_items = self._parse_plan_todos(original_answer)
                    if reporter:
                        if todo_items:
                            reporter.set_todo_items(todo_items)
                        reporter.send_initial_card(
                            f"[{agent_name}] 正在生成 Agent 文件..."
                        )
                        on_step = reporter.on_step

                    builder_agent = session.playground.agents.builder_agent
                    # 注入飞书工具到 builder agent（setup 时已注入，但确保可用）
                    if self._feishu_tools:
                        for tool in self._feishu_tools:
                            builder_agent.tools.register(tool)
                    # 构造 handoff 任务：将 planner 的方案摘要传递给 builder
                    plan_task = TaskInstance(
                        task_id=f"builder_{agent_name}",
                        task_type="builder",
                        description=(
                            "请根据以下设计方案生成 Agent 文件。\n\n"
                            f"## 方案摘要\n{original_answer}\n\n"
                            "请使用 feishu_doc_read 工具读取飞书文档获取完整方案，然后生成所有文件。"
                        ),
                    )
                    trajectory = builder_agent.run(plan_task, on_step=on_step)
                else:
                    trajectory = session.agent.continue_run(
                        task_text, on_step=on_step
                    )
                answer = _extract_final_answer(
                    {"trajectory": trajectory, "status": trajectory.status}
                )

                # 将结果注入 chat_agent 上下文
                chat_session = self._session_manager.get(chat_id)
                if chat_session and chat_session.initialized and chat_session.agent:
                    summary = (
                        f"[子任务结果 - {agent_name} Phase 2]\n"
                        f"结果: {answer}"
                    )
                    chat_session.agent.add_user_message(summary)

                if reporter:
                    try:
                        reporter.finalize("completed", answer)
                    except Exception:
                        logger.exception("Failed to finalize step reporter")

                # 更新 Phase 1 卡片：从 "生成中" 变为 "完成"，保留原始计划内容
                phase1_content = original_answer + "\n\n---\n> ✅ Agent 已成功创建。详情请查看下方回复。" if original_answer else "Agent 已成功创建。\n\n详情请查看下方回复。"
                self._patch_phase1_card(
                    card_message_id, "✅ Agent 创建完成",
                    phase1_content, "green",
                )

                return None

            except Exception as e:
                logger.exception(
                    "Card action subtask failed: key=%s", session_key
                )
                if reporter:
                    try:
                        reporter.finalize("failed")
                    except Exception:
                        logger.exception("Failed to finalize reporter on error")

                # 更新 Phase 1 卡片：显示失败状态，保留原始计划内容
                phase1_content = original_answer + f"\n\n---\n> ❌ Agent 创建过程中出错：{str(e)[:500]}" if original_answer else f"Agent 创建过程中出错。\n\n{str(e)[:500]}"
                self._patch_phase1_card(
                    card_message_id, "❌ Agent 创建失败",
                    phase1_content, "red",
                )

                return f"会话子任务执行出错: {e}"

    def _patch_phase1_card(
        self,
        card_message_id: str | None,
        title: str,
        content: str,
        header_template: str,
    ) -> None:
        """更新 Phase 1 卡片状态（Phase 2 完成/失败后调用）。"""
        if not card_message_id or not self._feishu_client:
            return
        try:
            from .sender import patch_card_message
            patch_card_message(
                self._feishu_client,
                card_message_id,
                title=title,
                content=content,
                header_template=header_template,
            )
        except Exception:
            logger.exception("Failed to update Phase 1 card: %s", card_message_id)

    @staticmethod
    def _parse_plan_todos(plan_text: str) -> list[str]:
        """从 planner 输出中解析 TODO 列表。

        格式::

            ---PLAN_TODO---
            - [ ] 创建目录结构
            - [ ] 创建 system_prompt.txt
            ---END_TODO---
        """
        todos: list[str] = []
        in_todo = False
        for line in plan_text.split("\n"):
            stripped = line.strip()
            if "---PLAN_TODO---" in stripped:
                in_todo = True
                continue
            if "---END_TODO---" in stripped:
                break
            if in_todo and stripped.startswith("- [ ]"):
                label = stripped[5:].strip()
                if label:
                    todos.append(label)
        return todos

    def _on_task_done(self, future, chat_id: str, message_id: str) -> None:
        """任务完成回调"""
        self._active_tasks.pop(message_id, None)

        try:
            result_text = future.result(timeout=0)
        except TimeoutError:
            result_text = f"任务超时（超过 {self._task_timeout} 秒）"
        except Exception as e:
            result_text = f"任务执行异常: {e}"

        # None 表示 reporter 卡片已包含回答，无需再发消息
        if result_text is None:
            return

        if self._on_result:
            try:
                self._on_result(chat_id, message_id, result_text)
            except Exception:
                logger.exception("Error in on_result callback")

    def _handle_shutdown(self, chat_id: str, message_id: str) -> None:
        """处理 /shutdown 命令：关闭 bot 进程。"""
        logger.info("Shutdown requested from chat_id=%s", chat_id)
        if self._on_result:
            try:
                self._on_result(chat_id, message_id, "Bot 正在关闭...")
            except Exception:
                logger.exception("Error sending shutdown message")

        # 在新线程中执行关闭，避免阻塞当前回调
        def _do_shutdown():
            time.sleep(1)  # 等待回复消息发送完成
            self.shutdown()
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=_do_shutdown, daemon=True).start()

    def shutdown(self, wait: bool = False) -> None:
        """关闭调度器和所有会话"""
        logger.info("Shutting down task dispatcher...")
        self._session_manager.shutdown()
        self._executor.shutdown(wait=wait)
        logger.info("Task dispatcher shut down")
