"""记忆管理器 — 业务逻辑层

负责自动提取（auto-capture）、自动召回（auto-recall）和工具 API。
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from .store import MemoryStore
from .types import MemoryCategory, MemoryEntry

if TYPE_CHECKING:
    from evomaster.utils.llm import BaseLLM

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Auto-capture 规则
# ------------------------------------------------------------------

CAPTURE_PATTERNS: list[tuple[re.Pattern, MemoryCategory]] = [
    # 中文 — 明确记忆指令
    (re.compile(r"记住[：:\s].+", re.S), "fact"),
    (re.compile(r"请?记住.+"), "fact"),
    # 中文 — 偏好
    (re.compile(r"我(喜欢|偏好|习惯|倾向于|一般用|经常用|更喜欢)"), "preference"),
    (re.compile(r"以后(都|总是|一直|请|帮我)"), "preference"),
    (re.compile(r"(每次|下次|将来)(都|请|帮我)"), "preference"),
    (re.compile(r"不要(再|总是|每次)"), "preference"),
    # 中文 — 事实
    (re.compile(r"我的.{1,15}(是|叫|为|用的是)"), "fact"),
    (re.compile(r"我(是|叫|在|住在|来自|从事|负责|管理)"), "fact"),
    # 中文 — 决策
    (re.compile(r"我(决定|选择|确定|打算)(了|要)?"), "decision"),
    # 英文
    (re.compile(r"(?i)\bremember\b.+"), "fact"),
    (re.compile(r"(?i)\bi (prefer|like|love|hate|always|never|usually)\b"), "preference"),
    (re.compile(r"(?i)\bmy .{1,30} is\b"), "fact"),
    (re.compile(r"(?i)\bi('m| am| work| live)\b"), "fact"),
    (re.compile(r"(?i)\bi (decided|chose|will use|want to use)\b"), "decision"),
]

# 过短或过长的消息不提取
_MIN_CAPTURE_LEN = 6
_MAX_CAPTURE_LEN = 500

# LLM 提取 prompt
_EXTRACT_PROMPT = """\
从以下对话内容中提取值得长期记忆的关键信息。

只提取以下类型的信息：
- preference: 用户的偏好、喜好、习惯
- fact: 关于用户的事实信息（姓名、职业、项目、技术栈等）
- decision: 用户做出的重要决策
- entity: 重要的实体名称（项目名、公司名等）

如果没有值得记忆的内容，返回空数组。

对话内容：
{content}

以 JSON 数组格式返回，每条记忆包含 content、category、importance(0-1)：
```json
[{{"content": "...", "category": "preference", "importance": 0.7}}]
```"""


class MemoryManager:
    """记忆管理器

    Args:
        store: 底层存储实例
        llm: 可选的 LLM 实例，用于 capture_with_llm 模式
        config: memory 配置字典
    """

    def __init__(
        self,
        store: MemoryStore,
        llm: BaseLLM | None = None,
        config: dict[str, Any] | None = None,
    ):
        self._store = store
        self._llm = llm
        self._config = config or {}

    @property
    def store(self) -> MemoryStore:
        return self._store

    # ------------------------------------------------------------------
    # Auto-recall
    # ------------------------------------------------------------------

    def recall_for_context(
        self, user_id: str, query: str, limit: int = 5
    ) -> str:
        """根据用户消息搜索相关记忆，返回格式化的 context 文本块。

        如果该用户没有记忆，返回空字符串。
        """
        entries = self._store.search(user_id, query, limit=limit)
        if not entries:
            return ""

        lines = ["## 用户记忆", "", "以下是关于当前用户的历史记忆，仅供参考："]
        for e in entries:
            lines.append(f"- [{e.category_label}] {e.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Auto-capture (规则匹配)
    # ------------------------------------------------------------------

    def extract_from_message(self, user_id: str, message: str) -> list[str]:
        """从用户消息中提取值得记忆的内容（规则匹配）。

        返回新增的记忆内容列表。
        """
        message = message.strip()
        if len(message) < _MIN_CAPTURE_LEN or len(message) > _MAX_CAPTURE_LEN:
            return []

        saved: list[str] = []
        for pattern, category in CAPTURE_PATTERNS:
            if pattern.search(message):
                memory_id = self._store.add(
                    user_id=user_id,
                    content=message,
                    category=category,
                    importance=0.6,
                    source="auto",
                )
                if memory_id:
                    saved.append(message)
                    self._enforce_user_limit(user_id)
                break  # 一条消息只匹配一次

        return saved

    # ------------------------------------------------------------------
    # Auto-capture (LLM 提取)
    # ------------------------------------------------------------------

    def extract_from_summary(self, user_id: str, summary: str) -> list[str]:
        """从 compaction summary 中用 LLM 提取关键事实/偏好/决策。

        如果 LLM 不可用，退化到规则匹配。
        """
        if not self._llm:
            return self.extract_from_message(user_id, summary)

        try:
            from evomaster.utils.types import Dialog, UserMessage

            prompt = _EXTRACT_PROMPT.format(content=summary[:4000])
            dialog = Dialog(messages=[UserMessage(content=prompt)], tools=[])
            response = self._llm.query(dialog)
            raw = response.content or ""

            items = self._parse_json_array(raw)
            saved: list[str] = []
            for item in items[:5]:  # 最多提取 5 条
                content = item.get("content", "").strip()
                category = item.get("category", "other")
                importance = float(item.get("importance", 0.5))
                if not content or len(content) < _MIN_CAPTURE_LEN:
                    continue
                if category not in ("preference", "fact", "decision", "entity", "other"):
                    category = "other"
                memory_id = self._store.add(
                    user_id=user_id,
                    content=content,
                    category=category,
                    importance=min(max(importance, 0.0), 1.0),
                    source="compaction",
                )
                if memory_id:
                    saved.append(content)

            self._enforce_user_limit(user_id)
            return saved

        except Exception:
            logger.exception("LLM memory extraction failed")
            return []

    # ------------------------------------------------------------------
    # Tool API
    # ------------------------------------------------------------------

    def search(self, user_id: str, query: str, limit: int = 5) -> list[MemoryEntry]:
        """搜索记忆（供 memory_search tool 调用）"""
        return self._store.search(user_id, query, limit=limit)

    def save(
        self, user_id: str, content: str, category: str = "other"
    ) -> str | None:
        """保存记忆（供 memory_save tool 调用）"""
        memory_id = self._store.add(
            user_id=user_id,
            content=content,
            category=category,
            importance=0.8,  # 用户/agent 显式保存的记忆重要性更高
            source="manual",
        )
        self._enforce_user_limit(user_id)
        return memory_id

    def forget(
        self, user_id: str, query: str | None = None, memory_id: str | None = None
    ) -> str:
        """删除记忆（供 memory_forget tool 调用）"""
        if memory_id:
            ok = self._store.delete(memory_id)
            return f"已删除记忆 {memory_id}" if ok else f"未找到记忆 {memory_id}"
        if query:
            count = self._store.delete_by_query(user_id, query)
            return f"已删除 {count} 条匹配的记忆" if count > 0 else "未找到匹配的记忆"
        return "请提供 query 或 memory_id"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enforce_user_limit(self, user_id: str) -> None:
        max_count = self._config.get("max_memories_per_user", 500)
        deleted = self._store.enforce_limit(user_id, max_count)
        if deleted > 0:
            logger.info(
                "Enforced memory limit for user %s: deleted %d oldest",
                user_id, deleted,
            )

    @staticmethod
    def _parse_json_array(text: str) -> list[dict]:
        """从 LLM 输出中解析 JSON 数组（容错 markdown code block）。"""
        # 去除 markdown code block
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # 去掉首尾 ``` 行
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # 尝试提取第一个 JSON 数组
        match = re.search(r"\[.*\]", text, re.S)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        return []
