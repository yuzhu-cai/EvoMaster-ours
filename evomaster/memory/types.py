"""记忆系统数据类型定义"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


MemoryCategory = Literal["preference", "fact", "decision", "entity", "other"]


@dataclass
class MemoryEntry:
    """一条记忆记录"""

    id: str
    user_id: str
    content: str
    category: MemoryCategory = "other"
    importance: float = 0.5
    source: str = "auto"  # auto / manual / compaction
    created_at: float = 0.0
    updated_at: float = 0.0
    access_count: int = 0
    score: float = 0.0  # 搜索时的匹配分数（不持久化）

    # 类别显示名（中文）
    _CATEGORY_LABELS: dict[str, str] = field(
        default_factory=lambda: {
            "preference": "偏好",
            "fact": "事实",
            "decision": "决策",
            "entity": "实体",
            "other": "其他",
        },
        init=False,
        repr=False,
    )

    @property
    def category_label(self) -> str:
        return self._CATEGORY_LABELS.get(self.category, self.category)
