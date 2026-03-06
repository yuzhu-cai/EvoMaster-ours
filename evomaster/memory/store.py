"""SQLite + FTS5 记忆存储层

单个 SQLite 文件存储所有用户记忆，通过 user_id 字段隔离。
FTS5 索引使用 jieba 分词预处理，支持中文搜索。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from difflib import SequenceMatcher
from pathlib import Path

import jieba

from .types import MemoryEntry

logger = logging.getLogger(__name__)

# 静默 jieba 的初始化日志
jieba.setLogLevel(logging.WARNING)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    content     TEXT NOT NULL,
    category    TEXT DEFAULT 'other',
    importance  REAL DEFAULT 0.5,
    source      TEXT DEFAULT 'auto',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    access_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(user_id, category);
"""

# 独立 FTS 表（不关联 content table，因为需要存 jieba 分词后的文本）
_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content);
"""

# 去重相似度阈值
_DEDUP_SIMILARITY = 0.90
# 时间衰减系数：每天降低 1% 的分数权重
_DECAY_RATE = 0.01


def _segment(text: str) -> str:
    """对文本做 jieba 分词，返回空格分隔的结果。

    例: "用户喜欢吃草莓" → "用户 喜欢 吃 草莓"
    """
    return " ".join(jieba.cut(text))


class MemoryStore:
    """SQLite + FTS5 记忆存储

    FTS5 索引存储 jieba 分词后的文本，搜索时也对 query 做分词。
    线程安全：通过 threading.Lock 保护所有数据库操作。
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = self._connect()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(_SCHEMA_SQL)

            # 迁移：如果存在旧的触发器驱动的 FTS 表，drop 并重建
            self._migrate_fts(cur)

            self._conn.commit()

    def _migrate_fts(self, cur: sqlite3.Cursor) -> None:
        """检测并迁移旧 FTS 表（从触发器同步迁移到 jieba 手动同步）。"""
        # 检查是否有旧触发器
        old_triggers = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN "
            "('memories_ai', 'memories_ad', 'memories_au')"
        ).fetchall()

        needs_rebuild = False

        if old_triggers:
            # 有旧触发器 → 需要迁移
            logger.info("Migrating FTS from trigger-based to jieba-segmented...")
            for trigger in old_triggers:
                cur.execute(f"DROP TRIGGER IF EXISTS {trigger['name']}")
            # Drop 旧 FTS 表并重建
            cur.execute("DROP TABLE IF EXISTS memories_fts")
            needs_rebuild = True

        # 确保 FTS 表存在
        try:
            cur.execute("SELECT * FROM memories_fts LIMIT 0")
        except sqlite3.OperationalError:
            needs_rebuild = True

        # 检查 FTS 内容是否已经用 jieba 分词（分词后的文本包含空格）
        if not needs_rebuild:
            sample = cur.execute(
                "SELECT content FROM memories_fts LIMIT 1"
            ).fetchone()
            if sample and sample["content"] and " " not in sample["content"]:
                logger.info("FTS index not jieba-segmented, rebuilding...")
                needs_rebuild = True

        if needs_rebuild:
            cur.execute("DROP TABLE IF EXISTS memories_fts")
            cur.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content)"
            )
            self._rebuild_fts_index(cur)

    def _rebuild_fts_index(self, cur: sqlite3.Cursor) -> None:
        """用 jieba 分词重建全部 FTS 索引。"""
        cur.execute("DELETE FROM memories_fts")
        rows = cur.execute("SELECT rowid, content FROM memories").fetchall()
        for row in rows:
            segmented = _segment(row["content"])
            cur.execute(
                "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
                (row["rowid"], segmented),
            )
        logger.info("Rebuilt FTS index with jieba segmentation: %d entries", len(rows))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        user_id: str,
        content: str,
        category: str = "other",
        importance: float = 0.5,
        source: str = "auto",
    ) -> str | None:
        """插入记忆。自动去重：如果已有高度相似的记忆，更新 updated_at 并返回 None。"""
        content = content.strip()
        if not content:
            return None

        # 去重检查
        existing = self.search(user_id, content, limit=1)
        if existing:
            top = existing[0]
            sim = SequenceMatcher(None, content.lower(), top.content.lower()).ratio()
            if sim >= _DEDUP_SIMILARITY:
                logger.debug(
                    "Duplicate memory (sim=%.2f), updating timestamp: %s",
                    sim, top.id,
                )
                with self._lock:
                    self._conn.execute(
                        "UPDATE memories SET updated_at = ?, access_count = access_count + 1 WHERE id = ?",
                        (time.time(), top.id),
                    )
                    self._conn.commit()
                return None

        now = time.time()
        memory_id = uuid.uuid4().hex[:16]
        segmented = _segment(content)

        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (id, user_id, content, category, importance, source, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (memory_id, user_id, content, category, importance, source, now, now),
            )
            # 手动同步 FTS 索引（用 jieba 分词后的文本）
            rowid = self._conn.execute(
                "SELECT rowid FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()["rowid"]
            self._conn.execute(
                "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
                (rowid, segmented),
            )
            self._conn.commit()
        logger.info("Saved memory %s for user %s: %s", memory_id, user_id, content[:80])
        return memory_id

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """FTS5 搜索 + BM25 排分 + 时间衰减。

        搜索时对 query 做 jieba 分词，再构建 FTS 查询。
        FTS5 返回空结果或报错时，回退到 LIKE 搜索。
        """
        query = query.strip()
        if not query:
            return self.get_recent(user_id, limit)

        # 用 jieba 分词构建 FTS 查询
        tokens = [t.strip() for t in jieba.cut(query) if t.strip()]
        if not tokens:
            return self.get_recent(user_id, limit)

        fts_query = " OR ".join(f'"{t}"' for t in tokens)

        sql = """
            SELECT m.*, bm25(memories_fts) AS rank
            FROM memories m
            JOIN memories_fts ON memories_fts.rowid = m.rowid
            WHERE memories_fts MATCH ? AND m.user_id = ?
            ORDER BY rank
            LIMIT ?
        """
        rows = []
        with self._lock:
            try:
                rows = self._conn.execute(sql, (fts_query, user_id, limit * 2)).fetchall()
            except sqlite3.OperationalError:
                logger.debug("FTS query failed, falling back to LIKE search")

        # FTS5 无结果，回退到 LIKE
        if not rows:
            return self._search_like(user_id, query, limit)

        now = time.time()
        entries = []
        for row in rows:
            entry = self._row_to_entry(row)
            # BM25 返回负值，越小越相关 → 取绝对值
            bm25 = abs(row["rank"])
            days = (now - entry.updated_at) / 86400
            decay = 1.0 / (1.0 + days * _DECAY_RATE)
            entry.score = bm25 * decay
            entries.append(entry)

        # 按 score 降序
        entries.sort(key=lambda e: e.score, reverse=True)
        return entries[:limit]

    def _search_like(self, user_id: str, query: str, limit: int) -> list[MemoryEntry]:
        """LIKE 回退搜索：用 jieba 分词后按关键词匹配。"""
        tokens = [t.strip() for t in jieba.cut(query) if t.strip()]
        if not tokens:
            return self.get_recent(user_id, limit)

        # 构建 OR 条件：任一关键词命中即可
        conditions = " OR ".join("content LIKE ?" for _ in tokens)
        params: list = [user_id] + [f"%{t}%" for t in tokens] + [limit]
        sql = f"""
            SELECT * FROM memories
            WHERE user_id = ? AND ({conditions})
            ORDER BY updated_at DESC
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_recent(self, user_id: str, limit: int = 10) -> list[MemoryEntry]:
        """获取最近的记忆"""
        sql = "SELECT * FROM memories WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (user_id, limit)).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_by_category(
        self, user_id: str, category: str, limit: int = 10
    ) -> list[MemoryEntry]:
        """按类别获取记忆"""
        sql = (
            "SELECT * FROM memories WHERE user_id = ? AND category = ? "
            "ORDER BY updated_at DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, (user_id, category, limit)).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def delete(self, memory_id: str) -> bool:
        """按 ID 删除"""
        with self._lock:
            # 先删 FTS 索引
            row = self._conn.execute(
                "SELECT rowid FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row:
                self._conn.execute(
                    "DELETE FROM memories_fts WHERE rowid = ?", (row["rowid"],)
                )
            # 再删主表
            cur = self._conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def delete_by_query(self, user_id: str, query: str) -> int:
        """搜索并删除匹配的记忆，返回删除数量"""
        matches = self.search(user_id, query, limit=3)
        if not matches:
            return 0
        deleted = 0
        for m in matches:
            # 只删除相似度较高的
            sim = SequenceMatcher(None, query.lower(), m.content.lower()).ratio()
            if sim >= 0.5:
                if self.delete(m.id):
                    deleted += 1
        return deleted

    def count(self, user_id: str) -> int:
        """用户记忆总数"""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM memories WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return row["cnt"] if row else 0

    def enforce_limit(self, user_id: str, max_count: int) -> int:
        """强制限制用户记忆数量，删除最旧的超出部分。返回删除数量。"""
        current = self.count(user_id)
        if current <= max_count:
            return 0
        to_delete = current - max_count

        with self._lock:
            # 找出要删除的记录的 rowid
            rows = self._conn.execute(
                "SELECT rowid FROM memories WHERE user_id = ? ORDER BY updated_at ASC LIMIT ?",
                (user_id, to_delete),
            ).fetchall()

            if not rows:
                return 0

            rowids = [r["rowid"] for r in rows]
            placeholders = ",".join("?" for _ in rowids)

            # 先删 FTS
            self._conn.execute(
                f"DELETE FROM memories_fts WHERE rowid IN ({placeholders})", rowids
            )
            # 再删主表
            cur = self._conn.execute(
                f"DELETE FROM memories WHERE rowid IN ({placeholders})", rowids
            )
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        """关闭连接"""
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            category=row["category"],
            importance=row["importance"],
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row["access_count"],
        )
