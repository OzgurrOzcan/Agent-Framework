"""SQLiteBackend — default zero-config persistent backend.

Single file, durable, no external deps. Uses `aiosqlite`-style access
through a thread pool (asyncio.to_thread) to keep the API uniform with the
async base class without adding a hard dependency.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from agent_framework.memory.backends.base import Fact, LongTermBackend


_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS idx_facts_updated ON facts(namespace, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_facts_tags ON facts(namespace, tags);

CREATE TABLE IF NOT EXISTS summaries (
    namespace TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class SQLiteBackend(LongTermBackend):
    def __init__(self, path: str | Path = "agent_memory.db") -> None:
        self._path = str(path)
        self._lock = threading.RLock()
        self._ensure_schema()

    # ----- sync core (runs inside to_thread) -----
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _put(self, fact: Fact) -> None:
        with self._lock, self._connect() as conn:
            now = datetime.now(timezone.utc).isoformat()
            existing = conn.execute(
                "SELECT created_at FROM facts WHERE namespace=? AND key=?",
                (fact.namespace, fact.key),
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            conn.execute(
                """
                INSERT INTO facts(namespace, key, value, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value=excluded.value,
                    tags=excluded.tags,
                    updated_at=excluded.updated_at
                """,
                (
                    fact.namespace,
                    fact.key,
                    fact.value,
                    json.dumps(list(fact.tags)),
                    created_at,
                    now,
                ),
            )
            conn.commit()

    def _get(self, namespace: str, key: str) -> Fact | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM facts WHERE namespace=? AND key=?", (namespace, key)
            ).fetchone()
            return _row_to_fact(row) if row else None

    def _delete(self, namespace: str, key: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM facts WHERE namespace=? AND key=?", (namespace, key)
            )
            conn.commit()

    def _search(
        self,
        namespace: str,
        query: str | None,
        tags: list[str],
        limit: int,
    ) -> list[Fact]:
        with self._lock, self._connect() as conn:
            sql = "SELECT * FROM facts WHERE namespace=?"
            params: list[object] = [namespace]
            if query:
                sql += " AND (LOWER(key) LIKE ? OR LOWER(value) LIKE ?)"
                pat = f"%{query.lower()}%"
                params.extend([pat, pat])
            if tags:
                for t in tags:
                    sql += " AND tags LIKE ?"
                    params.append(f'%"{t}"%')
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [_row_to_fact(r) for r in rows]

    def _set_summary(self, namespace: str, summary: str) -> None:
        with self._lock, self._connect() as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO summaries(namespace, summary, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(namespace) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at
                """,
                (namespace, summary, now),
            )
            conn.commit()

    def _get_summary(self, namespace: str) -> str:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM summaries WHERE namespace=?", (namespace,)
            ).fetchone()
            return row["summary"] if row else ""

    # ----- async surface -----
    async def put_fact(self, fact: Fact) -> None:
        await asyncio.to_thread(self._put, fact)

    async def get_fact(self, key: str, namespace: str = "default") -> Fact | None:
        return await asyncio.to_thread(self._get, namespace, key)

    async def delete_fact(self, key: str, namespace: str = "default") -> None:
        await asyncio.to_thread(self._delete, namespace, key)

    async def search_facts(
        self,
        *,
        query: str | None = None,
        tags: Iterable[str] | None = None,
        namespace: str = "default",
        limit: int = 20,
    ) -> list[Fact]:
        return await asyncio.to_thread(
            self._search, namespace, query, list(tags or []), limit
        )

    async def list_facts(self, namespace: str = "default", limit: int = 200) -> list[Fact]:
        return await asyncio.to_thread(self._search, namespace, None, [], limit)

    async def set_summary(self, namespace: str, summary: str) -> None:
        await asyncio.to_thread(self._set_summary, namespace, summary)

    async def get_summary(self, namespace: str) -> str:
        return await asyncio.to_thread(self._get_summary, namespace)


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        key=row["key"],
        value=row["value"],
        tags=tuple(json.loads(row["tags"] or "[]")),
        namespace=row["namespace"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
