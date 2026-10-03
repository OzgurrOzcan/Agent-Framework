"""SQLiteBackend — the default zero-config persistent backend.

One file, durable, no external dependency. All access goes through a single
connection, serialized by a lock and run in a worker thread
(`asyncio.to_thread`), so the async API never blocks the event loop and an
in-memory database (``":memory:"``) keeps its data for the backend's
lifetime. `close()` releases the connection and is idempotent.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from agent_framework.memory.backends.base import Fact, LongTermBackend
from agent_framework.memory.sessions import LogEntry, SessionRecord, SessionStore, encode

_T = TypeVar("_T")

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

-- Every message a session ever had, in order (see memory.sessions).
CREATE TABLE IF NOT EXISTS session_messages (
    namespace TEXT NOT NULL,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (namespace, session_id, seq)
);

-- What a resumed session continues from.
CREATE TABLE IF NOT EXISTS session_state (
    namespace TEXT NOT NULL,
    session_id TEXT NOT NULL,
    history TEXT NOT NULL,
    state TEXT NOT NULL,
    last_seq INTEGER NOT NULL DEFAULT 0,
    fingerprint TEXT NOT NULL DEFAULT '',
    summarized_upto INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (namespace, session_id)
);
CREATE INDEX IF NOT EXISTS idx_session_state_updated
    ON session_state(namespace, updated_at DESC);
"""


class SQLiteBackend(LongTermBackend, SessionStore):
    def __init__(self, path: str | Path = "agent_memory.db") -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = self._open()

    @property
    def path(self) -> str:
        return self._path

    @property
    def closed(self) -> bool:
        return self._conn is None

    # ----- connection -----
    def _open(self) -> sqlite3.Connection:
        # check_same_thread=False: the connection is used from worker
        # threads, one at a time under `_lock`.
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL lets readers in other processes proceed while one writes; a
        # busy timeout makes a concurrent writer wait instead of failing.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        with conn:
            conn.executescript(_SCHEMA)
        return conn

    def _run(self, work: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run *work* on the connection, in one transaction, under the lock."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError(
                    f"SQLiteBackend({self._path!r}) is closed; build a new backend "
                    "(or a new agent) to use the store again."
                )
            with self._conn:  # commits on success, rolls back on error
                return work(self._conn)

    async def _call(self, work: Callable[[sqlite3.Connection], _T]) -> _T:
        return await asyncio.to_thread(self._run, work)

    async def close(self) -> None:
        """Close the connection. Safe to call more than once."""

        def _close() -> None:
            with self._lock:
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None

        await asyncio.to_thread(_close)

    # ----- facts -----
    async def put_fact(self, fact: Fact) -> None:
        def work(conn: sqlite3.Connection) -> None:
            now = _now()
            conn.execute(
                """
                INSERT INTO facts(namespace, key, value, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value=excluded.value,
                    tags=excluded.tags,
                    updated_at=excluded.updated_at
                """,
                (fact.namespace, fact.key, fact.value, json.dumps(list(fact.tags)), now, now),
            )

        await self._call(work)

    async def get_fact(self, key: str, namespace: str = "default") -> Fact | None:
        def work(conn: sqlite3.Connection) -> Fact | None:
            row = conn.execute(
                "SELECT * FROM facts WHERE namespace=? AND key=?", (namespace, key)
            ).fetchone()
            return _row_to_fact(row) if row else None

        return await self._call(work)

    async def delete_fact(self, key: str, namespace: str = "default") -> None:
        await self._call(
            lambda conn: conn.execute(
                "DELETE FROM facts WHERE namespace=? AND key=?", (namespace, key)
            )
        )

    async def search_facts(
        self,
        *,
        query: str | None = None,
        tags: Iterable[str] | None = None,
        namespace: str = "default",
        limit: int = 20,
    ) -> list[Fact]:
        tag_list = list(tags or [])

        def work(conn: sqlite3.Connection) -> list[Fact]:
            sql = "SELECT * FROM facts WHERE namespace=?"
            params: list[object] = [namespace]
            if query:
                sql += " AND (LOWER(key) LIKE ? OR LOWER(value) LIKE ?)"
                pattern = f"%{query.lower()}%"
                params.extend([pattern, pattern])
            for tag in tag_list:
                sql += " AND tags LIKE ?"
                params.append(f'%"{tag}"%')
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            return [_row_to_fact(r) for r in conn.execute(sql, params).fetchall()]

        return await self._call(work)

    async def list_facts(self, namespace: str = "default", limit: int = 200) -> list[Fact]:
        return await self.search_facts(namespace=namespace, limit=limit)

    # ----- summary -----
    async def set_summary(self, namespace: str, summary: str) -> None:
        await self._call(
            lambda conn: conn.execute(
                """
                INSERT INTO summaries(namespace, summary, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(namespace) DO UPDATE SET
                    summary=excluded.summary, updated_at=excluded.updated_at
                """,
                (namespace, summary, _now()),
            )
        )

    async def get_summary(self, namespace: str) -> str:
        def work(conn: sqlite3.Connection) -> str:
            row = conn.execute(
                "SELECT summary FROM summaries WHERE namespace=?", (namespace,)
            ).fetchone()
            return row["summary"] if row else ""

        return await self._call(work)


    # ----- sessions -----
    async def load_session(self, namespace: str, session_id: str) -> SessionRecord | None:
        def work(conn: sqlite3.Connection) -> SessionRecord | None:
            row = conn.execute(
                "SELECT * FROM session_state WHERE namespace=? AND session_id=?",
                (namespace, session_id),
            ).fetchone()
            return _row_to_session(row) if row else None

        return await self._call(work)

    async def save_session(
        self,
        namespace: str,
        session_id: str,
        record: SessionRecord,
        new_log: list[LogEntry],
    ) -> None:
        def work(conn: sqlite3.Connection) -> None:
            now = _now()
            conn.executemany(
                "INSERT INTO session_messages(namespace, session_id, seq, message, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [(namespace, session_id, e.seq, encode(e.message), now) for e in new_log],
            )
            # MAX(): a counter another writer advanced (another process
            # summarizing this session) never moves backwards.
            conn.execute(
                """
                INSERT INTO session_state(namespace, session_id, history, state, last_seq,
                                          fingerprint, summarized_upto, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, session_id) DO UPDATE SET
                    history=excluded.history,
                    state=excluded.state,
                    last_seq=MAX(session_state.last_seq, excluded.last_seq),
                    fingerprint=excluded.fingerprint,
                    summarized_upto=MAX(session_state.summarized_upto, excluded.summarized_upto),
                    updated_at=excluded.updated_at
                """,
                (
                    namespace,
                    session_id,
                    encode(record.history),
                    encode(record.state),
                    record.last_seq,
                    record.fingerprint,
                    record.summarized_upto,
                    record.created_at or now,
                    now,
                ),
            )

        await self._call(work)

    async def session_log(
        self, namespace: str, session_id: str, *, after_seq: int = 0
    ) -> list[LogEntry]:
        def work(conn: sqlite3.Connection) -> list[LogEntry]:
            rows = conn.execute(
                "SELECT seq, message FROM session_messages "
                "WHERE namespace=? AND session_id=? AND seq>? ORDER BY seq",
                (namespace, session_id, after_seq),
            ).fetchall()
            return [LogEntry(seq=r["seq"], message=json.loads(r["message"])) for r in rows]

        return await self._call(work)


def _row_to_session(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        history=json.loads(row["history"]),
        state=json.loads(row["state"]),
        last_seq=row["last_seq"],
        fingerprint=row["fingerprint"],
        summarized_upto=row["summarized_upto"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        key=row["key"],
        value=row["value"],
        tags=tuple(json.loads(row["tags"] or "[]")),
        namespace=row["namespace"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
