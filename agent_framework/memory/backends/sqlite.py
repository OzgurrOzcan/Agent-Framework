"""SQLiteBackend — the default zero-config persistent backend.

One file, durable, no external dependency. All access goes through a single
connection, serialized by a lock and run in a worker thread
(`asyncio.to_thread`), so the async API never blocks the event loop and an
in-memory database (``":memory:"``) keeps its data for the backend's
lifetime. `close()` releases the connection and is idempotent.

Facts are also indexed in an FTS5 table (`facts_fts`) when the SQLite build
has the module — every standard CPython build does — so that fact retrieval
runs in the database: diacritic-insensitive (``iletisim`` finds
``iletişim``), ranked by BM25, and over every fact rather than the newest N.
Without FTS5 the manager falls back to matching in Python.

An older database gains the tables and columns it lacks on open; nothing it
already holds is touched.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from agent_framework.memory.backends.base import Fact, LongTermBackend
from agent_framework.memory.sessions import (
    IdleTail,
    LogEntry,
    SessionInfo,
    SessionRecord,
    SessionStore,
    SessionTakenOverError,
    SummarySegment,
    encode,
)

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
    owner TEXT NOT NULL DEFAULT '',
    owner_since TEXT NOT NULL DEFAULT '',
    claim_token TEXT NOT NULL DEFAULT '',
    claim_until TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (namespace, session_id)
);
CREATE INDEX IF NOT EXISTS idx_session_state_updated
    ON session_state(namespace, updated_at DESC);

-- A namespace's long-term summary, as a list of segments (oldest first).
CREATE TABLE IF NOT EXISTS summary_segments (
    namespace TEXT NOT NULL,
    seq INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (namespace, seq)
);
"""

# Columns added after the first release: (table, column, declaration).
# `CREATE TABLE IF NOT EXISTS` does not add them to an existing table.
_MIGRATIONS = (
    ("session_state", "owner", "TEXT NOT NULL DEFAULT ''"),
    ("session_state", "owner_since", "TEXT NOT NULL DEFAULT ''"),
    ("session_state", "claim_token", "TEXT NOT NULL DEFAULT ''"),
    ("session_state", "claim_until", "TEXT NOT NULL DEFAULT ''"),
)

# unicode61 splits on punctuation (so "user_name" is two words) and
# `remove_diacritics 2` folds accents on both the indexed text and the query.
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    namespace UNINDEXED, fact_key UNINDEXED, key, value, tags,
    tokenize='unicode61 remove_diacritics 2'
);
"""
_CAMEL_RX = re.compile(r"(?<=[^\W_])(?=[A-Z])")


class SQLiteBackend(LongTermBackend, SessionStore):
    def __init__(self, path: str | Path = "agent_memory.db") -> None:
        self._path = str(path)
        self._lock = threading.Lock()
        self._fts = False
        self._conn: sqlite3.Connection | None = self._open()

    @property
    def path(self) -> str:
        return self._path

    @property
    def closed(self) -> bool:
        return self._conn is None

    @property
    def full_text_search(self) -> bool:
        """Whether facts are matched in the database (FTS5 available)."""
        return self._fts

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
            for table, column, declaration in _MIGRATIONS:
                present = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                if column not in present:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
        self._fts = self._open_fts(conn)
        return conn

    def _open_fts(self, conn: sqlite3.Connection) -> bool:
        """Create the fact index if the build has FTS5; fill it for an
        existing database the first time."""
        existed = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='facts_fts'"
            ).fetchone()
            is not None
        )
        try:
            with conn:
                conn.executescript(_FTS_SCHEMA)
                if not existed:
                    conn.execute("DELETE FROM facts_fts")
                    conn.executemany(
                        "INSERT INTO facts_fts(namespace, fact_key, key, value, tags) "
                        "VALUES (?, ?, ?, ?, ?)",
                        [
                            (
                                r["namespace"],
                                r["key"],
                                _index_key(r["key"]),
                                r["value"],
                                _tag_text(r["tags"]),
                            )
                            for r in conn.execute("SELECT * FROM facts")
                        ],
                    )
        except sqlite3.OperationalError:
            return False  # no FTS5 in this build: match in Python instead
        return True

    def _run(self, work: Callable[[sqlite3.Connection], _T], *, immediate: bool = False) -> _T:
        """Run *work* on the connection, in one transaction, under the lock.

        *immediate* takes the database's write lock up front (BEGIN
        IMMEDIATE): for read-then-write steps that another process must not
        interleave with, such as claiming a session's tail.
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError(
                    f"SQLiteBackend({self._path!r}) is closed; build a new backend "
                    "(or a new agent) to use the store again."
                )
            with self._conn:  # commits on success, rolls back on error
                if immediate:
                    self._conn.execute("BEGIN IMMEDIATE")
                return work(self._conn)

    async def _call(
        self, work: Callable[[sqlite3.Connection], _T], *, immediate: bool = False
    ) -> _T:
        return await asyncio.to_thread(self._run, work, immediate=immediate)

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
            tags = json.dumps(list(fact.tags))
            conn.execute(
                """
                INSERT INTO facts(namespace, key, value, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value=excluded.value,
                    tags=excluded.tags,
                    updated_at=excluded.updated_at
                """,
                (fact.namespace, fact.key, fact.value, tags, now, now),
            )
            if self._fts:
                conn.execute(
                    "DELETE FROM facts_fts WHERE namespace=? AND fact_key=?",
                    (fact.namespace, fact.key),
                )
                conn.execute(
                    "INSERT INTO facts_fts(namespace, fact_key, key, value, tags) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (fact.namespace, fact.key, _index_key(fact.key), fact.value, _tag_text(tags)),
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
        def work(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM facts WHERE namespace=? AND key=?", (namespace, key))
            if self._fts:
                conn.execute(
                    "DELETE FROM facts_fts WHERE namespace=? AND fact_key=?", (namespace, key)
                )

        await self._call(work)

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

    async def match_facts(
        self, words: Iterable[str], *, namespace: str = "default", limit: int = 20
    ) -> list[Fact] | None:
        """The facts containing any of *words*, best match first (BM25, ties
        to the most recently updated). None without FTS5."""
        if not self._fts:
            return None
        terms = sorted({w for w in words if w})
        if not terms:
            return []
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)

        def work(conn: sqlite3.Connection) -> list[Fact]:
            rows = conn.execute(
                """
                SELECT f.*, bm25(facts_fts) AS rank FROM facts_fts
                JOIN facts f ON f.namespace = facts_fts.namespace
                             AND f.key = facts_fts.fact_key
                WHERE facts_fts MATCH ? AND facts_fts.namespace = ?
                ORDER BY rank, f.updated_at DESC LIMIT ?
                """,
                (match, namespace, limit),
            ).fetchall()
            return [_row_to_fact(r) for r in rows]

        return await self._call(work)

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

    async def take_session(self, namespace: str, session_id: str, owner: str) -> None:
        await self._call(
            lambda conn: conn.execute(
                "UPDATE session_state SET owner=?, owner_since=? "
                "WHERE namespace=? AND session_id=?",
                (owner, _now(), namespace, session_id),
            ),
            immediate=True,
        )

    async def save_session(
        self,
        namespace: str,
        session_id: str,
        record: SessionRecord,
        new_log: list[LogEntry],
        *,
        owner: str = "",
    ) -> None:
        def work(conn: sqlite3.Connection) -> None:
            now = _now()
            if owner:
                row = conn.execute(
                    "SELECT owner FROM session_state WHERE namespace=? AND session_id=?",
                    (namespace, session_id),
                ).fetchone()
                if row is not None and row["owner"] not in ("", owner):
                    raise SessionTakenOverError(
                        f"session {session_id!r} of namespace {namespace!r} is owned by "
                        "another agent"
                    )
            conn.executemany(
                "INSERT INTO session_messages(namespace, session_id, seq, message, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [(namespace, session_id, e.seq, encode(e.message), now) for e in new_log],
            )
            # MAX(): a counter another writer advanced (another process
            # summarizing this session) never moves backwards. An empty
            # owner leaves the stored owner alone.
            conn.execute(
                """
                INSERT INTO session_state(namespace, session_id, history, state, last_seq,
                                          fingerprint, summarized_upto, owner, owner_since,
                                          created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, session_id) DO UPDATE SET
                    history=excluded.history,
                    state=excluded.state,
                    last_seq=MAX(session_state.last_seq, excluded.last_seq),
                    fingerprint=excluded.fingerprint,
                    summarized_upto=MAX(session_state.summarized_upto, excluded.summarized_upto),
                    owner=CASE WHEN excluded.owner='' THEN session_state.owner
                               ELSE excluded.owner END,
                    owner_since=CASE WHEN excluded.owner='' OR excluded.owner=session_state.owner
                                     THEN session_state.owner_since ELSE excluded.owner_since END,
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
                    owner,
                    now if owner else "",
                    record.created_at or now,
                    now,
                ),
            )

        await self._call(work, immediate=True)

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

    # ----- housekeeping -----
    async def list_sessions(
        self, namespace: str, *, idle_before: str | None = None, limit: int = 100
    ) -> list[SessionInfo]:
        def work(conn: sqlite3.Connection) -> list[SessionInfo]:
            sql = (
                "SELECT session_id, created_at, updated_at, last_seq, summarized_upto "
                "FROM session_state WHERE namespace=?"
            )
            params: list[object] = [namespace]
            if idle_before is not None:
                sql += " AND updated_at < ?"
                params.append(idle_before)
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            return [
                SessionInfo(
                    session_id=r["session_id"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    last_seq=r["last_seq"],
                    summarized_upto=r["summarized_upto"],
                )
                for r in conn.execute(sql, params).fetchall()
            ]

        return await self._call(work)

    async def delete_session(self, namespace: str, session_id: str) -> bool:
        def work(conn: sqlite3.Connection) -> bool:
            conn.execute(
                "DELETE FROM session_messages WHERE namespace=? AND session_id=?",
                (namespace, session_id),
            )
            deleted = conn.execute(
                "DELETE FROM session_state WHERE namespace=? AND session_id=?",
                (namespace, session_id),
            ).rowcount
            return deleted > 0

        return await self._call(work, immediate=True)

    async def prune_sessions(
        self, namespace: str, *, idle_before: str, summarized_only: bool = True
    ) -> int:
        def work(conn: sqlite3.Connection) -> int:
            condition = "namespace=? AND updated_at < ?"
            if summarized_only:
                condition += " AND summarized_upto >= last_seq"
            conn.execute(
                "DELETE FROM session_messages WHERE namespace=? AND session_id IN "
                f"(SELECT session_id FROM session_state WHERE {condition})",
                (namespace, namespace, idle_before),
            )
            return conn.execute(
                f"DELETE FROM session_state WHERE {condition}", (namespace, idle_before)
            ).rowcount

        return await self._call(work, immediate=True)

    # ----- the namespace's summary -----
    async def summary_segments(self, namespace: str) -> list[SummarySegment]:
        def work(conn: sqlite3.Connection) -> list[SummarySegment]:
            rows = conn.execute(
                "SELECT seq, text FROM summary_segments WHERE namespace=? ORDER BY seq",
                (namespace,),
            ).fetchall()
            if rows:
                return [SummarySegment(seq=r["seq"], text=r["text"]) for r in rows]
            # A summary written before segments existed is the first one.
            legacy = conn.execute(
                "SELECT summary FROM summaries WHERE namespace=?", (namespace,)
            ).fetchone()
            return [SummarySegment(seq=0, text=legacy["summary"])] if legacy and legacy["summary"] else []

        return await self._call(work)

    async def append_summary(
        self,
        namespace: str,
        text: str,
        *,
        session_id: str | None = None,
        expected_upto: int | None = None,
        claim_token: str | None = None,
        summarized_upto: int = 0,
    ) -> bool:
        def work(conn: sqlite3.Connection) -> bool:
            now = _now()
            if session_id is not None:
                row = conn.execute(
                    "SELECT summarized_upto, claim_token, claim_until FROM session_state "
                    "WHERE namespace=? AND session_id=?",
                    (namespace, session_id),
                ).fetchone()
                if not _may_append(row, expected_upto, claim_token, now):
                    return False
            last = conn.execute(
                "SELECT MAX(seq) AS seq FROM summary_segments WHERE namespace=?", (namespace,)
            ).fetchone()["seq"]
            if last is None:
                last = 0
                legacy = conn.execute(
                    "SELECT summary FROM summaries WHERE namespace=?", (namespace,)
                ).fetchone()
                if legacy and legacy["summary"]:
                    last = 1
                    conn.execute(
                        "INSERT INTO summary_segments(namespace, seq, text, created_at) "
                        "VALUES (?, 1, ?, ?)",
                        (namespace, legacy["summary"], now),
                    )
            conn.execute(
                "INSERT INTO summary_segments(namespace, seq, text, created_at) VALUES (?, ?, ?, ?)",
                (namespace, last + 1, text, now),
            )
            if session_id is not None:
                conn.execute(
                    "UPDATE session_state SET summarized_upto=MAX(summarized_upto, ?) "
                    "WHERE namespace=? AND session_id=?",
                    (summarized_upto, namespace, session_id),
                )
                if claim_token is not None:
                    conn.execute(
                        "UPDATE session_state SET claim_token='', claim_until='' "
                        "WHERE namespace=? AND session_id=? AND claim_token=?",
                        (namespace, session_id, claim_token),
                    )
            return True

        return await self._call(work, immediate=True)

    async def merge_summaries(
        self, namespace: str, first_seq: int, second_seq: int, text: str
    ) -> bool:
        def work(conn: sqlite3.Connection) -> bool:
            found = conn.execute(
                "SELECT COUNT(*) AS n FROM summary_segments WHERE namespace=? AND seq IN (?, ?)",
                (namespace, first_seq, second_seq),
            ).fetchone()["n"]
            if found != 2:
                return False
            conn.execute(
                "DELETE FROM summary_segments WHERE namespace=? AND seq=?", (namespace, second_seq)
            )
            conn.execute(
                "UPDATE summary_segments SET text=? WHERE namespace=? AND seq=?",
                (text, namespace, first_seq),
            )
            return True

        return await self._call(work, immediate=True)

    async def claim_idle_session(
        self,
        namespace: str,
        *,
        idle_before: str,
        exclude_session: str,
        lease_until: str,
    ) -> IdleTail | None:
        def work(conn: sqlite3.Connection) -> IdleTail | None:
            now = _now()
            row = conn.execute(
                """
                SELECT session_id, summarized_upto, last_seq FROM session_state
                WHERE namespace=? AND session_id != ? AND updated_at < ?
                  AND summarized_upto < last_seq
                  AND (claim_until = '' OR claim_until < ?)
                ORDER BY updated_at ASC LIMIT 1
                """,
                (namespace, exclude_session, idle_before, now),
            ).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            claimed = conn.execute(
                "UPDATE session_state SET claim_token=?, claim_until=? "
                "WHERE namespace=? AND session_id=? AND (claim_until = '' OR claim_until < ?)",
                (token, lease_until, namespace, row["session_id"], now),
            ).rowcount
            if claimed != 1:
                return None
            return IdleTail(
                session_id=row["session_id"],
                after_seq=row["summarized_upto"],
                upto_seq=row["last_seq"],
                token=token,
            )

        return await self._call(work, immediate=True)

    async def release_claim(self, namespace: str, tail: IdleTail) -> None:
        await self._call(
            lambda conn: conn.execute(
                "UPDATE session_state SET claim_token='', claim_until='' "
                "WHERE namespace=? AND session_id=? AND claim_token=?",
                (namespace, tail.session_id, tail.token),
            ),
            immediate=True,
        )

    async def idle_backlog(self, namespace: str, *, idle_before: str, exclude_session: str) -> int:
        def work(conn: sqlite3.Connection) -> int:
            return conn.execute(
                """
                SELECT COUNT(*) AS n FROM session_state
                WHERE namespace=? AND session_id != ? AND updated_at < ?
                  AND summarized_upto < last_seq
                """,
                (namespace, exclude_session, idle_before),
            ).fetchone()["n"]

        return await self._call(work)


def _may_append(
    row: sqlite3.Row | None, expected_upto: int | None, claim_token: str | None, now: str
) -> bool:
    """The `append_summary` guards (see `SessionStore.append_summary`)."""
    if claim_token is not None:
        if row is None or row["claim_token"] != claim_token:
            return False  # the lease expired and went to someone else
    if expected_upto is not None and row is not None:
        if row["summarized_upto"] != expected_upto:
            return False  # summarized by another session meanwhile
        if claim_token is None and row["claim_token"] and row["claim_until"] > now:
            return False  # another session holds a lease on this tail
    return True


def _row_to_session(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        history=json.loads(row["history"]),
        state=json.loads(row["state"]),
        last_seq=row["last_seq"],
        fingerprint=row["fingerprint"],
        summarized_upto=row["summarized_upto"],
        owner=row["owner"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _index_key(key: str) -> str:
    """The key as indexed: camelCase split, so "deployTarget" has two words."""
    return _CAMEL_RX.sub(" ", key)


def _tag_text(tags: str | Iterable[str]) -> str:
    """The tags as indexed text (from the stored JSON list or a sequence)."""
    if isinstance(tags, str):
        try:
            loaded = json.loads(tags or "[]")
        except ValueError:
            return tags
        return " ".join(str(t) for t in loaded)
    return " ".join(str(t) for t in tags)


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        key=row["key"],
        value=row["value"],
        tags=tuple(json.loads(row["tags"] or "[]")),
        namespace=row["namespace"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
