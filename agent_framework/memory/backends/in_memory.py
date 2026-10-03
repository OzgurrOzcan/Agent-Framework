"""InMemoryBackend — for tests and ephemeral runs.

Sessions are kept as encoded JSON, exactly as SQLite keeps them, so nothing a
caller holds aliases the stored data and both backends behave alike.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from agent_framework.memory.backends.base import Fact, LongTermBackend
from agent_framework.memory.sessions import (
    LogEntry,
    SessionRecord,
    SessionStore,
    encode,
)


class InMemoryBackend(LongTermBackend, SessionStore):
    def __init__(self) -> None:
        self._facts: dict[tuple[str, str], Fact] = {}
        self._summaries: dict[str, str] = {}
        # (namespace, session_id) → encoded SessionRecord fields / log entries.
        self._sessions: dict[tuple[str, str], _StoredSession] = {}
        self._logs: dict[tuple[str, str], list[tuple[int, str]]] = {}
        self._lock = asyncio.Lock()

    async def put_fact(self, fact: Fact) -> None:
        async with self._lock:
            now = datetime.now(timezone.utc)
            existing = self._facts.get((fact.namespace, fact.key))
            self._facts[(fact.namespace, fact.key)] = Fact(
                key=fact.key,
                value=fact.value,
                tags=tuple(fact.tags),
                namespace=fact.namespace,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )

    async def get_fact(self, key: str, namespace: str = "default") -> Fact | None:
        async with self._lock:
            return self._facts.get((namespace, key))

    async def delete_fact(self, key: str, namespace: str = "default") -> None:
        async with self._lock:
            self._facts.pop((namespace, key), None)

    async def search_facts(
        self,
        *,
        query: str | None = None,
        tags: Iterable[str] | None = None,
        namespace: str = "default",
        limit: int = 20,
    ) -> list[Fact]:
        async with self._lock:
            tags_set = set(tags or ())
            results: list[Fact] = []
            ql = (query or "").lower()
            for (ns, _), fact in self._facts.items():
                if ns != namespace:
                    continue
                if tags_set and not tags_set.issubset(set(fact.tags)):
                    continue
                if ql:
                    hay = f"{fact.key} {fact.value} {' '.join(fact.tags)}".lower()
                    if ql not in hay:
                        continue
                results.append(fact)
            results.sort(key=lambda f: f.updated_at or datetime.min, reverse=True)
            return results[:limit]

    async def list_facts(self, namespace: str = "default", limit: int = 200) -> list[Fact]:
        return await self.search_facts(namespace=namespace, limit=limit)

    async def set_summary(self, namespace: str, summary: str) -> None:
        async with self._lock:
            self._summaries[namespace] = summary

    async def get_summary(self, namespace: str) -> str:
        async with self._lock:
            return self._summaries.get(namespace, "")

    # ----- sessions -----
    async def load_session(self, namespace: str, session_id: str) -> SessionRecord | None:
        async with self._lock:
            stored = self._sessions.get((namespace, session_id))
            if stored is None:
                return None
            return SessionRecord(
                history=json.loads(stored.history),
                state=json.loads(stored.state),
                last_seq=stored.last_seq,
                fingerprint=stored.fingerprint,
                summarized_upto=stored.summarized_upto,
                created_at=stored.created_at,
                updated_at=stored.updated_at,
            )

    async def save_session(
        self,
        namespace: str,
        session_id: str,
        record: SessionRecord,
        new_log: list[LogEntry],
    ) -> None:
        key = (namespace, session_id)
        async with self._lock:
            log = self._logs.setdefault(key, [])
            taken = {seq for seq, _ in log}
            for entry in new_log:
                if entry.seq in taken:
                    raise ValueError(
                        f"session {session_id!r} already has a log entry {entry.seq}"
                    )
            log.extend((entry.seq, encode(entry.message)) for entry in new_log)
            now = datetime.now(timezone.utc).isoformat()
            previous = self._sessions.get(key)
            self._sessions[key] = _StoredSession(
                history=encode(record.history),
                state=encode(record.state),
                # Never backwards: another writer may have advanced them.
                last_seq=max(record.last_seq, previous.last_seq if previous else 0),
                fingerprint=record.fingerprint,
                summarized_upto=max(
                    record.summarized_upto, previous.summarized_upto if previous else 0
                ),
                created_at=(previous.created_at if previous else "") or record.created_at or now,
                updated_at=now,
            )

    async def session_log(
        self, namespace: str, session_id: str, *, after_seq: int = 0
    ) -> list[LogEntry]:
        async with self._lock:
            entries = sorted(self._logs.get((namespace, session_id), []))
            return [
                LogEntry(seq=seq, message=json.loads(raw))
                for seq, raw in entries
                if seq > after_seq
            ]


@dataclass
class _StoredSession:
    history: str
    state: str
    last_seq: int
    fingerprint: str
    summarized_upto: int
    created_at: str
    updated_at: str
