"""InMemoryBackend — for tests and ephemeral runs.

Sessions are kept as encoded JSON, exactly as SQLite keeps them, so nothing a
caller holds aliases the stored data and both backends behave alike.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

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


class InMemoryBackend(LongTermBackend, SessionStore):
    def __init__(self) -> None:
        self._facts: dict[tuple[str, str], Fact] = {}
        self._summaries: dict[str, str] = {}
        # (namespace, session_id) → encoded SessionRecord fields / log entries.
        self._sessions: dict[tuple[str, str], _StoredSession] = {}
        self._logs: dict[tuple[str, str], list[tuple[int, str]]] = {}
        self._segments: dict[str, list[SummarySegment]] = {}
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
                owner=stored.owner,
                created_at=stored.created_at,
                updated_at=stored.updated_at,
            )

    async def take_session(self, namespace: str, session_id: str, owner: str) -> None:
        async with self._lock:
            stored = self._sessions.get((namespace, session_id))
            if stored is not None:
                stored.owner = owner

    async def save_session(
        self,
        namespace: str,
        session_id: str,
        record: SessionRecord,
        new_log: list[LogEntry],
        *,
        owner: str = "",
    ) -> None:
        key = (namespace, session_id)
        async with self._lock:
            previous = self._sessions.get(key)
            if owner and previous is not None and previous.owner not in ("", owner):
                raise SessionTakenOverError(
                    f"session {session_id!r} of namespace {namespace!r} is owned by another agent"
                )
            log = self._logs.setdefault(key, [])
            taken = {seq for seq, _ in log}
            for entry in new_log:
                if entry.seq in taken:
                    raise ValueError(
                        f"session {session_id!r} already has a log entry {entry.seq}"
                    )
            log.extend((entry.seq, encode(entry.message)) for entry in new_log)
            now = datetime.now(timezone.utc).isoformat()
            self._sessions[key] = _StoredSession(
                history=encode(record.history),
                state=encode(record.state),
                # Never backwards: another writer may have advanced them.
                last_seq=max(record.last_seq, previous.last_seq if previous else 0),
                fingerprint=record.fingerprint,
                summarized_upto=max(
                    record.summarized_upto, previous.summarized_upto if previous else 0
                ),
                owner=owner or (previous.owner if previous else ""),
                claim_token=previous.claim_token if previous else "",
                claim_until=previous.claim_until if previous else "",
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

    # ----- housekeeping -----
    async def list_sessions(
        self, namespace: str, *, idle_before: str | None = None, limit: int = 100
    ) -> list[SessionInfo]:
        async with self._lock:
            rows = [
                SessionInfo(
                    session_id=session_id,
                    created_at=stored.created_at,
                    updated_at=stored.updated_at,
                    last_seq=stored.last_seq,
                    summarized_upto=stored.summarized_upto,
                )
                for (ns, session_id), stored in self._sessions.items()
                if ns == namespace and (idle_before is None or stored.updated_at < idle_before)
            ]
            rows.sort(key=lambda info: info.updated_at, reverse=True)
            return rows[:limit]

    async def delete_session(self, namespace: str, session_id: str) -> bool:
        async with self._lock:
            self._logs.pop((namespace, session_id), None)
            return self._sessions.pop((namespace, session_id), None) is not None

    async def prune_sessions(
        self, namespace: str, *, idle_before: str, summarized_only: bool = True
    ) -> int:
        async with self._lock:
            doomed = [
                key
                for key, stored in self._sessions.items()
                if key[0] == namespace
                and stored.updated_at < idle_before
                and (not summarized_only or stored.summarized_upto >= stored.last_seq)
            ]
            for key in doomed:
                self._sessions.pop(key, None)
                self._logs.pop(key, None)
            return len(doomed)

    # ----- the namespace's summary -----
    async def summary_segments(self, namespace: str) -> list[SummarySegment]:
        async with self._lock:
            segments = self._segments.get(namespace)
            if segments:
                return list(segments)
            legacy = self._summaries.get(namespace, "")
            return [SummarySegment(seq=0, text=legacy)] if legacy else []

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
        async with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            stored = self._sessions.get((namespace, session_id)) if session_id else None
            if session_id is not None:
                if claim_token is not None and (
                    stored is None or stored.claim_token != claim_token
                ):
                    return False  # the lease expired and went to someone else
                if expected_upto is not None and stored is not None:
                    if stored.summarized_upto != expected_upto:
                        return False  # summarized by another session meanwhile
                    if claim_token is None and stored.claim_token and stored.claim_until > now:
                        return False  # another session holds a lease on this tail
            segments = self._segments.setdefault(namespace, [])
            if not segments and self._summaries.get(namespace):
                segments.append(SummarySegment(seq=1, text=self._summaries[namespace]))
            next_seq = (segments[-1].seq if segments else 0) + 1
            segments.append(SummarySegment(seq=next_seq, text=text))
            if stored is not None:
                stored.summarized_upto = max(stored.summarized_upto, summarized_upto)
                if claim_token is not None and stored.claim_token == claim_token:
                    stored.claim_token = stored.claim_until = ""
            return True

    async def merge_summaries(
        self, namespace: str, first_seq: int, second_seq: int, text: str
    ) -> bool:
        async with self._lock:
            segments = self._segments.get(namespace, [])
            seqs = {s.seq for s in segments}
            if first_seq not in seqs or second_seq not in seqs:
                return False
            self._segments[namespace] = [
                SummarySegment(seq=first_seq, text=text) if s.seq == first_seq else s
                for s in segments
                if s.seq != second_seq
            ]
            return True

    async def claim_idle_session(
        self,
        namespace: str,
        *,
        idle_before: str,
        exclude_session: str,
        lease_until: str,
    ) -> IdleTail | None:
        async with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            candidates = [
                (stored.updated_at, session_id, stored)
                for (ns, session_id), stored in self._sessions.items()
                if ns == namespace
                and session_id != exclude_session
                and stored.updated_at < idle_before
                and stored.summarized_upto < stored.last_seq
                and (not stored.claim_until or stored.claim_until < now)
            ]
            if not candidates:
                return None
            _, session_id, stored = min(candidates, key=lambda c: c[0])  # the oldest
            stored.claim_token = uuid.uuid4().hex
            stored.claim_until = lease_until
            return IdleTail(
                session_id=session_id,
                after_seq=stored.summarized_upto,
                upto_seq=stored.last_seq,
                token=stored.claim_token,
            )

    async def release_claim(self, namespace: str, tail: IdleTail) -> None:
        async with self._lock:
            stored = self._sessions.get((namespace, tail.session_id))
            if stored is not None and stored.claim_token == tail.token:
                stored.claim_token = stored.claim_until = ""

    async def idle_backlog(self, namespace: str, *, idle_before: str, exclude_session: str) -> int:
        async with self._lock:
            return sum(
                1
                for (ns, session_id), stored in self._sessions.items()
                if ns == namespace
                and session_id != exclude_session
                and stored.updated_at < idle_before
                and stored.summarized_upto < stored.last_seq
            )


@dataclass
class _StoredSession:
    history: str
    state: str
    last_seq: int
    fingerprint: str
    summarized_upto: int
    owner: str
    claim_token: str
    claim_until: str
    created_at: str
    updated_at: str
