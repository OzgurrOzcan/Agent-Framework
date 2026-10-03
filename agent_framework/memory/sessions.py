"""Saving and resuming conversations by `session_id`.

A session is one conversation of one namespace. Two records keep it:

  - the log: every message the conversation ever had, in order, each with a
    sequence number — kept even after a trim drops it from the history, so
    the long-term summary can be written from it later;
  - the state: the current history plus its `ContextState`, which is what a
    resumed agent continues from.

Each message is stamped with its sequence number (`metadata["log_seq"]`) the
first time it is logged, so it is never logged twice and a trim can tell
which of the turns it drops were already summarized.

`SessionStore` is the persistence interface; `SQLiteBackend` and
`InMemoryBackend` implement it. `ConversationSession` is the agent's side:
load once before the first run, flush after every run.
"""

from __future__ import annotations

import abc
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from agent_framework.core.message import AnyMessage, message_from_dict
from agent_framework.memory.short_term import LOG_SEQ_KEY, ContextState, ShortTermMemory


@dataclass
class SessionRecord:
    """The saved state of one session (plain data, safe to keep or change)."""

    history: list[dict[str, Any]]
    state: dict[str, Any]
    # Highest sequence number in the session's log (0 = nothing logged).
    last_seq: int = 0
    # Fingerprint of the system prompt and tool set the session ran with.
    fingerprint: str = ""
    # Log entries up to this sequence number are in the long-term summary.
    summarized_upto: int = 0
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class LogEntry:
    seq: int
    message: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SummarySegment:
    """One piece of a namespace's long-term summary, oldest first."""

    seq: int
    text: str


@dataclass(frozen=True)
class IdleTail:
    """A claimed, not yet summarized stretch of another session's log."""

    session_id: str
    after_seq: int  # summarized_upto before the claim
    upto_seq: int  # the session's last_seq, which the claim advanced it to


class SessionStore(abc.ABC):
    """Persistence for sessions. Implemented next to the long-term backend."""

    @abc.abstractmethod
    async def load_session(self, namespace: str, session_id: str) -> SessionRecord | None:
        """The saved session, or None if this namespace has no such session."""

    @abc.abstractmethod
    async def save_session(
        self,
        namespace: str,
        session_id: str,
        record: SessionRecord,
        new_log: list[LogEntry],
    ) -> None:
        """Append *new_log* and replace the saved state — atomically."""

    @abc.abstractmethod
    async def session_log(
        self, namespace: str, session_id: str, *, after_seq: int = 0
    ) -> list[LogEntry]:
        """The session's log entries with a sequence number above *after_seq*."""

    # ---- the namespace's long-term summary ----
    @abc.abstractmethod
    async def summary_segments(self, namespace: str) -> list[SummarySegment]:
        """The namespace's summary segments, oldest first."""

    @abc.abstractmethod
    async def append_summary(
        self,
        namespace: str,
        text: str,
        *,
        session_id: str | None = None,
        summarized_upto: int = 0,
    ) -> None:
        """Append a segment and, in the same transaction, advance *session_id*'s
        `summarized_upto` to at least *summarized_upto* — so a crash can never
        leave a summary whose messages still look unsummarized."""

    @abc.abstractmethod
    async def merge_summaries(
        self, namespace: str, first_seq: int, second_seq: int, text: str
    ) -> bool:
        """Replace two segments with *text* (at the first one's place).

        False, changing nothing, if either segment no longer exists — another
        writer merged them first.
        """

    @abc.abstractmethod
    async def claim_idle_session(
        self, namespace: str, *, idle_before: str, exclude_session: str
    ) -> IdleTail | None:
        """Claim the newest session idle since *idle_before* whose log runs
        past its `summarized_upto`, by advancing that marker to its last_seq.

        Atomic: of two callers, at most one gets the claim, so the same tail
        is never summarized twice.
        """

    @abc.abstractmethod
    async def release_claim(self, namespace: str, tail: IdleTail) -> None:
        """Undo a claim whose summary failed (only if nobody advanced it since)."""


def encode(value: Any) -> str:
    """JSON for storage. Values JSON cannot express are stored as text."""
    return json.dumps(value, default=str)


def tool_set_fingerprint(system_prompt: str, tools: Iterable[Any]) -> str:
    """What reasoning blocks are bound to besides the messages: the system
    prompt and the (name-sorted) tool definitions."""
    specs = sorted(
        (
            t.metadata.name,
            t.metadata.description,
            json.dumps(t.metadata.parameters_schema, sort_keys=True, default=str),
        )
        for t in tools
    )
    payload = json.dumps([system_prompt, specs], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class ConversationSession:
    """One agent's conversation in a `SessionStore`."""

    def __init__(self, store: SessionStore, *, namespace: str, session_id: str) -> None:
        self.store = store
        self.namespace = namespace
        self.session_id = session_id
        self._next_seq = 1
        self._summarized_upto = 0
        self._created_at = ""
        # Entries numbered before the next flush (see assign_seqs).
        self._pending_log: list[LogEntry] = []

    @property
    def summarized_upto(self) -> int:
        """Log entries up to this sequence number are in the long-term summary."""
        return self._summarized_upto

    def mark_summarized(self, upto: int) -> None:
        self._summarized_upto = max(self._summarized_upto, upto)

    def assign_seqs(self, messages: Iterable[AnyMessage]) -> None:
        """Give *messages* their log numbers now (in order) instead of at flush.

        A trim summarizes what it drops at once, and the summary must say
        which log entries it covers — even for messages the next flush has
        not written yet.
        """
        for message in messages:
            if LOG_SEQ_KEY in message.metadata:
                continue
            message.metadata[LOG_SEQ_KEY] = self._next_seq
            self._pending_log.append(LogEntry(seq=self._next_seq, message=message.to_dict()))
            self._next_seq += 1

    async def load(self, memory: ShortTermMemory, fingerprint: str) -> tuple[bool, bool]:
        """Restore *memory* from the store: ``(resumed, reasoning_dropped)``.

        Reasoning is bound to the system prompt and tool set it was produced
        with; if either changed since the session was saved, the restored
        history's reasoning is left out of the view (a leading run of blocks,
        so dropping it is always valid).
        """
        record = await self.store.load_session(self.namespace, self.session_id)
        if record is None:
            return False, False
        history = [message_from_dict(raw) for raw in record.history]
        state = ContextState.from_dict(record.state)
        changed = bool(record.fingerprint) and record.fingerprint != fingerprint
        dropped = False
        if changed and any(getattr(m, "reasoning", None) for m in history):
            state.reasoning_cut = len(history)
            dropped = True
        memory.replace_all(history, state=state)
        self._next_seq = record.last_seq + 1
        self._summarized_upto = record.summarized_upto
        self._created_at = record.created_at
        return True, dropped

    async def flush(self, memory: ShortTermMemory, fingerprint: str) -> int:
        """Log what is new and save the current state; returns entries logged.

        Messages are numbered tentatively: if the save fails (or is
        cancelled) the numbers are taken back, so the next flush logs the
        same messages again instead of skipping them as already logged.
        """
        seq = self._next_seq
        stamped: list[AnyMessage] = []
        new_log = list(self._pending_log)
        try:
            for message in [*memory.unlogged_dropped(), *memory.snapshot()]:
                if LOG_SEQ_KEY in message.metadata:
                    continue
                message.metadata[LOG_SEQ_KEY] = seq
                stamped.append(message)
                new_log.append(LogEntry(seq=seq, message=message.to_dict()))
                seq += 1
            record = SessionRecord(
                history=[m.to_dict() for m in memory.snapshot()],
                state=memory.state.to_dict(),
                last_seq=seq - 1,
                fingerprint=fingerprint,
                summarized_upto=self._summarized_upto,
                created_at=self._created_at,
            )
            await self.store.save_session(self.namespace, self.session_id, record, new_log)
        except BaseException:
            for message in stamped:
                message.metadata.pop(LOG_SEQ_KEY, None)
            raise
        self._next_seq = seq
        self._pending_log = []
        memory.forget_logged_dropped()
        return len(new_log)


def log_seq(message: AnyMessage) -> int | None:
    value = message.metadata.get(LOG_SEQ_KEY)
    return value if isinstance(value, int) else None
