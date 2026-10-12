"""MemoryManager — coordinates short-term + long-term memory.

Owns the working buffer (`ShortTermMemory`) and finds the long-term facts
relevant to a user message, which the context manager stores in the history
once per run.

Facts are matched by whole words in any script: "faturalar" finds a fact
that says "faturalar", "art" does not find "party", and the words inside a
key ("user_name", "deployTarget") count as words. A backend that can search
(SQLite with FTS5) does the matching itself, over every fact and with
diacritics folded, so "iletisim" finds "iletişim"; any other backend has its
facts matched here in Python. Words of one or two letters are ignored unless
written in capitals ("DB", "CI") or part of a key.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Iterable

from agent_framework.memory.long_term import LongTermMemory
from agent_framework.memory.sessions import SessionInfo, SessionStore
from agent_framework.memory.short_term import ShortTermMemory

if TYPE_CHECKING:
    from agent_framework.tracing.trace import ExecutionTrace

# Letters and digits of any script; underscores separate words in keys.
_WORD_RX = re.compile(r"[^\W_]+")
# A lower-case letter or digit followed by an upper-case one: deployTarget.
_CAMEL_RX = re.compile(r"(?<=[^\W_])(?=[A-Z])")
# Shorter words ("is", "ve", "de") match far too much to mean anything —
# unless written in capitals (an acronym: "DB", "CI") or taken from a key.
_MIN_WORD = 3
_MIN_KEY_WORD = 2
# How many facts a backend without search has matched in Python.
FACT_SCAN_LIMIT = 1_000


class MemoryManager:
    def __init__(
        self,
        *,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
        recent_query_limit: int = 8,
        owns_backend: bool = False,
        trace: ExecutionTrace | None = None,
    ) -> None:
        self.short_term = short_term or ShortTermMemory()
        self.long_term = long_term
        self.recent_query_limit = recent_query_limit
        # True when this manager built its backend (from MemoryConfig.store)
        # and so must close it; a backend passed in belongs to the caller.
        self._owns_backend = owns_backend
        self._trace = trace
        self._summarize: Callable[[], Awaitable[bool]] | None = None
        self._summarize_idle: Callable[[int], Awaitable[int]] | None = None

    def set_summarize(
        self,
        summarize: Callable[[], Awaitable[bool]],
        summarize_idle: Callable[[int], Awaitable[int]] | None = None,
    ) -> None:
        """Wire `summarize()` and `summarize_idle()` to the agent that owns this memory."""
        self._summarize = summarize
        self._summarize_idle = summarize_idle

    async def summarize(self) -> bool:
        """Summarize this conversation into long-term memory now.

        Covers what the conversation added since its last summary — a short
        conversation that never trimmed included — with one LLM call, and
        returns True if a summary was written. Call it between runs; without
        a long-term store it does nothing and returns False.
        """
        if self._summarize is None:
            return False
        return await self._summarize()

    async def summarize_idle(self, limit: int = 10) -> int:
        """Summarize up to *limit* idle conversations of this namespace that
        the summary does not cover yet, oldest first, one LLM call each.

        This is what a new conversation does for a couple of them before its
        first call (`MemoryConfig.idle_summaries_per_start`); a background
        job calling this keeps the backlog at zero and the first call fast.
        Returns how many were summarized. Without a long-term store: 0.
        """
        if self._summarize_idle is None:
            return 0
        return await self._summarize_idle(limit)

    async def close(self) -> None:
        """Release the long-term store (the SQLite connection). Idempotent.

        A backend passed as `Agent(memory_backend=...)` is left open: it
        belongs to the caller, who may share it between agents.
        """
        if self.long_term is not None and self._owns_backend:
            await self.long_term.backend.close()

    # ---- relevant context (Step 4 of context assembly) ----
    async def relevant_facts(self, current_user_input: str) -> list[dict[str, Any]]:
        """The facts that share words with *current_user_input*, best first.

        A backend with search ranks them itself; otherwise the facts sharing
        the most whole words win, ties going to the most recently updated
        fact (the listing is newest first and the sort is stable).
        """
        if self.long_term is None or not current_user_input:
            return []
        query = words(current_user_input)
        if not query:
            return []
        matched = await self.long_term.backend.match_facts(
            query, namespace=self.long_term.namespace, limit=self.recent_query_limit
        )
        if matched is not None:
            return [{"key": f.key, "value": f.value, "tags": list(f.tags)} for f in matched]
        all_facts = await self.long_term.search(limit=FACT_SCAN_LIMIT)
        scored: list[tuple[int, dict[str, Any]]] = []
        for f in all_facts:
            hay = words(f.key, split_case=True) | words(f.value) | words(" ".join(f.tags))
            score = len(query & hay)
            if score == 0:
                continue
            scored.append((score, {"key": f.key, "value": f.value, "tags": list(f.tags)}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[: self.recent_query_limit]]

    # ---- the long-term summary ----
    async def rolling_summary(self) -> str:
        """The namespace's long-term summary as one text ("" without one).

        With a store that keeps sessions this is the summary segments the
        agent writes, joined; a facts-only backend has a single text.
        """
        if self.long_term is None:
            return ""
        store = self._session_store()
        if store is not None:
            segments = await store.summary_segments(self.long_term.namespace)
            return "\n\n".join(s.text for s in segments)
        return await self.long_term.get_summary()

    async def fold_into_summary(self, addition: str) -> None:
        """Add *addition* to the long-term summary (no-op without one)."""
        if self.long_term is None or not addition.strip():
            return
        store = self._session_store()
        if store is not None:
            await store.append_summary(self.long_term.namespace, addition.strip())
        else:
            await self.long_term.append_to_summary(addition)

    # ---- facts ----
    async def remember(
        self, key: str, value: str, *, tags: Iterable[str] = ()
    ) -> None:
        """Store a fact in long-term memory (no-op without one)."""
        if self.long_term is not None:
            await self.long_term.remember(key, value, tags=tags)
            self._record_write("remember", key=key)

    async def forget(self, key: str) -> None:
        """Remove a fact from long-term memory (no-op without one)."""
        if self.long_term is not None:
            await self.long_term.forget(key)
            self._record_write("forget", key=key)

    # ---- sessions (housekeeping) ----
    async def sessions(self, *, idle_for: timedelta | None = None, limit: int = 100) -> list[SessionInfo]:
        """The namespace's saved conversations, most recently active first;
        with *idle_for*, only those inactive for at least that long."""
        store = self._session_store()
        if store is None or self.long_term is None:
            return []
        idle_before = _iso(datetime.now(UTC) - idle_for) if idle_for is not None else None
        return await store.list_sessions(self.long_term.namespace, idle_before=idle_before, limit=limit)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a saved conversation and its log. False if there was none."""
        store = self._session_store()
        if store is None or self.long_term is None:
            return False
        deleted = await store.delete_session(self.long_term.namespace, session_id)
        if deleted:
            self._record_write("delete_session", session_id=session_id)
        return deleted

    async def prune_sessions(self, *, idle_for: timedelta, summarized_only: bool = True) -> int:
        """Delete the conversations inactive for at least *idle_for*.

        With *summarized_only* (the default) a conversation the summary does
        not fully cover is kept — nothing is forgotten before it is
        remembered. Returns how many were deleted. `MemoryConfig.retention_days`
        does this on its own at every new conversation.
        """
        store = self._session_store()
        if store is None or self.long_term is None:
            return 0
        pruned = await store.prune_sessions(
            self.long_term.namespace,
            idle_before=_iso(datetime.now(UTC) - idle_for),
            summarized_only=summarized_only,
        )
        if pruned:
            self._record_write("prune_sessions", pruned=pruned)
        return pruned

    def _session_store(self) -> SessionStore | None:
        if self.long_term is not None and isinstance(self.long_term.backend, SessionStore):
            return self.long_term.backend
        return None

    def _record_write(self, name: str, **payload: Any) -> None:
        if self._trace is not None:
            from agent_framework.tracing.trace import EventType

            namespace = self.long_term.namespace if self.long_term is not None else None
            self._trace.record_event(
                EventType.MEMORY_WRITE, name=name, payload={"namespace": namespace, **payload}
            )


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def words(text: str, *, split_case: bool = False) -> set[str]:
    """The comparable words of *text*: Unicode-aware, case-folded.

    NFKC folds compatibility forms (full-width letters, ligatures) and
    casefold() handles scripts where lower() falls short. The combining dot
    that casefold() leaves on a dotted capital I ("İ" → "i̇") is removed, so
    "İstanbul" and "istanbul" are one word. Words under three letters are
    dropped, except capitalized ones ("DB", "CI") — and, with *split_case*
    (a key, which also splits camelCase such as "deployTarget"), two-letter
    ones: a key's words are deliberate.
    """
    if split_case:
        text = _CAMEL_RX.sub(" ", text)
    minimum = _MIN_KEY_WORD if split_case else _MIN_WORD
    out: set[str] = set()
    for token in _WORD_RX.findall(unicodedata.normalize("NFKC", text)):
        folded = token.casefold().replace("̇", "")
        if len(folded) >= minimum or (len(folded) >= _MIN_KEY_WORD and token.isupper()):
            out.add(folded)
    return out
