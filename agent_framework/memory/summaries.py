"""Long-term summaries: what a conversation leaves behind for the next one.

A namespace's summary is a list of segments, each written by one LLM call:

  - when a trim drops turns, everything in the session's log after its
    `summarized_upto` marker up to the dropped turns becomes a segment — so
    turns a failed summary left behind are covered by the next one;
  - when a conversation starts empty, the *oldest* idle sessions of the
    namespace (no activity for `MemoryConfig.idle_after_minutes`) whose log
    runs past their last summary are summarized first, up to
    `idle_summaries_per_start` of them — so short conversations that never
    trimmed are remembered too, and none waits forever behind newer ones;
  - `await agent.memory.summarize_idle()` works through that backlog from a
    background job, and `await agent.memory.summarize()` summarizes the
    current conversation's unsummarized part on demand.

An idle tail is summarized under a lease: the claim holds a token for
`CLAIM_LEASE`, and the summary is written only if the token still holds and
the session's marker has not moved, in the same transaction that advances
the marker. Two new conversations never summarize the same tail, and a
holder that crashes mid-call only lets the lease expire.

Once the segments exceed 8,000 tokens, the two oldest are merged (one more
call). Summaries never run on a dry run, and every call is cost-tracked and
traced (COMPACTION "summary", "summary_failed", "summary_merge").
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from agent_framework.core.context_filter import CONTEXT_KEY
from agent_framework.core.message import (
    AnyMessage,
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
    estimate_tokens,
    message_from_dict,
)
from agent_framework.memory.sessions import (
    ConversationSession,
    IdleTail,
    LogEntry,
    SessionStore,
    log_seq,
    run_to_end,
)
from agent_framework.memory.short_term import ShortTermMemory
from agent_framework.tracing.trace import (
    ErrorInfo,
    EventType,
    ExecutionTrace,
    SpanStatus,
)

if TYPE_CHECKING:
    from agent_framework.providers.base import BaseProviderAdapter
    from agent_framework.tracing.cost import CostTracker

# The length each summary is asked for (less on small windows).
SUMMARY_TOKENS = 800
# Above this total the two oldest segments are merged — or above a fifth of
# max_context_tokens, whichever is smaller: the summary heads every new
# conversation and no trim can remove it, so it must leave the window room.
SUMMARY_BUDGET_TOKENS = 8_000
# A session without activity for this long is finished enough to summarize
# (the default of MemoryConfig.idle_after_minutes).
IDLE_AFTER = timedelta(minutes=30)
# How long a claim on an idle tail holds: long enough for one summary call
# with retries, short enough that a crashed holder does not block the tail.
CLAIM_LEASE = timedelta(minutes=10)
# The most of a conversation one summary call reads (head and tail kept).
_EXCERPT_CHARS = 120_000


def _summary_prompt(tokens: int) -> str:
    return (
        "You write the long-term memory of an assistant. Summarize the conversation "
        "excerpt below so that a later conversation can continue from it. Keep what "
        "stays true and useful: names, identifiers, decisions and their reasons, the "
        "user's preferences, open tasks, errors and how they were resolved, numbers, "
        "dates and file paths. Leave out greetings and anything that no longer matters. "
        f"Write plain prose, under {tokens} tokens, with no preamble."
    )


def _merge_prompt(tokens: int) -> str:
    return (
        "You maintain the long-term memory of an assistant. Merge the two summaries "
        "of earlier conversations below into one, older events first. Keep every "
        "fact that still matters; drop what the later summary shows is outdated. "
        f"Write plain prose, under {tokens} tokens, with no preamble."
    )


class Summarizer:
    """Writes the namespace's summary segments for one agent's session."""

    def __init__(
        self,
        *,
        provider: BaseProviderAdapter,
        model: str,
        session: ConversationSession,
        cost: CostTracker,
        trace: ExecutionTrace | None = None,
        auto: bool = True,
        budget_tokens: int = SUMMARY_BUDGET_TOKENS,
        idle_after: timedelta = IDLE_AFTER,
        idle_per_start: int = 2,
        retention: timedelta | None = None,
    ) -> None:
        self._provider = provider
        self.model = model
        self._session = session
        self._cost = cost
        self._trace = trace
        # False (MemoryConfig.summarize=False): no summary is written on its
        # own — not at trims, not for idle sessions. summarize() still works.
        self.auto = auto
        self.budget_tokens = max(1, min(budget_tokens, SUMMARY_BUDGET_TOKENS))
        # Each summary leaves room for at least one more before a merge.
        self.summary_tokens = max(100, min(SUMMARY_TOKENS, self.budget_tokens // 2))
        self.idle_after = idle_after
        self.idle_per_start = max(0, idle_per_start)
        self.retention = retention

    @property
    def _store(self) -> SessionStore:
        return self._session.store

    @property
    def _namespace(self) -> str:
        return self._session.namespace

    @property
    def _unusable(self) -> bool:
        return self._session.broken or self._session.taken_over

    async def segments(self) -> list[str]:
        """The newest summary segments that fit the budget, oldest first.

        Segments written by an agent with a larger window may exceed this
        one's budget; the newest are kept, and a single segment too large for
        it is cut.
        """
        segments = [s.text for s in await self._store.summary_segments(self._namespace)]
        fitting: list[str] = []
        used = 0
        for text in reversed(segments):
            tokens = estimate_tokens(text)
            if used + tokens > self.budget_tokens:
                if not fitting:
                    fitting.append(text[: self.budget_tokens * 4].rstrip() + " …")
                break
            fitting.append(text)
            used += tokens
        return list(reversed(fitting))

    # ------------------------------------------------------------ the paths
    async def after_trim(self, memory: ShortTermMemory, dropped: list[AnyMessage]) -> bool:
        """Summarize what a trim dropped. True if the summary in view changed.

        The dropped messages are numbered in the log now, so the summary can
        record exactly what it covers. It covers everything in the log after
        the session's marker up to the newest dropped message — including
        turns an earlier, failed summary left behind. Never raises: on a
        failure the marker stays, and the next summary covers the turns.
        """
        if not self.auto or self._unusable:
            return False
        # Numbered in history order: messages dropped (or cleared) before
        # they were logged come first, then what this trim dropped.
        self._session.assign_seqs([*memory.unlogged_dropped(), *dropped])
        upto = max((seq for m in dropped if (seq := log_seq(m)) is not None), default=0)
        after = self._session.summarized_upto
        if upto <= after:
            return False  # all of it was summarized already
        try:
            todo = await self._log_between(after, upto)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed("trim", exc)
            return False
        if not todo:
            return False
        text = await self._summarize(
            _summary_prompt(self.summary_tokens), _excerpt(todo), source="trim", messages=len(todo)
        )
        if text is None or not await self._append_for_session(text, upto=upto, source="trim"):
            return False
        state = memory.state.copy()
        try:
            await self._merge_if_needed()
            state.summary = await self.segments()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed("trim", exc)  # the summary is saved; the view keeps the old one
        memory.state = state
        return True

    async def on_conversation_start(self) -> None:
        """Housekeeping before a new conversation's first call.

        Summarizes up to `idle_per_start` idle tails of the namespace (see
        `summarize_idle`) and, with a retention period, deletes sessions idle
        for longer whose whole log is in the summary. Never raises except
        for cancellation.
        """
        if not self.auto or self._unusable:
            return
        if self.idle_per_start:
            await self.summarize_idle(self.idle_per_start, source="idle_session")
        if self.retention is not None:
            await self._prune()

    async def summarize_idle(self, limit: int, *, source: str = "idle_backlog") -> int:
        """Summarize up to *limit* idle unsummarized tails, oldest first.

        Each tail is leased before the call and released again on any
        failure — cancellation included — so it is neither summarized twice
        nor lost. Stops at the first failure: a provider that is down is not
        asked *limit* times. Never raises except for cancellation.
        """
        if self._unusable:
            return 0
        done = 0
        for _ in range(max(0, limit)):
            try:
                tail = await self._claim()
                if tail is None:
                    break
                if not await self._summarize_tail(tail, source=source):
                    break
                done += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._failed(source, exc)
                break
        await self._record_backlog(done)
        return done

    async def _record_backlog(self, done: int) -> None:
        """What the idle queue looks like after a pass, for operators."""
        if self._trace is None:
            return
        try:
            remaining = await self._store.idle_backlog(
                self._namespace,
                idle_before=(datetime.now(UTC) - self.idle_after).isoformat(),
                exclude_session=self._session.session_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed("idle_backlog", exc)
            return
        self._trace.record_event(
            EventType.MEMORY_READ,
            name="idle_backlog",
            payload={"namespace": self._namespace, "summarized": done, "remaining": remaining},
        )

    async def summarize_session(self) -> bool:
        """Summarize the current session's log since its last summary.

        The caller saves the session first, so the log is complete. If another
        session summarized part of it meanwhile, the marker is re-read and
        what is still new is summarized once more.
        """
        if self._unusable:
            return False
        # This conversation may have sat idle long enough for another session
        # to summarize its tail: start from the stored marker.
        await self._session.refresh_marker()
        for _ in range(2):
            entries = await self._store.session_log(
                self._namespace, self._session.session_id, after_seq=self._session.summarized_upto
            )
            messages = [message_from_dict(e.message) for e in entries]
            if not messages:
                return False
            text = await self._summarize(
                _summary_prompt(self.summary_tokens),
                _excerpt(messages),
                source="explicit",
                messages=len(messages),
            )
            if text is None:
                return False
            if await self._append_for_session(text, upto=entries[-1].seq, source="explicit"):
                await self._merge_if_needed()
                return True
        return False

    # -------------------------------------------------------------- helpers
    async def _log_between(self, after: int, upto: int) -> list[AnyMessage]:
        """Log entries with *after* < seq <= *upto*: the saved ones plus those
        numbered by a trim but not yet saved."""
        by_seq: dict[int, LogEntry] = {
            e.seq: e for e in self._session.pending_entries() if after < e.seq <= upto
        }
        for entry in await self._store.session_log(
            self._namespace, self._session.session_id, after_seq=after
        ):
            if entry.seq <= upto:
                by_seq[entry.seq] = entry
        return [message_from_dict(by_seq[seq].message) for seq in sorted(by_seq)]

    async def _claim(self) -> IdleTail | None:
        now = datetime.now(UTC)
        tail, cancelled = await run_to_end(
            self._store.claim_idle_session(
                self._namespace,
                idle_before=(now - self.idle_after).isoformat(),
                exclude_session=self._session.session_id,
                lease_until=(now + CLAIM_LEASE).isoformat(),
            )
        )
        if cancelled:
            if tail is not None:
                await run_to_end(self._store.release_claim(self._namespace, tail))
            raise asyncio.CancelledError()
        return tail

    async def _summarize_tail(self, tail: IdleTail, *, source: str) -> bool:
        """One leased tail: read, summarize, append under the lease."""
        appended = False
        try:
            entries = await self._store.session_log(
                self._namespace, tail.session_id, after_seq=tail.after_seq
            )
            messages = [message_from_dict(e.message) for e in entries if e.seq <= tail.upto_seq]
            if not messages:
                self._failed(source, RuntimeError(f"session {tail.session_id!r} has no log to summarize"))
                return False
            text = await self._summarize(
                _summary_prompt(self.summary_tokens),
                _excerpt(messages),
                source=source,
                messages=len(messages),
                session_id=tail.session_id,
            )
            if text is None:
                return False
            ok, cancelled = await run_to_end(
                self._store.append_summary(
                    self._namespace,
                    text,
                    session_id=tail.session_id,
                    expected_upto=tail.after_seq,
                    claim_token=tail.token,
                    summarized_upto=tail.upto_seq,
                )
            )
            appended = bool(ok)
            if cancelled:
                raise asyncio.CancelledError()
            if not appended:
                self._failed(
                    source,
                    RuntimeError(
                        f"the lease on session {tail.session_id!r} was lost before the "
                        "summary was written; summary discarded"
                    ),
                )
                return False
        finally:
            if not appended:
                await run_to_end(self._store.release_claim(self._namespace, tail))
        await self._merge_if_needed()
        return True

    async def _prune(self) -> None:
        assert self.retention is not None
        try:
            pruned, cancelled = await run_to_end(
                self._store.prune_sessions(
                    self._namespace,
                    idle_before=(datetime.now(UTC) - self.retention).isoformat(),
                    summarized_only=True,
                )
            )
            if pruned and self._trace is not None:
                self._trace.record_event(
                    EventType.MEMORY_WRITE,
                    name="prune_sessions",
                    payload={"namespace": self._namespace, "pruned": pruned},
                )
            if cancelled:
                raise asyncio.CancelledError()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed("prune", exc)

    async def _append_for_session(self, text: str, *, upto: int, source: str) -> bool:
        """Append *text* as covering this session's log up to *upto*, unless
        another session got there first.

        The append is compare-and-set on this session's stored marker (and
        refused while another session holds a lease on it); on a conflict the
        marker is re-read, so the caller sees what is still unsummarized, and
        nothing is written. Errors are traced.
        """
        expected = self._session.summarized_upto
        try:
            appended, cancelled = await run_to_end(
                self._store.append_summary(
                    self._namespace,
                    text,
                    session_id=self._session.session_id,
                    expected_upto=expected,
                    summarized_upto=upto,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed(source, exc)
            return False
        if appended:
            self._session.mark_summarized(upto)
        if cancelled:
            raise asyncio.CancelledError()
        if not appended:
            self._failed(
                source,
                RuntimeError("another session is summarizing this one; summary discarded"),
            )
            try:
                await self._session.refresh_marker()
            except Exception as exc:
                self._failed(source, exc)
        return appended

    async def _merge_if_needed(self) -> None:
        segments = await self._store.summary_segments(self._namespace)
        total = sum(estimate_tokens(s.text) for s in segments)
        if total <= self.budget_tokens or len(segments) < 2:
            return
        first, second = segments[0], segments[1]
        text = await self._summarize(
            _merge_prompt(self.summary_tokens),
            f"<earlier>\n{first.text}\n</earlier>\n\n<later>\n{second.text}\n</later>",
            source="merge",
            messages=2,
        )
        if text is not None:
            _, cancelled = await run_to_end(
                self._store.merge_summaries(self._namespace, first.seq, second.seq, text)
            )
            if cancelled:
                raise asyncio.CancelledError()

    async def _summarize(
        self,
        prompt: str,
        excerpt: str,
        *,
        source: str,
        messages: int,
        session_id: str | None = None,
    ) -> str | None:
        """One LLM call. None (traced) on any failure or an empty answer."""
        try:
            response = await self._provider.call(
                system=prompt,
                messages=[UserMessage(content=excerpt)],
                tools=[],
                model=self.model,
                # Room for reasoning on top of the summary itself.
                max_tokens=max(1_024, self.summary_tokens * 4),
                temperature=None,
                enable_caching=False,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed(source, exc)
            return None
        cost = self._cost.record(self.model, response.usage)
        content = response.assistant_message.content
        text = content.strip() if isinstance(content, str) else ""
        if not text:
            self._failed(source, ValueError("the summarizer returned no text"))
            return None
        limit = self.summary_tokens * 8  # twice what was asked, in characters
        if len(text) > limit:
            text = text[:limit].rstrip() + " …"
        if self._trace is not None:
            payload = {
                "source": source,
                "messages": messages,
                "summary_tokens": estimate_tokens(text),
                "model": self.model,
            }
            if session_id is not None:
                payload["session_id"] = session_id
            self._trace.record_event(
                EventType.COMPACTION,
                name="summary_merge" if source == "merge" else "summary",
                payload=payload,
                token_usage=response.usage,
                estimated_cost_usd=cost,
            )
        return text

    def _failed(self, source: str, exc: BaseException) -> None:
        if self._trace is not None:
            self._trace.record_event(
                EventType.COMPACTION,
                name="summary_failed",
                status=SpanStatus.ERROR,
                payload={"source": source},
                error=ErrorInfo.from_exception(exc),
            )


def _excerpt(messages: list[AnyMessage]) -> str:
    """The conversation as plain lines for the summarizer."""
    lines: list[str] = []
    for message in messages:
        if message.metadata.get(CONTEXT_KEY):
            continue  # injected memory context: already long-term
        if isinstance(message, ToolResultMessage):
            for result in message.content:
                tag = f"{result.name} error" if result.is_error else result.name
                lines.append(f"[{tag}] {_clip(result.as_text(), 600)}")
        elif isinstance(message, AssistantMessage):
            if isinstance(message.content, str) and message.content:
                lines.append(f"[assistant] {message.content}")
            for call in message.tool_calls:
                lines.append(
                    f"[assistant → {call.name}] {_clip(json.dumps(call.arguments, default=str), 300)}"
                )
        elif isinstance(message, UserMessage):
            content = message.content if isinstance(message.content, str) else json.dumps(
                message.content, default=str
            )
            lines.append(f"[user] {content}")
    text = "\n".join(lines)
    if len(text) <= _EXCERPT_CHARS:
        return text
    half = _EXCERPT_CHARS // 2
    return f"{text[:half]}\n[... middle of the conversation omitted ...]\n{text[-half:]}"


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " …"
