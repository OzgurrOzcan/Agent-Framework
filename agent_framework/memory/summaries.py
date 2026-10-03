"""Long-term summaries: what a conversation leaves behind for the next one.

A namespace's summary is a list of segments, each written by one LLM call:

  - when a trim drops turns, those turns become a segment (unless they are
    already covered — see `ConversationSession.summarized_upto`);
  - when a conversation starts empty, the newest *idle* session of the
    namespace (no activity for 30 minutes) whose log runs past its last
    summary is summarized first — so short conversations that never trimmed
    are remembered too; it is claimed atomically, so two new sessions never
    summarize it twice;
  - `await agent.memory.summarize()` summarizes the current conversation's
    unsummarized part on demand.

Once the segments exceed 8,000 tokens, the two oldest are merged (one more
call). A failed summary keeps the dropped turns in `ContextState.pending`, and
the next trim tries again. Summaries never run on a dry run, and every call
is cost-tracked and traced (COMPACTION "summary", "summary_failed",
"summary_merge").
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
# A session without activity for this long is finished enough to summarize.
IDLE_AFTER = timedelta(minutes=30)
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

    @property
    def _store(self) -> SessionStore:
        return self._session.store

    @property
    def _namespace(self) -> str:
        return self._session.namespace

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
        record exactly what it covers; messages an earlier summary already
        covered are skipped. Never raises: a failure keeps the turns pending.
        """
        if not self.auto or self._session.broken:
            return False
        self._session.assign_seqs(dropped)
        state = memory.state.copy()
        todo = self._unsummarized([*state.pending, *dropped])
        if not todo:
            return False
        appended = False
        text = await self._summarize(
            _summary_prompt(self.summary_tokens), _excerpt(todo), source="trim", messages=len(todo)
        )
        if text is not None:
            appended = await self._append_for_session(text, todo, source="trim")
        if not appended:
            # Kept for the next trim (minus anything another session covered
            # meanwhile); already numbered, so never lost from the log.
            state.pending = self._unsummarized(todo)
            memory.state = state
            return False
        state.pending = []
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
        """Summarize the newest idle session's unsummarized tail, if any.

        The tail is claimed before the call and released again on any
        failure — cancellation included — so it is neither summarized twice
        nor lost. Never raises except for cancellation.
        """
        if not self.auto or self._session.broken:
            return
        try:
            idle_before = (datetime.now(UTC) - IDLE_AFTER).isoformat()
            tail, cancelled = await run_to_end(
                self._store.claim_idle_session(
                    self._namespace,
                    idle_before=idle_before,
                    exclude_session=self._session.session_id,
                )
            )
            if tail is None:
                if cancelled:
                    raise asyncio.CancelledError()
                return
            appended = False
            try:
                if cancelled:
                    raise asyncio.CancelledError()
                entries = await self._store.session_log(
                    self._namespace, tail.session_id, after_seq=tail.after_seq
                )
                messages = [
                    message_from_dict(e.message) for e in entries if e.seq <= tail.upto_seq
                ]
                text = (
                    await self._summarize(
                        _summary_prompt(self.summary_tokens),
                        _excerpt(messages),
                        source="idle_session",
                        messages=len(messages),
                    )
                    if messages
                    else None
                )
                if text is not None:
                    _, cancelled = await run_to_end(self._store.append_summary(self._namespace, text))
                    appended = True
                    if cancelled:
                        raise asyncio.CancelledError()
            finally:
                if not appended:
                    await run_to_end(self._store.release_claim(self._namespace, tail))
            await self._merge_if_needed()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed("idle_session", exc)

    async def summarize_session(self) -> bool:
        """Summarize the current session's log since its last summary.

        The caller saves the session first, so the log is complete. If another
        session summarized part of it meanwhile, the marker is re-read and
        what is still new is summarized once more.
        """
        if self._session.broken:
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
            if await self._append_for_session(text, messages, source="explicit", upto=entries[-1].seq):
                await self._merge_if_needed()
                return True
        return False

    # -------------------------------------------------------------- helpers
    def _unsummarized(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        upto = self._session.summarized_upto
        return [m for m in messages if (seq := log_seq(m)) is None or seq > upto]

    async def _append_for_session(
        self,
        text: str,
        covered: list[AnyMessage],
        *,
        source: str,
        upto: int | None = None,
    ) -> bool:
        """Append *text* as covering *covered*, unless another session got there first.

        The append is compare-and-set on this session's stored marker; on a
        conflict the marker is re-read (so the caller sees what is still
        unsummarized) and nothing is written. Errors are traced.
        """
        if upto is None:
            upto = max((seq for m in covered if (seq := log_seq(m)) is not None), default=0)
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
                RuntimeError("another session summarized this one meanwhile; summary discarded"),
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

    async def _summarize(self, prompt: str, excerpt: str, *, source: str, messages: int) -> str | None:
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
            self._trace.record_event(
                EventType.COMPACTION,
                name="summary_merge" if source == "merge" else "summary",
                payload={
                    "source": source,
                    "messages": messages,
                    "summary_tokens": estimate_tokens(text),
                    "model": self.model,
                },
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
