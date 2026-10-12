"""ContextManager — what the model sees on every call.

The history in `ShortTermMemory` is stored in full and only grows between
trims. Before every LLM call the context manager renders the *view* — what is
actually sent — from that history and its `ContextState`:

    [summary]? [pinned first message]? history
                     (old tool results in the cleared zone become a one-line
                      note; reasoning before the cut is left out)

and estimates the prompt: system prompt + tool definitions + messages.

Trimming makes no LLM call and happens only when the estimate exceeds
`max_context_tokens` — then deeply, down to 60 % of the limit — so the prompt
cache is rebuilt once per trim instead of on every call:

  1. Clear: all but the newest k tool results become a byte-stable note.
     Errors, small results, `ask_user` and tools declared with
     `clear_old_results=False` are never cleared.
  2. Drop: if still above 60 %, whole oldest turns go; the conversation's
     first user message is pinned and the newest three turns always stay.
  3. Reasoning: if 1 or 2 changed anything, reasoning before the current turn
     is left out — models that bind reasoning to the conversation prefix
     reject it once that prefix changed, and most others ignore old
     reasoning. Not on a model that *requires* it (Kimi: a kept tool-call
     message without its reasoning is rejected, editing earlier messages is
     fine) — there steps 1 and 2 do all the trimming.

Afterwards the view only grows again. A trim that cannot get usefully below
the limit (a huge current turn or pinned message) changes nothing, and is
recorded once rather than on every call.

Token estimates are anchored on the prompt size the provider reported for the
previous call, plus a calibrated chars/4 estimate of what was appended since.
The anchor is dropped whenever the history is trimmed or replaced. The
calibration factor is an EWMA of actual/estimated, because chars/4 drifts
badly on non-English text and code.

Nothing already sent ever changes between trims — the prompt cache depends
on it, and models that bind reasoning to the prefix reject the next request
otherwise. So memory context is never generated per call: facts relevant to
a run are stored in the history once, just before the user's message
(`start_run`), and the long-term summary is a snapshot in the state, shown at
the head of the view.
"""

from __future__ import annotations

import asyncio
import bisect
import copy
import dataclasses
import inspect
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_framework.core.message import (
    AnyMessage,
    AssistantMessage,
    ToolResult,
    ToolResultMessage,
    UserMessage,
    estimate_message_tokens,
    estimate_tokens,
)
from agent_framework.core.context_filter import (
    CONTEXT_KEY,
    ContextFilter,
    ContextInfo,
    FilterOutcome,
    current_turn_start,
    repair,
    same_messages,
    validate_output,
    without_reasoning_before,
)
from agent_framework.memory.short_term import ContextState, ShortTermMemory
from agent_framework.tools.builtin.ask_user import ASK_USER_TOOL_NAME
from agent_framework.tracing.trace import ErrorInfo, EventType, ExecutionTrace, SpanStatus

if TYPE_CHECKING:
    from agent_framework.memory.manager import MemoryManager
    from agent_framework.memory.summaries import Summarizer
    from agent_framework.tools.registry import ToolRegistry


# What a cleared tool result is replaced with. Derived only from the result
# itself, so the note is byte-identical on every call (prompt cache) and the
# tool_use / tool_result pair stays intact.
CLEAR_NOTE = (
    "[cleared to save context: {name} returned ≈{tokens} tokens. "
    "Call the tool again if you need it.]"
)

# A trim aims this far below the limit, so the view can grow for a while
# before the next one: one cache rebuild per trim, not one per call.
TRIM_TARGET_SHARE = 0.60
# The newest tool results that stay complete (fewer on small windows).
KEEP_RECENT_RESULTS = 5
# The newest turns that are never dropped; the current turn is one of them.
KEEP_RECENT_TURNS = 3
# Results this small are never cleared: the note would save next to nothing.
PROTECTED_RESULT_TOKENS = 64
# A trim that cannot reach the target is applied only if it leaves at least
# this share of the limit free; a smaller gain would trigger the next trim —
# and the next cache rebuild — almost immediately.
MIN_TRIM_HEADROOM = 0.10

_CALIBRATION_BOUNDS = (0.5, 4.0)
# A reported prompt size this far from the estimate does not describe the
# prompt that was sent (a proxy reporting placeholder usage, a provider
# counting something else). chars/4 is rough, but not off by more than this
# — even dense CJK text or highly repetitive output stays within it. Such a
# sample is ignored: trusting it would anchor every later estimate on noise
# and silently stop the trim from firing.
_PLAUSIBLE_RATIO = (0.2, 10.0)


@dataclass(frozen=True)
class TrimOutcome:
    """What one trim did (or why it did nothing), for traces and reports."""

    applied: bool
    estimated_before: int
    estimated_after: int
    limit: int
    target: int
    cleared_results: int = 0
    dropped_messages: int = 0
    dropped_turns: int = 0
    pinned: bool = False
    reasoning_cut: bool = False

    @property
    def reached_target(self) -> bool:
        return self.estimated_after <= self.target

    def payload(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "estimated_before": self.estimated_before,
            "estimated_after": self.estimated_after,
            "reclaimed_tokens": self.estimated_before - self.estimated_after,
            "limit": self.limit,
            "target": self.target,
            "reached_target": self.reached_target,
            "cleared_results": self.cleared_results,
            "dropped_messages": self.dropped_messages,
            "dropped_turns": self.dropped_turns,
            "pinned_first_message": self.pinned,
            "reasoning_cut": self.reasoning_cut,
        }


@dataclass
class AssembledContext:
    system_prompt: str
    messages: list[AnyMessage]
    estimated_tokens: int
    # Set when this call's estimate exceeded the limit.
    trim: TrimOutcome | None = None
    # Messages in the stored history (the view adds summary/pinned ones).
    messages_stored: int = 0
    # Tool results shown as a "cleared" note in this view.
    cleared_results: int = 0
    # Assistant messages whose reasoning this view leaves out.
    reasoning_dropped: int = 0
    # Whether a context_filter changed what is sent.
    filtered: bool = False


@dataclass(frozen=True)
class ContextReport:
    """What one LLM call was sent, and how the provider's cache served it.

    Recorded on every LLM_CALL span and returned as `AgentResult.context` for
    the run's last call (or, with `run(dry_run=True)`, for the call that
    would be made).
    """

    messages_sent: int
    messages_stored: int
    estimated_tokens: int
    max_context_tokens: int
    cleared_results: int
    reasoning_dropped: int
    trimmed: bool
    # Whether a context_filter changed what is sent.
    filtered: bool = False
    # Provider-reported; zero until the call has been made.
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def cache_hit_ratio(self) -> float:
        """Share of the prompt served from the provider's cache (0.0–1.0)."""
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages_sent": self.messages_sent,
            "messages_stored": self.messages_stored,
            "estimated_tokens": self.estimated_tokens,
            "max_context_tokens": self.max_context_tokens,
            "cleared_results": self.cleared_results,
            "reasoning_dropped": self.reasoning_dropped,
            "trimmed": self.trimmed,
            "filtered": self.filtered,
            "input_tokens": self.input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_hit_ratio": round(self.cache_hit_ratio, 4),
        }


@dataclass
class _RenderStats:
    cleared_results: int = 0
    reasoning_dropped: int = 0


@dataclass(frozen=True)
class _Anchor:
    """The provider-reported size of the previous call's prompt."""

    epoch: int
    view_len: int  # view messages that call covered
    tokens: int  # reported prompt size
    tools_raw: int  # raw tool-definition estimate at that call


@dataclass(frozen=True)
class _PendingCall:
    """What the call about to be made was built from, for `note_actual_tokens`."""

    epoch: int
    view_len: int
    raw_total: int
    tools_raw: int


class ContextManager:
    def __init__(
        self,
        *,
        system_prompt: str,
        max_context_tokens: int,
        max_tool_result_tokens: int,
        tools: ToolRegistry | None = None,
        memory: MemoryManager | None = None,
        trace: ExecutionTrace | None = None,
        context_filter: ContextFilter | None = None,
        binds_reasoning: bool = False,
        keeps_reasoning: bool = False,
        model: str = "",
        summarizer: Summarizer | None = None,
        usable_tokens: int | None = None,
    ) -> None:
        if max_context_tokens < 1 or max_tool_result_tokens < 1:
            raise ValueError("max_context_tokens and max_tool_result_tokens must be >= 1")
        self._system_prompt = system_prompt
        self._system_raw = estimate_tokens(system_prompt)
        self._limit = max_context_tokens
        self._result_cap = max_tool_result_tokens
        # The most the prompt may use at all (window minus the response). A
        # trim that brings an over-full prompt within it is always worth it.
        self._usable = usable_tokens if usable_tokens is not None else max_context_tokens
        self._tools = tools
        self._memory = memory
        self._trace = trace
        # EWMA correction factor: actual_tokens / estimated_tokens.
        self._calibration = 1.0
        self._anchor: _Anchor | None = None
        self._pending: _PendingCall | None = None
        # Epoch at which an unsatisfiable trim was last recorded.
        self._noop_epoch: int | None = None
        self._tools_raw_cache: tuple[tuple[str, ...], int] | None = None
        self._summary_cache: tuple[tuple[str, ...], UserMessage] | None = None
        # (epoch, tokens) the provider cleared server-side at the last call.
        self._provider_cleared: tuple[int, int] = (0, 0)
        self._filter = context_filter
        # Long-term summaries (a store that keeps sessions): what a trim drops
        # is summarized, and a new conversation starts with the summary.
        self._summarizer = summarizer
        # Reasoning is invalidated by editing earlier messages: nothing
        # already sent may change mid-run, so trims wait for a run's start.
        self._binds_reasoning = binds_reasoning
        # The opposite case: reasoning is not invalidated by a trim but a
        # kept tool-call message without it is rejected (Kimi), so a trim
        # clears results and drops turns but never cuts reasoning.
        self._keeps_reasoning = keeps_reasoning and not binds_reasoning
        self._model = model
        # What the last call actually sent, and the view it was made from —
        # with a filter, the two differ, and the sent one is what the
        # model's reasoning is bound to.
        self._last_sent: list[AnyMessage] | None = None
        self._last_view_len = 0
        self._last_epoch = 0
        # This run's first call left out reasoning before the current turn: the
        # filter's later candidates are compared with that applied too.
        self._stripped_this_run = False

    # ------------------------------------------------------------ properties
    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def calibration(self) -> float:
        return self._calibration

    @property
    def max_context_tokens(self) -> int:
        return self._limit

    @property
    def trim_target(self) -> int:
        return int(self._limit * TRIM_TARGET_SHARE)

    def keep_recent_results(self) -> int:
        """How many of the newest clearable results a trim keeps complete.

        Five, unless five results at the cap would take more than half of the
        trim target — then fewer (never less than one), so that a small
        window can still reach the target.
        """
        fit = (self.trim_target // 2) // self._result_cap
        return max(1, min(KEEP_RECENT_RESULTS, fit))

    # ------------------------------------------------------------- the view
    def render_view(
        self,
        history: list[AnyMessage],
        state: ContextState,
        stats: _RenderStats | None = None,
    ) -> list[AnyMessage]:
        """What the model sees for *history* under *state*. Pure; never mutates."""
        view: list[AnyMessage] = []
        if state.summary:
            view.append(self._summary_message(state.summary))
        if state.pinned is not None:
            view.append(state.pinned)
        view.extend(self._render_history(history, state, stats))
        return view

    def _summary_message(self, segments: list[str]) -> UserMessage:
        # Cached so that every render returns the very same message.
        key = tuple(segments)
        if self._summary_cache is None or self._summary_cache[0] != key:
            body = "\n\n".join(segments)
            message = UserMessage(
                content=(
                    "<conversation_summary>\nSummary of earlier conversations:\n\n"
                    f"{body}\n</conversation_summary>"
                ),
                metadata={CONTEXT_KEY: "summary"},
            )
            self._summary_cache = (key, message)
        return self._summary_cache[1]

    def _render_history(
        self,
        history: list[AnyMessage],
        state: ContextState,
        stats: _RenderStats | None = None,
    ) -> list[AnyMessage]:
        stats = stats if stats is not None else _RenderStats()
        out: list[AnyMessage] = []
        ordinal = 0  # tool results seen so far, in history order
        for index, message in enumerate(history):
            if isinstance(message, ToolResultMessage):
                rendered: list[ToolResult] = []
                changed = False
                for result in message.content:
                    if ordinal < state.cleared_results and self._clearable(result):
                        rendered.append(_cleared(result))
                        stats.cleared_results += 1
                        changed = True
                    else:
                        rendered.append(result)
                    ordinal += 1
                out.append(dataclasses.replace(message, content=rendered) if changed else message)
            elif (
                isinstance(message, AssistantMessage)
                and index < state.reasoning_cut
                and message.reasoning
            ):
                out.append(dataclasses.replace(message, reasoning=[]))
                stats.reasoning_dropped += 1
            else:
                out.append(message)
        return out

    def _clearable(self, result: ToolResult) -> bool:
        if result.is_error or result.name == ASK_USER_TOOL_NAME:
            return False
        if estimate_tokens(result.as_text()) <= PROTECTED_RESULT_TOKENS:
            return False
        tool = self._tools.get(result.name) if self._tools is not None else None
        return tool is None or tool.metadata.clear_old_results

    # ------------------------------------------------------------ estimates
    def _tools_raw(self) -> int:
        """Raw token estimate of the tool definitions sent with every call."""
        if self._tools is None:
            return 0
        names = tuple(self._tools.names)
        if self._tools_raw_cache is not None and self._tools_raw_cache[0] == names:
            return self._tools_raw_cache[1]
        total = 0
        for t in self._tools:
            meta = t.metadata
            spec = {
                "name": meta.name,
                "description": meta.description,
                "input_schema": meta.parameters_schema,
            }
            total += 8 + estimate_tokens(json.dumps(spec, default=str))
        self._tools_raw_cache = (names, total)
        return total

    def _calibrated(self, raw: int) -> int:
        return int(raw * self._calibration)

    def estimate(self, memory: ShortTermMemory, view: list[AnyMessage]) -> int:
        """Estimated prompt size for sending *view* rendered from *memory*.

        Anchored on the previous call's reported size when the history has
        only grown since; otherwise the calibrated estimate of everything.
        """
        tools_raw = self._tools_raw()
        anchor = self._anchor
        if anchor is not None and anchor.epoch == memory.epoch and anchor.view_len <= len(view):
            appended = _raw(view[anchor.view_len :])
            return anchor.tokens + self._calibrated(appended + tools_raw - anchor.tools_raw)
        return self._calibrated(self._system_raw + tools_raw + _raw(view))

    def note_actual_tokens(self, actual_input_tokens: int, *, cleared_by_provider: int = 0) -> None:
        """Calibrate against the provider-reported size of the call just made.

        Pass the *full* prompt size (input + cache_read + cache_creation
        tokens) and what the provider cleared server-side before counting:
        the two together are the size of the prompt that was sent, which is
        what the estimate describes. Called by the loop after every LLM
        response; it also anchors the next call's estimate.
        """
        pending = self._pending
        if pending is None:
            return
        self._provider_cleared = (pending.epoch, max(0, cleared_by_provider))
        actual_input_tokens += max(0, cleared_by_provider)
        if actual_input_tokens <= 0 or pending.raw_total <= 0:
            return
        ratio = actual_input_tokens / pending.raw_total
        if not _PLAUSIBLE_RATIO[0] <= ratio <= _PLAUSIBLE_RATIO[1]:
            return
        low, high = _CALIBRATION_BOUNDS
        self._calibration = min(high, max(low, 0.8 * self._calibration + 0.2 * ratio))
        self._anchor = _Anchor(
            epoch=pending.epoch,
            view_len=pending.view_len,
            tokens=actual_input_tokens,
            tools_raw=pending.tools_raw,
        )

    def provider_cleared_tokens(self, memory: ShortTermMemory) -> int:
        """Tokens the provider cleared server-side at the last call.

        The provider clears again on every call, so this much of the next
        prompt will not reach the model either. Zero once *memory* was trimmed
        or replaced: what was cleared then may no longer be there.
        """
        epoch, cleared = self._provider_cleared
        return cleared if epoch == memory.epoch else 0

    # ------------------------------------------------------------- assembly
    async def start_run(
        self, memory: ShortTermMemory, user_input: str, *, dry_run: bool = False
    ) -> None:
        """Store this run's memory context in the history, once.

        Called at the start of a run, before the user's message is appended.
        A new conversation (empty history) first lets the newest idle session
        of the namespace be summarized, then takes a snapshot of the long-term
        summary into the state; facts relevant to *user_input* are appended as
        one message — part of the history from then on, so every call of the
        run sends exactly the same prefix. Facts already present in the
        history are not repeated. A dry run reads but never summarizes.
        """
        if self._memory is None or self._memory.long_term is None:
            return
        # Errors are feedback, never crashes: an unreadable store costs this
        # run its memory context, not the run itself.
        if len(memory) == 0 and memory.state.pinned is None:
            try:
                if self._summarizer is not None:
                    if not dry_run:
                        await self._summarizer.on_conversation_start()
                    segments = await self._summarizer.segments()
                else:
                    legacy = await self._memory.rolling_summary()
                    segments = [legacy] if legacy else []
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._memory_read_failed("summary", exc)
                segments = []
            if segments:
                state = memory.state.copy()
                state.summary = segments
                memory.state = state
                self._record_memory_read(
                    "summary", segments=len(segments), chars=sum(len(s) for s in segments)
                )
        try:
            facts = await self._memory.relevant_facts(user_input)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._memory_read_failed("relevant_facts", exc)
            return
        seen = _injected_facts(memory.snapshot())
        # A fact already in the history is not repeated; one whose value
        # changed since is stored again, saying what it replaces — the old
        # line stays in the history (nothing sent is ever edited), so the
        # model must be told which one holds.
        new = [f for f in facts if f.get("key") not in seen or seen[f.get("key")] != f.get("value")]
        if new:
            memory.append(
                UserMessage(
                    content=_relevant_context_block(new, seen),
                    metadata={
                        CONTEXT_KEY: "relevant_facts",
                        "facts": [[f.get("key"), f.get("value")] for f in new],
                    },
                )
            )
            self._record_memory_read(
                "relevant_facts",
                facts=len(new),
                updated=sum(1 for f in new if f.get("key") in seen),
            )

    def _memory_read_failed(self, name: str, exc: BaseException) -> None:
        if self._trace is not None:
            self._trace.record_event(
                EventType.ERROR,
                name=f"memory_read_failed:{name}",
                status=SpanStatus.ERROR,
                error=ErrorInfo.from_exception(exc),
            )

    def _record_memory_read(self, name: str, **payload: Any) -> None:
        if self._trace is not None:
            self._trace.record_event(EventType.MEMORY_READ, name=name, payload=payload)

    async def assemble(
        self,
        memory: ShortTermMemory,
        *,
        call: int = 1,
        dry_run: bool = False,
    ) -> AssembledContext:
        """Render what the run's *call*-th LLM call sends, trimming first if needed.

        On models that bind reasoning to the prefix the history is trimmed
        only at a run's first call; in the middle of a run it is rendered
        as-is even above the limit (the pre-flight guard and server-side
        clearing take over there). A trim
        mutates *memory* (pass a copy to look without touching).
        `dry_run=True` is for looking only: no call follows, so this
        manager's own state (the token anchor, the once-only trim record, the
        last sent view) is left alone and trim events are marked `dry_run`.
        """
        run_start = call == 1
        stats = _RenderStats()
        view = self.render_view(memory.snapshot(), memory.state, stats)
        estimated = self.estimate(memory, view)

        trim: TrimOutcome | None = None
        if (run_start or not self._binds_reasoning) and estimated > self._limit:
            trim, summary_changed = await self._trim(memory, estimated, dry_run=dry_run)
            if trim.applied:
                stats = _RenderStats()
                view = self.render_view(memory.snapshot(), memory.state, stats)
                estimated = trim.estimated_after
                if summary_changed:
                    # The summary at the head changed too: measure it all again.
                    estimated = self._calibrated(self._system_raw + self._tools_raw() + _raw(view))

        sent, filtered = view, False
        if self._filter is not None:
            info = ContextInfo(
                call=call,
                run_start=run_start,
                estimated_tokens=estimated,
                max_context_tokens=self._limit,
                model=self._model,
                reasoning_binds_to_prefix=self._binds_reasoning,
                trimmed=trim is not None and trim.applied,
            )
            sent = await self._apply_filter(memory, view, info, dry_run=dry_run)
            filtered = not same_messages(sent, view)
            if filtered:
                estimated = self._calibrated(self._system_raw + self._tools_raw() + _raw(sent))

        if not dry_run:
            tools_raw = self._tools_raw()
            self._pending = _PendingCall(
                # A filtered view is not the rendered one the next call extends,
                # so it cannot anchor that call's estimate.
                epoch=-1 if filtered else memory.epoch,
                view_len=len(view),
                raw_total=self._system_raw + tools_raw + _raw(sent),
                tools_raw=tools_raw,
            )
            if self._filter is not None:
                self._last_sent, self._last_view_len = sent, len(view)
                self._last_epoch = memory.epoch
        return AssembledContext(
            system_prompt=self._system_prompt,
            messages=sent,
            estimated_tokens=estimated,
            trim=trim,
            messages_stored=len(memory),
            cleared_results=stats.cleared_results,
            reasoning_dropped=stats.reasoning_dropped,
            filtered=filtered,
        )

    # -------------------------------------------------------- context filter
    async def _apply_filter(
        self,
        memory: ShortTermMemory,
        view: list[AnyMessage],
        info: ContextInfo,
        *,
        dry_run: bool,
    ) -> list[AnyMessage]:
        """Run the filter, repair its output, and keep the sent prefix intact."""
        assert self._filter is not None
        outcome = FilterOutcome()
        candidate = view
        try:
            result = self._filter([copy.copy(m) for m in view], info)
            if inspect.isawaitable(result):
                result = await result
            candidate, outcome.repairs = repair(validate_output(result))
        except Exception as exc:
            # Errors are feedback, never crashes: the unfiltered view is sent.
            outcome.error = f"{type(exc).__name__}: {exc}"
            candidate = view
        if self._binds_reasoning:
            candidate = self._guard_prefix(memory, view, candidate, info.run_start, outcome)
        outcome.changed = not same_messages(candidate, view)
        self._record_filter(outcome, dry_run=dry_run)
        return candidate

    def _guard_prefix(
        self,
        memory: ShortTermMemory,
        view: list[AnyMessage],
        candidate: list[AnyMessage],
        run_start: bool,
        outcome: FilterOutcome,
    ) -> list[AnyMessage]:
        """Keep reasoning valid on a model that binds it to the prefix.

        Its reasoning blocks were produced against what was actually *sent*,
        so the reference is the last sent view extended by everything stored
        since — the model's own new output included. A candidate that keeps
        all of that unchanged is fine. Otherwise: at a run's start the
        reasoning before the current turn is left out (removing a leading run
        of blocks keeps the rest valid); in the middle of a run the edit is
        rejected and the reference itself is sent.
        """
        if run_start:
            self._stripped_this_run = False
        elif self._stripped_this_run:
            # Same rule as at the run's start, so a pure filter's output still
            # extends what was sent.
            candidate, _ = without_reasoning_before(candidate, current_turn_start(candidate))
        reference = self._sent_so_far(memory, view)
        if reference is not None:
            sent, appended = reference
            frozen = len(sent) + _through_last_assistant(appended)
            expected = (sent + appended)[:frozen]
            if same_messages(candidate[:frozen], expected):
                return candidate
            if not run_start:
                outcome.rejected = True
                return sent + appended
        stripped, count = without_reasoning_before(candidate, current_turn_start(candidate))
        outcome.reasoning_stripped = count
        if run_start:
            self._stripped_this_run = True
        return stripped

    def _sent_so_far(
        self, memory: ShortTermMemory, view: list[AnyMessage]
    ) -> tuple[list[AnyMessage], list[AnyMessage]] | None:
        """(last sent view, messages appended since) — None if not comparable."""
        if self._last_sent is None or self._last_epoch != memory.epoch:
            return None
        if len(view) < self._last_view_len:
            return None
        return self._last_sent, view[self._last_view_len :]

    def _record_filter(self, outcome: FilterOutcome, *, dry_run: bool) -> None:
        if self._trace is None:
            return
        extra = {"dry_run": True} if dry_run else {}
        if outcome.error is not None:
            self._trace.record_event(
                EventType.ERROR,
                name="context_filter_error",
                status=SpanStatus.ERROR,
                payload={"error": outcome.error[:500], "fallback": "unfiltered view", **extra},
            )
        if outcome.rejected:
            self._trace.record_event(
                EventType.ERROR,
                name="context_filter_rejected",
                status=SpanStatus.ERROR,
                payload={
                    "reason": (
                        "the filter changed messages already sent to a model that "
                        "binds reasoning to the prefix, in the middle of a run"
                    ),
                    "fallback": "last sent view + new messages",
                    **extra,
                },
            )
        if outcome.changed or outcome.repairs.any() or outcome.reasoning_stripped:
            self._trace.record_event(
                EventType.COMPACTION,
                name="context_filter",
                payload={
                    "changed": outcome.changed,
                    **outcome.repairs.to_dict(),
                    "reasoning_stripped": outcome.reasoning_stripped,
                    **extra,
                },
            )

    # ----------------------------------------------------------------- trim
    async def _trim(
        self, memory: ShortTermMemory, estimated_before: int, *, dry_run: bool = False
    ) -> tuple[TrimOutcome, bool]:
        """Plan and apply a trim: ``(outcome, summary_changed)``.

        With long-term summaries, the turns it drops are summarized right
        away (one LLM call; never on a dry run).
        """
        state, drop, outcome = self.plan_trim(memory.snapshot(), memory.state, estimated_before)
        summary_changed = False
        if outcome.applied:
            memory.state = state
            dropped = memory.drop_head(drop)
            self._record(outcome, dry_run=dry_run)
            if not dry_run:
                self._noop_epoch = None
                if self._summarizer is not None and dropped:
                    summary_changed = await self._summarizer.after_trim(memory, dropped)
        elif dry_run:
            self._record(outcome, dry_run=True)
        elif self._noop_epoch != memory.epoch:
            # Recorded once: the same history would give the same answer on
            # every following call until something is removed or replaced.
            self._noop_epoch = memory.epoch
            self._record(outcome)
        return outcome, summary_changed

    def _record(self, outcome: TrimOutcome, *, dry_run: bool = False) -> None:
        if self._trace is not None:
            payload = outcome.payload()
            if dry_run:
                payload["dry_run"] = True
            self._trace.record_event(EventType.COMPACTION, name="trim", payload=payload)

    def plan_trim(
        self,
        history: list[AnyMessage],
        state: ContextState,
        estimated_before: int,
    ) -> tuple[ContextState, int, TrimOutcome]:
        """Decide a trim without applying it: ``(new_state, drop_count, outcome)``.

        *new_state* is expressed against *history* as it is now — apply it
        first, then `drop_head(drop_count)`, which shifts its counters.
        Savings are measured in raw tokens and scaled by the calibration, then
        subtracted from *estimated_before*, so the outcome stays on the same
        (anchored) scale as the estimate that triggered it.
        """
        limit, target = self._limit, self.trim_target
        starts = _turn_starts(history)
        current_turn = starts[-1] if starts else len(history)
        raw_before = _raw(self.render_view(history, state))

        # 1. Clear all but the newest k clearable results — and never one the
        #    model has not seen yet (those after its last message), however
        #    many parallel calls returned at once.
        clearable = self._clearable_ordinals(history)
        keep = self.keep_recent_results()
        cleared_zone = state.cleared_results
        if len(clearable) > keep:
            cleared_zone = max(cleared_zone, min(clearable[-keep], _first_unseen_result(history)))
        cleared_now = _count_below(clearable, cleared_zone) - _count_below(
            clearable, state.cleared_results
        )

        # Whatever changes from here on, reasoning before the current turn goes
        # with it: its signatures cover the prefix being edited — unless the
        # model wants its reasoning kept, where it is not bound to the prefix
        # but a kept tool-call message without it is rejected.
        # A copy, so whatever else the state carries (summary, pending
        # summaries) survives the trim.
        candidate = state.copy()
        candidate.cleared_results = cleared_zone
        if not self._keeps_reasoning:
            candidate.reasoning_cut = max(state.reasoning_cut, current_turn)
        tokens = [estimate_message_tokens(m) for m in self._render_history(history, candidate)]
        suffix = [0] * (len(tokens) + 1)
        for i in range(len(tokens) - 1, -1, -1):
            suffix[i] = suffix[i + 1] + tokens[i]

        # The summary at the head of the view stays whatever is dropped.
        head = _raw([self._summary_message(state.summary)]) if state.summary else 0

        def after(pinned: AnyMessage | None, drop: int) -> int:
            kept = head + suffix[drop]
            if pinned is not None:
                kept += estimate_message_tokens(pinned)
            return estimated_before - self._calibrated(raw_before - kept)

        estimated = after(state.pinned, 0) if cleared_now else estimated_before

        # 2. Drop whole oldest turns until the target is reached, keeping the
        #    newest turns and pinning the conversation's first user message.
        drop = 0
        if estimated > target:
            first_index, first = _first_user_message(history)
            droppable = starts[: max(0, len(starts) - (KEEP_RECENT_TURNS - 1))]
            for cut in (s for s in droppable if s > 0):
                if state.pinned is None and first is not None and first_index < cut:
                    candidate.pinned = first
                drop, estimated = cut, after(candidate.pinned, cut)
                if estimated <= target:
                    break

        changed = cleared_now > 0 or drop > 0
        useful = estimated <= int(limit * (1 - MIN_TRIM_HEADROOM)) or (
            # Without it the prompt could not be sent at all.
            estimated_before > self._usable and estimated <= self._usable
        )
        applied = changed and useful
        if not applied:
            return state, 0, TrimOutcome(
                applied=False,
                estimated_before=estimated_before,
                estimated_after=estimated_before,
                limit=limit,
                target=target,
            )
        return candidate, drop, TrimOutcome(
            applied=True,
            estimated_before=estimated_before,
            estimated_after=estimated,
            limit=limit,
            target=target,
            cleared_results=cleared_now,
            dropped_messages=drop,
            dropped_turns=sum(1 for s in starts if s < drop),
            pinned=state.pinned is None and candidate.pinned is not None,
            reasoning_cut=candidate.reasoning_cut > state.reasoning_cut,
        )

    def _clearable_ordinals(self, history: list[AnyMessage]) -> list[int]:
        """Ordinals (over all tool results, in history order) of clearable ones."""
        ordinals: list[int] = []
        ordinal = 0
        for message in history:
            if isinstance(message, ToolResultMessage):
                for result in message.content:
                    if self._clearable(result):
                        ordinals.append(ordinal)
                    ordinal += 1
        return ordinals


# ---------------------------------------------------------------- helpers
def _raw(messages: list[AnyMessage]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def _cleared(result: ToolResult) -> ToolResult:
    note = CLEAR_NOTE.format(name=result.name, tokens=estimate_tokens(result.as_text()))
    return ToolResult(
        tool_use_id=result.tool_use_id,
        name=result.name,
        content=note,
        is_error=False,
        duration_ms=result.duration_ms,
    )


def _is_user_turn(message: AnyMessage) -> bool:
    return isinstance(message, UserMessage) and not isinstance(message, ToolResultMessage)


def _turn_starts(history: list[AnyMessage]) -> list[int]:
    """Indices where a turn begins.

    A turn begins at a user message, except one that directly follows
    context the framework injected for it (facts before the user's own
    message): cutting at a turn start therefore never separates the two, and
    never a tool call from its result. A user message that follows another
    user message the model never answered (a run stopped before replying)
    begins its own turn, so that one can still be dropped later.
    """
    return [
        i
        for i, message in enumerate(history)
        if _is_user_turn(message)
        and (
            i == 0
            or not _is_user_turn(history[i - 1])
            or CONTEXT_KEY not in history[i - 1].metadata
        )
    ]


def _first_unseen_result(history: list[AnyMessage]) -> int:
    """Ordinal of the first tool result after the last assistant message.

    Those results answer calls the model made but have not been shown to it
    yet; clearing one would only make it call the tool again.
    """
    last_assistant = max(
        (i for i, m in enumerate(history) if isinstance(m, AssistantMessage)), default=-1
    )
    ordinal = 0
    for message in history[: last_assistant + 1]:
        if isinstance(message, ToolResultMessage):
            ordinal += len(message.content)
    return ordinal


def _through_last_assistant(messages: list[AnyMessage]) -> int:
    """Length of the prefix of *messages* up to its last assistant message."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], AssistantMessage):
            return index + 1
    return 0


def _first_user_message(messages: list[AnyMessage]) -> tuple[int, AnyMessage | None]:
    """The first message the user wrote (not context the framework injected)."""
    for index, message in enumerate(messages):
        if _is_user_turn(message) and CONTEXT_KEY not in message.metadata:
            return index, message
    return len(messages), None


def _injected_facts(history: list[AnyMessage]) -> dict[Any, Any]:
    """key → the value most recently stored in *history* for each fact."""
    seen: dict[Any, Any] = {}
    for message in history:
        if message.metadata.get(CONTEXT_KEY) == "relevant_facts":
            for pair in message.metadata.get("facts") or []:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    seen[pair[0]] = pair[1]
    return seen


def _count_below(ordinals: list[int], bound: int) -> int:
    return bisect.bisect_left(ordinals, bound)


def _relevant_context_block(facts: list[dict[str, Any]], seen: dict[Any, Any]) -> str:
    lines = []
    for f in facts:
        key, value = f.get("key", ""), f.get("value", "")
        line = f"- {key}: {value}"
        if key in seen:
            line += f" (updated; replaces: {seen[key]})"
        lines.append(line)
    body = "\n".join(lines)
    return f"<relevant_context>\n{body}\n</relevant_context>"
