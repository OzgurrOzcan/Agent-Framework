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
     reject it once that prefix changed, and the rest ignore old reasoning.

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

import bisect
import dataclasses
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
from agent_framework.memory.short_term import ContextState, ShortTermMemory
from agent_framework.tools.builtin.ask_user import ASK_USER_TOOL_NAME
from agent_framework.tracing.trace import EventType, ExecutionTrace

if TYPE_CHECKING:
    from agent_framework.memory.manager import MemoryManager
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
# Metadata marking a message the framework injected (not written by the user).
CONTEXT_KEY = "context"

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
    ) -> None:
        if max_context_tokens < 1 or max_tool_result_tokens < 1:
            raise ValueError("max_context_tokens and max_tool_result_tokens must be >= 1")
        self._system_prompt = system_prompt
        self._system_raw = estimate_tokens(system_prompt)
        self._limit = max_context_tokens
        self._result_cap = max_tool_result_tokens
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
    def render_view(self, history: list[AnyMessage], state: ContextState) -> list[AnyMessage]:
        """What the model sees for *history* under *state*. Pure; never mutates."""
        view: list[AnyMessage] = []
        if state.summary:
            view.append(self._summary_message(state.summary))
        if state.pinned is not None:
            view.append(state.pinned)
        view.extend(self._render_history(history, state))
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
        self, history: list[AnyMessage], state: ContextState
    ) -> list[AnyMessage]:
        out: list[AnyMessage] = []
        ordinal = 0  # tool results seen so far, in history order
        for index, message in enumerate(history):
            if isinstance(message, ToolResultMessage):
                rendered: list[ToolResult] = []
                changed = False
                for result in message.content:
                    if ordinal < state.cleared_results and self._clearable(result):
                        rendered.append(_cleared(result))
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

    def note_actual_tokens(self, actual_input_tokens: int) -> None:
        """Calibrate against the provider-reported size of the call just made.

        Pass the *full* prompt size (input + cache_read + cache_creation
        tokens). Called by the loop after every LLM response; it also anchors
        the next call's estimate.
        """
        pending = self._pending
        if actual_input_tokens <= 0 or pending is None or pending.raw_total <= 0:
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

    # ------------------------------------------------------------- assembly
    async def start_run(self, memory: ShortTermMemory, user_input: str) -> None:
        """Store this run's memory context in the history, once.

        Called at the start of a run, before the user's message is appended.
        A new conversation (empty history) takes a snapshot of the long-term
        summary into the state; facts relevant to *user_input* are appended as
        one message — part of the history from then on, so every call of the
        run sends exactly the same prefix. Facts already present in the
        history are not repeated.
        """
        if self._memory is None or self._memory.long_term is None:
            return
        if len(memory) == 0 and memory.state.pinned is None:
            summary = await self._memory.rolling_summary()
            if summary:
                state = memory.state.copy()
                state.summary = [summary]
                memory.state = state
                self._record_memory_read("summary", chars=len(summary))
        facts = await self._memory.relevant_facts(user_input)
        seen = _injected_facts(memory.snapshot())
        new = [f for f in facts if (f.get("key"), f.get("value")) not in seen]
        if new:
            memory.append(
                UserMessage(
                    content=_relevant_context_block(new),
                    metadata={
                        CONTEXT_KEY: "relevant_facts",
                        "facts": [[f.get("key"), f.get("value")] for f in new],
                    },
                )
            )
            self._record_memory_read("relevant_facts", facts=len(new))

    def _record_memory_read(self, name: str, **payload: Any) -> None:
        if self._trace is not None:
            self._trace.record_event(EventType.MEMORY_READ, name=name, payload=payload)

    async def assemble(
        self,
        memory: ShortTermMemory,
        *,
        allow_trim: bool = True,
    ) -> AssembledContext:
        """Render the view for the next call, trimming *memory* first if needed.

        A trim mutates *memory* (pass a copy to look without touching).
        `allow_trim=False` renders as-is even above the limit — for models that
        bind reasoning to the prefix, in the middle of a run.
        """
        view = self.render_view(memory.snapshot(), memory.state)
        estimated = self.estimate(memory, view)

        trim: TrimOutcome | None = None
        if allow_trim and estimated > self._limit:
            trim = self._trim(memory, estimated)
            if trim.applied:
                view = self.render_view(memory.snapshot(), memory.state)
                estimated = trim.estimated_after

        tools_raw = self._tools_raw()
        self._pending = _PendingCall(
            epoch=memory.epoch,
            view_len=len(view),
            raw_total=self._system_raw + tools_raw + _raw(view),
            tools_raw=tools_raw,
        )
        return AssembledContext(
            system_prompt=self._system_prompt,
            messages=view,
            estimated_tokens=estimated,
            trim=trim,
        )

    # ----------------------------------------------------------------- trim
    def _trim(self, memory: ShortTermMemory, estimated_before: int) -> TrimOutcome:
        state, drop, outcome = self.plan_trim(memory.snapshot(), memory.state, estimated_before)
        if outcome.applied:
            memory.state = state
            memory.drop_head(drop)
            self._noop_epoch = None
            self._record(outcome)
        elif self._noop_epoch != memory.epoch:
            # Recorded once: the same history would give the same answer on
            # every following call until something is removed or replaced.
            self._noop_epoch = memory.epoch
            self._record(outcome)
        return outcome

    def _record(self, outcome: TrimOutcome) -> None:
        if self._trace is not None:
            self._trace.record_event(EventType.COMPACTION, name="trim", payload=outcome.payload())

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

        # 1. Clear all but the newest k clearable results.
        clearable = self._clearable_ordinals(history)
        keep = self.keep_recent_results()
        cleared_zone = state.cleared_results
        if len(clearable) > keep:
            cleared_zone = max(cleared_zone, clearable[-keep])
        cleared_now = _count_below(clearable, cleared_zone) - _count_below(
            clearable, state.cleared_results
        )

        # Whatever changes from here on, reasoning before the current turn goes
        # with it: its signatures cover the prefix being edited.
        candidate = ContextState(
            pinned=state.pinned,
            cleared_results=cleared_zone,
            reasoning_cut=max(state.reasoning_cut, current_turn),
            summary=list(state.summary),
        )
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
        useful = estimated <= int(limit * (1 - MIN_TRIM_HEADROOM))
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
    """Indices where a turn begins: a user message not preceded by one.

    Consecutive user messages (injected context followed by the user's own
    message) open a single turn, so cutting at a turn start never separates
    them — and never separates a tool call from its result.
    """
    return [
        i
        for i, message in enumerate(history)
        if _is_user_turn(message) and (i == 0 or not _is_user_turn(history[i - 1]))
    ]


def _first_user_message(messages: list[AnyMessage]) -> tuple[int, AnyMessage | None]:
    """The first message the user wrote (not context the framework injected)."""
    for index, message in enumerate(messages):
        if _is_user_turn(message) and CONTEXT_KEY not in message.metadata:
            return index, message
    return len(messages), None


def _injected_facts(history: list[AnyMessage]) -> set[tuple[Any, Any]]:
    """(key, value) of every fact already stored in *history*."""
    seen: set[tuple[Any, Any]] = set()
    for message in history:
        if message.metadata.get(CONTEXT_KEY) == "relevant_facts":
            for pair in message.metadata.get("facts") or []:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    seen.add((pair[0], pair[1]))
    return seen


def _count_below(ordinals: list[int], bound: int) -> int:
    return bisect.bisect_left(ordinals, bound)


def _relevant_context_block(facts: list[dict[str, Any]]) -> str:
    body = "\n".join(f"- {f.get('key', '')}: {f.get('value', '')}" for f in facts)
    return f"<relevant_context>\n{body}\n</relevant_context>"
