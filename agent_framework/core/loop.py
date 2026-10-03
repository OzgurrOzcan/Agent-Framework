"""AgentLoop — the core Observe → Reason → Act → Observe while-loop.

Design laws enforced here:
  - Messages are the only durable state. Without memory (the default) a
    run starts from an empty history; with it, the history carries over.
  - Errors are feedback, never crashes (denials, timeouts, exceptions,
    provider failures all become `ToolResult(is_error=True)` or an
    `AgentResult(error=...)` — never an unhandled traceback).
  - `tool_use` and `tool_result` are inseparable — every assistant
    `tool_call` is followed by exactly one `ToolResult`, even on denial
    and even on cancellation mid-flight.
  - Tools run in parallel within a single turn.
  - Four independent termination layers: natural / max_turns / stall / budget.
  - Each loop iteration is its own `turn[N]` span in the trace tree.
  - `pause_turn` continues the loop; `refusal` and context-window overflows
    terminate it with their own stop reasons.
  - Nothing already sent is edited in the middle of a run on a model that
    binds reasoning to the prefix (see `ProviderSpec.binds_reasoning_to_prefix`):
    the history is trimmed only at a run's first call there. A call that
    would not fit the window stops the run with CONTEXT_EXCEEDED *before*
    it is sent, instead of surfacing the provider's 400. Where the provider
    can clear old tool results server-side (Anthropic context editing), that
    is requested on every call: it keeps long runs in bounds without editing
    what was sent.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from typing import Any, AsyncGenerator

from agent_framework.core.config import AgentConfig
from agent_framework.core.context_manager import (
    AssembledContext,
    ContextManager,
    ContextReport,
)
from agent_framework.core.message import (
    AnyMessage,
    AssistantMessage,
    ReasoningContent,
    StopReason,
    TokenUsage,
    ToolCallRequest,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from agent_framework.core.policy import (
    BudgetExceededError,
    ContextWindowExceededError,
    LoopGuardError,
    LoopPolicy,
)
from agent_framework.memory.short_term import ShortTermMemory
from agent_framework.permissions.models import Verdict
from agent_framework.permissions.resolver import PermissionResolver
from agent_framework.providers.base import (
    AppliedEdit,
    BaseProviderAdapter,
    ContextEditsRejectedError,
    NormalizedResponse,
    StreamChunk,
    StreamChunkType,
    ToolResultClearing,
)
from agent_framework.tools.builtin.ask_user import ASK_USER_TOOL_NAME
from agent_framework.tools.executor import ToolCancellationError, ToolExecutor
from agent_framework.tools.models import ToolMetadata
from agent_framework.tools.registry import ToolRegistry
from agent_framework.tracing.cost import CostTracker
from agent_framework.tracing.trace import (
    ErrorInfo,
    EventType,
    ExecutionTrace,
    SpanStatus,
)


@dataclass
class LoopState:
    """Tiny serializable record for resumption / introspection."""

    turn: int = 0
    tool_rounds: int = 0
    cumulative_cost_usd: float = 0.0
    recent_tool_hashes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "tool_rounds": self.tool_rounds,
            "cumulative_cost_usd": self.cumulative_cost_usd,
            "recent_tool_hashes": list(self.recent_tool_hashes),
        }


@dataclass
class AgentResult:
    """The result of an agent run."""

    final_message: str
    messages: list[AnyMessage]
    stop_reason: StopReason
    state: LoopState
    error: Exception | None = None
    trace: ExecutionTrace | None = None
    # What the run's last LLM call was sent (with dry_run: would be sent).
    context: ContextReport | None = None

    def __str__(self) -> str:
        return self.final_message

    @property
    def tool_calls(self) -> list[ToolCallRequest]:
        """Tool calls from the last assistant message.

        Primarily useful with `agent.complete()`, where requested calls are
        returned as data instead of being executed.
        """
        for msg in reversed(self.messages):
            if isinstance(msg, AssistantMessage):
                return list(msg.tool_calls)
        return []


def _policy_payload(
    policy: LoopPolicy, *, streaming: bool, retain_history: bool
) -> dict[str, Any]:
    """The effective shape of one run, for the RUN span.

    `AgentConfig` records the defaults, but `run(max_turns=3)` overrides them
    per call — so these values are recoverable from nowhere else. Without them
    a trace cannot distinguish "the caller capped this run" from "the deployed
    config changed", nor `max_tool_rounds` from a permission wall when a turn
    requests tools and none execute.
    """
    choice = policy.tool_choice
    return {
        "max_turns": policy.max_turns,
        # None = unlimited (bounded only by max_turns).
        "max_tool_rounds": policy.max_tool_rounds,
        "execute_tools": policy.execute_tools,
        "tool_choice": choice.mode if choice is not None else None,
        "tool_choice_name": choice.tool_name if choice is not None else None,
        "streaming": streaming,
        # False = this run started from an empty history (no memory).
        "retain_history": retain_history,
    }


class AgentLoop:
    """The core engine. One instance per agent."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        provider: BaseProviderAdapter,
        tools: ToolRegistry,
        short_term: ShortTermMemory,
        context_manager: ContextManager,
        permission_resolver: PermissionResolver,
        tool_executor: ToolExecutor,
        cost_tracker: CostTracker,
        trace: ExecutionTrace,
        agent_id: str,
        retain_history: bool = True,
        native_clearing: bool = False,
    ) -> None:
        self.config = config
        self.provider = provider
        self.tools = tools
        self.short_term = short_term
        self.context = context_manager
        self.permissions = permission_resolver
        self.executor = tool_executor
        self.cost = cost_tracker
        self.trace = trace
        self.agent_id = agent_id
        # False = stateless: each run starts from an empty history.
        self.retain_history = retain_history
        # Server-side tool-result clearing; switched off for the rest of the
        # session if the provider rejects it once.
        self.native_clearing = native_clearing
        # Concurrent runs would interleave into the same short-term memory.
        self._running = False

    def _acquire_run(self) -> None:
        if self._running:
            raise RuntimeError(
                "This agent is already executing a run/stream; concurrent runs "
                "share conversation history and are not supported. Create a "
                "separate agent for parallel work."
            )
        self._running = True

    async def _begin_run(self, user_input: str, memory: ShortTermMemory | None = None) -> None:
        """Start a run's history: fresh without memory, repaired with it, then
        this run's memory context, then the user's message."""
        memory = memory if memory is not None else self.short_term
        if not self.retain_history:
            memory.clear()
        else:
            _answer_unanswered_calls(memory)
        await self.context.start_run(memory, user_input)
        memory.append(UserMessage(content=user_input))

    async def _assemble(self, turn: int) -> AssembledContext:
        """The view for this run's *turn*-th call. The context manager decides
        whether it may be trimmed: on a model that binds reasoning to the
        prefix, only at a run's first call."""
        return await self.context.assemble(self.short_term, call=turn)

    def _window_exceeded(
        self, estimated_tokens: int, turn_num: int
    ) -> ContextWindowExceededError | None:
        """The pre-flight guard: a prompt that cannot fit is never sent.

        With server-side clearing on, the provider clears the old results
        again on every call, so the amount it cleared last time does not
        reach the model either.
        """
        usable = self.config.usable_context_tokens
        cleared = (
            self.context.provider_cleared_tokens(self.short_term) if self.native_clearing else 0
        )
        if estimated_tokens - cleared <= usable:
            return None
        self.trace.record_event(
            EventType.LOOP_GUARD,
            name="context_exceeded",
            status=SpanStatus.ERROR,
            turn_number=turn_num,
            payload={
                "estimated_tokens": estimated_tokens,
                "cleared_by_provider": cleared,
                "usable_tokens": usable,
                "max_context_tokens": self.context.max_context_tokens,
            },
        )
        return ContextWindowExceededError(
            f"the prompt (≈{estimated_tokens} tokens) would not fit the "
            f"{usable} tokens the context window leaves next to "
            f"max_tokens_per_response",
            turn=turn_num,
            estimated_tokens=estimated_tokens,
            limit=usable,
        )

    def _tool_choice_for_turn(self, policy: LoopPolicy, turn: int):
        # A forced tool choice only applies to the first LLM call — repeating
        # it on later turns would force the model into an infinite tool loop.
        if policy.tool_choice is None:
            return None
        if policy.tool_choice.mode in ("tool", "any") and turn > 1:
            return None
        return policy.tool_choice

    def _note_usage(self, usage: TokenUsage, applied_edits: list[AppliedEdit]) -> float:
        cost = self.cost.record(self.config.model, usage)
        ctx_actual = (
            usage.input_tokens
            + usage.cache_read_input_tokens
            + usage.cache_creation_input_tokens
        )
        cleared = 0
        for edit in applied_edits:
            cleared += edit.cleared_input_tokens
            self.trace.record_event(
                EventType.COMPACTION, name="native_clear", payload=edit.to_dict()
            )
        # The provider counted the prompt *after* clearing; calibrate against
        # the prompt that was sent.
        self.context.note_actual_tokens(ctx_actual, cleared_by_provider=cleared)
        return cost

    # ---------- server-side tool-result clearing ----------

    def _context_edits(self) -> ToolResultClearing | None:
        if not self.native_clearing:
            return None
        limit = self.context.max_context_tokens
        protected = {ASK_USER_TOOL_NAME} | {
            t.metadata.name for t in self.tools if not t.metadata.clear_old_results
        }
        return ToolResultClearing(
            trigger_tokens=limit,
            keep_results=self.context.keep_recent_results(),
            # Each clear rebuilds the prompt cache: only worth it in bulk.
            clear_at_least_tokens=max(1, limit // 10),
            exclude_tools=tuple(sorted(protected)),
        )

    def _disable_native_clearing(self, exc: ContextEditsRejectedError) -> None:
        self.native_clearing = False
        self.trace.record_event(
            EventType.COMPACTION,
            name="native_clear",
            payload={"disabled": True, "reason": str(exc)[:500]},
        )

    async def _call_provider(self, **request: Any) -> NormalizedResponse:
        """One provider call, with server-side clearing where available.

        A rejection of the clearing (a gateway that drops the beta, a model
        without it) is retried once without it and switches it off for the
        rest of the session.
        """
        edits = self._context_edits()
        if edits is None:
            return await self.provider.call(**request)
        try:
            return await self.provider.call(**request, context_edits=edits)
        except ContextEditsRejectedError as exc:
            self._disable_native_clearing(exc)
            return await self.provider.call(**request)

    async def _stream_provider(self, **request: Any) -> AsyncGenerator[StreamChunk, None]:
        """`_call_provider` for streaming. A rejection arrives before any
        chunk, so the retry never duplicates output."""
        edits = self._context_edits()
        if edits is None:
            async with _closing(self.provider.stream(**request)) as chunks:
                async for chunk in chunks:
                    yield chunk
            return
        started = False
        try:
            async with _closing(self.provider.stream(**request, context_edits=edits)) as chunks:
                async for chunk in chunks:
                    started = True
                    yield chunk
        except ContextEditsRejectedError as exc:
            if started:
                raise
            self._disable_native_clearing(exc)
            async with _closing(self.provider.stream(**request)) as chunks:
                async for chunk in chunks:
                    yield chunk

    # ===================== non-streaming run =====================

    async def run(
        self,
        user_input: str,
        *,
        policy: LoopPolicy,
    ) -> AgentResult:
        self._acquire_run()
        try:
            # One RUN span per run, so every turn below it is attributable to
            # the policy that shaped it (session = how the agent was built).
            with self.trace.start_span(
                EventType.RUN,
                name="run",
                payload=_policy_payload(
                    policy, streaming=False, retain_history=self.retain_history
                ),
            ) as run_span:
                result = await self._run_inner(user_input, policy=policy)
                run_span.update(
                    stop_reason=result.stop_reason.value,
                    turns=result.state.turn,
                    tool_rounds=result.state.tool_rounds,
                    cost_usd=result.state.cumulative_cost_usd,
                )
                # The loop turns operational failures into AgentResult.error
                # rather than raising, so the span has to be marked by hand.
                if result.error is not None:
                    run_span.set_error(result.error)
                return result
        finally:
            self._running = False

    async def _run_inner(
        self,
        user_input: str,
        *,
        policy: LoopPolicy,
    ) -> AgentResult:
        state = LoopState()
        await self._begin_run(user_input)

        stop_reason = StopReason.END_TURN
        report: ContextReport | None = None
        try:
            while True:
                # Layer 2: hard iteration ceiling.
                if state.turn >= policy.max_turns:
                    self.trace.record_event(
                        EventType.LOOP_GUARD,
                        name="max_turns",
                        status=SpanStatus.ERROR,
                        payload={"max_turns": policy.max_turns, "turn": state.turn},
                    )
                    raise LoopGuardError(
                        f"max_turns={policy.max_turns} exceeded", turn=state.turn
                    )

                state.turn += 1
                turn_num = self.trace.bump_turn()

                # Each loop iteration is its own turn span.
                with self.trace.start_span(
                    EventType.TURN,
                    name=f"turn[{turn_num}]",
                    turn_number=turn_num,
                ) as turn_span:
                    # Render what is sent; trims the history first when the
                    # prompt has outgrown max_context_tokens.
                    assembled = await self._assemble(state.turn)
                    report = _report(assembled, self.context.max_context_tokens, None)
                    overflow = self._window_exceeded(assembled.estimated_tokens, turn_num)
                    if overflow is not None:
                        raise overflow

                    # LLM call — nested under the turn span.
                    with self.trace.start_span(
                        EventType.LLM_CALL,
                        name=self.config.model,
                        turn_number=turn_num,
                    ) as llm_span:
                        try:
                            response = await self._call_provider(
                                system=assembled.system_prompt,
                                messages=assembled.messages,
                                tools=[t.metadata for t in self.tools],
                                model=self.config.model,
                                max_tokens=self.config.max_tokens_per_response,
                                temperature=self.config.temperature,
                                tool_choice=self._tool_choice_for_turn(policy, state.turn),
                                enable_caching=self.config.enable_prompt_caching,
                                extra=self.config.request_extra or None,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            err = ErrorInfo.from_exception(exc)
                            llm_span.set_error(err)
                            self.trace.record_event(
                                EventType.ERROR,
                                name="provider_error",
                                status=SpanStatus.ERROR,
                                error=err,
                                turn_number=turn_num,
                            )
                            return AgentResult(
                                final_message=(
                                    f"[provider error: {type(exc).__name__}: {exc}]"
                                ),
                                messages=self.short_term.snapshot(),
                                stop_reason=StopReason.ERROR,
                                state=state,
                                error=exc,
                                trace=self.trace,
                                context=report,
                            )
                        cost = self._note_usage(response.usage, response.applied_edits)
                        state.cumulative_cost_usd = self.cost.snapshot()[0]
                        llm_span.set_usage(response.usage, cost)
                        report = _report(
                            assembled, self.context.max_context_tokens, response.usage
                        )
                        llm_span.update(
                            model=self.config.model,
                            stop_reason=response.stop_reason.value,
                            **report.to_dict(),
                        )

                    # Layer 4: budget guard.
                    if (
                        self.config.max_budget_usd is not None
                        and state.cumulative_cost_usd > self.config.max_budget_usd
                    ):
                        self.trace.record_event(
                            EventType.LOOP_GUARD,
                            name="budget_exceeded",
                            status=SpanStatus.ERROR,
                            turn_number=turn_num,
                            payload={
                                "spent_usd": state.cumulative_cost_usd,
                                "limit_usd": self.config.max_budget_usd,
                            },
                        )
                        raise BudgetExceededError(
                            f"budget ${state.cumulative_cost_usd:.4f} > "
                            f"${self.config.max_budget_usd}",
                            turn=state.turn,
                            spent=state.cumulative_cost_usd,
                            limit=self.config.max_budget_usd,
                        )

                    assistant_msg = response.assistant_message
                    self.short_term.append(assistant_msg)
                    stop_reason = response.stop_reason
                    turn_span.update(stop_reason=stop_reason.value)

                    # Server-side tool loop paused mid-turn — re-send to resume.
                    if stop_reason == StopReason.PAUSE_TURN:
                        continue

                    # Terminal stop reasons with their own semantics.
                    if stop_reason in (StopReason.REFUSAL, StopReason.CONTEXT_EXCEEDED):
                        break

                    # Natural termination — model issued no tool calls.
                    if not assistant_msg.tool_calls:
                        break

                    # call_once shape — return tool calls as data, do not execute.
                    if not policy.execute_tools:
                        break

                    # Tool-round cap — feed denial results and let the model synthesize.
                    if (
                        policy.max_tool_rounds is not None
                        and state.tool_rounds >= policy.max_tool_rounds
                    ):
                        deny_results = [
                            self.executor.deny_result(
                                tool=self._meta_or_synth(tc),
                                call=tc,
                                feedback=(
                                    "Tool-round cap reached. Respond with a final answer "
                                    "using the information gathered so far."
                                ),
                            )
                            for tc in assistant_msg.tool_calls
                        ]
                        self.short_term.append(ToolResultMessage(content=deny_results))
                        continue

                    # Layer 3: stall detection — same tool+args N times in a row.
                    new_hashes = [_call_hash(tc) for tc in assistant_msg.tool_calls]
                    state.recent_tool_hashes.extend(new_hashes)
                    # Keep only the sliding window needed for detection.
                    keep = self.config.stall_threshold * 2
                    state.recent_tool_hashes = state.recent_tool_hashes[-keep:]

                    if _is_stalled(state.recent_tool_hashes, self.config.stall_threshold):
                        self.trace.record_event(
                            EventType.LOOP_GUARD,
                            name="stall",
                            status=SpanStatus.STALLED,
                            turn_number=turn_num,
                            payload={
                                "threshold": self.config.stall_threshold,
                                "repeated_hash": state.recent_tool_hashes[-1],
                            },
                        )
                        # Satisfy the tool_use → tool_result invariant with stall feedback.
                        stall_results = [
                            self.executor.deny_result(
                                tool=self._meta_or_synth(tc),
                                call=tc,
                                feedback=(
                                    "Stall detected: this exact tool call was repeated "
                                    f"{self.config.stall_threshold} times in a row. "
                                    "Reconsider your approach — use different arguments "
                                    "or a different strategy."
                                ),
                            )
                            for tc in assistant_msg.tool_calls
                        ]
                        self.short_term.append(ToolResultMessage(content=stall_results))
                        state.recent_tool_hashes.clear()
                        continue

                    # Permission resolution + parallel execution. On cancellation
                    # mid-flight, persist the partial results first so the
                    # tool_use ↔ tool_result invariant survives.
                    try:
                        tool_results = await self._dispatch_tools(
                            assistant_msg.tool_calls, turn_number=turn_num
                        )
                    except ToolCancellationError as exc:
                        self.short_term.append(
                            ToolResultMessage(content=exc.partial_results)
                        )
                        raise
                    self.short_term.append(ToolResultMessage(content=tool_results))
                    state.tool_rounds += 1

        except LoopGuardError as exc:
            return AgentResult(
                final_message=f"[loop guard: {exc}]",
                messages=self.short_term.snapshot(),
                stop_reason=(
                    StopReason.CONTEXT_EXCEEDED
                    if isinstance(exc, ContextWindowExceededError)
                    else StopReason.ERROR
                ),
                state=state,
                error=exc,
                trace=self.trace,
                context=report,
            )
        except asyncio.CancelledError:
            self.trace.record_event(
                EventType.LOOP_GUARD,
                name="cancelled",
                status=SpanStatus.CANCELLED,
            )
            raise

        # Extract the last assistant text.
        final_text = ""
        for msg in reversed(self.short_term.snapshot()):
            if isinstance(msg, AssistantMessage) and isinstance(msg.content, str):
                final_text = msg.content
                break

        return AgentResult(
            final_message=final_text,
            messages=self.short_term.snapshot(),
            stop_reason=stop_reason,
            state=state,
            error=None,
            trace=self.trace,
            context=report,
        )

    # ===================== dry run =====================

    async def dry_run(self, user_input: str, *, policy: LoopPolicy) -> AgentResult:
        """What the first call of `run(user_input)` would send — without
        calling the model, summarizing, or saving anything.

        Built on a copy: the stored history and its state are untouched. A
        stateless agent shows a fresh run, as its next run would be.
        """
        self._acquire_run()
        try:
            with self.trace.start_span(
                EventType.RUN,
                name="run",
                payload={
                    **_policy_payload(
                        policy, streaming=False, retain_history=self.retain_history
                    ),
                    "dry_run": True,
                },
            ) as run_span:
                memory = self.short_term.copy() if self.retain_history else ShortTermMemory()
                await self._begin_run(user_input, memory)
                assembled = await self.context.assemble(memory, call=1, dry_run=True)
                report = _report(assembled, self.context.max_context_tokens, None)
                run_span.update(stop_reason=StopReason.DRY_RUN.value, **report.to_dict())
                return AgentResult(
                    final_message="",
                    messages=assembled.messages,
                    stop_reason=StopReason.DRY_RUN,
                    state=LoopState(),
                    trace=self.trace,
                    context=report,
                )
        finally:
            self._running = False

    # ===================== streaming run =====================

    async def stream(
        self,
        user_input: str,
        *,
        policy: LoopPolicy,
    ) -> AsyncGenerator[StreamChunk, None]:
        self._acquire_run()
        # Owned here rather than inside _stream_inner so the RUN span can
        # report this run's own numbers; the trace-wide turn counter is
        # cumulative across runs.
        state = LoopState()
        try:
            with self.trace.start_span(
                EventType.RUN,
                name="run",
                payload=_policy_payload(
                    policy, streaming=True, retain_history=self.retain_history
                ),
            ) as run_span:
                try:
                    # aclosing: if our consumer closes the stream early, close
                    # the inner generator now, so its spans end and its
                    # cleanup runs instead of waiting for garbage collection.
                    async with contextlib.aclosing(
                        self._stream_inner(user_input, policy=policy, state=state)
                    ) as chunks:
                        async for chunk in chunks:
                            if chunk.type is StreamChunkType.ERROR:
                                # Streaming reports failure as an ERROR chunk,
                                # not an AgentResult, so mark the span here.
                                # Provider chunks are forwarded verbatim, so
                                # do not trust the payload's shape: tracing
                                # must never break the stream it observes.
                                payload = chunk.payload
                                reason = (
                                    payload.get("reason", "")
                                    if isinstance(payload, dict)
                                    else payload
                                )
                                run_span.set_error(
                                    ErrorInfo(
                                        exception_type="StreamError",
                                        message=str(reason),
                                        traceback="",
                                    )
                                )
                            yield chunk
                finally:
                    run_span.update(
                        turns=state.turn,
                        tool_rounds=state.tool_rounds,
                        cost_usd=state.cumulative_cost_usd,
                    )
        finally:
            self._running = False

    async def _stream_inner(
        self,
        user_input: str,
        *,
        policy: LoopPolicy,
        state: LoopState,
    ) -> AsyncGenerator[StreamChunk, None]:
        await self._begin_run(user_input)

        try:
            while True:
                # Layer 2: max-turns guard.
                if state.turn >= policy.max_turns:
                    yield StreamChunk(
                        StreamChunkType.ERROR,
                        payload={"reason": "max_turns_exceeded", "turn": state.turn},
                    )
                    return

                state.turn += 1
                turn_num = self.trace.bump_turn()

                assembled = await self._assemble(state.turn)
                overflow = self._window_exceeded(assembled.estimated_tokens, turn_num)
                if overflow is not None:
                    yield StreamChunk(
                        StreamChunkType.ERROR,
                        payload={
                            "reason": "context_exceeded",
                            "estimated_tokens": overflow.estimated_tokens,
                            "limit": overflow.limit,
                        },
                    )
                    return

                tool_calls_this_turn: list[ToolCallRequest] = []
                text_parts: list[str] = []
                reasoning_blocks: list[ReasoningContent] = []
                stop_reason = StopReason.END_TURN
                call_usage: TokenUsage | None = None

                with self.trace.start_span(
                    EventType.LLM_CALL,
                    name=self.config.model,
                    turn_number=turn_num,
                ) as llm_span:
                    try:
                        provider_chunks = self._stream_provider(
                            system=assembled.system_prompt,
                            messages=assembled.messages,
                            tools=[t.metadata for t in self.tools],
                            model=self.config.model,
                            max_tokens=self.config.max_tokens_per_response,
                            temperature=self.config.temperature,
                            tool_choice=self._tool_choice_for_turn(policy, state.turn),
                            enable_caching=self.config.enable_prompt_caching,
                            extra=self.config.request_extra or None,
                        )
                        async with contextlib.aclosing(provider_chunks) as chunks:
                            async for chunk in chunks:
                                yield chunk
                                ctype = chunk.type
                                if ctype == StreamChunkType.TEXT_DELTA:
                                    text_parts.append(chunk.payload.get("text", ""))
                                elif ctype == StreamChunkType.TOOL_USE_COMPLETE:
                                    tool_calls_this_turn.append(
                                        ToolCallRequest(
                                            id=chunk.payload["id"],
                                            name=chunk.payload["name"],
                                            arguments=chunk.payload.get("arguments", {}),
                                        )
                                    )
                                elif ctype == StreamChunkType.USAGE:
                                    usage = TokenUsage(
                                        input_tokens=chunk.payload.get("input_tokens", 0),
                                        output_tokens=chunk.payload.get("output_tokens", 0),
                                        cache_read_input_tokens=chunk.payload.get(
                                            "cache_read_input_tokens", 0
                                        ),
                                        cache_creation_input_tokens=chunk.payload.get(
                                            "cache_creation_input_tokens", 0
                                        ),
                                    )
                                    edits = [
                                        AppliedEdit.from_dict(e)
                                        for e in chunk.payload.get("applied_edits") or []
                                        if isinstance(e, dict)
                                    ]
                                    cost = self._note_usage(usage, edits)
                                    state.cumulative_cost_usd = self.cost.snapshot()[0]
                                    llm_span.set_usage(usage, cost)
                                    call_usage = usage
                                elif ctype == StreamChunkType.STOP:
                                    sr = chunk.payload.get("stop_reason", "end_turn")
                                    try:
                                        stop_reason = StopReason(sr)
                                    except ValueError:
                                        stop_reason = StopReason.END_TURN
                                    # Preserve reasoning/thinking blocks for round-trip:
                                    # Anthropic rejects the next tool-use turn if the
                                    # prior thinking block was stripped.
                                    for r in chunk.payload.get("reasoning") or []:
                                        if isinstance(r, dict):
                                            reasoning_blocks.append(
                                                ReasoningContent(
                                                    provider=self.provider.name,  # type: ignore[arg-type]
                                                    raw=r,
                                                    text=r.get("thinking") or r.get("text"),
                                                )
                                            )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        err = ErrorInfo.from_exception(exc)
                        llm_span.set_error(err)
                        self.trace.record_event(
                            EventType.ERROR,
                            name="provider_error",
                            status=SpanStatus.ERROR,
                            error=err,
                            turn_number=turn_num,
                        )
                        yield StreamChunk(
                            StreamChunkType.ERROR,
                            payload={
                                "reason": "provider_error",
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                        return

                    llm_span.update(
                        model=self.config.model,
                        stop_reason=stop_reason.value,
                        **_report(
                            assembled, self.context.max_context_tokens, call_usage
                        ).to_dict(),
                    )

                # Layer 4: budget guard.
                if (
                    self.config.max_budget_usd is not None
                    and state.cumulative_cost_usd > self.config.max_budget_usd
                ):
                    yield StreamChunk(
                        StreamChunkType.ERROR,
                        payload={
                            "reason": "budget_exceeded",
                            "spent_usd": state.cumulative_cost_usd,
                        },
                    )
                    return

                self.short_term.append(
                    AssistantMessage(
                        content="".join(text_parts),
                        tool_calls=tool_calls_this_turn,
                        reasoning=reasoning_blocks,
                        stop_reason=stop_reason,
                    )
                )

                # Server-side tool loop paused — re-send to resume.
                if stop_reason == StopReason.PAUSE_TURN:
                    continue

                # Terminal stop reasons.
                if stop_reason in (StopReason.REFUSAL, StopReason.CONTEXT_EXCEEDED):
                    return

                # Natural termination.
                if not tool_calls_this_turn or not policy.execute_tools:
                    return

                # Tool-round cap.
                if (
                    policy.max_tool_rounds is not None
                    and state.tool_rounds >= policy.max_tool_rounds
                ):
                    deny_results = [
                        self.executor.deny_result(
                            tool=self._meta_or_synth(tc),
                            call=tc,
                            feedback="Tool-round cap reached. Respond with a final answer.",
                        )
                        for tc in tool_calls_this_turn
                    ]
                    self.short_term.append(ToolResultMessage(content=deny_results))
                    continue

                # Layer 3: stall detection (same as non-streaming).
                new_hashes = [_call_hash(tc) for tc in tool_calls_this_turn]
                state.recent_tool_hashes.extend(new_hashes)
                keep = self.config.stall_threshold * 2
                state.recent_tool_hashes = state.recent_tool_hashes[-keep:]

                if _is_stalled(state.recent_tool_hashes, self.config.stall_threshold):
                    self.trace.record_event(
                        EventType.LOOP_GUARD,
                        name="stall",
                        status=SpanStatus.STALLED,
                        turn_number=turn_num,
                        payload={"threshold": self.config.stall_threshold},
                    )
                    stall_results = [
                        self.executor.deny_result(
                            tool=self._meta_or_synth(tc),
                            call=tc,
                            feedback=(
                                "Stall detected: same tool call repeated. "
                                "Try a different approach."
                            ),
                        )
                        for tc in tool_calls_this_turn
                    ]
                    self.short_term.append(ToolResultMessage(content=stall_results))
                    state.recent_tool_hashes.clear()
                    continue

                # Dispatch tools + stream the results back.
                try:
                    results = await self._dispatch_tools(
                        tool_calls_this_turn, turn_number=turn_num
                    )
                except ToolCancellationError as exc:
                    self.short_term.append(ToolResultMessage(content=exc.partial_results))
                    raise
                self.short_term.append(ToolResultMessage(content=results))
                for r in results:
                    yield StreamChunk(
                        StreamChunkType.TOOL_USE_COMPLETE,
                        payload={
                            "id": r.tool_use_id,
                            "name": r.name,
                            "result": r.content,
                            "is_error": r.is_error,
                        },
                    )
                state.tool_rounds += 1

        except LoopGuardError as exc:
            yield StreamChunk(StreamChunkType.ERROR, payload={"reason": str(exc)})
        except asyncio.CancelledError:
            self.trace.record_event(
                EventType.LOOP_GUARD, name="cancelled", status=SpanStatus.CANCELLED
            )
            raise

    # ===================== helpers =====================

    async def _dispatch_tools(
        self,
        tool_calls: list[ToolCallRequest],
        *,
        turn_number: int,
    ) -> list[ToolResult]:
        """Resolve permissions → run allowed tools in parallel → return one
        ToolResult per call in original order. Denials and unknown tools produce
        is_error=True results; the loop invariant (tool_use ↔ tool_result) holds."""
        from agent_framework.tools.models import Tool as ToolObj

        # Step 1: permission resolution (serial — interrupts are inherently serial).
        verdicts: list[tuple[ToolCallRequest, ToolObj | None, Any]] = []
        for tc in tool_calls:
            t = self.tools.get(tc.name)
            if t is None:
                verdicts.append((tc, None, None))
                continue
            decision = await self.permissions.resolve(
                t.metadata, tc.arguments, turn_number=turn_number
            )
            verdicts.append((tc, t, decision))

        # Step 2: split into runnable vs. denied.
        runnable: list[tuple[ToolObj, ToolCallRequest]] = []
        results_map: dict[str, ToolResult] = {}

        for tc, t, decision in verdicts:
            if t is None:
                results_map[tc.id] = ToolResult(
                    tool_use_id=tc.id,
                    name=tc.name,
                    content=(
                        f"Unknown tool `{tc.name}`. "
                        f"Available: {', '.join(self.tools.names)}"
                    ),
                    is_error=True,
                )
            elif decision.verdict == Verdict.ALLOW:
                runnable.append((t, tc))
            else:
                results_map[tc.id] = self.executor.deny_result(
                    tool=t.metadata,
                    call=tc,
                    feedback=decision.deny_feedback or "Denied.",
                )

        # Step 3: run allowed tools concurrently. On ToolCancellationError the
        # already-computed denial/unknown results are merged into the partial
        # set so every tool_use still gets a result.
        if runnable:
            try:
                executed = await self.executor.run_many(runnable, turn_number=turn_number)
            except ToolCancellationError as exc:
                merged: dict[str, ToolResult] = dict(results_map)
                for r in exc.partial_results:
                    merged[r.tool_use_id] = r
                ordered = [merged[tc.id] for tc in tool_calls if tc.id in merged]
                raise ToolCancellationError(ordered) from None
            for r in executed:
                results_map[r.tool_use_id] = r

        # Step 4: reassemble in original call order.
        return [results_map[tc.id] for tc in tool_calls]

    def _meta_or_synth(self, tc: ToolCallRequest) -> ToolMetadata:
        """Return the tool's real metadata, or a synthetic deny-category stub."""
        t = self.tools.get(tc.name)
        if t is not None:
            return t.metadata
        from agent_framework.permissions.models import PermissionCategory
        return ToolMetadata(
            name=tc.name,
            description="(unknown)",
            parameters_schema={},
            permission=PermissionCategory.DENY,
        )


# ---- pure helpers ----

def _answer_unanswered_calls(memory: ShortTermMemory) -> None:
    """Give tool calls left without results a result before going on.

    `complete()` returns tool calls as data without running them, and a
    response cut off by the context window can end in one. Sending such a
    call again without its result is rejected by every provider, so a
    remembered conversation would fail on its next run.
    """
    last = memory.last
    if not isinstance(last, AssistantMessage) or not last.tool_calls:
        return
    memory.append(
        ToolResultMessage(
            content=[
                ToolResult(
                    tool_use_id=tc.id,
                    name=tc.name,
                    content="Not executed: this tool call was never run.",
                    is_error=True,
                )
                for tc in last.tool_calls
            ]
        )
    )


def _report(assembled: AssembledContext, limit: int, usage: TokenUsage | None) -> ContextReport:
    return ContextReport(
        messages_sent=len(assembled.messages),
        messages_stored=assembled.messages_stored,
        estimated_tokens=assembled.estimated_tokens,
        max_context_tokens=limit,
        cleared_results=assembled.cleared_results,
        reasoning_dropped=assembled.reasoning_dropped,
        trimmed=assembled.trim is not None and assembled.trim.applied,
        filtered=assembled.filtered,
        input_tokens=usage.input_tokens if usage is not None else 0,
        cache_read_tokens=usage.cache_read_input_tokens if usage is not None else 0,
        cache_write_tokens=usage.cache_creation_input_tokens if usage is not None else 0,
    )


@contextlib.asynccontextmanager
async def _closing(
    chunks: AsyncIterator[StreamChunk],
) -> AsyncGenerator[AsyncIterator[StreamChunk], None]:
    """`contextlib.aclosing` for any adapter stream.

    Closing promptly releases the provider's HTTP stream when the consumer
    stops early. Adapters normally return async generators, but the
    interface only promises an async iterator, which may have no `aclose`.
    """
    try:
        yield chunks
    finally:
        aclose = getattr(chunks, "aclose", None)
        if aclose is not None:
            await aclose()


def _call_hash(tc: ToolCallRequest) -> str:
    canonical = json.dumps(tc.arguments, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(f"{tc.name}\x1f{canonical}".encode()).hexdigest()[:24]


def _is_stalled(hashes: list[str], threshold: int) -> bool:
    """True iff the last `threshold` hashes are all identical."""
    if threshold <= 0 or len(hashes) < threshold:
        return False
    last = hashes[-1]
    consecutive = 0
    for h in reversed(hashes):
        if h == last:
            consecutive += 1
        else:
            break
    return consecutive >= threshold
