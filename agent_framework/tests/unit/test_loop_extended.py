"""Extended loop tests: provider errors, new stop reasons, streaming
reasoning round-trip, cancellation invariant, tool_choice, budget guard,
concurrency guard, persistent compaction integration."""

from __future__ import annotations

import asyncio

import pytest

from agent_framework.core.config import AgentConfig, CompactionConfig
from agent_framework.core.context_manager import ContextManager
from agent_framework.core.loop import AgentLoop
from agent_framework.core.message import (
    AssistantMessage,
    StopReason,
    ToolResultMessage,
    UserMessage,
)
from agent_framework.core.policy import LoopPolicy
from agent_framework.permissions.config import PermissionConfig
from agent_framework.permissions.models import PermissionCategory
from agent_framework.permissions.resolver import PermissionResolver
from agent_framework.providers.base import StreamChunkType, ToolChoice
from agent_framework.memory.short_term import ShortTermMemory
from agent_framework.tools.decorator import tool
from agent_framework.tools.executor import ToolExecutor
from agent_framework.tools.registry import ToolRegistry
from agent_framework.tracing.cost import CostTracker, ModelPricing, PricingTable
from agent_framework.tracing.trace import EventType, ExecutionTrace, SpanStatus
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn


def make_loop(
    mock: MockLLMProvider,
    tools: list | None = None,
    *,
    pricing: PricingTable | None = None,
    max_budget_usd: float | None = None,
    context_manager: ContextManager | None = None,
    max_parallel_tools: int = 8,
) -> AgentLoop:
    registry = ToolRegistry(tools or [])
    trace = ExecutionTrace(agent_id="test")
    config = AgentConfig(
        model="claude-opus-5",
        api_key="test",
        system_prompt="You are a test agent.",
        max_turns=10,
        stall_threshold=3,
        max_budget_usd=max_budget_usd,
        max_parallel_tools=max_parallel_tools,
    )
    ctx = context_manager or ContextManager(
        system_prompt="You are a test agent.",
        context_window=10000,
        response_buffer_tokens=512,
        compaction_buffer_tokens=256,
        compaction=CompactionConfig(enabled=False),
    )
    return AgentLoop(
        config=config,
        provider=mock,
        tools=registry,
        short_term=ShortTermMemory(),
        context_manager=ctx,
        permission_resolver=PermissionResolver(PermissionConfig(), trace=trace),
        tool_executor=ToolExecutor(trace=trace, max_parallel=max_parallel_tools),
        cost_tracker=CostTracker(pricing=pricing or PricingTable()),
        trace=trace,
        agent_id="test",
    )


# ---------------- provider error contract (T2.1) ----------------

async def test_provider_error_returns_result_not_exception():
    mock = MockLLMProvider([ScriptedTurn(raise_exc=RuntimeError("rate limited"))])
    loop = make_loop(mock)
    result = await loop.run("hi", policy=LoopPolicy.full())
    assert isinstance(result.error, RuntimeError)
    assert result.stop_reason == StopReason.ERROR
    assert "rate limited" in result.final_message
    # History is clean: only the user message; no dangling assistant turn.
    assert len(result.messages) == 1
    assert isinstance(result.messages[0], UserMessage)
    # Trace carries the error event.
    assert any(
        s.event_type == EventType.ERROR and s.name == "provider_error"
        for s in loop.trace.all_spans()
    )


async def test_provider_error_streaming_yields_error_chunk():
    mock = MockLLMProvider([ScriptedTurn(raise_exc=RuntimeError("boom"))])
    loop = make_loop(mock)
    chunks = [c async for c in loop.stream("hi", policy=LoopPolicy.full())]
    assert chunks[-1].type == StreamChunkType.ERROR
    assert chunks[-1].payload["reason"] == "provider_error"


async def test_cancelled_error_is_not_swallowed():
    mock = MockLLMProvider([ScriptedTurn(raise_exc=asyncio.CancelledError())])
    loop = make_loop(mock)
    with pytest.raises(asyncio.CancelledError):
        await loop.run("hi", policy=LoopPolicy.full())


# ---------------- new stop reasons (T2.5) ----------------

async def test_pause_turn_continues_loop():
    mock = MockLLMProvider([
        ScriptedTurn(text="working...", stop_reason=StopReason.PAUSE_TURN),
        ScriptedTurn(text="done"),
    ])
    loop = make_loop(mock)
    result = await loop.run("go", policy=LoopPolicy.full())
    assert mock._call_count == 2
    assert result.final_message == "done"


async def test_refusal_terminates_with_own_stop_reason():
    mock = MockLLMProvider([ScriptedTurn(text="", stop_reason=StopReason.REFUSAL)])
    loop = make_loop(mock)
    result = await loop.run("go", policy=LoopPolicy.full())
    assert result.stop_reason == StopReason.REFUSAL
    assert result.error is None
    assert mock._call_count == 1


async def test_context_exceeded_terminates():
    mock = MockLLMProvider([
        ScriptedTurn(text="partial", stop_reason=StopReason.CONTEXT_EXCEEDED),
    ])
    loop = make_loop(mock)
    result = await loop.run("go", policy=LoopPolicy.full())
    assert result.stop_reason == StopReason.CONTEXT_EXCEEDED
    assert mock._call_count == 1


# ---------------- budget guard (T1.1 companion) ----------------

async def test_budget_guard_fires_with_priced_model():
    pricing = PricingTable({"claude-opus-5": ModelPricing(1_000_000.0, 1_000_000.0)})
    mock = MockLLMProvider([
        ScriptedTurn(text="...", tool_calls=[
            {"id": "b1", "name": "nope", "arguments": {}}]),
        ScriptedTurn(text="never reached"),
    ])
    loop = make_loop(mock, pricing=pricing, max_budget_usd=0.01)
    result = await loop.run("go", policy=LoopPolicy.full())
    assert result.error is not None
    assert "budget" in str(result.error).lower()
    assert result.stop_reason == StopReason.ERROR
    assert any(
        s.name == "budget_exceeded" and s.status == SpanStatus.ERROR
        for s in loop.trace.all_spans()
    )


# ---------------- streaming reasoning round-trip (T1.2) ----------------

async def test_stream_preserves_reasoning_blocks():
    thinking = {"type": "thinking", "thinking": "let me think", "signature": "sig1"}
    mock = MockLLMProvider([
        ScriptedTurn(
            text="calling tool",
            tool_calls=[{"id": "r1", "name": "unknown_tool", "arguments": {}}],
            reasoning=[thinking],
        ),
        ScriptedTurn(text="done"),
    ])
    loop = make_loop(mock)
    async for _ in loop.stream("go", policy=LoopPolicy.full()):
        pass
    assistants = [m for m in loop.short_term.snapshot() if isinstance(m, AssistantMessage)]
    assert assistants[0].reasoning, "reasoning must survive the streaming path"
    assert assistants[0].reasoning[0].raw == thinking


async def test_stream_without_reasoning_keeps_empty_list():
    mock = MockLLMProvider([ScriptedTurn(text="plain")])
    loop = make_loop(mock)
    async for _ in loop.stream("go", policy=LoopPolicy.full()):
        pass
    assistants = [m for m in loop.short_term.snapshot() if isinstance(m, AssistantMessage)]
    assert assistants[0].reasoning == []


# ---------------- cancellation invariant (T1.7) ----------------

async def test_cancellation_preserves_tool_result_invariant():
    started = asyncio.Event()

    @tool(description="fast", permission=PermissionCategory.AUTO)
    async def fast() -> str:
        return "quick"

    @tool(description="slow", permission=PermissionCategory.AUTO)
    async def slow() -> str:
        started.set()
        await asyncio.sleep(30)
        return "never"

    mock = MockLLMProvider([
        ScriptedTurn(text="running both", tool_calls=[
            {"id": "f1", "name": "fast", "arguments": {}},
            {"id": "s1", "name": "slow", "arguments": {}},
        ]),
    ])
    loop = make_loop(mock, tools=[fast, slow])
    task = asyncio.create_task(loop.run("go", policy=LoopPolicy.full()))
    await started.wait()
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    history = loop.short_term.snapshot()
    last = history[-1]
    assert isinstance(last, ToolResultMessage)
    ids = {r.tool_use_id for r in last.content}
    assert ids == {"f1", "s1"}, "every tool_use must have a tool_result after cancel"


# ---------------- tool_choice pass-through (T2.3) ----------------

async def test_forced_tool_choice_applies_to_first_turn_only():
    @tool(description="Search", permission=PermissionCategory.AUTO)
    async def search(q: str) -> str:
        return "res"

    tc = ToolChoice(mode="tool", tool_name="search")
    mock = MockLLMProvider([
        ScriptedTurn(text="", tool_calls=[
            {"id": "t1", "name": "search", "arguments": {"q": "x"}}]),
        ScriptedTurn(text="done"),
    ])
    loop = make_loop(mock, tools=[search])
    await loop.run("go", policy=LoopPolicy.full(tool_choice=tc))
    assert mock.call_log[0]["tool_choice"] is tc
    assert mock.call_log[1]["tool_choice"] is None


# ---------------- concurrency guard (T3.2) ----------------

async def test_concurrent_run_rejected():
    release = asyncio.Event()

    @tool(description="block", permission=PermissionCategory.AUTO)
    async def block() -> str:
        await release.wait()
        return "ok"

    mock = MockLLMProvider([
        ScriptedTurn(text="", tool_calls=[{"id": "x", "name": "block", "arguments": {}}]),
        ScriptedTurn(text="done"),
    ])
    loop = make_loop(mock, tools=[block])
    task = asyncio.create_task(loop.run("first", policy=LoopPolicy.full()))
    await asyncio.sleep(0.05)
    with pytest.raises(RuntimeError, match="already executing"):
        await loop.run("second", policy=LoopPolicy.full())
    release.set()
    result = await task
    assert result.final_message == "done"


# ---------------- parallel tool cap (T3.2) ----------------

async def test_max_parallel_tools_respected():
    active = 0
    peak = 0

    @tool(description="slowish", permission=PermissionCategory.AUTO)
    async def slowish(n: int) -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return n

    calls = [{"id": f"p{i}", "name": "slowish", "arguments": {"n": i}} for i in range(8)]
    mock = MockLLMProvider([ScriptedTurn(text="", tool_calls=calls), ScriptedTurn(text="done")])
    loop = make_loop(mock, tools=[slowish], max_parallel_tools=2)
    result = await loop.run("go", policy=LoopPolicy.full())
    assert result.final_message == "done"
    assert peak <= 2


# ---------------- persistent compaction integration (T1.3) ----------------

async def test_loop_persists_compaction_summary():
    summarizer = MockLLMProvider([ScriptedTurn(text="COMPACT-SUMMARY")])
    ctx = ContextManager(
        system_prompt="sys",
        context_window=800,          # tiny window to force compaction
        response_buffer_tokens=50,
        compaction_buffer_tokens=50,
        compaction=CompactionConfig(
            enabled=True, trigger_ratio=0.3, tool_result_truncate_threshold=100_000,
            sliding_window_min_turns=50, full_compaction_min_recent_turns=1,
            summary_max_tokens=50,
        ),
        summarizer=summarizer,
        summarizer_model="mock-model",
    )
    mock = MockLLMProvider([ScriptedTurn(text="ok")])
    loop = make_loop(mock, context_manager=ctx)
    # Pre-fill history with bulky turns so assembly triggers summarization.
    for i in range(6):
        loop.short_term.append(UserMessage(content=f"question {i} " + "x" * 400))
        loop.short_term.append(AssistantMessage(content=f"answer {i} " + "y" * 400))

    before = len(loop.short_term)
    await loop.run("final question", policy=LoopPolicy.full())
    after = loop.short_term.snapshot()
    assert len(after) < before, "compaction result must be persisted"
    assert any(
        isinstance(m, UserMessage) and "COMPACT-SUMMARY" in str(m.content)
        for m in after
    ), "persisted history must contain the summary message"
    # (The single-summary-call guarantee is covered in
    # test_context_manager_v2.test_summary_computed_once_then_persisted.)
