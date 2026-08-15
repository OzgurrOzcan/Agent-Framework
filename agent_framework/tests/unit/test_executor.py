"""Tests for ToolExecutor argument validation (T2.8) and the
cancellation contract `ToolCancellationError` (T1.7).
"""
import asyncio

import pytest

from agent_framework.core.message import ToolCallRequest
from agent_framework.permissions.models import PermissionCategory
from agent_framework.tools.decorator import tool
from agent_framework.tools.executor import ToolCancellationError, ToolExecutor
from agent_framework.tools.models import Tool, ToolMetadata
from agent_framework.tracing.trace import EventType, ExecutionTrace, SpanStatus


# ---- argument validation (T2.8) ----

@pytest.mark.asyncio
async def test_invalid_argument_type_rejected_before_invoke():
    calls_made = 0

    @tool(description="calc", permission=PermissionCategory.AUTO)
    async def calc(expression: str) -> str:
        nonlocal calls_made
        calls_made += 1
        return expression

    executor = ToolExecutor()
    call = ToolCallRequest(id="v1", name="calc", arguments={"expression": 123})
    result = await executor.run(calc, call)
    assert result.is_error is True
    assert calls_made == 0  # tool function was never invoked
    assert "Invalid arguments" in result.content
    assert "expression" in result.content  # field path is reported
    assert result.tool_use_id == "v1"
    assert result.duration_ms == 0


@pytest.mark.asyncio
async def test_valid_arguments_pass_validation():
    @tool(description="calc", permission=PermissionCategory.AUTO)
    async def calc(expression: str) -> str:
        return f"ok:{expression}"

    executor = ToolExecutor()
    call = ToolCallRequest(id="v2", name="calc", arguments={"expression": "1+1"})
    result = await executor.run(calc, call)
    assert result.is_error is False
    assert result.content == "ok:1+1"


@pytest.mark.asyncio
async def test_additional_properties_violation_rejected():
    calls_made = 0

    @tool(description="calc", permission=PermissionCategory.AUTO)
    async def calc(expression: str) -> str:
        nonlocal calls_made
        calls_made += 1
        return expression

    executor = ToolExecutor()
    call = ToolCallRequest(
        id="v3", name="calc", arguments={"expression": "1+1", "bogus": True}
    )
    result = await executor.run(calc, call)
    assert result.is_error is True
    assert calls_made == 0
    assert "Invalid arguments" in result.content


@pytest.mark.asyncio
async def test_empty_schema_skips_validation():
    async def mcp_like(**kwargs) -> str:
        return "ok"

    meta = ToolMetadata(
        name="mcp_like",
        description="MCP-style tool with an empty schema",
        parameters_schema={},
        permission=PermissionCategory.AUTO,
    )
    executor = ToolExecutor()
    call = ToolCallRequest(id="v4", name="mcp_like", arguments={"anything": "goes"})
    result = await executor.run(Tool(metadata=meta, fn=mcp_like), call)
    assert result.is_error is False
    assert result.content == "ok"


@pytest.mark.asyncio
async def test_validation_error_recorded_in_trace():
    @tool(description="calc", permission=PermissionCategory.AUTO)
    async def calc(expression: str) -> str:
        return expression

    trace = ExecutionTrace(agent_id="a1", session_id="s1")
    executor = ToolExecutor(trace=trace)
    call = ToolCallRequest(id="v5", name="calc", arguments={"expression": 5})
    result = await executor.run(calc, call)
    assert result.is_error is True

    events = [
        ev
        for ev in trace.all_spans()
        if ev.event_type == EventType.TOOL_RESULT and ev.status == SpanStatus.ERROR
    ]
    assert len(events) == 1
    assert events[0].payload.get("tool_name") == "calc"
    assert events[0].payload.get("validation_error") is True


# ---- cancellation contract (T1.7) ----

def test_tool_cancellation_error_is_cancelled_error():
    err = ToolCancellationError([])
    assert isinstance(err, asyncio.CancelledError)
    assert err.partial_results == []


@pytest.mark.asyncio
async def test_run_many_cancellation_carries_partial_results():
    @tool(description="fast", permission=PermissionCategory.AUTO)
    async def fast() -> str:
        return "done"

    @tool(description="slow", permission=PermissionCategory.AUTO)
    async def slow() -> str:
        await asyncio.sleep(10)
        return "never"

    executor = ToolExecutor()
    calls = [
        (fast, ToolCallRequest(id="f1", name="fast", arguments={})),
        (slow, ToolCallRequest(id="s1", name="slow", arguments={})),
    ]
    task = asyncio.create_task(executor.run_many(calls))
    await asyncio.sleep(0.05)  # let the fast tool finish
    task.cancel()

    with pytest.raises(ToolCancellationError) as excinfo:
        await task

    partial = excinfo.value.partial_results
    assert len(partial) == 2
    by_id = {r.tool_use_id: r for r in partial}
    assert set(by_id) == {"f1", "s1"}  # every tool_use_id has a result
    # The fast tool's real result survived.
    assert by_id["f1"].is_error is False
    assert by_id["f1"].content == "done"
    # The slow tool got the "cancelled" placeholder.
    assert by_id["s1"].is_error is True
    assert "cancelled" in str(by_id["s1"].content).lower()
