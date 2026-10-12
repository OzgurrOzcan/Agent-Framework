"""Tool results are capped where they are produced.

A result over its tool's token cap keeps its head and tail with a marker in
between. The cap is applied once, before the result is cached, recorded or
sent anywhere — so an oversized result never reaches the history.
"""

from __future__ import annotations

import json

import pytest

from agent_framework.core.message import ToolCallRequest, estimate_tokens
from agent_framework.permissions.models import PermissionCategory
from agent_framework.tools.decorator import tool
from agent_framework.tools.executor import ToolExecutor
from agent_framework.tools.models import Tool, ToolMetadata
from agent_framework.tracing.trace import EventType, ExecutionTrace, SpanStatus


def _call(name: str = "dump", **arguments) -> ToolCallRequest:
    return ToolCallRequest(id="c1", name=name, arguments=arguments)


def _returning(value, *, name: str = "dump", max_result_tokens: int | None = None) -> Tool:
    async def fn() -> object:
        return value

    meta = ToolMetadata(
        name=name,
        description="returns a fixed value",
        parameters_schema={"type": "object", "properties": {}},
        permission=PermissionCategory.AUTO,
        max_result_tokens=max_result_tokens,
    )
    return Tool(metadata=meta, fn=fn)


def _tool_results(trace: ExecutionTrace):
    return [e for e in trace.all_spans() if e.event_type is EventType.TOOL_RESULT]


async def test_large_result_keeps_head_and_tail():
    text = "HEAD-" + "x" * 20_000 + "-TAIL"
    result = await ToolExecutor(default_max_result_tokens=100).run(_returning(text), _call())

    content = result.content
    assert result.is_error is False
    assert content.startswith("HEAD-")
    assert content.endswith("-TAIL")
    assert "truncated" in content
    # 400 characters of the original survive: 60 % from the head, 40 % from the tail.
    head, _, rest = content.partition("\n\n[...")
    tail = rest.rpartition("...]\n\n")[2]
    assert (len(head), len(tail)) == (240, 160)
    # The marker tells the model how big the result really was.
    assert f"≈{estimate_tokens(text):,} tokens" in content


async def test_per_tool_cap_overrides_the_default():
    text = "y" * 8_000  # ≈2000 tokens
    executor = ToolExecutor(default_max_result_tokens=100)

    loose = await executor.run(_returning(text, max_result_tokens=5_000), _call())
    tight = await executor.run(_returning(text, max_result_tokens=10), _call())

    assert loose.content == text
    # 10 tokens = 40 characters kept: 24 from the head, 16 from the tail.
    assert tight.content.startswith("y" * 24 + "\n\n[...")
    assert tight.content.endswith("...]\n\n" + "y" * 16)


async def test_uncapped_without_a_cap():
    text = "z" * 100_000
    result = await ToolExecutor().run(_returning(text), _call())
    assert result.content == text


@pytest.mark.parametrize(
    "value",
    ["short text", {"rows": [1, 2, 3]}, [1, 2, 3], 42, None, "w" * 400],
)
async def test_small_and_non_str_results_are_untouched(value):
    """Within the cap the value is returned as-is — even a dict stays a dict."""
    result = await ToolExecutor(default_max_result_tokens=100).run(_returning(value), _call())
    assert result.content == value
    assert type(result.content) is type(value)


async def test_non_str_result_over_cap_becomes_text():
    rows = [{"id": i, "name": f"row-{i}"} for i in range(2_000)]
    result = await ToolExecutor(default_max_result_tokens=50).run(_returning(rows), _call())

    assert isinstance(result.content, str)
    # The serialized JSON is what was cut — its first bytes survive.
    assert result.content.startswith(json.dumps(rows)[:50])
    assert "truncated" in result.content


async def test_cap_applies_before_the_cache():
    """A cache hit must not resurrect the full result the cap kept out."""
    calls = 0

    @tool(
        description="big",
        permission=PermissionCategory.AUTO,
        cache_ttl_seconds=60,
        max_result_tokens=20,
    )
    async def big(q: str) -> str:
        nonlocal calls
        calls += 1
        return "b" * 10_000

    executor = ToolExecutor()
    first = await executor.run(big, _call("big", q="x"))
    second = await executor.run(big, _call("big", q="x"))

    assert calls == 1  # the second call was served from the cache
    assert second.content == first.content
    assert len(second.content) < 1_000
    [entry] = executor._cache.values()
    assert entry.value == first.content


async def test_error_results_are_capped_too():
    @tool(description="explodes", permission=PermissionCategory.AUTO, max_result_tokens=30)
    async def explodes() -> str:
        raise RuntimeError("response body: " + "e" * 50_000)

    result = await ToolExecutor().run(explodes, _call("explodes"))
    assert result.is_error is True
    assert result.content.startswith("Tool `explodes` failed: RuntimeError")
    assert "truncated" in result.content
    assert len(result.content) < 1_000


async def test_denial_feedback_is_capped_too():
    meta = ToolMetadata(
        name="guarded",
        description="",
        parameters_schema={},
        max_result_tokens=10,
    )
    result = ToolExecutor().deny_result(meta, _call("guarded"), "no. " * 5_000)
    assert result.is_error is True
    assert "truncated" in result.content


async def test_truncation_is_traced():
    trace = ExecutionTrace()
    executor = ToolExecutor(trace=trace, default_max_result_tokens=10)
    await executor.run(_returning("t" * 4_000), _call())
    await executor.run(_returning("small"), _call())

    big, small = _tool_results(trace)
    assert big.status is SpanStatus.OK
    assert big.payload["truncated"] is True
    assert big.payload["original_tokens"] == 1_000
    assert small.payload["truncated"] is False
    assert small.payload["original_tokens"] == estimate_tokens("small")


@pytest.mark.parametrize("bad", [0, -5, 1.5, True, "100"])
def test_max_result_tokens_must_be_positive(bad):
    with pytest.raises(ValueError, match="max_result_tokens"):
        tool(description="x", max_result_tokens=bad)(lambda: "x")
    with pytest.raises(ValueError, match="default_max_result_tokens"):
        ToolExecutor(default_max_result_tokens=bad)


async def test_an_unknown_tool_result_is_capped_too():
    """It lists every available tool — long with many MCP servers."""
    from agent_framework.core.agent import Agent
    from agent_framework.core.config import AgentConfig
    from agent_framework.core.message import ToolResultMessage
    from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn

    many = [_returning("x", name=f"tool_with_a_long_name_{i}") for i in range(200)]
    mock = MockLLMProvider(
        [
            ScriptedTurn(tool_calls=[{"id": "u1", "name": "no_such_tool", "arguments": {}}]),
            ScriptedTurn(text="done"),
        ]
    )
    agent = Agent(
        AgentConfig(model="claude-haiku-4-5", api_key="k", max_tool_result_tokens=100),
        tools=many,
        provider=mock,
    )
    await agent.run("go")

    [result] = next(m for m in agent.get_history() if isinstance(m, ToolResultMessage)).content
    assert result.is_error and result.content.startswith("Unknown tool `no_such_tool`")
    assert "truncated" in result.content and len(result.content) < 1_000
