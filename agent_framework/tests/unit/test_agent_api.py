"""Public execution-surface tests: run / run_stream / complete (API v2).

The surface contract:
  - `complete()` = ONE model call, tools never execute, calls come back as
    data on `result.tool_calls`.
  - `run()` = the agent loop; shape via max_turns / max_tool_rounds /
    tool_choice parameters.
  - `run_stream()` = run with a streaming transport.
  - Legacy shape-per-method API (call_once / call_with_tools / stream_once /
    stream_with_tools / stream) is gone; LoopPolicy is internal.
  - Construction is `Agent(model_or_config, ...)`; there is no create()
    factory and no provider enum.
"""

from __future__ import annotations


import agent_framework
from agent_framework.core.agent import Agent
from agent_framework.core.message import StopReason
from agent_framework.permissions.models import PermissionCategory
from agent_framework.providers.base import StreamChunkType, ToolChoice
from agent_framework.tools.decorator import tool
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn


def make_agent(script, tools=None):
    agent = Agent(
        "claude-opus-5",
        api_key="test-key",
        system_prompt="api surface test",
        tools=tools,
    )
    mock = MockLLMProvider(script)
    agent._provider = mock
    agent._loop.provider = mock
    return agent, mock


def counting_tool():
    calls = {"n": 0}

    @tool(description="Search", permission=PermissionCategory.AUTO)
    async def search(q: str) -> str:
        calls["n"] += 1
        return f"results for {q}"

    return search, calls


# ---------------- complete(): one call, tools as data ----------------

async def test_complete_returns_tool_calls_as_data():
    search, calls = counting_tool()
    agent, mock = make_agent(
        [ScriptedTurn(text="I would search.", tool_calls=[
            {"id": "c1", "name": "search", "arguments": {"q": "ai"}}])],
        tools=[search],
    )
    result = await agent.complete("find ai news")
    assert mock._call_count == 1, "complete() must make exactly one model call"
    assert calls["n"] == 0, "complete() must never execute tools"
    assert [tc.name for tc in result.tool_calls] == ["search"]
    assert result.tool_calls[0].arguments == {"q": "ai"}


async def test_complete_plain_text():
    agent, mock = make_agent([ScriptedTurn(text="just an answer")])
    result = await agent.complete("q")
    assert result.final_message == "just an answer"
    assert result.tool_calls == []
    assert result.stop_reason == StopReason.END_TURN


async def test_complete_forwards_tool_choice():
    search, _ = counting_tool()
    tc = ToolChoice(mode="any")
    agent, mock = make_agent([ScriptedTurn(text="", tool_calls=[
        {"id": "c1", "name": "search", "arguments": {"q": "x"}}])], tools=[search])
    await agent.complete("q", tool_choice=tc)
    assert mock.call_log[0]["tool_choice"] is tc


# ---------------- run(): the loop, shaped by parameters ----------------

async def test_run_full_loop_executes_tools():
    search, calls = counting_tool()
    agent, mock = make_agent(
        [
            ScriptedTurn(text="searching", tool_calls=[
                {"id": "t1", "name": "search", "arguments": {"q": "ai"}}]),
            ScriptedTurn(text="here you go"),
        ],
        tools=[search],
    )
    result = await agent.run("find ai")
    assert result.final_message == "here you go"
    assert calls["n"] == 1
    assert mock._call_count == 2


async def test_run_max_tool_rounds_one_replaces_call_with_tools():
    """max_tool_rounds=1 == the old call_with_tools: one executed round,
    later tool requests are denied with feedback, model must synthesize."""
    search, calls = counting_tool()
    agent, mock = make_agent(
        [
            ScriptedTurn(text="round1", tool_calls=[
                {"id": "t1", "name": "search", "arguments": {"q": "a"}}]),
            ScriptedTurn(text="round2", tool_calls=[
                {"id": "t2", "name": "search", "arguments": {"q": "b"}}]),
            ScriptedTurn(text="final synthesis"),
        ],
        tools=[search],
    )
    result = await agent.run("go", max_tool_rounds=1)
    assert result.final_message == "final synthesis"
    assert calls["n"] == 1, "only the first tool round may execute"
    assert mock._call_count == 3
    # The second round's denial feedback reached the model as a tool_result.
    from agent_framework.core.message import ToolResultMessage
    denials = [
        r for m in result.messages if isinstance(m, ToolResultMessage)
        for r in m.content if r.is_error and "cap" in str(r.content).lower()
    ]
    assert denials, "round-cap denial must be fed back to the model"


async def test_run_max_turns_overrides_config():
    agent, mock = make_agent(
        [ScriptedTurn(text="loop", tool_calls=[
            {"id": f"c{i}", "name": "missing", "arguments": {}}]) for i in range(5)],
    )
    result = await agent.run("go", max_turns=2)
    assert result.error is not None
    assert "max_turns" in str(result.error)


async def test_run_forwards_tool_choice_first_turn_only():
    search, _ = counting_tool()
    tc = ToolChoice(mode="tool", tool_name="search")
    agent, mock = make_agent(
        [
            ScriptedTurn(text="", tool_calls=[
                {"id": "t1", "name": "search", "arguments": {"q": "x"}}]),
            ScriptedTurn(text="done"),
        ],
        tools=[search],
    )
    await agent.run("go", tool_choice=tc)
    assert mock.call_log[0]["tool_choice"] is tc
    assert mock.call_log[1]["tool_choice"] is None


# ---------------- run_stream(): same loop, streaming transport ----------------

async def test_run_stream_full_loop():
    search, calls = counting_tool()
    agent, mock = make_agent(
        [
            ScriptedTurn(text="searching", tool_calls=[
                {"id": "t1", "name": "search", "arguments": {"q": "ai"}}]),
            ScriptedTurn(text="streamed answer"),
        ],
        tools=[search],
    )
    chunks = [c async for c in agent.run_stream("find ai")]
    assert calls["n"] == 1
    texts = [c.payload.get("text", "") for c in chunks
             if c.type == StreamChunkType.TEXT_DELTA]
    assert "streamed answer" in "".join(texts)


async def test_run_stream_accepts_shape_parameters():
    agent, mock = make_agent([ScriptedTurn(text="hi")])
    chunks = [c async for c in agent.run_stream(
        "q", max_turns=3, max_tool_rounds=1)]
    assert any(c.type == StreamChunkType.STOP for c in chunks)


# ---------------- surface hygiene ----------------

def test_legacy_methods_are_gone():
    agent, _ = make_agent([])
    for legacy in ("call_once", "call_with_tools", "stream", "stream_once",
                   "stream_with_tools"):
        assert not hasattr(agent, legacy), f"legacy method {legacy} must not exist"


def test_loop_policy_is_not_public_api():
    assert "LoopPolicy" not in agent_framework.__all__
    assert not hasattr(agent_framework, "LoopPolicy")


def test_public_surface_is_exactly_four_verbs():
    public_exec = [
        n for n in ("complete", "run", "run_stream", "to_json")
        if hasattr(Agent, n)
    ]
    assert public_exec == ["complete", "run", "run_stream", "to_json"]
