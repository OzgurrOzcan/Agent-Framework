"""What is sent is visible: per call in the trace, per run on the result,
and before anything is sent with `run(dry_run=True)`."""

from __future__ import annotations

from agent_framework.core.agent import Agent
from agent_framework.core.config import AgentConfig, MemoryConfig
from agent_framework.core.message import (
    AssistantMessage,
    ReasoningContent,
    StopReason,
    TokenUsage,
    ToolCallRequest,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from agent_framework.permissions.models import PermissionCategory
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn
from agent_framework.tools.decorator import tool
from agent_framework.tracing.trace import EventType


@tool(description="search", permission=PermissionCategory.AUTO)
async def search(q: str) -> str:
    return f"results for {q}"


def make_agent(script=(), *, memory=None, **config) -> tuple[Agent, MockLLMProvider]:
    mock = MockLLMProvider(list(script))
    agent = Agent(
        AgentConfig(model="claude-haiku-4-5", api_key="k", **config),
        tools=[search],
        provider=mock,
        memory=memory,
    )
    return agent, mock


def llm_spans(agent: Agent):
    return [e for e in agent.get_trace().all_spans() if e.event_type is EventType.LLM_CALL]


def trims(agent: Agent):
    return [
        e
        for e in agent.get_trace().all_spans()
        if e.event_type is EventType.COMPACTION and e.name == "trim"
    ]


def _bulky_history(agent: Agent, rounds: int = 8) -> None:
    """One earlier turn of large tool results ending in reasoning. A single
    turn cannot be dropped, so a trim has to clear results instead."""
    memory = agent.memory.short_term
    memory.append(UserMessage(content="research this"))
    for t in range(rounds):
        call = ToolCallRequest(id=f"t{t}", name="search", arguments={"q": str(t)})
        memory.extend(
            [
                AssistantMessage(content="", tool_calls=[call]),
                ToolResultMessage(
                    content=[ToolResult(tool_use_id=f"t{t}", name="search", content="r" * 2_000)]
                ),
            ]
        )
    memory.append(
        AssistantMessage(
            content="a" * 400,
            reasoning=[ReasoningContent(provider="anthropic", raw={"type": "thinking"})],
        )
    )


async def test_llm_span_records_what_was_sent():
    agent, mock = make_agent(
        [ScriptedTurn(text="ok")],
        memory=MemoryConfig(),
        max_context_tokens=4_000,
        max_tool_result_tokens=400,
    )
    _bulky_history(agent)
    await agent.run("next")

    [span] = llm_spans(agent)
    p = span.payload
    sent = mock.call_log[0]["messages"]
    assert p["messages_sent"] == len(sent)
    assert p["messages_stored"] == len(agent.get_history()) - 1  # the answer came after
    assert p["trimmed"] is True
    assert p["cleared_results"] == sum(
        1
        for m in sent
        if isinstance(m, ToolResultMessage)
        for r in m.content
        if str(r.content).startswith("[cleared")
    )
    assert p["cleared_results"] > 0
    assert p["reasoning_dropped"] > 0
    assert 0 < p["estimated_tokens"] <= p["max_context_tokens"] == 4_000


async def test_cache_hit_ratio_is_recorded():
    usage = TokenUsage(
        input_tokens=100, output_tokens=10, cache_read_input_tokens=800, cache_creation_input_tokens=100
    )
    agent, _ = make_agent([ScriptedTurn(text="ok", usage=usage), ScriptedTurn(text="ok", usage=usage)])
    result = await agent.run("go")
    async for _ in agent.run_stream("go"):
        pass

    for span in llm_spans(agent):
        assert span.payload["cache_hit_ratio"] == 0.8
        assert span.payload["cache_read_tokens"] == 800
        assert span.payload["cache_write_tokens"] == 100
    assert result.context.cache_hit_ratio == 0.8


async def test_result_carries_a_context_report():
    agent, mock = make_agent(
        [
            ScriptedTurn(tool_calls=[{"id": "c1", "name": "search", "arguments": {"q": "x"}}]),
            ScriptedTurn(text="done"),
        ]
    )
    result = await agent.run("go")

    report = result.context
    last_sent = mock.call_log[-1]["messages"]
    assert report.messages_sent == len(last_sent) == 3  # user, tool call, its result
    assert report.messages_stored == 3
    assert report.trimmed is False and report.cleared_results == 0
    assert report.estimated_tokens > 0
    assert report.to_dict()["messages_sent"] == 3


async def test_dry_run_calls_nothing_and_changes_nothing():
    agent, mock = make_agent(
        [ScriptedTurn(text="real")],
        memory=MemoryConfig(),
        max_context_tokens=4_000,
        max_tool_result_tokens=400,
    )
    _bulky_history(agent)
    history = agent.get_history()
    state, epoch = agent.memory.short_term.state, agent.memory.short_term.epoch

    preview = await agent.run("next", dry_run=True)

    assert mock.call_log == []  # no model call
    assert agent.get_history() == history  # nothing stored ...
    assert agent.memory.short_term.state == state  # ... or trimmed
    assert agent.memory.short_term.epoch == epoch
    assert preview.stop_reason is StopReason.DRY_RUN
    assert preview.messages[-1].content == "next"
    assert preview.context.trimmed is True
    [dry] = trims(agent)
    assert dry.payload["dry_run"] is True

    # The real run then trims for itself — and records it as a real trim.
    await agent.run("next")

    def shape(messages):
        return [(type(m).__name__, str(m.content)) for m in messages]

    assert shape(mock.call_log[0]["messages"]) == shape(preview.messages)
    real = trims(agent)[-1]
    assert real.payload["applied"] is True and "dry_run" not in real.payload


async def test_dry_run_on_a_stateless_agent_shows_a_fresh_run():
    agent, mock = make_agent([ScriptedTurn(text="first answer")])
    await agent.run("first")

    preview = await agent.run("second", dry_run=True)

    assert [m.content for m in preview.messages] == ["second"]
    assert preview.context.messages_stored == 1
    assert len(mock.call_log) == 1  # only the real run called the model
    assert [m.content for m in agent.get_history()] == ["first", "first answer"]
    run = [e for e in agent.get_trace().all_spans() if e.event_type is EventType.RUN][-1]
    assert run.payload["dry_run"] is True
