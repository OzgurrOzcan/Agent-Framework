"""Reasoning stays valid on models that bind it to the conversation prefix.

On such models (Claude Fable 5.1, Opus 5.5, Sonnet 5.5) a reasoning block is
rejected once any message before it changed. So nothing already sent changes
in the middle of a run: trims happen only at a run's first call, memory
context is stored once instead of regenerated per call, and a prompt that
cannot fit is stopped before it is sent.
"""

from __future__ import annotations

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.config import AgentConfig, MemoryConfig
from agent_framework.core.context_manager import ContextManager
from agent_framework.core.message import (
    AssistantMessage,
    ReasoningContent,
    StopReason,
    ToolResultMessage,
    UserMessage,
)
from agent_framework.core.policy import ContextWindowExceededError
from agent_framework.memory.backends.in_memory import InMemoryBackend
from agent_framework.memory.short_term import ShortTermMemory
from agent_framework.permissions.models import PermissionCategory
from agent_framework.providers.base import StreamChunkType
from agent_framework.providers.catalog import get_provider, provider_names
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn
from agent_framework.tools.decorator import tool
from agent_framework.tracing.trace import EventType


@tool(description="fetch a page", permission=PermissionCategory.AUTO)
async def fetch(n: int) -> str:
    return f"page {n} " + "x" * 4_000  # ≈1000 tokens


def _tool_rounds(n: int) -> list[ScriptedTurn]:
    rounds = [
        ScriptedTurn(
            text="",
            tool_calls=[{"id": f"c{i}", "name": "fetch", "arguments": {"n": i}}],
            reasoning=[{"type": "thinking", "thinking": "", "signature": f"sig{i}"}],
        )
        for i in range(n)
    ]
    return rounds + [ScriptedTurn(text="done")]


def make_agent(
    model: str, script, *, memory: MemoryConfig | None = None, **config
) -> tuple[Agent, MockLLMProvider]:
    config.setdefault("context_window", 40_000)
    config.setdefault("max_tokens_per_response", 1_000)
    config.setdefault("max_context_tokens", 8_000)
    config.setdefault("max_tool_result_tokens", 1_000)
    config.setdefault("max_turns", 40)
    mock = MockLLMProvider(list(script))
    agent = Agent(
        AgentConfig(model=model, api_key="k", **config),
        tools=[fetch],
        provider=mock,
        memory=memory,
    )
    return agent, mock


def applied_trims(agent: Agent):
    return [
        e
        for e in agent.get_trace().all_spans()
        if e.event_type is EventType.COMPACTION and e.payload.get("applied")
    ]


def _with_reasoning(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=text,
        reasoning=[ReasoningContent(provider="anthropic", raw={"type": "thinking"})],
    )


# ---------------- the binding table ----------------

def test_binding_table_covers_the_catalog():
    anthropic = get_provider("anthropic")
    for bound in ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5"):
        assert anthropic.binds_reasoning_to_prefix(bound), bound
    for unbound in (
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-8",
        "claude-opus-4-5-20251101",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-mythos-5-1",  # does not run the conversation check
        "claude-3-7-sonnet-latest",
    ):
        assert not anthropic.binds_reasoning_to_prefix(unbound), unbound
    # A Claude model the table does not know yet is assumed to bind.
    assert anthropic.binds_reasoning_to_prefix("claude-opus-6")
    assert anthropic.binds_reasoning_to_prefix("claude-opus-5-6")
    # Chat Completions never sends reasoning back.
    openai = get_provider("openai")
    assert not any(openai.binds_reasoning_to_prefix(m) for m in ("gpt-5", "o3", "gpt-4o"))
    # Every registered provider answers for every model it has a window for.
    for name in provider_names():
        spec = get_provider(name)
        for prefix, _ in spec.context_windows:
            assert isinstance(spec.binds_reasoning_to_prefix(prefix), bool)


# ---------------- where trims may happen ----------------

async def test_no_mid_run_trim_on_bound_models():
    agent, mock = make_agent("claude-opus-5-5", _tool_rounds(12))
    result = await agent.run("research")

    assert result.final_message == "done"
    llm_calls = [e for e in agent.get_trace().all_spans() if e.event_type is EventType.LLM_CALL]
    assert llm_calls[-1].payload["estimated_tokens"] > 8_000  # the run did outgrow the limit ...
    assert not applied_trims(agent)  # ... and still nothing already sent was edited
    for earlier, later in zip(mock.call_log, mock.call_log[1:]):
        sent_before = earlier["messages"]
        assert later["messages"][: len(sent_before)] == sent_before


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "gpt-5"])
async def test_mid_run_trim_on_unbound_models(model):
    agent, mock = make_agent(model, _tool_rounds(12))
    result = await agent.run("research")

    assert result.final_message == "done"
    trims = applied_trims(agent)
    assert trims and all(t.payload["cleared_results"] > 0 for t in trims)
    edited = [
        later
        for earlier, later in zip(mock.call_log, mock.call_log[1:])
        if later["messages"][: len(earlier["messages"])] != earlier["messages"]
    ]
    assert edited  # some call sent an edited prefix: the trim happened mid-run


async def test_a_bound_model_is_still_trimmed_at_a_run_start():
    agent, mock = make_agent(
        "claude-opus-5-5", [ScriptedTurn(text="ok")], memory=MemoryConfig()
    )
    for i in range(12):
        agent.memory.short_term.extend(
            [UserMessage(content=f"q{i}"), _with_reasoning("a" * 4_000)]
        )
    await agent.run("next")

    assert applied_trims(agent)
    # No reasoning from before this run survives the edit.
    assert not any(
        isinstance(m, AssistantMessage) and m.reasoning for m in mock.call_log[0]["messages"]
    )


# ---------------- reasoning ----------------

async def test_reasoning_dropped_at_a_trim():
    mgr = ContextManager(system_prompt="", max_context_tokens=3_000, max_tool_result_tokens=200)
    memory = ShortTermMemory()
    for i in range(6):
        memory.extend([UserMessage(content=f"q{i}"), _with_reasoning("a" * 2_000)])
    memory.append(UserMessage(content="current"))
    out = await mgr.assemble(memory)

    assert out.trim.applied and out.trim.reasoning_cut
    assert not any(isinstance(m, AssistantMessage) and m.reasoning for m in out.messages)
    # The stored history keeps every block; only the view leaves them out.
    assert all(m.reasoning for m in memory if isinstance(m, AssistantMessage))


async def test_reasoning_kept_without_trims():
    mgr = ContextManager(system_prompt="", max_context_tokens=100_000, max_tool_result_tokens=200)
    memory = ShortTermMemory()
    for i in range(3):
        memory.extend([UserMessage(content=f"q{i}"), _with_reasoning("a" * 400)])
    out = await mgr.assemble(memory)

    assert out.trim is None
    assert out.messages == memory.snapshot()
    assert all(m.reasoning for m in out.messages if isinstance(m, AssistantMessage))


# ---------------- the pre-flight guard ----------------

async def test_preflight_context_exceeded_stops_the_run_without_a_request():
    # 12k window, 2k for the response: a prompt over 10k is never sent.
    agent, mock = make_agent(
        "claude-opus-5-5",
        _tool_rounds(20),
        context_window=12_000,
        max_tokens_per_response=2_000,
        max_context_tokens=5_000,
    )
    result = await agent.run("research")

    assert result.stop_reason is StopReason.CONTEXT_EXCEEDED
    assert isinstance(result.error, ContextWindowExceededError)
    assert result.error.estimated_tokens > result.error.limit == 10_000
    assert len(mock.call_log) < 20  # stopped before sending, not after a 400
    # The history is still a valid conversation: the last call got its result.
    assert isinstance(agent.get_history()[-1], ToolResultMessage)
    [guard] = [
        e
        for e in agent.get_trace().all_spans()
        if e.event_type is EventType.LOOP_GUARD and e.name == "context_exceeded"
    ]
    assert guard.payload["usable_tokens"] == 10_000


async def test_preflight_in_a_stream_yields_an_error_chunk():
    agent, mock = make_agent(
        "claude-opus-5-5",
        _tool_rounds(20),
        context_window=12_000,
        max_tokens_per_response=2_000,
        max_context_tokens=5_000,
    )
    chunks = [c async for c in agent.run_stream("research")]

    assert chunks[-1].type is StreamChunkType.ERROR
    assert chunks[-1].payload["reason"] == "context_exceeded"
    assert len(mock.call_log) < 20


async def test_the_next_run_recovers_with_a_trim_at_its_start():
    """With memory, the overflowing run's history is trimmed when the next
    run starts — the one point where a bound model may be edited."""
    agent, mock = make_agent(
        "claude-opus-5-5",
        _tool_rounds(20),
        context_window=12_000,
        max_tokens_per_response=2_000,
        max_context_tokens=5_000,
        memory=MemoryConfig(),
    )
    first = await agent.run("research")
    assert first.stop_reason is StopReason.CONTEXT_EXCEEDED

    mock._script = [ScriptedTurn(text="summary of what I found")]
    mock._call_count = 0
    second = await agent.run("summarize what you found")
    assert second.stop_reason is StopReason.END_TURN
    assert applied_trims(agent)


# ---------------- memory context is stored once ----------------

def _memory_agent(script, backend) -> tuple[Agent, MockLLMProvider]:
    mock = MockLLMProvider(list(script))
    agent = Agent(
        "claude-opus-5-5",
        api_key="k",
        tools=[fetch],
        provider=mock,
        memory=MemoryConfig(),
        memory_backend=backend,
    )
    return agent, mock


async def test_facts_are_stored_once_per_run_not_per_call():
    backend = InMemoryBackend()
    agent, mock = _memory_agent(_tool_rounds(1) + [ScriptedTurn(text="again")], backend)
    await agent.memory.remember("deploy_target", "the eu-1 cluster")

    await agent.run("which deploy_target do we use?")
    first, second = mock.call_log[0]["messages"], mock.call_log[1]["messages"]
    # The second call of the run sends the first call's messages unchanged.
    assert second[: len(first)] == first
    history = agent.get_history()
    facts = [m for m in history if m.metadata.get("context") == "relevant_facts"]
    assert len(facts) == 1
    assert "deploy_target" in facts[0].content
    assert history[history.index(facts[0]) + 1].content == "which deploy_target do we use?"

    # A later run asking about the same fact does not store it again.
    await agent.run("remind me: deploy_target?")
    again = [m for m in agent.get_history() if m.metadata.get("context") == "relevant_facts"]
    assert len(again) == 1


async def test_the_summary_is_a_stable_snapshot_at_the_head():
    backend = InMemoryBackend()
    await backend.set_summary("default", "The user prefers short answers.")
    agent, mock = _memory_agent([ScriptedTurn(text="one"), ScriptedTurn(text="two")], backend)

    await agent.run("hello")
    await backend.set_summary("default", "Something newer.")  # another session wrote
    await agent.run("again")

    first, second = mock.call_log[0]["messages"], mock.call_log[1]["messages"]
    assert "short answers" in first[0].content
    assert second[: len(first)] == first  # unchanged: a snapshot, not re-read per call


async def test_an_oversized_message_does_not_brick_a_conversation():
    """A message too large to send is not kept unanswered in the history,
    where every later message would join it in an untrimmable turn."""
    agent, _ = make_agent(
        "claude-haiku-4-5",
        [ScriptedTurn(text="ok")] * 6,
        memory=MemoryConfig(),
        context_window=20_000,
        max_tokens_per_response=2_000,
        max_context_tokens=None,
    )
    for i in range(3):
        await agent.run(f"hello {i}")
    huge = await agent.run("x" * 100_000)
    assert huge.stop_reason is StopReason.CONTEXT_EXCEEDED
    assert all("xxxx" not in str(m.content) for m in agent.get_history())
    run = [e for e in agent.get_trace().all_spans() if e.event_type is EventType.RUN][-1]
    assert run.payload["input_rolled_back"] is True

    for i in range(3):
        follow_up = await agent.run(f"small follow-up {i}")
        assert follow_up.stop_reason is StopReason.END_TURN


async def test_a_cancelled_run_keeps_its_message():
    """Stopping a reply is not a failure: the message was sent and stays."""
    agent, _ = make_agent("claude-haiku-4-5", [ScriptedTurn(text="ok")], memory=MemoryConfig())
    stream = agent.run_stream("keep me")
    async for _ in stream:
        break
    await stream.aclose()
    assert [m.content for m in agent.get_history()] == ["keep me"]
