"""Stateless by default; conversation memory only when configured.

Without `memory`, every run/run_stream/complete starts from an empty history
and nothing is remembered. `memory=MemoryConfig()` keeps the conversation in
RAM across runs; the history is trimmed by tokens — never by a turn count —
and trimming makes no LLM call.
"""

from __future__ import annotations

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.config import AgentConfig, MemoryConfig
from agent_framework.core.message import (
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from agent_framework.memory.backends.in_memory import InMemoryBackend
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn
from agent_framework.tracing.trace import EventType, ExecutionTrace


def make_agent(script=(), *, max_context_tokens=None, **kwargs) -> tuple[Agent, MockLLMProvider]:
    mock = MockLLMProvider(list(script))
    config = AgentConfig(model="claude-opus-5", api_key="k", max_context_tokens=max_context_tokens)
    agent = Agent(config, provider=mock, **kwargs)
    return agent, mock


def sent(mock: MockLLMProvider, call: int = -1) -> list:
    return mock.call_log[call]["messages"]


def contents(messages) -> list:
    return [m.content for m in messages]


def events(agent: Agent, event_type: EventType):
    return [e for e in agent.get_trace().all_spans() if e.event_type is event_type]


def _turn(i: int, answer_chars: int):
    return [UserMessage(content=f"question {i}"), AssistantMessage(content="a" * answer_chars)]


# ---------------- stateless by default ----------------

async def test_agent_is_stateless_by_default():
    agent, mock = make_agent([ScriptedTurn(text="one"), ScriptedTurn(text="two")])
    await agent.run("first")
    await agent.run("second")

    assert contents(sent(mock)) == ["second"]
    assert contents(agent.get_history()) == ["second", "two"]
    assert agent.config.memory is None
    assert [e.payload["retain_history"] for e in events(agent, EventType.RUN)] == [False, False]


async def test_memory_config_keeps_history():
    agent, mock = make_agent(
        [ScriptedTurn(text="one"), ScriptedTurn(text="two")], memory=MemoryConfig()
    )
    await agent.run("first")
    await agent.run("second")

    assert contents(sent(mock)) == ["first", "one", "second"]
    assert events(agent, EventType.RUN)[-1].payload["retain_history"] is True


def test_memory_override_kwarg_works():
    base = AgentConfig(model="claude-opus-5", api_key="k")
    agent = Agent(base, memory=MemoryConfig())
    assert agent.config.memory == MemoryConfig()
    assert agent._loop.retain_history is True
    assert base.memory is None  # the caller's config is never mutated


async def test_stateless_stream_and_complete_start_fresh():
    agent, mock = make_agent()
    async for _ in agent.run_stream("streamed 1"):
        pass
    async for _ in agent.run_stream("streamed 2"):
        pass
    await agent.complete("completed 1")
    await agent.complete("completed 2")

    assert [contents(call["messages"]) for call in mock.call_log] == [
        ["streamed 1"],
        ["streamed 2"],
        ["completed 1"],
        ["completed 2"],
    ]


async def test_restore_does_not_leak_into_a_stateless_run():
    agent, mock = make_agent([ScriptedTurn(text="A"), ScriptedTurn(text="B")])
    await agent.run("a")
    cid = agent.checkpoint()
    await agent.run("b")
    agent.restore(cid)
    assert contents(agent.get_history()) == ["a", "A"]

    await agent.run("c")
    assert contents(sent(mock)) == ["c"]


# ---------------- trimming by tokens, not turns ----------------

async def test_many_small_turns_are_never_trimmed():
    """A turn count says nothing about size: 200 short turns fit easily."""
    agent, mock = make_agent(memory=MemoryConfig())
    for i in range(200):
        await agent.run(f"q{i}")

    assert len(sent(mock)) == 200 + 199  # every question and every earlier answer
    assert not events(agent, EventType.COMPACTION)


async def test_few_large_turns_are_trimmed_by_tokens():
    """8 turns of ≈8k tokens with a 50k limit: trimmed to ≤ 30k in one step."""
    agent, mock = make_agent(memory=MemoryConfig(), max_context_tokens=50_000)
    for i in range(8):
        agent.memory.short_term.extend(_turn(i, answer_chars=32_000))
    await agent.run("question 8")

    [trim] = events(agent, EventType.COMPACTION)
    assert trim.payload["applied"] is True
    assert trim.payload["estimated_after"] <= 30_000
    assert trim.payload["dropped_turns"] == 5
    # First message pinned, the oldest five turns gone, the newest three kept.
    questions = [m.content for m in sent(mock) if isinstance(m, UserMessage)]
    assert questions == ["question 0", "question 5", "question 6", "question 7", "question 8"]


async def test_memory_trim_makes_no_llm_calls():
    agent, mock = make_agent(memory=MemoryConfig(), max_context_tokens=50_000)
    for i in range(8):
        agent.memory.short_term.extend(_turn(i, answer_chars=32_000))
    await agent.run("question 8")

    assert events(agent, EventType.COMPACTION)[0].payload["applied"] is True
    assert len(mock.call_log) == 1  # the run's own call, no summary call


async def test_first_message_is_pinned_once():
    agent, mock = make_agent(memory=MemoryConfig(), max_context_tokens=20_000)
    for i in range(6):
        agent.memory.short_term.extend(_turn(i, answer_chars=16_000))
    await agent.run("question 6")
    first_trim_head = agent.get_history()[0].content
    assert first_trim_head != "question 0"  # dropped from the history ...
    assert agent.memory.short_term.state.pinned.content == "question 0"  # ... and pinned

    for i in range(7, 12):
        agent.memory.short_term.extend(_turn(i, answer_chars=16_000))
    await agent.run("question 12")

    assert len([e for e in events(agent, EventType.COMPACTION) if e.payload["applied"]]) == 2
    view = sent(mock)
    assert view[0].content == "question 0"
    # The head of the history after the first trim was dropped by the second,
    # and not pinned in its place.
    assert first_trim_head not in contents(view)
    assert agent.memory.short_term.state.pinned.content == "question 0"


def test_memory_config_has_no_turn_limit():
    with pytest.raises(TypeError):
        MemoryConfig(max_history_turns=10)  # type: ignore[call-arg]


# ---------------- long-term settings need a long-term store ----------------

@pytest.mark.parametrize(
    "memory,field",
    [
        (MemoryConfig(namespace="customer-42"), "namespace"),
        (MemoryConfig(summarize=False), "summarize"),
        (MemoryConfig(summary_model="claude-haiku-4-5"), "summary_model"),
    ],
)
def test_long_term_fields_require_a_store(memory, field):
    with pytest.raises(ValueError, match=f"MemoryConfig.{field}") as exc:
        Agent("claude-opus-5", api_key="k", memory=memory)
    assert "store=" in str(exc.value)


async def test_long_term_fields_are_accepted_with_a_store_or_a_backend(tmp_path):
    with_store = Agent(
        "claude-opus-5",
        api_key="k",
        memory=MemoryConfig(store=str(tmp_path / "m.db"), namespace="customer-42"),
    )
    with_backend = Agent(
        "claude-opus-5",
        api_key="k",
        memory=MemoryConfig(namespace="customer-42"),
        memory_backend=InMemoryBackend(),
    )
    assert with_store.memory.long_term.namespace == "customer-42"
    assert isinstance(with_backend.memory.long_term.backend, InMemoryBackend)
    await with_store.memory.close()


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"memory_backend": InMemoryBackend()}, ValueError, "needs memory=MemoryConfig"),
        (
            {"memory": MemoryConfig(store="x.db"), "memory_backend": InMemoryBackend()},
            ValueError,
            "not both",
        ),
        ({"memory": MemoryConfig(), "memory_backend": object()}, TypeError, "LongTermBackend"),
        ({"memory": object()}, TypeError, "MemoryConfig"),
        ({"session_id": ""}, ValueError, "session_id"),
    ],
)
def test_memory_wiring_is_validated(kwargs, error, match):
    with pytest.raises(error, match=match):
        Agent("claude-opus-5", api_key="k", **kwargs)


def test_memory_config_values_are_validated():
    with pytest.raises(ValueError, match="namespace"):
        MemoryConfig(namespace=" ")
    with pytest.raises(ValueError, match="store"):
        MemoryConfig(store="")
    assert MemoryConfig(store=__import__("pathlib").Path("a.db")).store == "a.db"


def test_memory_config_round_trips_through_to_dict():
    config = AgentConfig(
        model="claude-opus-5",
        api_key="k",
        memory=MemoryConfig(store="s.db", namespace="n", summarize=False),
    )
    restored = AgentConfig.from_dict(config.to_dict())
    assert restored.memory == config.memory


# ---------------- provenance ----------------

def test_session_span_records_memory():
    stateless, _ = make_agent()
    remembering, _ = make_agent(
        memory=MemoryConfig(namespace="n"),
        memory_backend=InMemoryBackend(),
        session_id="ticket-4412",
    )
    [plain] = events(stateless, EventType.SESSION)
    [stateful] = events(remembering, EventType.SESSION)

    assert plain.payload["memory"] is None
    assert plain.payload["retain_history"] is False
    assert plain.payload["memory_backend"] is None
    assert plain.payload["session_id"].startswith("sess_")
    assert stateful.payload["memory"]["namespace"] == "n"
    assert stateful.payload["retain_history"] is True
    assert stateful.payload["memory_backend"] == "InMemoryBackend"
    assert stateful.payload["session_id"] == "ticket-4412"


def test_two_agents_on_one_trace_get_their_own_session():
    trace = ExecutionTrace()
    first = Agent("claude-opus-5", api_key="k", trace=trace)
    second = Agent("claude-opus-5", api_key="k", trace=trace)
    assert first.session_id != second.session_id
    assert trace.session_id not in (first.session_id, second.session_id)


# ---------------- a remembered conversation stays valid ----------------

async def test_unanswered_tool_calls_are_answered_before_the_next_run():
    """complete() returns tool calls unexecuted; the next run must not resend
    them without results — every provider rejects that."""
    agent, mock = make_agent(
        [
            ScriptedTurn(
                text="I would search",
                tool_calls=[{"id": "c1", "name": "search", "arguments": {"q": "x"}}],
            ),
            ScriptedTurn(text="ok"),
        ],
        memory=MemoryConfig(),
    )
    await agent.complete("look this up")
    await agent.run("never mind")

    view = sent(mock)
    assert isinstance(view[1], AssistantMessage) and view[1].tool_calls
    assert isinstance(view[2], ToolResultMessage)
    [result] = view[2].content
    assert (result.tool_use_id, result.is_error) == ("c1", True)
    assert view[3].content == "never mind"
