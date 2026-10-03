"""`context_filter`: the developer decides what is sent, never what is stored.

The filter's output is repaired into a valid conversation, its failures fall
back to the framework's own view, and on models that bind reasoning to the
prefix nothing already sent changes in the middle of a run.
"""

from __future__ import annotations

import dataclasses

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.config import AgentConfig, MemoryConfig
from agent_framework.core.context_filter import OMITTED_RESULT, ContextInfo
from agent_framework.core.message import (
    AssistantMessage,
    ReasoningContent,
    ToolCallRequest,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from agent_framework.permissions.models import PermissionCategory
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn
from agent_framework.tools.decorator import tool
from agent_framework.tracing.trace import EventType, SpanStatus


@tool(description="search", permission=PermissionCategory.AUTO)
async def search(q: str) -> str:
    return f"results for {q}"


def make_agent(script, context_filter, *, model="claude-haiku-4-5", memory=None):
    mock = MockLLMProvider(list(script))
    agent = Agent(
        AgentConfig(model=model, api_key="k"),
        tools=[search],
        provider=mock,
        memory=memory,
        context_filter=context_filter,
    )
    return agent, mock


def one_tool_round():
    return [
        ScriptedTurn(
            tool_calls=[{"id": "c1", "name": "search", "arguments": {"q": "x"}}],
            reasoning=[{"type": "thinking", "thinking": "", "signature": "s1"}],
        ),
        ScriptedTurn(text="done"),
    ]


def events(agent, name):
    return [e for e in agent.get_trace().all_spans() if e.name == name]


def contents(messages):
    return [str(m.content) for m in messages]


def _thinking(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=text, reasoning=[ReasoningContent(provider="anthropic", raw={"type": "thinking"})]
    )


# ---------------- the basics ----------------

async def test_filter_changes_only_what_is_sent():
    def current_turn_only(messages, info):
        return [m for m in messages if not str(m.content).startswith("old")]

    agent, mock = make_agent([ScriptedTurn(text="ok")], current_turn_only, memory=MemoryConfig())
    agent.memory.short_term.extend(
        [UserMessage(content="old question"), AssistantMessage(content="old answer")]
    )
    result = await agent.run("new question")

    assert contents(mock.call_log[0]["messages"]) == ["new question"]
    assert contents(agent.get_history()) == ["old question", "old answer", "new question", "ok"]
    assert result.context.filtered is True


async def test_async_filter():
    async def drop_old(messages, info):
        return messages[-1:]

    agent, mock = make_agent([ScriptedTurn(text="ok")], drop_old, memory=MemoryConfig())
    agent.memory.short_term.extend([UserMessage(content="a"), AssistantMessage(content="b")])
    await agent.run("c")
    assert contents(mock.call_log[0]["messages"]) == ["c"]


@pytest.mark.parametrize("model,binds", [("claude-opus-5-5", True), ("claude-haiku-4-5", False)])
async def test_filter_receives_info(model, binds):
    seen: list[ContextInfo] = []

    def record(messages, info):
        seen.append(info)
        return messages

    agent, _ = make_agent(one_tool_round(), record, model=model)
    await agent.run("go")

    assert [(i.call, i.run_start) for i in seen] == [(1, True), (2, False)]
    assert all(i.model == model and i.reasoning_binds_to_prefix is binds for i in seen)
    assert all(i.max_context_tokens == agent.config.resolved_max_context_tokens for i in seen)
    assert all(i.estimated_tokens > 0 and i.trimmed is False for i in seen)


async def test_the_filter_cannot_mutate_the_history():
    def vandal(messages, info):
        for m in messages:
            m.content = "overwritten"  # on a copy: the stored message is safe
        return messages

    agent, _ = make_agent([ScriptedTurn(text="ok")], vandal)
    await agent.run("original")
    assert agent.get_history()[0].content == "original"


# ---------------- repair ----------------

async def test_broken_pairs_are_repaired_and_traced():
    a = ToolCallRequest(id="a", name="search", arguments={})
    c2 = ToolCallRequest(id="c2", name="search", arguments={})
    c3 = ToolCallRequest(id="c3", name="search", arguments={})

    def broken(messages, info):
        return [
            AssistantMessage(content="", tool_calls=[a]),  # before any user message
            ToolResultMessage(content=[ToolResult(tool_use_id="a", name="search", content="r")]),
            messages[-1],
            AssistantMessage(content="", tool_calls=[c2, c3]),
            ToolResultMessage(
                content=[
                    ToolResult(tool_use_id="c2", name="search", content="r2"),
                    ToolResult(tool_use_id="c9", name="search", content="orphan"),
                ]
            ),
        ]

    agent, mock = make_agent([ScriptedTurn(text="ok")], broken)
    await agent.run("q")

    sent = mock.call_log[0]["messages"]
    assert isinstance(sent[0], UserMessage) and sent[0].content == "q"
    results = {r.tool_use_id: r.content for r in sent[-1].content}
    assert results == {"c2": "r2", "c3": OMITTED_RESULT}
    [event] = events(agent, "context_filter")
    assert event.payload["leading_dropped"] == 2
    assert event.payload["orphan_results_dropped"] == 1
    assert event.payload["missing_results_added"] == 1


@pytest.mark.parametrize(
    "bad_filter",
    [
        pytest.param(lambda messages, info: 1 / 0, id="raises"),
        pytest.param(lambda messages, info: "not a list", id="wrong-type"),
        pytest.param(lambda messages, info: [AssistantMessage(content="hi")], id="no-user-turn"),
        pytest.param(
            lambda messages, info: messages + [AssistantMessage(content="prefill")],
            id="ends-with-assistant",
        ),
    ],
)
async def test_filter_exception_falls_back_and_is_traced(bad_filter):
    agent, mock = make_agent([ScriptedTurn(text="ok")], bad_filter)
    result = await agent.run("q")

    assert result.error is None and result.final_message == "ok"
    assert contents(mock.call_log[0]["messages"]) == ["q"]  # the unfiltered view
    [error] = events(agent, "context_filter_error")
    assert error.event_type is EventType.ERROR and error.status is SpanStatus.ERROR
    assert result.context.filtered is False


# ---------------- models that bind reasoning to the prefix ----------------

def _edit_first_message_from_call_2(messages, info):
    if info.call < 2:
        return messages
    return [dataclasses.replace(messages[0], content="rewritten")] + messages[1:]


async def test_mid_run_prefix_edit_is_rejected_on_anthropic():
    agent, mock = make_agent(one_tool_round(), _edit_first_message_from_call_2, model="claude-opus-5-5")
    await agent.run("original task")

    first, second = mock.call_log[0]["messages"], mock.call_log[1]["messages"]
    assert second[: len(first)] == first  # what was sent stays as it was sent
    assert second[0].content == "original task"
    assert second[1].reasoning  # the tool loop's reasoning is echoed unchanged
    [rejected] = events(agent, "context_filter_rejected")
    assert rejected.status is SpanStatus.ERROR


async def test_the_same_edit_mid_run_is_fine_on_other_models():
    agent, mock = make_agent(one_tool_round(), _edit_first_message_from_call_2)
    await agent.run("original task")

    assert mock.call_log[1]["messages"][0].content == "rewritten"
    assert not events(agent, "context_filter_rejected")


async def test_a_consistent_filter_is_not_rejected_mid_run():
    """Dropping the same old message on every call keeps the sent prefix
    stable — it must not be mistaken for an edit."""

    def drop_old(messages, info):
        return [m for m in messages if m.content != "old"]

    agent, mock = make_agent(
        one_tool_round(), drop_old, model="claude-opus-5-5", memory=MemoryConfig()
    )
    agent.memory.short_term.extend([UserMessage(content="old"), _thinking("old answer")])
    await agent.run("task")

    first, second = mock.call_log[0]["messages"], mock.call_log[1]["messages"]
    assert "old" not in contents(second)
    assert second[: len(first)] == first
    assert not events(agent, "context_filter_rejected")


async def test_a_run_start_edit_strips_older_reasoning_on_bound_models():
    def drop_first(messages, info):
        return messages[1:] if info.run_start else messages

    agent, mock = make_agent(
        [ScriptedTurn(text="one"), ScriptedTurn(text="two")],
        drop_first,
        model="claude-opus-5-5",
        memory=MemoryConfig(),
    )
    agent.memory.short_term.extend(
        [UserMessage(content="q0"), _thinking("a0"), UserMessage(content="q1"), _thinking("a1")]
    )
    await agent.run("q2")

    sent = mock.call_log[0]["messages"]
    # q0 removed by the filter; a0 then led the view and was repaired away.
    assert contents(sent) == ["q1", "a1", "q2"]
    assert not any(isinstance(m, AssistantMessage) and m.reasoning for m in sent)
    [event] = events(agent, "context_filter")
    assert event.payload["reasoning_stripped"] == 1
    assert event.payload["leading_dropped"] == 1
    # The stored history keeps its reasoning.
    assert all(m.reasoning for m in agent.get_history()[:4] if isinstance(m, AssistantMessage))


async def test_an_unchanged_view_keeps_reasoning_at_a_run_start():
    """A filter that leaves the sent prefix alone costs no reasoning."""

    def identity(messages, info):
        return messages

    agent, mock = make_agent(
        [
            ScriptedTurn(text="one", reasoning=[{"type": "thinking", "signature": "s"}]),
            ScriptedTurn(text="two"),
        ],
        identity,
        model="claude-opus-5-5",
        memory=MemoryConfig(),
    )
    await agent.run("q1")
    await agent.run("q2")

    first, second = mock.call_log[0]["messages"], mock.call_log[1]["messages"]
    assert second[: len(first)] == first
    assert second[1].reasoning  # run 1's reasoning is sent again, unchanged
    assert not events(agent, "context_filter")


async def test_a_per_call_reminder_stays_where_it_was_sent_mid_run():
    """Re-appending a reminder on every call moves it — an edit of what was
    sent. Mid-run on a bound model it is rejected: the reminder stays at its
    first position and the new messages follow it."""

    def remind(messages, info):
        return messages + [UserMessage(content="(be brief)")]

    agent, mock = make_agent(one_tool_round(), remind, model="claude-opus-5-5")
    await agent.run("task")

    first, second = mock.call_log[0]["messages"], mock.call_log[1]["messages"]
    assert contents(first) == ["task", "(be brief)"]
    assert second[: len(first)] == first
    assert events(agent, "context_filter_rejected")
