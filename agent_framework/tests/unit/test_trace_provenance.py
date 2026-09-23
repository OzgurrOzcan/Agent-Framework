"""A trace must explain its own behaviour.

Two levels, because there are two facts:
  - SESSION: how the agent was built (one per agent).
  - RUN:     how one run was shaped (one per run/stream/complete).

Config defaults and per-call overrides can disagree, so the effective policy
is recoverable from nowhere but the RUN span.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.config import AgentConfig, CompactionConfig
from agent_framework.core.message import StopReason
from agent_framework.permissions.models import PermissionCategory
from agent_framework.providers.base import ToolChoice
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn
from agent_framework.tools.decorator import tool
from agent_framework.tracing.cost import PricingTable
from agent_framework.tracing.trace import (
    ErrorInfo,
    EventType,
    ExecutionTrace,
    SpanStatus,
)


@tool(description="Search", permission=PermissionCategory.AUTO)
async def search(q: str) -> str:
    return f"results for {q}"


def make_agent(script=None, **kwargs) -> Agent:
    kwargs.setdefault("api_key", "test-key")
    agent = Agent(kwargs.pop("config", "claude-opus-5"), **kwargs)
    mock = MockLLMProvider(list(script or []))
    agent._provider = mock
    agent._loop.provider = mock
    return agent


def spans(agent: Agent, event_type: EventType):
    return [s for s in agent.get_trace().all_spans() if s.event_type is event_type]


def session_payload(agent: Agent) -> dict:
    recorded = spans(agent, EventType.SESSION)
    assert len(recorded) == 1, f"expected exactly one session span, got {len(recorded)}"
    return recorded[0].payload


# ---------------- the session span ----------------

def test_one_session_span_is_recorded_per_agent():
    agent = make_agent()
    assert spans(agent, EventType.SESSION)[0].name == "session"


def test_the_session_span_carries_the_settings_that_explain_a_run():
    agent = make_agent(
        config=AgentConfig(
            model="claude-opus-5",
            api_key="test-key",
            max_turns=4,
            max_budget_usd=0.25,
            temperature=0.3,
            compaction=CompactionConfig(trigger_ratio=0.85),
        ),
        tools=[search],
    )
    payload = session_payload(agent)
    assert payload["model"] == "claude-opus-5"
    assert payload["provider"] == "anthropic"
    assert payload["max_turns"] == 4
    assert payload["max_budget_usd"] == 0.25
    assert payload["temperature"] == 0.3
    assert payload["compaction"]["trigger_ratio"] == 0.85
    assert payload["tools"] == ["ask_user", "search"]
    assert payload["framework_version"]


def test_the_session_span_records_the_resolved_context_window():
    """A compaction span alone cannot say whether the window was defaulted."""
    explicit = make_agent(
        config=AgentConfig(
            model="claude-opus-5", api_key="k", context_window=32_000
        )
    )
    inferred = make_agent()
    assert session_payload(explicit)["resolved_context_window"] == 32_000
    assert session_payload(inferred)["resolved_context_window"] == 1_000_000


def test_the_session_span_says_whether_the_budget_guard_is_armed():
    """An unpriced model records 0.0 cost and disables max_budget_usd."""
    priced = make_agent(max_budget_usd=1.0)
    unpriced = make_agent(
        config=AgentConfig(model="claude-opus-5", api_key="k"),
        pricing=PricingTable({}),
    )
    assert session_payload(priced)["pricing_resolved"] is True
    assert session_payload(unpriced)["pricing_resolved"] is False


def test_the_session_span_never_contains_the_api_key():
    agent = make_agent(api_key="sk-super-secret")
    payload = session_payload(agent)
    assert "sk-super-secret" not in json.dumps(payload)
    assert payload["api_key_set"] is True


def test_the_session_span_is_json_serializable():
    """It has to survive the JSONL and SQLite exporters unchanged."""
    event = spans(make_agent(tools=[search]), EventType.SESSION)[0]
    assert json.loads(json.dumps(event.to_dict()))["payload"]["model"]


def test_the_session_span_reports_the_mcp_discovery_mode():
    assert session_payload(make_agent())["mcp_discovery"] == "none"


def test_two_agents_sharing_a_trace_each_record_their_own_session():
    first = make_agent(system_prompt="first")
    second = Agent(
        "claude-opus-5", api_key="k", system_prompt="second", trace=first.get_trace()
    )
    recorded = spans(second, EventType.SESSION)
    assert [s.payload["system_prompt"] for s in recorded] == ["first", "second"]


# ---------------- the run span ----------------

async def test_a_run_records_its_effective_policy():
    agent = make_agent([ScriptedTurn(text="done")])
    await agent.run("go")
    payload = spans(agent, EventType.RUN)[0].payload
    assert payload["max_turns"] == 10
    assert payload["execute_tools"] is True
    assert payload["max_tool_rounds"] is None
    assert payload["streaming"] is False


async def test_a_per_call_override_is_visible_in_the_run_not_the_session():
    """The config says 10; this run was capped at 3. Both facts are recorded."""
    agent = make_agent([ScriptedTurn(text="done")], max_turns=10)
    await agent.run("go", max_turns=3)
    assert session_payload(agent)["max_turns"] == 10
    assert spans(agent, EventType.RUN)[0].payload["max_turns"] == 3


async def test_max_tool_rounds_is_distinguishable_from_a_permission_wall():
    """Both produce tool_use with no TOOL_CALL span; only the policy says why."""
    agent = make_agent(
        [
            ScriptedTurn(
                text="searching",
                tool_calls=[{"id": "c1", "name": "search", "arguments": {"q": "x"}}],
            ),
            ScriptedTurn(text="done"),
        ],
        tools=[search],
    )
    await agent.run("go", max_tool_rounds=1)
    assert spans(agent, EventType.RUN)[0].payload["max_tool_rounds"] == 1


async def test_complete_records_that_tools_were_never_going_to_run():
    agent = make_agent(
        [
            ScriptedTurn(
                text="I would search",
                tool_calls=[{"id": "c1", "name": "search", "arguments": {"q": "x"}}],
            )
        ],
        tools=[search],
    )
    await agent.complete("go")
    payload = spans(agent, EventType.RUN)[0].payload
    assert payload["execute_tools"] is False
    assert payload["max_turns"] == 1
    assert not spans(agent, EventType.TOOL_CALL)


async def test_a_forced_tool_choice_is_recorded():
    agent = make_agent([ScriptedTurn(text="done")], tools=[search])
    await agent.run("go", tool_choice=ToolChoice(mode="tool", tool_name="search"))
    payload = spans(agent, EventType.RUN)[0].payload
    assert payload["tool_choice"] == "tool"
    assert payload["tool_choice_name"] == "search"


async def test_the_run_span_records_the_outcome():
    agent = make_agent([ScriptedTurn(text="done")])
    await agent.run("go")
    payload = spans(agent, EventType.RUN)[0].payload
    assert payload["stop_reason"] == StopReason.END_TURN.value
    assert payload["turns"] == 1
    assert payload["tool_rounds"] == 0


async def test_turns_nest_under_their_run():
    agent = make_agent([ScriptedTurn(text="done")])
    await agent.run("go")
    run_span = spans(agent, EventType.RUN)[0]
    turn_spans = spans(agent, EventType.TURN)
    assert turn_spans
    assert all(t.parent_span_id == run_span.span_id for t in turn_spans)


async def test_each_run_gets_its_own_span():
    agent = make_agent([ScriptedTurn(text="a"), ScriptedTurn(text="b")])
    await agent.run("first", max_turns=5)
    await agent.run("second", max_turns=2)
    assert [s.payload["max_turns"] for s in spans(agent, EventType.RUN)] == [5, 2]


async def test_a_guard_stop_marks_the_run_span_as_failed():
    """The loop returns errors instead of raising, so the span is set by hand."""
    agent = make_agent(
        [
            ScriptedTurn(
                text="looping",
                tool_calls=[{"id": "c1", "name": "search", "arguments": {"q": "x"}}],
            )
        ]
        * 6,
        tools=[search],
    )
    result = await agent.run("go", max_turns=2)
    assert result.stop_reason is StopReason.ERROR
    run_span = spans(agent, EventType.RUN)[0]
    assert run_span.status is SpanStatus.ERROR
    assert run_span.error is not None
    assert agent.get_trace().where_it_failed()


async def test_the_run_span_has_a_duration():
    agent = make_agent([ScriptedTurn(text="done")])
    await agent.run("go")
    assert spans(agent, EventType.RUN)[0].duration_ms is not None


@pytest.mark.parametrize("streaming", [False, True])
async def test_both_transports_record_a_run_span(streaming):
    agent = make_agent([ScriptedTurn(text="done")])
    if streaming:
        async for _ in agent.run_stream("go"):
            pass
    else:
        await agent.run("go")
    payload = spans(agent, EventType.RUN)[0].payload
    assert payload["streaming"] is streaming


# ---------------- where_it_failed() must name the cause ----------------

def _looping_script(n: int = 6):
    return [
        ScriptedTurn(
            text="looping",
            tool_calls=[{"id": "c1", "name": "search", "arguments": {"q": "x"}}],
        )
    ] * n


async def test_where_it_failed_ends_at_the_guard_not_at_the_run_wrapper():
    """A RUN span fails *because* its guard failed; the chain must reach the guard."""
    agent = make_agent(_looping_script(), tools=[search])
    await agent.run("go", max_turns=2)
    chain = agent.get_trace().where_it_failed()
    assert [s.event_type for s in chain] == [EventType.RUN, EventType.LOOP_GUARD]
    assert chain[-1].name == "max_turns"


async def test_where_it_failed_reaches_a_provider_error_through_its_ancestors():
    agent = make_agent([ScriptedTurn(raise_exc=RuntimeError("upstream 503"))])
    await agent.run("go")
    chain = agent.get_trace().where_it_failed()
    assert chain[0].event_type is EventType.RUN
    # The RUN span carries the same message, so check the chain goes deeper.
    assert chain[-1].event_type is not EventType.RUN
    assert chain[-1].error is not None
    assert "upstream 503" in chain[-1].error.message


def _err(message: str = "boom") -> ErrorInfo:
    return ErrorInfo(exception_type="RuntimeError", message=message, traceback="")


def test_where_it_failed_prefers_a_failing_child_over_its_failing_parent():
    trace = ExecutionTrace()
    with trace.start_span(EventType.RUN, name="run") as parent:
        trace.record_event(EventType.ERROR, name="cause", status=SpanStatus.ERROR, error=_err())
        parent.set_error(_err("consequence"))
    assert [s.name for s in trace.where_it_failed()] == ["run", "cause"]


def test_where_it_failed_returns_a_failing_span_that_has_no_failing_child():
    trace = ExecutionTrace()
    with trace.start_span(EventType.RUN, name="run") as parent:
        trace.record_event(EventType.TURN, name="fine", status=SpanStatus.OK)
        parent.set_error(_err())
    assert [s.name for s in trace.where_it_failed()] == ["run"]


def test_where_it_failed_picks_the_earliest_root_cause():
    trace = ExecutionTrace()
    trace.record_event(EventType.ERROR, name="first", status=SpanStatus.ERROR, error=_err())
    trace.record_event(EventType.ERROR, name="second", status=SpanStatus.ERROR, error=_err())
    assert [s.name for s in trace.where_it_failed()] == ["first"]


# ---------------- cancellation is not an error ----------------

def test_a_span_interrupted_by_cancellation_is_cancelled_not_failed():
    trace = ExecutionTrace()
    with pytest.raises(asyncio.CancelledError), trace.start_span(EventType.TURN, name="t"):
        raise asyncio.CancelledError()
    [span] = trace.all_spans()
    assert span.status is SpanStatus.CANCELLED
    assert span.duration_ms is not None


def test_a_span_closed_by_generator_exit_is_cancelled_not_failed():
    trace = ExecutionTrace()
    with pytest.raises(GeneratorExit), trace.start_span(EventType.TURN, name="t"):
        raise GeneratorExit()
    assert trace.all_spans()[0].status is SpanStatus.CANCELLED


def test_a_span_raising_an_ordinary_exception_is_still_an_error():
    trace = ExecutionTrace()
    with pytest.raises(ValueError), trace.start_span(EventType.TURN, name="t"):
        raise ValueError("bad")
    assert trace.all_spans()[0].status is SpanStatus.ERROR


async def test_a_cancelled_run_marks_its_run_span_cancelled():
    agent = make_agent([ScriptedTurn(raise_exc=asyncio.CancelledError())])
    with pytest.raises(asyncio.CancelledError):
        await agent.run("go")
    [run_span] = spans(agent, EventType.RUN)
    assert run_span.status is SpanStatus.CANCELLED
    assert run_span.duration_ms is not None


# ---------------- streaming parity ----------------

async def _drain(agent: Agent, prompt: str, **kwargs) -> list:
    return [chunk async for chunk in agent.run_stream(prompt, **kwargs)]


async def test_streamed_runs_count_their_own_turns():
    """The trace-wide turn counter is cumulative; a run must report its own."""
    agent = make_agent([ScriptedTurn(text="a"), ScriptedTurn(text="b")])
    await _drain(agent, "one")
    await _drain(agent, "two")
    assert [s.payload["turns"] for s in spans(agent, EventType.RUN)] == [1, 1]


async def test_streamed_run_records_tool_rounds_and_cost():
    agent = make_agent(
        [
            ScriptedTurn(
                text="searching",
                tool_calls=[{"id": "c1", "name": "search", "arguments": {"q": "x"}}],
            ),
            ScriptedTurn(text="done"),
        ],
        tools=[search],
    )
    await _drain(agent, "go")
    payload = spans(agent, EventType.RUN)[0].payload
    assert (payload["turns"], payload["tool_rounds"]) == (2, 1)
    assert "cost_usd" in payload


async def test_a_streamed_guard_stop_marks_the_run_span_as_failed():
    """Streaming reports failure as an ERROR chunk; the span must still say so."""
    agent = make_agent(_looping_script(), tools=[search])
    chunks = await _drain(agent, "go", max_turns=2)
    assert any(c.type.value == "error" for c in chunks)
    [run_span] = spans(agent, EventType.RUN)
    assert run_span.status is SpanStatus.ERROR
    assert "max_turns" in run_span.error.message


async def test_an_early_closed_stream_leaves_the_agent_usable():
    """A "stop generating" button must not brick the agent (pre-existing bug)."""
    agent = make_agent([ScriptedTurn(text="first"), ScriptedTurn(text="second")])
    stream = agent.run_stream("go")
    async for _ in stream:
        break
    await stream.aclose()
    result = await agent.run("next")
    assert result.final_message == "second"


async def test_an_early_closed_stream_ends_its_run_span_as_cancelled():
    agent = make_agent([ScriptedTurn(text="hello")])
    stream = agent.run_stream("go")
    async for _ in stream:
        break
    await stream.aclose()
    [run_span] = spans(agent, EventType.RUN)
    assert run_span.status is SpanStatus.CANCELLED
    assert run_span.duration_ms is not None
