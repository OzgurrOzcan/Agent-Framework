"""The rendered view and the deterministic, cache-aware trim.

The history is stored in full; what the model sees is rendered from it and a
small `ContextState`. Between trims the view only grows (so the prompt cache
holds); a trim fires only above `max_context_tokens` and goes down to 60 % of
it in one step, without an LLM call.
"""

from __future__ import annotations

import random

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.config import AgentConfig, MemoryConfig
from agent_framework.core.context_manager import (
    CLEAR_NOTE,
    ContextManager,
    _turn_starts,
)
from agent_framework.core.message import (
    AssistantMessage,
    ReasoningContent,
    ToolCallRequest,
    ToolResult,
    ToolResultMessage,
    UserMessage,
    estimate_message_tokens,
    estimate_tokens,
)
from agent_framework.memory.short_term import ContextState, ShortTermMemory
from agent_framework.permissions.models import PermissionCategory
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn
from agent_framework.tools.decorator import tool
from agent_framework.tools.registry import ToolRegistry
from agent_framework.tracing.trace import EventType, ExecutionTrace

# ---------------------------------------------------------------- helpers

def _mgr(limit: int = 10_000, cap: int = 200, **kw) -> ContextManager:
    return ContextManager(
        system_prompt="sys", max_context_tokens=limit, max_tool_result_tokens=cap, **kw
    )


def _memory(*messages) -> ShortTermMemory:
    memory = ShortTermMemory()
    memory.extend(messages)
    return memory


def _round(call_id: str, chars: int, *, name: str = "search", is_error: bool = False):
    """One tool round: the assistant's call and its result."""
    call = ToolCallRequest(id=call_id, name=name, arguments={"q": call_id})
    return [
        AssistantMessage(content="", tool_calls=[call]),
        ToolResultMessage(
            content=[
                ToolResult(tool_use_id=call_id, name=name, content="r" * chars, is_error=is_error)
            ]
        ),
    ]


def _turn(i: int, answer_chars: int = 40):
    return [UserMessage(content=f"question {i}"), AssistantMessage(content="a" * answer_chars)]


def _results(view):
    return [r for m in view if isinstance(m, ToolResultMessage) for r in m.content]


def _is_note(result: ToolResult) -> bool:
    return isinstance(result.content, str) and result.content.startswith("[cleared to save")


def _trims(trace: ExecutionTrace):
    return [e for e in trace.all_spans() if e.event_type is EventType.COMPACTION]


def _assert_pairs_intact(view) -> None:
    calls = [tc.id for m in view if isinstance(m, AssistantMessage) for tc in m.tool_calls]
    results = [r.tool_use_id for r in _results(view)]
    assert sorted(calls) == sorted(results)
    for i, message in enumerate(view):
        if isinstance(message, AssistantMessage) and message.tool_calls:
            follower = view[i + 1]
            assert isinstance(follower, ToolResultMessage)
            assert {r.tool_use_id for r in follower.content} == {
                tc.id for tc in message.tool_calls
            }


# ---------------------------------------------------------------- tests

async def test_no_trim_below_the_limit():
    trace = ExecutionTrace()
    history = [UserMessage(content="hi"), *_round("c1", 400), AssistantMessage(content="ok")]
    memory = _memory(*history)
    out = await _mgr(trace=trace).assemble(memory)

    assert out.trim is None
    assert out.messages == history
    assert memory.state == ContextState()
    assert not _trims(trace)


async def test_view_is_append_only_between_trims():
    mgr = _mgr(limit=4_000)
    memory = _memory(UserMessage(content="go"))
    for i in range(10):
        memory.extend(_round(f"c{i}", 2_000))
    first = await mgr.assemble(memory)
    assert first.trim is not None and first.trim.applied

    previous = first.messages
    for i in range(10, 13):
        memory.extend(_round(f"c{i}", 40))
        out = await mgr.assemble(memory)
        assert out.trim is None
        # Everything sent before is sent again unchanged; new content only appends.
        assert out.messages[: len(previous)] == previous
        assert len(out.messages) == len(previous) + 2
        previous = out.messages


async def test_trim_clears_old_tool_results_first():
    trace = ExecutionTrace()
    memory = _memory(UserMessage(content="research this"))
    for i in range(8):
        memory.extend(_round(f"c{i}", 2_000))  # 8 results of ≈500 tokens
    out = await _mgr(limit=4_000, cap=200, trace=trace).assemble(memory)

    results = _results(out.messages)
    assert [_is_note(r) for r in results] == [True] * 3 + [False] * 5
    assert out.trim.applied and out.trim.cleared_results == 3
    assert out.trim.dropped_messages == 0  # one turn: nothing to drop
    # The stored history keeps every result in full.
    assert all(not _is_note(r) for r in _results(memory.snapshot()))
    [event] = _trims(trace)
    assert event.name == "trim" and event.payload["cleared_results"] == 3


async def test_clear_note_is_byte_stable_and_keeps_pairs():
    mgr = _mgr(limit=4_000)
    memory = _memory(UserMessage(content="go"))
    for i in range(8):
        memory.extend(_round(f"c{i}", 2_000))
    first = await mgr.assemble(memory)
    second = await mgr.assemble(memory)

    note = _results(first.messages)[0]
    assert note.content == CLEAR_NOTE.format(name="search", tokens=500)
    assert note.tool_use_id == "c0" and note.is_error is False
    assert first.messages == second.messages
    _assert_pairs_intact(first.messages)


async def test_protected_results_are_never_cleared():
    @tool(description="pinned docs", permission=PermissionCategory.AUTO, clear_old_results=False)
    async def read_spec() -> str:
        return ""

    memory = _memory(UserMessage(content="go"))
    memory.extend(_round("err", 2_000, is_error=True))
    memory.extend(_round("tiny", 200))  # ≈50 tokens
    memory.extend(_round("ask", 2_000, name="ask_user"))
    memory.extend(_round("spec", 2_000, name="read_spec"))
    for i in range(8):
        memory.extend(_round(f"c{i}", 2_000))
    mgr = _mgr(limit=5_000, cap=200, tools=ToolRegistry([read_spec]))
    out = await mgr.assemble(memory)

    by_id = {r.tool_use_id: r for r in _results(out.messages)}
    for protected in ("err", "tiny", "ask", "spec"):
        assert not _is_note(by_id[protected]), protected
    assert [_is_note(by_id[f"c{i}"]) for i in range(8)] == [True] * 3 + [False] * 5


async def test_trim_drops_whole_old_turns_when_needed():
    trace = ExecutionTrace()
    memory = _memory()
    for i in range(8):
        memory.extend(_turn(i, answer_chars=2_000))  # no tool results: nothing to clear
    out = await _mgr(limit=3_000, trace=trace).assemble(memory)

    view = out.messages
    assert out.trim.applied and out.trim.dropped_turns > 0
    # The first user message is pinned at the head; whole turns follow.
    assert view[0].content == "question 0"
    assert isinstance(view[1], UserMessage)
    assert [m.content for m in view if isinstance(m, UserMessage)][-3:] == [
        "question 5",
        "question 6",
        "question 7",
    ]
    stored = memory.snapshot()
    assert stored[0].content == view[1].content  # dropped from the history itself
    assert memory.state.pinned.content == "question 0"


def _reasoned_turn(i: int, answer_chars: int = 2_000):
    """A turn whose assistant called a tool with `reasoning_content` ahead of it."""
    block = ReasoningContent.from_raw("openai", {"reasoning_content": f"why {i}"})
    call = ToolCallRequest(id=f"c{i}", name="search", arguments={"q": str(i)})
    return [
        UserMessage(content=f"question {i}"),
        AssistantMessage(content="", tool_calls=[call], reasoning=[block]),
        ToolResultMessage(content=[
            ToolResult(tool_use_id=f"c{i}", name="search", content="r" * 300, is_error=False)
        ]),
        AssistantMessage(content="a" * answer_chars, reasoning=[block]),
    ]


@pytest.mark.parametrize("keeps", [False, True])
async def test_trim_keeps_old_reasoning_only_when_the_model_requires_it(keeps):
    """Kimi rejects a kept tool-call message without its reasoning_content:
    a trim still clears results and drops whole turns, but step 3 is off."""
    trace = ExecutionTrace()
    memory = _memory()
    for i in range(8):
        memory.extend(_reasoned_turn(i))
    out = await _mgr(limit=3_000, trace=trace, keeps_reasoning=keeps).assemble(memory)

    assert out.trim.applied and out.trim.dropped_turns > 0 and out.trim.cleared_results > 0
    assert out.trim.reasoning_cut is (not keeps)
    kept = [m for m in out.messages if isinstance(m, AssistantMessage)]
    current = memory.snapshot()[_turn_starts(memory.snapshot())[-1]:]
    old = kept[: len(kept) - sum(isinstance(m, AssistantMessage) for m in current)]
    assert old, "the trim must leave some earlier turns to test"
    if keeps:
        assert all(m.reasoning for m in kept)
        assert out.reasoning_dropped == 0
    else:
        assert not any(m.reasoning for m in old)
        assert out.reasoning_dropped == len(old)
    # Either way the current turn's reasoning is sent.
    assert all(m.reasoning for m in kept[len(old):])


@pytest.mark.parametrize("model,kept", [("openai:kimi-k3", True), ("openai:gpt-5", False)])
async def test_the_config_decides_whether_a_trim_keeps_reasoning(model, kept):
    """End to end: the catalog names Kimi, the config resolves it, the agent
    hands it to the manager, and the trimmed request still carries every
    kept message's reasoning_content."""
    mock = MockLLMProvider([ScriptedTurn(text="ok")])
    agent = Agent(
        AgentConfig(model=model, api_key="k", base_url="https://example.invalid/v1",
                    context_window=131_072, max_context_tokens=3_000),
        provider=mock, memory=MemoryConfig(),
    )
    for i in range(8):
        agent.memory.short_term.extend(_reasoned_turn(i))
    before = len(agent.get_history())
    result = await agent.run("question 8")
    assert len(agent.get_history()) < before  # the trim fired
    sent = [m for m in mock.call_log[0]["messages"] if isinstance(m, AssistantMessage)]
    assert sent
    assert all(bool(m.reasoning) for m in sent) is kept
    assert (result.context.reasoning_dropped == 0) is kept


def test_a_bound_model_never_keeps_old_reasoning():
    """Both flags at once is a contradiction the config already resolves;
    the manager resolves it the same way if constructed directly."""
    mgr = _mgr(binds_reasoning=True, keeps_reasoning=True)
    assert mgr._binds_reasoning and not mgr._keeps_reasoning


async def test_one_trim_reaches_60_percent():
    trace = ExecutionTrace()
    mgr = _mgr(limit=3_000, trace=trace)
    memory = _memory()
    for i in range(10):
        memory.extend(_turn(i, answer_chars=1_200))
    out = await mgr.assemble(memory)

    assert out.trim.reached_target
    assert out.estimated_tokens <= mgr.trim_target
    # An independent estimate of what is now sent agrees.
    fresh = _mgr(limit=3_000)
    assert fresh.estimate(memory, out.messages) <= mgr.trim_target
    # And the next call has room to grow: no second trim.
    memory.extend(_turn(99))
    assert (await mgr.assemble(memory)).trim is None
    assert len(_trims(trace)) == 1


async def test_current_turn_is_never_dropped():
    memory = _memory()
    for i in range(4):
        memory.extend(_turn(i))
    memory.append(UserMessage(content="the current task"))
    for i in range(6):
        memory.extend(_round(f"c{i}", 2_000))
    out = await _mgr(limit=3_000, cap=100).assemble(memory)

    view = out.messages
    assert any(m.content == "the current task" for m in view if isinstance(m, UserMessage))
    # Every message of the current turn is still there (results may be notes).
    assert [r.tool_use_id for r in _results(view)] == [f"c{i}" for i in range(6)]
    # The two turns before it stay as well.
    assert [m.content for m in view if isinstance(m, UserMessage)][-3:] == [
        "question 2",
        "question 3",
        "the current task",
    ]


async def test_a_cleared_result_never_comes_back():
    mgr = _mgr(limit=4_000)
    memory = _memory(UserMessage(content="go"))
    for i in range(8):
        memory.extend(_round(f"c{i}", 2_000))
    first = await mgr.assemble(memory)
    cleared = {r.tool_use_id for r in _results(first.messages) if _is_note(r)}
    assert cleared

    for i in range(8, 20):
        memory.extend(_round(f"c{i}", 2_000))
        out = await mgr.assemble(memory)
        still = {r.tool_use_id for r in _results(out.messages) if _is_note(r)}
        assert cleared <= still
        cleared = still


async def test_unsatisfiable_trigger_is_a_recorded_noop():
    trace = ExecutionTrace()
    mgr = _mgr(limit=1_000, trace=trace)
    memory = _memory(UserMessage(content="x" * 8_000))  # one huge current turn
    first = await mgr.assemble(memory)
    second = await mgr.assemble(memory)

    assert first.trim is not None and not first.trim.applied
    assert second.trim is not None and not second.trim.applied
    assert memory.state == ContextState() and len(memory) == 1
    [event] = _trims(trace)  # recorded once, not on every call
    assert event.payload["applied"] is False


async def test_tool_definitions_count_toward_the_limit():
    def schema_tool(i: int):
        @tool(name=f"t{i}", description="d" * 2_000, permission=PermissionCategory.AUTO)
        async def t(q: str) -> str:
            return q

        return t

    history = [UserMessage(content="q" * 2_000)]
    bare = _mgr(limit=2_000)
    with_tools = _mgr(limit=2_000, tools=ToolRegistry([schema_tool(i) for i in range(3)]))

    bare_est = bare.estimate(_memory(*history), history)
    tools_est = with_tools.estimate(_memory(*history), history)
    assert tools_est - bare_est >= 3 * 500
    assert bare_est <= 2_000 < tools_est


async def test_estimate_is_anchored_and_reset_on_replace():
    mgr = _mgr()
    memory = _memory(UserMessage(content="hello " * 100))
    await mgr.assemble(memory)
    mgr.note_actual_tokens(300)  # the provider counts ≈2x what chars/4 says

    appended = AssistantMessage(content="z" * 400)
    memory.append(appended)
    view = mgr.render_view(memory.snapshot(), memory.state)
    expected = 300 + int(estimate_message_tokens(appended) * mgr.calibration)
    assert mgr.calibration > 1.0
    assert mgr.estimate(memory, view) == expected

    # Any history replacement invalidates the anchor.
    memory.replace_all(memory.snapshot())
    view = mgr.render_view(memory.snapshot(), memory.state)
    raw = estimate_tokens("sys") + sum(estimate_message_tokens(m) for m in view)
    assert mgr.estimate(memory, view) == int(raw * mgr.calibration)


@pytest.mark.parametrize(
    "kwargs,context,result",
    [
        ({"model": "claude-opus-5"}, 150_000, 16_000),  # 1M window
        ({"model": "claude-haiku-4-5"}, 140_000, 16_000),  # 200k window
        ({"model": "gpt-4o"}, 89_600, 12_800),  # 128k window
        (
            {"model": "openai:local", "base_url": "http://x/v1", "context_window": 8192},
            4_096,  # 70 % would exceed what the response leaves
            819,
        ),
        ({"model": "claude-opus-5", "max_context_tokens": 20_000}, 20_000, 4_000),
    ],
)
def test_auto_limits_scale_with_the_window(kwargs, context, result):
    config = AgentConfig(api_key="k", **kwargs)
    assert config.resolved_max_context_tokens == context
    assert config.resolved_max_tool_result_tokens == result


def test_small_windows_keep_fewer_recent_results():
    assert _mgr(limit=150_000, cap=16_000).keep_recent_results() == 2
    assert _mgr(limit=150_000, cap=4_000).keep_recent_results() == 5
    assert _mgr(limit=4_096, cap=819).keep_recent_results() == 1


async def test_trim_invariants_random_histories():
    rng = random.Random(7)
    for trial in range(60):
        memory = _memory()
        n_turns = rng.randint(1, 12)
        for t in range(n_turns):
            memory.append(UserMessage(content=f"u{t} " + "w" * rng.randint(10, 600)))
            for r in range(rng.randint(0, 3)):
                memory.extend(_round(f"t{t}r{r}", rng.randint(10, 3_000)))
            memory.append(
                AssistantMessage(
                    content="a" * rng.randint(10, 600),
                    reasoning=[ReasoningContent(provider="anthropic", raw={"type": "thinking"})],
                )
            )
        before = memory.snapshot()
        current_turn = before[_turn_starts(before)[-1] :]
        out = await _mgr(limit=rng.choice([800, 1_500, 3_000, 8_000])).assemble(memory)
        view = out.messages

        assert isinstance(view[0], UserMessage) and not isinstance(view[0], ToolResultMessage)
        _assert_pairs_intact(view)
        # The current turn is always sent (results possibly as notes).
        assert view[-len(current_turn) :][0] is current_turn[0]
        if out.trim is not None and out.trim.applied:
            # No reasoning survives from before the current turn.
            head = view[: len(view) - len(current_turn)]
            assert not any(isinstance(m, AssistantMessage) and m.reasoning for m in head)
            # Kept turns are whole: the stored history starts at a turn.
            stored = memory.snapshot()
            assert _turn_starts(stored)[0] == 0


async def test_clear_resets_the_context_state():
    memory = _memory(UserMessage(content="go"))
    for i in range(8):
        memory.extend(_round(f"c{i}", 2_000))
    await _mgr(limit=4_000).assemble(memory)
    assert memory.state != ContextState()

    epoch = memory.epoch
    memory.clear()
    assert memory.state == ContextState() and len(memory) == 0
    assert memory.epoch != epoch


def _priced_agent(**config) -> tuple[Agent, MockLLMProvider]:
    mock = MockLLMProvider([])
    agent = Agent(
        AgentConfig(model="claude-opus-5", api_key="k", **config),
        provider=mock,
    )
    return agent, mock


async def test_checkpoint_restores_the_context_state():
    agent, _ = _priced_agent()
    short_term = agent.memory.short_term
    short_term.extend(_turn(0) + _turn(1))
    state = ContextState(pinned=UserMessage(content="first"), cleared_results=3, reasoning_cut=2)
    short_term.state = state

    cid = agent.checkpoint()
    await agent.clear_history()
    assert short_term.state == ContextState()

    agent.restore(cid)
    restored = short_term.state
    assert (restored.cleared_results, restored.reasoning_cut) == (3, 2)
    assert restored.pinned.content == "first"
    assert [m.content for m in agent.get_history()] == [
        m.content for m in _turn(0) + _turn(1)
    ]


@pytest.mark.parametrize("streaming", [False, True])
async def test_loop_applies_the_trim(streaming):
    agent, mock = _priced_agent(
        max_context_tokens=2_000, max_tool_result_tokens=200, memory=MemoryConfig()
    )
    mock.add_turn(ScriptedTurn(text="done"))
    for i in range(10):
        agent.memory.short_term.extend(_turn(i, answer_chars=1_000))
    stored_before = len(agent.get_history())

    if streaming:
        async for _ in agent.run_stream("next"):
            pass
    else:
        await agent.run("next")

    trace = agent.get_trace()
    [trim] = _trims(trace)
    assert trim.payload["applied"] is True
    [llm] = [e for e in trace.all_spans() if e.event_type is EventType.LLM_CALL]
    assert llm.payload["trimmed"] is True
    # The provider saw the trimmed view, and the history itself was trimmed.
    assert mock.call_log[0]["message_count"] < stored_before + 1
    assert len(agent.get_history()) < stored_before + 2


# ---------------- review regressions ----------------

def test_an_unanswered_message_begins_its_own_turn():
    """Injected context joins the message it was injected for; an earlier
    user message that never got a reply does not — so it stays droppable."""
    facts = UserMessage(content="<relevant_context>…</relevant_context>", metadata={"context": "relevant_facts"})
    history = [
        UserMessage(content="never answered"),
        UserMessage(content="next question"),
        AssistantMessage(content="answer"),
        facts,
        UserMessage(content="with facts"),
    ]
    assert _turn_starts(history) == [0, 1, 3]


async def test_results_the_model_has_not_seen_are_never_cleared():
    """Six parallel results arrive at once: all are new to the model, and
    clearing any would only make it call the tool again."""
    calls = [ToolCallRequest(id=f"p{i}", name="search", arguments={"q": str(i)}) for i in range(6)]
    memory = _memory(
        UserMessage(content="read all six"),
        AssistantMessage(content="", tool_calls=calls),
        ToolResultMessage(
            content=[ToolResult(tool_use_id=f"p{i}", name="search", content="r" * 4_000) for i in range(6)]
        ),
    )
    out = await _mgr(limit=4_000, cap=900).assemble(memory, call=2)

    assert not any(_is_note(r) for r in _results(out.messages))


async def test_a_trim_that_makes_the_prompt_fit_is_applied():
    """Short of the usual headroom, but without it the prompt could not be sent."""
    mgr = ContextManager(
        system_prompt="", max_context_tokens=18_000, max_tool_result_tokens=1_000, usable_tokens=18_000
    )
    memory = _memory(*_turn(0, answer_chars=12_000), *_turn(1, 8), *_turn(2, 8))
    memory.append(UserMessage(content="y" * 66_000))  # ≈16.5k: fits on its own
    out = await mgr.assemble(memory)  # ≈19.7k → dropping turn 0 gives ≈16.7k

    assert out.trim.applied
    assert out.trim.estimated_after > 18_000 * 0.9  # above the usual headroom ...
    assert out.estimated_tokens <= 18_000  # ... yet within what can be sent
