"""Server-side tool-result clearing (Anthropic context editing) in long runs.

The provider clears old tool results in its own copy of the prompt, so the
history that was sent stays unchanged — the one way to keep a long run in
bounds on a model that binds reasoning to the prefix. It is requested
automatically where the adapter supports it and the vendor's own API is used,
and switched off for the session after a rejection.
"""

from __future__ import annotations

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.config import AgentConfig
from agent_framework.core.context_manager import ContextManager
from agent_framework.core.message import UserMessage
from agent_framework.memory.short_term import ShortTermMemory
from agent_framework.permissions.models import PermissionCategory
from agent_framework.providers.base import ContextEditsRejectedError
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn
from agent_framework.tools.decorator import tool
from agent_framework.tracing.trace import EventType


@tool(description="read the spec", permission=PermissionCategory.AUTO, clear_old_results=False)
async def read_spec() -> str:
    return "spec"


@tool(description="search", permission=PermissionCategory.AUTO)
async def search(q: str) -> str:
    return f"results for {q}"


def make_agent(script, *, supports=True, **config) -> tuple[Agent, MockLLMProvider]:
    config.setdefault("max_context_tokens", 50_000)
    mock = MockLLMProvider(list(script), supports_tool_result_clearing=supports)
    agent = Agent(
        AgentConfig(model="claude-opus-5-5", api_key="k", **config),
        tools=[search, read_spec],
        provider=mock,
    )
    return agent, mock


def native_events(agent: Agent):
    return [
        e
        for e in agent.get_trace().all_spans()
        if e.event_type is EventType.COMPACTION and e.name == "native_clear"
    ]


def _search_round(i: int) -> ScriptedTurn:
    return ScriptedTurn(tool_calls=[{"id": f"c{i}", "name": "search", "arguments": {"q": str(i)}}])


# ---------------- when it is requested ----------------

def test_native_clearing_is_on_for_the_vendor_api():
    agent = Agent("claude-opus-5-5", api_key="k")
    assert agent._loop.native_clearing is True
    session = next(e for e in agent.get_trace().all_spans() if e.event_type is EventType.SESSION)
    assert session.payload["native_tool_result_clearing"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_url": "https://gateway.example.com"},
        {"request_extra": {"context_management": {"edits": []}}},
    ],
)
def test_custom_base_url_disables_native_clearing(kwargs):
    """A gateway may not forward the beta; the caller's own context_management
    is theirs to control."""
    agent = Agent(AgentConfig(model="claude-opus-5-5", api_key="k", **kwargs))
    assert agent._loop.native_clearing is False


def test_adapters_without_the_feature_never_get_it():
    assert Agent("gpt-5", api_key="k")._loop.native_clearing is False
    agent, _ = make_agent([], supports=False)
    assert agent._loop.native_clearing is False


async def test_the_request_describes_the_agent():
    agent, mock = make_agent([ScriptedTurn(text="ok")])
    await agent.run("go")

    edits = mock.call_log[0]["context_edits"]
    assert edits.trigger_tokens == 50_000  # max_context_tokens
    assert edits.clear_at_least_tokens == 5_000  # a clear must be worth a cache rebuild
    assert edits.keep_results == agent._context_manager.keep_recent_results()
    # ask_user and tools that keep their results are never cleared.
    assert edits.exclude_tools == ("ask_user", "read_spec")


# ---------------- rejection ----------------

@pytest.mark.parametrize("streaming", [False, True])
async def test_beta_rejection_falls_back_once_and_disables(streaming):
    rejection = ContextEditsRejectedError("400 context_management: not available")
    agent, mock = make_agent(
        [ScriptedTurn(raise_exc=rejection), ScriptedTurn(text="first"), ScriptedTurn(text="second")]
    )

    async def run(prompt):
        if streaming:
            async for _ in agent.run_stream(prompt):
                pass
        else:
            await agent.run(prompt)

    await run("one")
    await run("two")

    sent_edits = [call["context_edits"] is not None for call in mock.call_log]
    # Rejected once, retried at once without, never requested again.
    assert sent_edits == [True, False, False]
    assert agent.get_history()[-1].content == "second"
    [disabled] = native_events(agent)
    assert disabled.payload["disabled"] is True
    assert "context_management" in disabled.payload["reason"]


# ---------------- what was cleared ----------------

async def test_native_clears_are_traced():
    edit = {"type": "clear_tool_uses_20250919", "cleared_tool_uses": 4, "cleared_input_tokens": 9_000}
    agent, _ = make_agent([_search_round(0), ScriptedTurn(text="done", applied_edits=[edit])])
    await agent.run("go")

    [event] = native_events(agent)
    assert event.payload == edit


async def test_calibration_and_anchor_include_the_cleared_tokens():
    """The provider counts the prompt after clearing; the estimate describes
    the prompt as sent, so the two are compared with the clearing added back."""
    mgr = ContextManager(system_prompt="", max_context_tokens=10_000, max_tool_result_tokens=500)
    memory = ShortTermMemory()
    memory.append(UserMessage(content="w" * 4_000))  # ≈1000 tokens
    await mgr.assemble(memory)

    mgr.note_actual_tokens(400, cleared_by_provider=600)
    assert mgr._anchor.tokens == 1_000
    assert mgr.calibration == pytest.approx(1.0, abs=0.01)  # 400 + 600 ≈ the estimate
    assert mgr.provider_cleared_tokens(memory) == 600


async def test_the_preflight_guard_counts_what_the_provider_clears():
    agent, _ = make_agent(
        [ScriptedTurn(text="ok")],
        context_window=12_000,
        max_tokens_per_response=2_000,
        max_context_tokens=5_000,
    )
    loop, memory = agent._loop, agent.memory.short_term
    await agent._context_manager.assemble(memory)
    agent._context_manager.note_actual_tokens(100, cleared_by_provider=4_000)

    over = agent.config.usable_context_tokens + 1_000
    assert loop._window_exceeded(over, turn_num=1) is None  # 4k of it never reaches the model
    memory.clear()  # what was cleared may be gone: no longer counted
    assert loop._window_exceeded(over, turn_num=1) is not None
