"""Tests for the core agent loop using MockLLMProvider (no API calls)."""
import pytest
from agent_framework.core.config import AgentConfig, ProviderConfig, ProviderKind
from agent_framework.core.loop import AgentLoop
from agent_framework.core.policy import LoopPolicy
from agent_framework.core.context_manager import ContextManager
from agent_framework.core.config import CompactionConfig
from agent_framework.memory.short_term import ShortTermMemory
from agent_framework.permissions.config import PermissionConfig
from agent_framework.permissions.models import PermissionCategory
from agent_framework.permissions.resolver import PermissionResolver
from agent_framework.tools.decorator import tool
from agent_framework.tools.executor import ToolExecutor
from agent_framework.tools.registry import ToolRegistry
from agent_framework.tracing.cost import CostTracker, PricingTable
from agent_framework.tracing.trace import ExecutionTrace
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn


def _make_loop(mock: MockLLMProvider, tools: list | None = None) -> AgentLoop:
    registry = ToolRegistry(tools or [])
    trace = ExecutionTrace(agent_id="test")
    config = AgentConfig(
        provider=ProviderConfig(
            provider=ProviderKind.ANTHROPIC,
            model="claude-opus-4-7",
            api_key="test",
        ),
        system_prompt="You are a test agent.",
        max_turns=10,
        stall_threshold=3,
    )
    mem = ShortTermMemory()
    ctx = ContextManager(
        system_prompt="You are a test agent.",
        context_window=10000,
        response_buffer_tokens=512,
        compaction_buffer_tokens=256,
        compaction=CompactionConfig(enabled=False),
    )
    resolver = PermissionResolver(PermissionConfig(), trace=trace)
    executor = ToolExecutor(trace=trace)
    cost = CostTracker(pricing=PricingTable())
    loop = AgentLoop(
        config=config,
        provider=mock,
        tools=registry,
        short_term=mem,
        context_manager=ctx,
        permission_resolver=resolver,
        tool_executor=executor,
        cost_tracker=cost,
        trace=trace,
        agent_id="test",
    )
    return loop


@pytest.mark.asyncio
async def test_simple_text_response():
    mock = MockLLMProvider([ScriptedTurn(text="Hello, world!")])
    loop = _make_loop(mock)
    result = await loop.run("Hi", policy=LoopPolicy.full())
    assert result.final_message == "Hello, world!"
    assert result.error is None


@pytest.mark.asyncio
async def test_tool_call_and_result():
    @tool(description="Calculator", permission=PermissionCategory.AUTO)
    async def calculator(expression: str) -> float:
        return eval(expression)  # noqa: S307 — test only

    mock = MockLLMProvider([
        ScriptedTurn(
            text="Let me calculate.",
            tool_calls=[{"id": "c1", "name": "calculator", "arguments": {"expression": "2+2"}}],
        ),
        ScriptedTurn(text="The result is 4."),
    ])
    loop = _make_loop(mock, tools=[calculator])
    result = await loop.run("What is 2+2?", policy=LoopPolicy.full())
    assert result.final_message == "The result is 4."
    assert mock._call_count == 2


@pytest.mark.asyncio
async def test_max_turns_guard():
    # Script that always requests a tool → forces the turn ceiling.
    mock = MockLLMProvider([
        ScriptedTurn(text="thinking...", tool_calls=[
            {"id": f"c{i}", "name": "unknown_tool", "arguments": {}}
        ])
        for i in range(20)
    ])
    loop = _make_loop(mock)
    loop.config.__dict__["max_turns"] = 3
    result = await loop.run("Go forever", policy=LoopPolicy.full(max_turns=3))
    assert result.error is not None
    assert "max_turns" in str(result.error).lower()


@pytest.mark.asyncio
async def test_call_once_does_not_execute_tools():
    mock = MockLLMProvider([
        ScriptedTurn(
            text="I would search...",
            tool_calls=[{"id": "c1", "name": "web_search", "arguments": {"query": "test"}}],
        ),
    ])
    loop = _make_loop(mock)
    result = await loop.run("Search for python", policy=LoopPolicy.once())
    # Tools were NOT executed — no second LLM call.
    assert mock._call_count == 1


@pytest.mark.asyncio
async def test_stall_detection():
    # Same tool+args repeated 3 times → stall.
    same_turn = ScriptedTurn(
        text="Searching...",
        tool_calls=[{"id": "s1", "name": "web_search", "arguments": {"q": "same"}}],
    )
    # After stall injection the model stops.
    mock = MockLLMProvider([same_turn, same_turn, same_turn, ScriptedTurn(text="Done.")])
    loop = _make_loop(mock)
    result = await loop.run("Search forever", policy=LoopPolicy.full())
    # Loop continued after the stall nudge — final message came from the 4th turn.
    assert result.final_message == "Done."


@pytest.mark.asyncio
async def test_tool_denial_is_feedback_not_crash():

    @tool(description="Risky", permission=PermissionCategory.ASK)
    async def risky_tool() -> str:
        return "executed"

    mock = MockLLMProvider([
        ScriptedTurn(
            text="I'll use risky_tool.",
            tool_calls=[{"id": "r1", "name": "risky_tool", "arguments": {}}],
        ),
        ScriptedTurn(text="OK, I'll try something else."),
    ])
    # No renderer attached → risky_tool ASK resolves to DENY with feedback.
    loop = _make_loop(mock, tools=[risky_tool])
    result = await loop.run("Do the risky thing", policy=LoopPolicy.full())
    assert result.error is None
    assert result.final_message == "OK, I'll try something else."


@pytest.mark.asyncio
async def test_single_round_policy():
    @tool(description="Search", permission=PermissionCategory.AUTO)
    async def search(q: str) -> str:
        return f"results for {q}"

    mock = MockLLMProvider([
        ScriptedTurn(
            text="Searching...",
            tool_calls=[{"id": "t1", "name": "search", "arguments": {"q": "ai"}}],
        ),
        ScriptedTurn(text="Here are the AI results."),
    ])
    loop = _make_loop(mock, tools=[search])
    result = await loop.run("Find info about AI", policy=LoopPolicy.single_round())
    assert result.final_message == "Here are the AI results."
    assert mock._call_count == 2
