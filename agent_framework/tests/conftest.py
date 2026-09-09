"""Shared pytest fixtures."""
import pytest
from agent_framework.core.config import AgentConfig, CompactionConfig
from agent_framework.core.context_manager import ContextManager
from agent_framework.memory.short_term import ShortTermMemory
from agent_framework.permissions.config import PermissionConfig
from agent_framework.permissions.resolver import PermissionResolver
from agent_framework.tools.executor import ToolExecutor
from agent_framework.tools.registry import ToolRegistry
from agent_framework.tracing.cost import CostTracker, PricingTable
from agent_framework.tracing.trace import ExecutionTrace
from agent_framework.tests.mocks.mock_llm import MockLLMProvider
from agent_framework.core.loop import AgentLoop


@pytest.fixture
def anthropic_config() -> AgentConfig:
    return AgentConfig(
        model="claude-opus-4-7",
        api_key="test-key",
        system_prompt="You are a test agent.",
        max_turns=10,
        stall_threshold=3,
        compaction=CompactionConfig(enabled=False),
    )


@pytest.fixture
def openai_config() -> AgentConfig:
    return AgentConfig(
        model="gpt-4o",
        api_key="test-key",
        system_prompt="You are a test agent.",
        max_turns=10,
        stall_threshold=3,
        compaction=CompactionConfig(enabled=False),
    )


@pytest.fixture
def trace() -> ExecutionTrace:
    return ExecutionTrace(agent_id="test-agent", session_id="test-session")


@pytest.fixture
def empty_registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def make_loop(anthropic_config):
    """Factory fixture: returns a function that builds an AgentLoop with a mock provider."""
    def _make(mock: MockLLMProvider, tools: ToolRegistry | None = None) -> AgentLoop:
        registry = tools or ToolRegistry()
        trace = ExecutionTrace(agent_id="test")
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
        short_term = ShortTermMemory()
        return AgentLoop(
            config=anthropic_config,
            provider=mock,
            tools=registry,
            short_term=short_term,
            context_manager=ctx,
            permission_resolver=resolver,
            tool_executor=executor,
            cost_tracker=cost,
            trace=trace,
            agent_id="test",
        )
    return _make
