"""Agent construction: the flat surface, config overrides, and MCP discovery.

The contract:
  - `Agent("model-id", ...)` is the whole common path — no provider enum, no
    nested config objects, no factory method.
  - the keyword arguments that shadow `AgentConfig` fields override a passed
    config without mutating it.
  - MCP tools are discovered lazily by the constructor and eagerly by
    `await Agent.connect(...)`.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings

import pytest

from agent_framework.core.agent import _CONFIG_OVERRIDES, Agent
from agent_framework.core.config import AgentConfig
from agent_framework.mcp.provider import MCPToolProvider
from agent_framework.mcp.registry import MCPServerRegistry
from agent_framework.permissions.config import AllowRule, DenyRule, PermissionConfig
from agent_framework.permissions.models import PermissionCategory
from agent_framework.providers.catalog import ProviderResolutionError
from agent_framework.tests.mocks.mock_mcp import MockMCPServer
from agent_framework.tools.decorator import tool
from agent_framework.tools.registry import ToolRegistry


@tool(description="Add two numbers", permission=PermissionCategory.AUTO)
async def add(a: int, b: int) -> int:
    return a + b


def _mcp_registry(name: str = "srv", tool_name: str = "search") -> MCPServerRegistry:
    """A registry backed by a mock client — no real MCP SDK, no subprocess."""
    server = MockMCPServer(name)
    server.register(tool_name, lambda query: f"{name}:{query}")
    client = server.as_client()
    registry = MCPServerRegistry()
    registry._clients[client.name] = client  # type: ignore[assignment]
    registry._providers[client.name] = MCPToolProvider(client)  # type: ignore[arg-type]
    return registry


# ---------------- the flat surface ----------------

def test_a_model_id_is_enough(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    agent = Agent("claude-opus-5")
    assert agent.config.model == "claude-opus-5"
    assert agent.config.provider == "anthropic"


def test_the_adapter_comes_from_the_model_id():
    from agent_framework.providers.anthropic_agent import AnthropicAdapter
    from agent_framework.providers.openai_agent import OpenAIAdapter

    assert isinstance(Agent("claude-opus-5", api_key="k")._provider, AnthropicAdapter)
    assert isinstance(Agent("gpt-5", api_key="k")._provider, OpenAIAdapter)


def test_an_explicit_prefix_selects_the_adapter_for_a_compatible_endpoint():
    from agent_framework.providers.openai_agent import OpenAIAdapter

    agent = Agent(
        "openai:llama-3.3-70b", api_key="k", base_url="http://localhost:8000/v1"
    )
    assert isinstance(agent._provider, OpenAIAdapter)
    assert agent.config.model == "llama-3.3-70b"


def test_an_unresolvable_model_fails_at_construction():
    with pytest.raises(ProviderResolutionError):
        Agent("llama-3.3-70b", api_key="k")


def test_a_missing_api_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        Agent("claude-opus-5")


@pytest.mark.parametrize("bad", [123, None, ["claude-opus-5"], {"model": "x"}])
def test_the_first_argument_must_be_a_model_id_or_a_config(bad):
    with pytest.raises((TypeError, ProviderResolutionError)):
        Agent(bad, api_key="k")


def test_ask_user_is_always_registered():
    assert "ask_user" in Agent("claude-opus-5", api_key="k").tools


def test_tools_may_be_a_list_or_a_registry():
    from_list = Agent("claude-opus-5", api_key="k", tools=[add])
    from_registry = Agent("claude-opus-5", api_key="k", tools=ToolRegistry([add]))
    assert "add" in from_list.tools
    assert "add" in from_registry.tools


def test_a_shared_registry_is_never_mutated():
    """Registering ask_user must not reach into the caller's registry."""
    shared = ToolRegistry([add])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # ToolRegistry.add warns on overwrite
        first = Agent("claude-opus-5", api_key="k", tools=shared)
        second = Agent("claude-opus-5", api_key="k", tools=shared)
    assert shared.names == ["add"]
    assert "ask_user" in first.tools
    assert "ask_user" in second.tools


# ---------------- config overrides ----------------

def test_every_override_is_a_real_config_field():
    """Guards against the flat signature drifting from AgentConfig."""
    fields = {f.name for f in dataclasses.fields(AgentConfig)}
    assert set(_CONFIG_OVERRIDES) <= fields


def test_every_override_is_an_actual_keyword_argument():
    params = inspect.signature(Agent.__init__).parameters
    for name in _CONFIG_OVERRIDES:
        assert name in params, name
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    "name,value",
    [
        ("system_prompt", "overridden"),
        ("max_turns", 2),
        ("max_budget_usd", 0.25),
        ("temperature", 0.5),
        ("base_url", "http://localhost:8000/v1"),
    ],
)
def test_an_override_beats_the_config(name, value):
    base = AgentConfig(model="claude-opus-5", api_key="k", system_prompt="base")
    agent = Agent(base, **{name: value})
    assert getattr(agent.config, name) == value


def test_overrides_do_not_mutate_the_config_they_came_from():
    base = AgentConfig(model="claude-opus-5", api_key="k", max_turns=10)
    Agent(base, max_turns=2)
    assert base.max_turns == 10


def test_omitting_an_override_leaves_the_config_value():
    base = AgentConfig(
        model="claude-opus-5", api_key="k", max_turns=7, system_prompt="kept"
    )
    agent = Agent(base)
    assert (agent.config.max_turns, agent.config.system_prompt) == (7, "kept")


@pytest.mark.parametrize("name", ["max_budget_usd", "temperature"])
def test_none_is_a_real_value_not_absence(name):
    """`None` means "no budget" / "omit the parameter", so it must be passable."""
    base = AgentConfig(model="claude-opus-5", api_key="k", **{name: 0.5})
    agent = Agent(base, **{name: None})
    assert getattr(agent.config, name) is None


def test_a_sub_agent_is_one_line():
    base = AgentConfig(
        model="claude-opus-5", api_key="k", system_prompt="parent", max_turns=10
    )
    parent = Agent(base, tools=[add])
    child = Agent(base, system_prompt="Summarize.", max_turns=2)
    assert (child.config.system_prompt, child.config.max_turns) == ("Summarize.", 2)
    assert parent.config.system_prompt == "parent"


@pytest.mark.parametrize(
    "model_id", ["openai:llama-3.3-70b", "openai:claude-opus-5", "openai:llama3.2:3b"]
)
def test_a_sub_agent_is_one_line_for_explicit_model_ids(model_id):
    """The headline pattern must hold for local and gateway models too."""
    base = AgentConfig(
        model=model_id, base_url="http://127.0.0.1:1/v1", context_window=32_768
    )
    child = Agent(base, system_prompt="Summarize.", max_turns=2)
    assert child.config.provider == "openai"
    assert child.config.model == base.model


def test_an_override_is_validated():
    with pytest.raises(ValueError, match="max_turns"):
        Agent("claude-opus-5", api_key="k", max_turns=0)


# ---------------- wiring ----------------

def test_permissions_and_trace_are_injected():
    perms = PermissionConfig(deny=[DenyRule(tool="shell")])
    agent = Agent("claude-opus-5", api_key="k", permissions=perms, agent_id="fixed")
    assert agent.permissions is perms
    assert agent.agent_id == "fixed"
    assert agent.get_trace().agent_id == "fixed"


def test_a_sub_agent_may_not_widen_parent_permissions():
    parent = PermissionConfig(deny=[DenyRule(tool="shell")])
    with pytest.raises(PermissionError, match="subset"):
        Agent(
            "claude-opus-5",
            api_key="k",
            permissions=PermissionConfig(allow=[AllowRule(tool="shell")]),
            parent_permissions=parent,
        )


def test_a_non_renderer_on_ask_is_rejected():
    with pytest.raises(TypeError, match="UIRenderer"):
        Agent(
            "claude-opus-5",
            api_key="k",
            permissions=PermissionConfig(on_ask=lambda *_: None),
        )


def test_a_prebuilt_provider_bypasses_the_catalog():
    """No API key needed, and no catalog lookup, when the adapter is given."""
    from agent_framework.tests.mocks.mock_llm import MockLLMProvider

    mock = MockLLMProvider([])
    agent = Agent("claude-opus-5", api_key="k", provider=mock)
    assert agent._provider is mock
    assert agent._loop.provider is mock


# ---------------- MCP discovery ----------------

async def test_the_constructor_defers_mcp_discovery_to_the_first_run():
    registry = _mcp_registry()
    agent = Agent("claude-opus-5", api_key="k", mcp=registry, tools=[add])
    # Nothing has connected yet, so only the local tools are known.
    assert "srv__search" not in agent.tools
    await agent._ensure_mcp()
    assert "srv__search" in agent.tools
    assert "add" in agent.tools


async def test_lazy_discovery_happens_only_once():
    registry = _mcp_registry()
    agent = Agent("claude-opus-5", api_key="k", mcp=registry)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a second add() of the same name warns
        await agent._ensure_mcp()
        await agent._ensure_mcp()
    assert "srv__search" in agent.tools


async def test_connect_discovers_eagerly():
    """Boot-time discovery: the toolset is complete before the first run."""
    registry = _mcp_registry()
    agent = await Agent.connect("claude-opus-5", api_key="k", mcp=registry, tools=[add])
    assert "srv__search" in agent.tools
    assert "add" in agent.tools
    assert agent._mcp is registry


async def test_connect_works_inside_a_running_event_loop():
    """The old sync factory raised here; this is the whole point of connect()."""
    agent = await Agent.connect("claude-opus-5", api_key="k", mcp=_mcp_registry())
    assert "srv__search" in agent.tools


def _capabilities_config() -> AgentConfig:
    """include_capabilities_block is long-tail: it lives on AgentConfig only."""
    return AgentConfig(
        model="claude-opus-5",
        api_key="k",
        system_prompt="you are a test",
        include_capabilities_block=True,
    )


async def test_connect_lets_mcp_tools_reach_the_capabilities_block():
    agent = await Agent.connect(_capabilities_config(), mcp=_mcp_registry())
    assert "srv__search" in agent._context_manager._system_prompt


def test_lazy_mcp_with_a_capabilities_block_warns():
    """The system prompt is byte-stable, so it cannot gain tools later."""
    with pytest.warns(UserWarning, match="Agent.connect"):
        Agent(_capabilities_config(), mcp=_mcp_registry())


def test_no_warning_without_a_capabilities_block():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Agent("claude-opus-5", api_key="k", mcp=_mcp_registry())
