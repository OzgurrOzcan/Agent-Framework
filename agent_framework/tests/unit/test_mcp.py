"""Tests for MCP integration: sync/async discovery, tool name namespacing,
registry overwrite warnings, and transport selection.

None of these tests require the real `mcp` SDK — transport tests inject
fake `mcp.*` modules into `sys.modules`, everything else uses the mocks in
`tests/mocks/mock_mcp.py`.
"""

from __future__ import annotations

import sys
import types
import warnings
from typing import Any

import pytest

from agent_framework.mcp.client import MCPClient, MCPConnectionConfig
from agent_framework.mcp.provider import MCPToolProvider
from agent_framework.mcp.registry import MCPServerRegistry
from agent_framework.tools.models import Tool, ToolMetadata
from agent_framework.tools.registry import ToolRegistry
from agent_framework.tests.mocks.mock_mcp import MockMCPServer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_mock_server_client(name: str, tool_name: str = "search"):
    server = MockMCPServer(name)
    server.register(tool_name, lambda query: f"{name}:{query}")
    return server.as_client()


def _registry_with(*clients) -> MCPServerRegistry:
    """Build an MCPServerRegistry backed by mock clients (no real MCPClient)."""
    registry = MCPServerRegistry()
    for client in clients:
        registry._clients[client.name] = client
        registry._providers[client.name] = MCPToolProvider(client)
    return registry


def _native_tool(name: str) -> Tool:
    return Tool(
        metadata=ToolMetadata(
            name=name,
            description=f"tool {name}",
            parameters_schema={"type": "object", "properties": {}},
        ),
        fn=lambda: name,
    )


# ---------------------------------------------------------------------------
# T1.6 — discover_all_sync
# ---------------------------------------------------------------------------

async def test_discover_all_sync_raises_inside_running_loop():
    """Blocking discovery cannot happen inside async code; point at the fix."""
    registry = _registry_with(_make_mock_server_client("alpha"))
    with pytest.raises(RuntimeError, match=r"Agent\.connect"):
        registry.discover_all_sync()


def test_discover_all_sync_works_without_running_loop():
    registry = _registry_with(_make_mock_server_client("alpha"))
    tools = registry.discover_all_sync()
    assert [t.name for t in tools] == ["alpha__search"]
    assert tools[0].metadata.source == "mcp:alpha"


# ---------------------------------------------------------------------------
# T2.6 — tool name namespacing
# ---------------------------------------------------------------------------

async def test_tool_name_is_namespaced_but_server_call_uses_original_name():
    client = _make_mock_server_client("alpha")
    provider = MCPToolProvider(client)

    tools = await provider.discover()
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "alpha__search"
    # Description must stay untouched (no "[server] " prefix).
    assert tool.metadata.description == "Mock tool: search"

    result = await tool.fn(query="x")
    assert result == "alpha:x"
    # The wire call must use the ORIGINAL tool name, not the namespaced one.
    assert client.calls == [("search", {"query": "x"})]


async def test_two_servers_with_same_tool_name_do_not_collide():
    client_a = _make_mock_server_client("alpha")
    client_b = _make_mock_server_client("beta")
    registry = _registry_with(client_a, client_b)

    tools = await registry.discover_all()
    reg = ToolRegistry()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for t in tools:
            reg.add(t)
    assert caught == []  # namespaced names never clash -> no overwrite warning

    assert "alpha__search" in reg
    assert "beta__search" in reg
    assert len(reg) == 2

    # Each namespaced tool routes to its own server with the original name.
    assert await reg.get("alpha__search").fn(query="a") == "alpha:a"
    assert await reg.get("beta__search").fn(query="b") == "beta:b"
    assert client_a.calls == [("search", {"query": "a"})]
    assert client_b.calls == [("search", {"query": "b"})]


# ---------------------------------------------------------------------------
# T2.6 — ToolRegistry overwrite warning
# ---------------------------------------------------------------------------

def test_registry_add_warns_on_overwrite():
    first = _native_tool("dup")
    second = _native_tool("dup")
    reg = ToolRegistry()
    reg.add(first)
    with pytest.warns(UserWarning, match="Tool 'dup' is being overwritten"):
        reg.add(second)
    assert reg.get("dup") is second


def test_registry_add_same_object_does_not_warn():
    t = _native_tool("same")
    reg = ToolRegistry()
    reg.add(t)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reg.add(t)
    assert caught == []
    assert reg.get("same") is t


def test_registry_replace_is_silent():
    first = _native_tool("dup")
    second = _native_tool("dup")
    reg = ToolRegistry()
    reg.add(first)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reg.replace(second)
    assert caught == []
    assert reg.get("dup") is second


# ---------------------------------------------------------------------------
# T2.6 — transport selection (fake `mcp` SDK injected into sys.modules)
# ---------------------------------------------------------------------------

class _FakeSession:
    """Stands in for mcp.ClientSession: async CM + async initialize()."""

    def __init__(self, read: Any, write: Any) -> None:
        self.read = read
        self.write = write
        self.initialized = False

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def initialize(self) -> None:
        self.initialized = True


class _FakeStreamCM:
    """Async context manager yielding a fixed tuple (read/write streams)."""

    def __init__(self, ret: Any) -> None:
        self._ret = ret

    async def __aenter__(self) -> Any:
        return self._ret

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _FakeStdioParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _install_fake_mcp(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject fake `mcp`, `mcp.client.*` modules; return a call-recording dict."""
    calls: dict[str, Any] = {}

    def fake_streamablehttp_client(url: str, headers: Any = None) -> _FakeStreamCM:
        calls["streamable_http"] = (url, headers)
        # Streamable HTTP yields a 3-tuple: (read, write, get_session_id).
        return _FakeStreamCM(("read", "write", lambda: "session-id"))

    def fake_sse_client(url: str, headers: Any = None) -> _FakeStreamCM:
        calls["sse"] = (url, headers)
        return _FakeStreamCM(("read", "write"))

    def fake_stdio_client(params: Any) -> _FakeStreamCM:
        calls["stdio_params"] = params
        return _FakeStreamCM(("read", "write"))

    mcp_mod = types.ModuleType("mcp")
    mcp_mod.ClientSession = _FakeSession
    client_mod = types.ModuleType("mcp.client")
    sh_mod = types.ModuleType("mcp.client.streamable_http")
    sh_mod.streamablehttp_client = fake_streamablehttp_client
    sse_mod = types.ModuleType("mcp.client.sse")
    sse_mod.sse_client = fake_sse_client
    stdio_mod = types.ModuleType("mcp.client.stdio")
    stdio_mod.stdio_client = fake_stdio_client
    stdio_mod.StdioServerParameters = _FakeStdioParams

    mcp_mod.client = client_mod
    client_mod.streamable_http = sh_mod
    client_mod.sse = sse_mod
    client_mod.stdio = stdio_mod

    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.client", client_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", sh_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.sse", sse_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_mod)
    return calls


@pytest.mark.parametrize("transport", ["http", "streamable_http"])
async def test_http_transport_uses_streamablehttp_client(monkeypatch, transport):
    calls = _install_fake_mcp(monkeypatch)
    config = MCPConnectionConfig(
        transport=transport,
        url="http://example.test/mcp",
        auth_headers={"Authorization": "Bearer t"},
    )
    client = MCPClient(name="srv", config=config)
    await client.connect()
    try:
        assert calls["streamable_http"] == (
            "http://example.test/mcp",
            {"Authorization": "Bearer t"},
        )
        assert "sse" not in calls
        assert client.connected
        assert client._session.initialized
    finally:
        await client.close()


async def test_sse_transport_warns_deprecation_and_uses_sse_client(monkeypatch):
    calls = _install_fake_mcp(monkeypatch)
    config = MCPConnectionConfig(transport="sse", url="http://example.test/sse")
    client = MCPClient(name="srv", config=config)
    with pytest.warns(DeprecationWarning, match="SSE transport is deprecated"):
        await client.connect()
    try:
        assert calls["sse"] == ("http://example.test/sse", {})
        assert "streamable_http" not in calls
        assert client.connected
    finally:
        await client.close()


async def test_stdio_transport_passes_env_none_through(monkeypatch):
    calls = _install_fake_mcp(monkeypatch)
    config = MCPConnectionConfig(transport="stdio", command=["echo", "hi"], env=None)
    client = MCPClient(name="srv", config=config)
    await client.connect()
    try:
        params = calls["stdio_params"]
        assert params.kwargs["command"] == "echo"
        assert params.kwargs["args"] == ["hi"]
        # None must NOT be collapsed to {} — the SDK then supplies its
        # own default environment for the subprocess.
        assert params.kwargs["env"] is None
    finally:
        await client.close()


async def test_stdio_transport_passes_explicit_env(monkeypatch):
    calls = _install_fake_mcp(monkeypatch)
    config = MCPConnectionConfig(
        transport="stdio", command=["echo"], env={"FOO": "bar"}
    )
    client = MCPClient(name="srv", config=config)
    await client.connect()
    try:
        assert calls["stdio_params"].kwargs["env"] == {"FOO": "bar"}
    finally:
        await client.close()
