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
from typing import Any, Literal

import pytest

from agent_framework.mcp.client import MCPClient, MCPConnectionConfig, _sole_error
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


def _broken_client(name: str):
    """Connects, then fails to list tools — no retry backoff, so tests stay fast."""
    client = _make_mock_server_client(name)

    async def _unreachable(*_args, **_kwargs):
        raise ConnectionError(f"{name} is not reachable")

    client.list_tools = _unreachable
    return client


async def test_discover_all_tolerates_a_failing_server_by_default():
    from agent_framework.tracing.trace import EventType, ExecutionTrace

    registry = _registry_with(_make_mock_server_client("alpha"), _broken_client("beta"))
    registry._trace = ExecutionTrace()
    tools = await registry.discover_all()
    assert [t.name for t in tools] == ["alpha__search"]
    errors = [e for e in registry._trace.all_spans() if e.event_type is EventType.ERROR]
    assert [e.name for e in errors] == ["mcp:beta"]


async def test_strict_discovery_raises_naming_every_failed_server():
    from agent_framework.mcp.registry import MCPDiscoveryError

    registry = _registry_with(
        _make_mock_server_client("alpha"), _broken_client("beta"), _broken_client("gamma")
    )
    with pytest.raises(MCPDiscoveryError) as exc:
        await registry.discover_all(strict=True)
    assert set(exc.value.failures) == {"beta", "gamma"}
    assert "beta" in str(exc.value) and "gamma" in str(exc.value)
    assert isinstance(exc.value.__cause__, ConnectionError)


async def test_strict_discovery_still_records_trace_events():
    from agent_framework.mcp.registry import MCPDiscoveryError
    from agent_framework.tracing.trace import EventType, ExecutionTrace

    registry = _registry_with(_broken_client("beta"))
    registry._trace = ExecutionTrace()
    with pytest.raises(MCPDiscoveryError):
        await registry.discover_all(strict=True)
    assert any(e.event_type is EventType.ERROR for e in registry._trace.all_spans())


async def test_strict_discovery_returns_normally_when_all_servers_work():
    registry = _registry_with(_make_mock_server_client("alpha"))
    assert [t.name for t in await registry.discover_all(strict=True)] == ["alpha__search"]


def test_discover_all_sync_passes_strict_through():
    from agent_framework.mcp.registry import MCPDiscoveryError

    registry = _registry_with(_broken_client("beta"))
    assert registry.discover_all_sync() == []
    with pytest.raises(MCPDiscoveryError):
        registry.discover_all_sync(strict=True)


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

    def __init__(self, ret: Any, on_exit: Any = None) -> None:
        self._ret = ret
        self._on_exit = on_exit

    async def __aenter__(self) -> Any:
        return self._ret

    async def __aexit__(self, *exc: Any) -> bool:
        if self._on_exit is not None:
            self._on_exit()
        return False


class _FakeHttpClient:
    """Stands in for the httpx2.AsyncClient that mcp>=2 builds for headers."""

    def __init__(self, headers: Any, teardown: list[str]) -> None:
        self.headers = headers
        self._teardown = teardown

    async def __aenter__(self) -> "_FakeHttpClient":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        self._teardown.append("http-client")
        return False


class _FakeStdioParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _install_fake_mcp(
    monkeypatch: pytest.MonkeyPatch, *, sdk: str = "modern"
) -> dict[str, Any]:
    """Inject fake `mcp`, `mcp.client.*` modules; return a call-recording dict.

    *sdk* picks the streamable HTTP API generation to model, because the two
    are called differently:

      - "modern" (mcp>=2, the default because it is what installs today):
        `streamable_http_client(url, *, http_client=None,
        terminate_on_close=True)` — keyword-only, no `headers`, yields
        (read, write). Headers ride on `create_mcp_http_client(headers=...)`.
      - "legacy" (mcp<2): `streamablehttp_client(url, headers=None)`,
        yielding (read, write, get_session_id).

    Modelling only the legacy shape is how an HTTP transport that could not
    work against the installed SDK once passed every test here.
    """
    calls: dict[str, Any] = {"teardown": []}

    def fake_streamablehttp_client(url: str, headers: Any = None) -> _FakeStreamCM:
        calls["streamable_http"] = (url, headers)
        # mcp<2 streamable HTTP yields a 3-tuple: (read, write, get_session_id).
        return _FakeStreamCM(("read", "write", lambda: "session-id"))

    def fake_streamable_http_client(
        url: str, *, http_client: Any = None, terminate_on_close: bool = True
    ) -> _FakeStreamCM:
        calls["streamable_http"] = (url, http_client)
        return _FakeStreamCM(
            ("read", "write"), on_exit=lambda: calls["teardown"].append("transport")
        )

    def fake_create_mcp_http_client(
        headers: Any = None, timeout: Any = None, auth: Any = None
    ) -> _FakeHttpClient:
        client = _FakeHttpClient(headers, calls["teardown"])
        calls["http_client"] = client
        return client

    def fake_sse_client(url: str, headers: Any = None) -> _FakeStreamCM:
        calls["sse"] = (url, headers)
        return _FakeStreamCM(("read", "write"))

    def fake_stdio_client(params: Any) -> _FakeStreamCM:
        calls["stdio_params"] = params
        return _FakeStreamCM(("read", "write"))

    mcp_mod: Any = types.ModuleType("mcp")
    mcp_mod.ClientSession = _FakeSession
    client_mod: Any = types.ModuleType("mcp.client")
    sh_mod: Any = types.ModuleType("mcp.client.streamable_http")
    if sdk == "modern":
        sh_mod.streamable_http_client = fake_streamable_http_client
    elif sdk == "legacy":
        sh_mod.streamablehttp_client = fake_streamablehttp_client
    else:
        raise ValueError(f"unknown fake sdk {sdk!r}")
    shared_mod: Any = types.ModuleType("mcp.shared")
    httpx_utils_mod: Any = types.ModuleType("mcp.shared._httpx_utils")
    httpx_utils_mod.create_mcp_http_client = fake_create_mcp_http_client
    shared_mod._httpx_utils = httpx_utils_mod
    sse_mod: Any = types.ModuleType("mcp.client.sse")
    sse_mod.sse_client = fake_sse_client
    stdio_mod: Any = types.ModuleType("mcp.client.stdio")
    stdio_mod.stdio_client = fake_stdio_client
    stdio_mod.StdioServerParameters = _FakeStdioParams

    mcp_mod.client = client_mod
    mcp_mod.shared = shared_mod
    client_mod.streamable_http = sh_mod
    client_mod.sse = sse_mod
    client_mod.stdio = stdio_mod

    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.client", client_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", sh_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.sse", sse_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_mod)
    monkeypatch.setitem(sys.modules, "mcp.shared", shared_mod)
    monkeypatch.setitem(sys.modules, "mcp.shared._httpx_utils", httpx_utils_mod)
    return calls


def _http_client_config(
    transport: Literal["http", "streamable_http"] = "http", **kw: Any
) -> MCPConnectionConfig:
    kw.setdefault("auth_headers", {"Authorization": "Bearer t"})
    return MCPConnectionConfig(transport=transport, url="http://example.test/mcp", **kw)


@pytest.mark.parametrize("transport", ["http", "streamable_http"])
async def test_http_on_mcp2_sends_headers_through_the_http_client(monkeypatch, transport):
    """mcp>=2 has no headers= on the transport; they ride on the client."""
    calls = _install_fake_mcp(monkeypatch, sdk="modern")
    client = MCPClient(name="srv", config=_http_client_config(transport))
    await client.connect()
    try:
        http_client = calls["http_client"]
        assert http_client.headers == {"Authorization": "Bearer t"}
        assert calls["streamable_http"] == ("http://example.test/mcp", http_client)
        assert "sse" not in calls
        assert client.connected
        assert client._session.initialized
    finally:
        await client.close()


@pytest.mark.parametrize("transport", ["http", "streamable_http"])
async def test_http_on_legacy_mcp_passes_headers_to_the_transport(monkeypatch, transport):
    """mcp<2 keeps working: headers go to the transport, 3-tuple is accepted."""
    calls = _install_fake_mcp(monkeypatch, sdk="legacy")
    client = MCPClient(name="srv", config=_http_client_config(transport))
    await client.connect()
    try:
        assert calls["streamable_http"] == (
            "http://example.test/mcp",
            {"Authorization": "Bearer t"},
        )
        assert "http_client" not in calls
        assert client.connected
    finally:
        await client.close()


async def test_http_without_auth_headers_builds_a_client_without_headers(monkeypatch):
    calls = _install_fake_mcp(monkeypatch, sdk="modern")
    client = MCPClient(name="srv", config=_http_client_config(auth_headers={}))
    await client.connect()
    try:
        assert calls["http_client"].headers is None
    finally:
        await client.close()


async def test_http_client_is_closed_after_the_transport(monkeypatch):
    """The SDK never closes a client it was handed, and the transport sends
    its session-terminating DELETE on close — so the client must outlive it."""
    calls = _install_fake_mcp(monkeypatch, sdk="modern")
    client = MCPClient(name="srv", config=_http_client_config())
    await client.connect()
    await client.close()
    assert calls["teardown"] == ["transport", "http-client"]


async def test_http_client_is_closed_when_the_handshake_fails(monkeypatch):
    calls = _install_fake_mcp(monkeypatch, sdk="modern")

    class _FailingSession(_FakeSession):
        async def initialize(self) -> None:
            raise TimeoutError("no answer")

    monkeypatch.setattr(sys.modules["mcp"], "ClientSession", _FailingSession)
    client = MCPClient(name="srv", config=_http_client_config())
    with pytest.raises(TimeoutError):
        await client.connect()
    assert calls["teardown"] == ["transport", "http-client"]
    assert client.connected is False


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


# ---------------------------------------------------------------------------
# connect() failure path — nothing is published until the handshake succeeds
# ---------------------------------------------------------------------------

class _RecordingTransport:
    """Transport CM that records whether it was torn down."""

    def __init__(self, log: list[str], *, fail_on_enter: bool = False) -> None:
        self._log = log
        self._fail = fail_on_enter

    async def __aenter__(self) -> Any:
        if self._fail:
            raise ConnectionRefusedError("transport could not open")
        self._log.append("opened")
        return ("read", "write")

    async def __aexit__(self, *exc: Any) -> bool:
        self._log.append("closed")
        return False


def _install_handshake_fake(
    monkeypatch: pytest.MonkeyPatch,
    *,
    handshake: Any,
    fail_on_enter: bool = False,
) -> dict[str, Any]:
    """Fake SDK whose stdio transport records teardown and whose
    `initialize()` is *handshake* (an async callable)."""
    state: dict[str, Any] = {"log": [], "handshakes": 0}

    class _Session:
        def __init__(self, read: Any, write: Any) -> None:
            pass

        async def __aenter__(self) -> "_Session":
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            state["log"].append("session-closed")
            return False

        async def initialize(self) -> None:
            state["handshakes"] += 1
            await handshake(state["handshakes"])

    calls = _install_fake_mcp(monkeypatch)
    monkeypatch.setattr(sys.modules["mcp"], "ClientSession", _Session)
    monkeypatch.setattr(
        sys.modules["mcp.client.stdio"],
        "stdio_client",
        lambda params: _RecordingTransport(state["log"], fail_on_enter=fail_on_enter),
    )
    state["calls"] = calls
    return state


def _stdio_client() -> MCPClient:
    return MCPClient(name="srv", config=MCPConnectionConfig(transport="stdio", command=["x"]))


async def _always_fail(n: int) -> None:
    raise TimeoutError("server never answered initialize")


async def test_a_failed_handshake_leaves_the_client_disconnected(monkeypatch):
    """It used to report connected=True with a session that never handshook."""
    _install_handshake_fake(monkeypatch, handshake=_always_fail)
    client = _stdio_client()
    with pytest.raises(TimeoutError):
        await client.connect()
    assert client.connected is False
    assert client._exit_stack is None


async def test_a_failed_handshake_tears_down_the_transport(monkeypatch):
    """For stdio the transport is a subprocess; it must not outlive the failure."""
    state = _install_handshake_fake(monkeypatch, handshake=_always_fail)
    with pytest.raises(TimeoutError):
        await _stdio_client().connect()
    assert state["log"] == ["opened", "session-closed", "closed"]


async def test_a_retry_after_a_failed_handshake_handshakes_again(monkeypatch):
    """The retry used to return "success" without ever handshaking."""

    async def fail_then_succeed(n: int) -> None:
        if n == 1:
            raise TimeoutError("first handshake times out")

    state = _install_handshake_fake(monkeypatch, handshake=fail_then_succeed)
    client = _stdio_client()
    with pytest.raises(TimeoutError):
        await client.connect()
    await client.connect()
    try:
        assert state["handshakes"] == 2
        assert client.connected
    finally:
        await client.close()


async def test_a_transport_that_cannot_open_leaves_nothing_behind(monkeypatch):
    state = _install_handshake_fake(
        monkeypatch, handshake=_always_fail, fail_on_enter=True
    )
    client = _stdio_client()
    with pytest.raises(ConnectionRefusedError):
        await client.connect()
    assert client.connected is False
    assert state["handshakes"] == 0


async def test_cancellation_during_the_handshake_still_cleans_up(monkeypatch):
    import asyncio

    started = asyncio.Event()

    async def hang(n: int) -> None:
        started.set()
        await asyncio.Event().wait()

    state = _install_handshake_fake(monkeypatch, handshake=hang)
    client = _stdio_client()
    task = asyncio.create_task(client.connect())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.connected is False
    assert state["log"][-1] == "closed"


async def test_the_registry_retry_now_really_reconnects(monkeypatch, mcp_backoff):
    """What strict discovery (and so Agent.connect) relies on at boot."""

    async def fail_then_succeed(n: int) -> None:
        if n == 1:
            raise TimeoutError("first handshake times out")

    state = _install_handshake_fake(monkeypatch, handshake=fail_then_succeed)
    registry = MCPServerRegistry()
    client = _stdio_client()
    await registry._connect_with_retry("srv", client, max_attempts=2)
    try:
        assert state["handshakes"] == 2
        assert client.connected
    finally:
        await client.close()



# ---------------------------------------------------------------------------
# exception groups from the transport's task group
# ---------------------------------------------------------------------------



def test_sole_error_unwraps_nested_single_member_groups():
    cause = PermissionError("401 Unauthorized")
    group = ExceptionGroup("outer", [ExceptionGroup("inner", [cause])])
    assert _sole_error(group) is cause


def test_sole_error_keeps_a_group_with_several_errors():
    group = ExceptionGroup("g", [ValueError("a"), KeyError("b")])
    assert _sole_error(group) is None
    nested = ExceptionGroup("outer", [ExceptionGroup("inner", [ValueError(), KeyError()])])
    assert _sole_error(nested) is None


def test_sole_error_never_unwraps_a_cancellation():
    import asyncio

    group = BaseExceptionGroup("g", [asyncio.CancelledError()])
    assert _sole_error(group) is None


async def test_connect_surfaces_the_one_error_inside_a_task_group(monkeypatch):
    cause = PermissionError("401 Unauthorized")

    class _GroupRaisingTransport:
        async def __aenter__(self) -> Any:
            raise ExceptionGroup("unhandled errors in a TaskGroup", [cause])

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    _install_fake_mcp(monkeypatch)
    monkeypatch.setattr(
        sys.modules["mcp.client.stdio"], "stdio_client", lambda p: _GroupRaisingTransport()
    )
    client = _stdio_client()
    with pytest.raises(PermissionError) as caught:
        await client.connect()
    assert caught.value is cause
    assert client.connected is False


async def test_connect_leaves_a_multi_error_group_intact(monkeypatch):
    class _GroupRaisingTransport:
        async def __aenter__(self) -> Any:
            raise ExceptionGroup("two failures", [ValueError("a"), KeyError("b")])

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    _install_fake_mcp(monkeypatch)
    monkeypatch.setattr(
        sys.modules["mcp.client.stdio"], "stdio_client", lambda p: _GroupRaisingTransport()
    )
    with pytest.raises(ExceptionGroup):
        await _stdio_client().connect()


# ---------------------------------------------------------------------------
# retry backoff
# ---------------------------------------------------------------------------

async def test_retry_does_not_sleep_after_the_final_attempt(monkeypatch, mcp_backoff):
    """It used to wait 0.5 + 1 + 2s before raising; the last 2s bought nothing
    but a slower boot failure."""
    state = _install_handshake_fake(monkeypatch, handshake=_always_fail)
    with pytest.raises(TimeoutError):
        await MCPServerRegistry()._connect_with_retry("srv", _stdio_client(), max_attempts=3)
    assert state["handshakes"] == 3
    assert mcp_backoff == [0.5, 1.0]


async def test_a_single_attempt_never_sleeps(monkeypatch, mcp_backoff):
    _install_handshake_fake(monkeypatch, handshake=_always_fail)
    with pytest.raises(TimeoutError):
        await MCPServerRegistry()._connect_with_retry("srv", _stdio_client(), max_attempts=1)
    assert mcp_backoff == []


async def test_backoff_stops_once_an_attempt_succeeds(monkeypatch, mcp_backoff):
    async def fail_then_succeed(n: int) -> None:
        if n == 1:
            raise TimeoutError("first handshake times out")

    _install_handshake_fake(monkeypatch, handshake=fail_then_succeed)
    client = _stdio_client()
    await MCPServerRegistry()._connect_with_retry("srv", client, max_attempts=3)
    try:
        assert mcp_backoff == [0.5]
    finally:
        await client.close()
