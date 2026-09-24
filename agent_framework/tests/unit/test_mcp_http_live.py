"""MCP over streamable HTTP, against a real server from the installed SDK.

The fake-SDK tests in test_mcp.py pin our calling conventions, but a fake can
only be as right as its author's model of the SDK. That model once described
mcp<2 alone, so an HTTP transport that raised TypeError against every real
server passed all of them. These tests stand up an actual MCP server in-process
(uvicorn on an ephemeral localhost port — still fully offline) and go through
the real SDK client end to end, so SDK drift fails here instead of in
production. They skip when the optional `mcp` extra is not installed; CI
installs it for exactly this reason.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import socket
import threading
import time
from typing import Any

import pytest

pytest.importorskip("mcp")
uvicorn = pytest.importorskip("uvicorn")

from mcp.types import ToolAnnotations

from agent_framework.core.agent import Agent
from agent_framework.core.message import ToolResultMessage
from agent_framework.mcp.client import MCPClient, MCPConnectionConfig
from agent_framework.mcp.registry import MCPServerRegistry
from agent_framework.permissions.models import PermissionCategory
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn
from agent_framework.tracing.trace import ExecutionTrace

TOKEN = "Bearer s3cret-token"


def _server_class() -> Any:
    import mcp.server

    # mcp>=2 ships MCPServer; mcp<2 called the same thing FastMCP.
    cls = getattr(mcp.server, "MCPServer", None)
    if cls is None:
        cls = importlib.import_module("mcp.server.fastmcp").FastMCP
    return cls


class _Recorder:
    """ASGI wrapper: demands the bearer token and records every exchange."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.exchanges: list[dict[str, Any]] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.inner(scope, receive, send)
            return
        exchange = {
            "method": scope["method"],
            "auth": dict(scope["headers"]).get(b"authorization", b"").decode(),
            "status": None,
        }
        self.exchanges.append(exchange)

        async def recording_send(message: Any) -> None:
            if message["type"] == "http.response.start":
                exchange["status"] = message["status"]
            await send(message)

        if exchange["auth"] != TOKEN:
            await recording_send(
                {"type": "http.response.start", "status": 401, "headers": []}
            )
            await send({"type": "http.response.body", "body": b"unauthorized"})
            return
        await self.inner(scope, receive, recording_send)


@pytest.fixture(scope="module")
def mcp_server():
    """A real MCP server with one read-only tool, on an ephemeral port."""
    logging.getLogger("mcp").setLevel(logging.WARNING)
    server = _server_class()("stock")

    @server.tool(
        description="Stock for a SKU.",
        annotations=ToolAnnotations.model_validate({"readOnlyHint": True}),
    )
    def get_stock(sku: str) -> str:
        return f"{sku}: 7"

    app = _Recorder(server.streamable_http_app())
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    sock.setblocking(False)
    httpd = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    thread = threading.Thread(target=httpd.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not httpd.started and time.monotonic() < deadline:
        time.sleep(0.01)
    if not httpd.started:
        pytest.fail("in-process MCP server did not start")
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}/mcp", app
    finally:
        httpd.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def server(mcp_server):
    url, recorder = mcp_server
    recorder.exchanges.clear()
    return url, recorder


def _client(url: str, token: str = TOKEN) -> MCPClient:
    return MCPClient(
        name="stock",
        config=MCPConnectionConfig(
            transport="http",
            url=url,
            auth_headers={"Authorization": token},
            init_timeout_seconds=10,
        ),
    )


# ---------------- the transport itself ----------------

async def test_connects_and_lists_tools_with_bearer_auth(server):
    url, _ = server
    client = _client(url)
    await client.connect()
    try:
        assert [t.name for t in await client.list_tools()] == ["get_stock"]
    finally:
        await client.close()


async def test_calls_a_tool_over_http(server):
    url, _ = server
    client = _client(url)
    await client.connect()
    try:
        assert await client.call_tool("get_stock", {"sku": "LAP-14"}) == "LAP-14: 7"
    finally:
        await client.close()


async def test_every_request_carries_the_auth_header(server):
    """Headers now ride on the HTTP client, so they must reach every request."""
    url, recorder = server
    client = _client(url)
    await client.connect()
    await client.list_tools()
    await client.close()
    assert recorder.exchanges
    assert {e["auth"] for e in recorder.exchanges} == {TOKEN}
    assert all(e["status"] != 401 for e in recorder.exchanges)


async def test_close_terminates_the_session_on_the_server(server):
    """The transport sends a DELETE on close using our HTTP client, so the
    client must still be open then — it is torn down after the transport."""
    url, recorder = server
    client = _client(url)
    await client.connect()
    await client.close()
    deletes = [e for e in recorder.exchanges if e["method"] == "DELETE"]
    assert len(deletes) == 1
    assert deletes[0]["auth"] == TOKEN
    assert deletes[0]["status"] is not None and deletes[0]["status"] < 400


async def test_a_wrong_token_fails_as_an_ordinary_error(server):
    """A rejected handshake must surface as an Exception, never as a spurious
    CancelledError. On mcp 1.x the transport's task group cancels its scope
    when a request fails; unless that cancellation passes back through the
    scope's exit, it escapes as CancelledError — which retry and discovery,
    catching Exception, cannot see, and which run() would then propagate as if
    the caller had cancelled it."""
    url, recorder = server
    client = _client(url, token="Bearer wrong")
    error: Exception | None = None
    try:
        await client.connect()
    except asyncio.CancelledError:  # pragma: no cover - the regression itself
        pytest.fail("a rejected token escaped as CancelledError")
    except Exception as exc:  # noqa: BLE001 - the SDK's error type varies by version
        error = exc
    assert error is not None, "connect() accepted a rejected token"
    # The SDK's task group wraps even one failure in (nested) groups; the
    # client unwraps it so callers see the real error, not the wrapper.
    assert not isinstance(error, BaseExceptionGroup)
    assert client.connected is False
    assert recorder.exchanges and recorder.exchanges[0]["status"] == 401


# ---------------- through the whole framework ----------------

async def test_an_agent_runs_an_mcp_tool_over_http(server):
    """Agent.connect discovers over HTTP; the loop then executes the tool."""
    url, _ = server
    registry = MCPServerRegistry(trace=ExecutionTrace()).add(
        "stock",
        MCPConnectionConfig(
            transport="http", url=url, auth_headers={"Authorization": TOKEN}
        ),
    )
    mock = MockLLMProvider(
        [
            ScriptedTurn(
                text="Checking.",
                tool_calls=[
                    {"id": "c1", "name": "stock__get_stock", "arguments": {"sku": "LAP-14"}}
                ],
            ),
            ScriptedTurn(text="LAP-14 has 7 in stock."),
        ]
    )
    try:
        agent = await Agent.connect("claude-opus-5", mcp=registry, provider=mock)
        tool = agent.tools.get("stock__get_stock")
        assert tool is not None
        # readOnlyHint travels over HTTP and becomes AUTO, so no human is asked.
        assert tool.metadata.permission is PermissionCategory.AUTO
        result = await agent.run("Is LAP-14 in stock?")
    finally:
        await registry.close_all()

    assert result.error is None
    assert result.final_message == "LAP-14 has 7 in stock."
    tool_results = [
        r for m in result.messages if isinstance(m, ToolResultMessage) for r in m.content
    ]
    assert len(tool_results) == 1
    assert tool_results[0].is_error is False
    assert "LAP-14: 7" in str(tool_results[0].content)


async def test_agent_connect_fails_at_boot_on_a_rejected_token(server, mcp_backoff):
    """The promise Agent.connect() exists to keep, checked for real: a server
    that refuses us fails construction — with a message naming the cause."""
    from agent_framework.mcp.registry import MCPDiscoveryError

    url, _ = server
    registry = MCPServerRegistry(trace=ExecutionTrace()).add(
        "stock",
        MCPConnectionConfig(
            transport="http", url=url, auth_headers={"Authorization": "Bearer wrong"}
        ),
    )
    try:
        with pytest.raises(MCPDiscoveryError) as caught:
            await Agent.connect("claude-opus-5", mcp=registry, provider=MockLLMProvider([]))
    finally:
        await registry.close_all()
    message = str(caught.value)
    assert "stock" in message
    assert "TaskGroup" not in message
