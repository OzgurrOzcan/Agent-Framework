"""MockMCPServer — in-process MCP stub for testing without real servers.

Register tools with a callable, then use `as_client()` to get an
`MCPClient`-compatible object that routes `call_tool()` through the
registered handlers.

Usage:
    mock_mcp = MockMCPServer("test_server")
    mock_mcp.register("read_file", lambda path: f"Contents of {path}")

    client = mock_mcp.as_client()
    tools = await client.list_tools()
    result = await client.call_tool("read_file", {"path": "/etc/hosts"})
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

from agent_framework.mcp.client import MCPToolDescriptor, MCPConnectionConfig


class MockMCPServer:
    def __init__(self, name: str = "mock") -> None:
        self.name = name
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._annotations: dict[str, dict[str, Any]] = {}

    def register(
        self,
        tool_name: str,
        handler: Callable[..., Any],
        *,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        annotations: dict[str, Any] | None = None,
    ) -> None:
        self._handlers[tool_name] = handler
        self._schemas[tool_name] = input_schema or {"type": "object", "properties": {}}
        self._annotations[tool_name] = annotations or {"readOnlyHint": True}
        if not description:
            description = f"Mock tool: {tool_name}"
        self._schemas[tool_name]["_description"] = description

    def as_client(self) -> "_MockMCPClient":
        return _MockMCPClient(self)


class _MockMCPClient:
    """Quacks like MCPClient for testing.

    Every `call_tool()` invocation is recorded in `self.calls` as a
    `(tool_name, arguments)` tuple so tests can assert routing/naming.
    """

    def __init__(self, server: MockMCPServer) -> None:
        self._server = server
        self.name = server.name
        self.config = MCPConnectionConfig()
        self._connected = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    async def list_tools(self, *, refresh: bool = False) -> list[MCPToolDescriptor]:
        return [
            MCPToolDescriptor(
                name=name,
                description=schema.get("_description", name),
                input_schema={k: v for k, v in schema.items() if k != "_description"},
                annotations=self._server._annotations.get(name, {}),
            )
            for name, schema in self._server._schemas.items()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        handler = self._server._handlers.get(name)
        if handler is None:
            raise ValueError(f"MockMCPServer: unknown tool '{name}'")
        if inspect.iscoroutinefunction(handler):
            return await handler(**arguments)
        return await asyncio.to_thread(handler, **arguments)
