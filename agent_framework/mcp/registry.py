"""MCPServerRegistry — named collection of MCP server connections.

Supports lazy connection (connect on first list/call) and connection retry
with exponential backoff at discovery time. Connection failures are
emitted as `error` trace events when a trace is attached.
"""

from __future__ import annotations

import asyncio

from agent_framework.mcp.client import MCPClient, MCPConnectionConfig
from agent_framework.mcp.provider import MCPToolProvider
from agent_framework.tools.models import Tool
from agent_framework.tracing.trace import EventType, ExecutionTrace, SpanStatus


class MCPServerRegistry:
    def __init__(self, *, trace: ExecutionTrace | None = None) -> None:
        self._clients: dict[str, MCPClient] = {}
        self._providers: dict[str, MCPToolProvider] = {}
        self._configs: dict[str, MCPConnectionConfig] = {}
        self._trace = trace

    def add(self, name: str, config: MCPConnectionConfig) -> "MCPServerRegistry":
        client = MCPClient(name=name, config=config)
        self._clients[name] = client
        self._providers[name] = MCPToolProvider(client)
        self._configs[name] = config
        return self

    def remove(self, name: str) -> None:
        self._clients.pop(name, None)
        self._providers.pop(name, None)
        self._configs.pop(name, None)

    def names(self) -> list[str]:
        return list(self._clients.keys())

    def get(self, name: str) -> MCPClient | None:
        return self._clients.get(name)

    async def connect_all(self) -> None:
        for name, client in self._clients.items():
            await self._connect_with_retry(name, client)

    async def discover_all(self) -> list[Tool]:
        out: list[Tool] = []
        for name, provider in self._providers.items():
            try:
                await self._connect_with_retry(name, self._clients[name])
                tools = await provider.discover()
                out.extend(tools)
            except Exception as exc:
                if self._trace is not None:
                    from agent_framework.tracing.trace import ErrorInfo
                    self._trace.record_event(
                        EventType.ERROR,
                        name=f"mcp:{name}",
                        status=SpanStatus.ERROR,
                        error=ErrorInfo.from_exception(exc),
                    )
        return out

    def discover_all_sync(self) -> list[Tool]:
        """Synchronous convenience for `BaseAgent.create()`.

        Runs `discover_all()` in a new event loop. If an event loop is
        already running, raises `RuntimeError` — synchronous discovery
        cannot block inside async code; use `await BaseAgent.acreate(...)`
        (which awaits `discover_all()`) instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.discover_all())
        raise RuntimeError(
            "BaseAgent.create(mcp=...) cannot discover MCP tools inside a running "
            "event loop. Use `await BaseAgent.acreate(...)` instead."
        )

    async def close_all(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                pass

    async def _connect_with_retry(
        self, name: str, client: MCPClient, *, max_attempts: int = 3
    ) -> None:
        if client.connected:
            return
        delay = 0.5
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                await client.connect()
                return
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)
        if last_exc is not None:
            raise last_exc
