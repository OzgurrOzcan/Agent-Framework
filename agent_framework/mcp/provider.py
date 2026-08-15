"""MCPToolProvider — adapts MCP-advertised tools into framework `Tool`s.

MCP tools become indistinguishable from native tools to the agent: they
flow through the same `ToolExecutor`, permission engine, tracing, and
result handling.

Permission category defaults from MCP annotations:
  - readOnlyHint=True       → AUTO
  - destructiveHint=True    → ASK
  - otherwise               → ASK
"""

from __future__ import annotations

from typing import Any

from agent_framework.mcp.client import MCPClient, MCPToolDescriptor
from agent_framework.permissions.models import PermissionCategory
from agent_framework.tools.models import Tool, ToolMetadata


class MCPToolProvider:
    def __init__(self, client: MCPClient) -> None:
        self.client = client

    async def discover(self) -> list[Tool]:
        descriptors = await self.client.list_tools()
        return [self._build_tool(d) for d in descriptors]

    def _build_tool(self, d: MCPToolDescriptor) -> Tool:
        category = _category_from_annotations(d.annotations)
        meta = ToolMetadata(
            # Namespace by server name so identically-named tools from
            # different servers coexist (e.g. "github__search_issues").
            # The server itself is still called with the original name.
            name=f"{self.client.name}__{d.name}",
            description=d.description,
            parameters_schema=d.input_schema or {"type": "object", "properties": {}},
            permission=category,
            timeout_seconds=self.client.config.request_timeout_seconds,
            max_retries=0,
            idempotent=bool(d.annotations.get("idempotentHint", False)),
            cache_ttl_seconds=None,
            tags=("mcp", self.client.name),
            is_async=True,
            source=f"mcp:{self.client.name}",
        )

        client = self.client

        async def _invoke(**kwargs: Any) -> Any:
            return await client.call_tool(d.name, kwargs)

        return Tool(metadata=meta, fn=_invoke)


def _category_from_annotations(ann: dict[str, Any]) -> PermissionCategory:
    if ann.get("readOnlyHint"):
        return PermissionCategory.AUTO
    if ann.get("destructiveHint"):
        return PermissionCategory.ASK
    return PermissionCategory.ASK
