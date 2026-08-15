"""Tool data model.

`ToolMetadata` is the canonical description of a tool (name, JSON schema,
permission category, retry/timeout/idempotency/cache config, tags).
`Tool` binds metadata to a callable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agent_framework.permissions.models import PermissionCategory


class RetryableToolError(Exception):
    """Raise this from a tool to opt into retry. The executor only retries
    tools marked `idempotent=True` and only on this exception type."""


class ToolTimeoutError(Exception):
    """Raised by the executor when `timeout_seconds` elapses."""


@dataclass
class ToolMetadata:
    name: str
    description: str
    parameters_schema: dict[str, Any]  # JSON-Schema
    permission: PermissionCategory = PermissionCategory.ASK
    timeout_seconds: float = 60.0
    max_retries: int = 0
    idempotent: bool = False
    cache_ttl_seconds: float | None = None
    tags: tuple[str, ...] = ()
    is_async: bool = True
    source: str = "native"  # "native" | "mcp:<server>"

    def with_overrides(self, **changes: Any) -> "ToolMetadata":
        from dataclasses import replace
        return replace(self, **changes)


# A tool function is either sync or async; the executor handles both.
ToolFn = Callable[..., Awaitable[Any] | Any]


@dataclass
class Tool:
    metadata: ToolMetadata
    fn: ToolFn

    @property
    def name(self) -> str:
        return self.metadata.name
