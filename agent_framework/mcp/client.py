"""MCPClient — lifecycle of a single MCP server connection.

Supports `stdio` (local subprocess), `http`/`streamable_http` (streamable
HTTP) and deprecated `sse` transports, on both `mcp<2` and `mcp>=2`. Soft
dependency on the `mcp` Python package — the rest of the framework functions
fine without it.
"""

from __future__ import annotations

import asyncio
import warnings
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class MCPConnectionConfig:
    transport: Literal["stdio", "sse", "http", "streamable_http"] = "stdio"
    # stdio fields
    command: list[str] | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None
    # sse/http fields
    url: str | None = None
    auth_headers: dict[str, str] = field(default_factory=dict)
    # universal
    init_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 60.0


class MCPToolError(RuntimeError):
    """An MCP server reported a failed tool call (`CallToolResult.isError`)."""


def _raise_if_error(tool_name: str, result: Any, flattened: Any) -> None:
    """Turn an `isError` result into an exception.

    MCP reports tool failures in-band: the call succeeds at the protocol level
    and sets `isError` on the result. Ignoring that flag makes every failure look
    like a success — the trace records OK, `tool_stats()` reports 100%, and the
    model is handed an error message dressed up as data. Raising here lets the
    normal `ToolExecutor` path convert it into `ToolResult(is_error=True)`, which
    is what the rest of the framework already knows how to handle.
    """
    if not (getattr(result, "is_error", None) or getattr(result, "isError", None)):
        return
    detail = flattened if isinstance(flattened, str) else repr(flattened)
    raise MCPToolError(f"{tool_name}: {detail}")


def _normalize_annotations(raw: Any) -> dict[str, Any]:
    """Return MCP tool annotations as a camelCase dict.

    The SDK changed shape: `mcp<2` handed back a plain dict, `mcp>=2` hands back
    a pydantic `ToolAnnotations` whose fields are snake_case. Everything
    downstream reads the wire names (`readOnlyHint`, `idempotentHint`), so both
    forms are collapsed here rather than at every call site.
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    model_dump = getattr(raw, "model_dump", None)
    if callable(model_dump):
        return model_dump(by_alias=True, exclude_none=True)
    return dict(getattr(raw, "__dict__", {}))


def _sole_error(exc: BaseException) -> Exception | None:
    """The single error inside nested one-member exception groups, else None.

    None when some level holds several errors, or when the innermost member
    is not an `Exception` (a real cancellation must propagate untouched).
    """
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    if isinstance(exc, BaseExceptionGroup) or not isinstance(exc, Exception):
        return None
    return exc


@dataclass
class MCPToolDescriptor:
    """Description of a tool advertised by an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)


class MCPClient:
    """Lifecycle wrapper around a single MCP server connection.

    Uses the official `mcp` package when available. Otherwise raises a
    descriptive ImportError at connect time.
    """

    def __init__(self, name: str, config: MCPConnectionConfig) -> None:
        self.name = name
        self.config = config
        self._session: Any = None
        self._exit_stack: Any = None
        self._tools_cache: list[MCPToolDescriptor] | None = None
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> None:
        async with self._lock:
            if self._session is not None:
                return
            try:
                from mcp import ClientSession
            except ImportError as e:
                raise ImportError(
                    "The `mcp` package is required for MCP support. Install with "
                    "`pip install mcp`."
                ) from e

            # Nothing is published until the handshake succeeds. Assigning
            # `_session` before `initialize()` made a failed handshake look like
            # a live connection: `connected` turned True, the next attempt in
            # `_connect_with_retry` returned "success" without handshaking, and
            # the transport — for stdio, a subprocess — was never closed.
            #
            # On failure the stack must be exited *with* the exception rather
            # than closed as if nothing happened: an HTTP transport runs an
            # anyio task group, and when one of its requests fails (a 401, say)
            # it cancels its scope, which delivers a CancelledError into this
            # task. Only when that exception passes through the scope's exit can
            # anyio recognize its own cancellation and raise the real error in
            # its place — otherwise a rejected token escapes as a spurious
            # CancelledError, which no caller treats as a connection failure.
            try:
                async with AsyncExitStack() as stack:
                    read, write = await self._open_transport(stack)
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await asyncio.wait_for(
                        session.initialize(), timeout=self.config.init_timeout_seconds
                    )
                    # Success: take ownership, so leaving this block closes nothing.
                    self._exit_stack = stack.pop_all()
                    self._session = session
            except BaseExceptionGroup as group:
                # The task group reports even a single failure as a group, often
                # nested, so the cause reads "unhandled errors in a TaskGroup (1
                # sub-exception)" instead of "401 Unauthorized". That wrapper is
                # an SDK implementation detail; surface the one real error.
                # Genuine multi-error groups are left intact.
                leaf = _sole_error(group)
                if leaf is None:
                    raise
                raise leaf from None

    async def _open_transport(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        """Enter the configured transport on *stack*; return (read, write)."""
        if self.config.transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client
            if not self.config.command:
                raise ValueError("stdio transport requires `command`")
            params = StdioServerParameters(
                command=self.config.command[0],
                args=self.config.command[1:],
                # None means "inherit the SDK's default environment";
                # do not collapse it to {}.
                env=self.config.env,
                # Several servers resolve relative paths and sandbox roots
                # against their own working directory, so `cwd` has to reach
                # the subprocess rather than sit unused on the config.
                cwd=self.config.cwd,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            return read, write
        if self.config.transport in ("http", "streamable_http"):
            if not self.config.url:
                raise ValueError(f"{self.config.transport} transport requires `url`")
            streams = await self._open_streamable_http(stack, self.config.url)
            # mcp>=2 yields (read, write); mcp<2 appended a get_session_id
            # callable. Index rather than unpack so both shapes work.
            return streams[0], streams[1]
        if self.config.transport == "sse":
            warnings.warn(
                "SSE transport is deprecated in MCP; prefer 'http' "
                "(streamable HTTP).",
                DeprecationWarning,
            )
            from mcp.client.sse import sse_client
            if not self.config.url:
                raise ValueError(f"{self.config.transport} transport requires `url`")
            read, write = await stack.enter_async_context(
                sse_client(self.config.url, headers=self.config.auth_headers)
            )
            return read, write
        raise ValueError(f"Unsupported transport: {self.config.transport}")

    async def _open_streamable_http(self, stack: AsyncExitStack, url: str) -> Any:
        """Enter the streamable HTTP transport, for either SDK generation.

        mcp 2 renamed `streamablehttp_client` to `streamable_http_client` and
        also changed how it is called — it no longer takes `headers=`, and
        yields a 2-tuple instead of a 3-tuple. Selecting the new name but
        calling it the old way made every HTTP server fail with a TypeError.
        """
        import mcp.client.streamable_http as _sh

        modern = getattr(_sh, "streamable_http_client", None)
        if modern is None:
            # mcp<2: headers are an argument of the transport itself.
            legacy = _sh.streamablehttp_client  # type: ignore[attr-defined]
            return await stack.enter_async_context(
                legacy(url, headers=self.config.auth_headers)
            )

        # mcp>=2: headers travel on a pre-built HTTP client. The SDK's own
        # factory is used because it returns the client class that SDK version
        # expects (`httpx2` in mcp 2, not `httpx`) with MCP's recommended
        # timeouts, including a long read timeout for streamed responses.
        # The SDK does not close a client it was handed, so it goes on our
        # stack — entered first, so that the transport, which sends the
        # session-terminating DELETE on close, is torn down before its client.
        from mcp.shared._httpx_utils import create_mcp_http_client

        http = await stack.enter_async_context(
            create_mcp_http_client(headers=self.config.auth_headers or None)
        )
        return await stack.enter_async_context(modern(url, http_client=http))

    async def close(self) -> None:
        async with self._lock:
            if self._exit_stack is not None:
                try:
                    await self._exit_stack.__aexit__(None, None, None)
                except Exception:
                    pass
            self._session = None
            self._exit_stack = None
            self._tools_cache = None

    async def list_tools(self, *, refresh: bool = False) -> list[MCPToolDescriptor]:
        if self._tools_cache is not None and not refresh:
            return self._tools_cache
        if self._session is None:
            await self.connect()
        listed = await asyncio.wait_for(
            self._session.list_tools(), timeout=self.config.request_timeout_seconds
        )
        descriptors: list[MCPToolDescriptor] = []
        for t in getattr(listed, "tools", []) or []:
            descriptors.append(
                MCPToolDescriptor(
                    name=getattr(t, "name", ""),
                    description=getattr(t, "description", "") or "",
                    # mcp>=2 exposes the pydantic field `input_schema`; older
                    # SDKs used the wire name `inputSchema`. Reading only the
                    # latter silently advertised every MCP tool with an empty
                    # parameter schema, leaving the model no idea what to pass.
                    input_schema=(
                        getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
                    ),
                    annotations=_normalize_annotations(getattr(t, "annotations", None)),
                )
            )
        self._tools_cache = descriptors
        return descriptors

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._session is None:
            await self.connect()
        result = await asyncio.wait_for(
            self._session.call_tool(name, arguments),
            timeout=self.config.request_timeout_seconds,
        )
        # Flatten the MCP result content blocks into a single text/JSON payload.
        content = getattr(result, "content", None)
        if content is None:
            _raise_if_error(name, result, result)
            return result
        parts: list[Any] = []
        for c in content:
            ctype = getattr(c, "type", None)
            if ctype == "text":
                parts.append(getattr(c, "text", ""))
            elif ctype == "json":
                parts.append(getattr(c, "json", None))
            else:
                parts.append(getattr(c, "__dict__", str(c)))
        flattened = parts[0] if len(parts) == 1 else parts
        _raise_if_error(name, result, flattened)
        return flattened
