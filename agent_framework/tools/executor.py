"""ToolExecutor — async execution of tool calls.

Responsibilities (all traced):
  - Run async/sync tools (sync → thread).
  - Validate call arguments against the tool's JSON schema before
    invoking (invalid arguments become an `is_error` ToolResult the
    model can react to — the tool function is never called).
  - Apply per-tool timeout via `asyncio.wait_for()`.
  - Retry only `idempotent=True` tools on `RetryableToolError`.
  - Cap every result at the tool's token limit (head + tail kept) at the
    single place results are produced, before the cache stores it — so a
    huge result never reaches the history, the cache, or the provider.
  - Check/populate optional result cache.
  - Always return a structured `ToolResult` — including for exceptions,
    timeouts, denials, and cancellations. The provider-level invariant
    (one tool_use → one tool_result) is non-negotiable.
  - On cancellation, `run_many` raises `ToolCancellationError` carrying
    the partial results so the loop can keep the invariant intact.
  - Intercept the built-in `ask_user` tool and route through the
    InterruptManager as a CLARIFICATION interrupt.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
from dataclasses import dataclass
from typing import Any

import jsonschema

from agent_framework.interrupts.manager import InterruptManager
from agent_framework.interrupts.models import (
    InterruptClass,
    InterruptEvent,
)
from agent_framework.tools.builtin.ask_user import ASK_USER_TOOL_NAME
from agent_framework.tools.models import (
    RetryableToolError,
    Tool,
    ToolMetadata,
    ToolTimeoutError,
    validate_result_cap,
)
from agent_framework.core.message import (
    ToolCallRequest,
    ToolResult,
    content_as_text,
    estimate_tokens,
)
from agent_framework.tracing.trace import (
    ErrorInfo,
    EventType,
    ExecutionTrace,
    SpanStatus,
)


# Share of the character budget kept from the start of a capped result; the
# rest comes from the end. The head usually says what the output is, the tail
# where it ended up (the last log lines, the final error).
_HEAD_SHARE = 0.6


def cap_result_content(content: Any, max_tokens: int | None) -> tuple[Any, bool, int]:
    """Cap *content* at *max_tokens*: ``(content, truncated, original_tokens)``.

    Content within the cap is returned unchanged — including non-string
    values, which the adapters serialize later. Over the cap, the serialized
    text keeps its first 60 % and last 40 % of ``max_tokens * 4`` characters
    with a marker in between, so the model can see both what the output was
    and how it ended, and knows something is missing.
    """
    text = content_as_text(content)
    original_tokens = estimate_tokens(text)
    if max_tokens is None or original_tokens <= max_tokens:
        return content, False, original_tokens
    budget = max_tokens * 4
    head_len = int(budget * _HEAD_SHARE)
    tail_len = budget - head_len
    omitted = len(text) - head_len - tail_len
    marker = (
        f"\n\n[... {omitted:,} characters truncated: the result was "
        f"≈{original_tokens:,} tokens and this tool's results are capped at "
        f"{max_tokens:,}. Call it again with narrower arguments if you need "
        "the missing part ...]\n\n"
    )
    return text[:head_len] + marker + text[len(text) - tail_len :], True, original_tokens


class ToolCancellationError(asyncio.CancelledError):
    """Cancellation that carries the partial ToolResults produced so far.

    `run_many` raises this instead of a bare CancelledError so the loop can
    append the placeholder results and keep the tool_use/tool_result
    invariant intact before re-raising.
    """
    def __init__(self, partial_results: list[ToolResult]) -> None:
        super().__init__()
        self.partial_results = partial_results


@dataclass
class _CacheEntry:
    expires_at: float
    value: Any


class ToolExecutor:
    def __init__(
        self,
        *,
        trace: ExecutionTrace | None = None,
        interrupt_manager: InterruptManager | None = None,
        session_id: str = "",
        max_parallel: int | None = None,
        default_max_result_tokens: int | None = None,
    ) -> None:
        validate_result_cap(default_max_result_tokens, "default_max_result_tokens")
        self._trace = trace
        self._interrupt = interrupt_manager
        self._session_id = session_id
        self._cache: dict[str, _CacheEntry] = {}
        self._max_parallel = max_parallel if (max_parallel or 0) > 0 else None
        self._default_cap = default_max_result_tokens

    def result_cap(self, meta: ToolMetadata) -> int | None:
        """The token cap applied to *meta*'s results (None = uncapped)."""
        if meta.max_result_tokens is not None:
            return meta.max_result_tokens
        return self._default_cap

    async def run(
        self,
        tool: Tool,
        call: ToolCallRequest,
        *,
        turn_number: int = 0,
    ) -> ToolResult:
        """Execute a single tool call. Always returns a ToolResult."""
        meta = tool.metadata

        # Special handling: the built-in ask_user.
        if meta.name == ASK_USER_TOOL_NAME:
            answer = await self._run_ask_user(meta, call, turn_number=turn_number)
            answer.content, _, _ = cap_result_content(answer.content, self.result_cap(meta))
            return answer

        # Cache lookup (opt-in per tool).
        cache_key = (
            self._cache_key(meta.name, call.arguments)
            if meta.cache_ttl_seconds
            else None
        )
        if cache_key is not None:
            entry = self._cache.get(cache_key)
            if entry is not None and entry.expires_at > time.monotonic():
                # Stored already capped (see the success path below).
                return ToolResult(
                    tool_use_id=call.id,
                    name=meta.name,
                    content=entry.value,
                    is_error=False,
                    duration_ms=0,
                )

        # Validate arguments against the tool's schema before invoking.
        # Tools with an empty/propertyless schema (e.g. MCP `{}`) are skipped.
        schema = meta.parameters_schema
        if isinstance(schema, dict) and schema.get("properties"):
            try:
                jsonschema.validate(call.arguments, schema)
            except jsonschema.ValidationError as e:
                path = "/".join(map(str, e.absolute_path)) or "<root>"
                return self._finish(
                    meta,
                    call,
                    (
                        f"Invalid arguments for `{meta.name}`: {e.message} "
                        f"(at: {path}). Fix the arguments and retry."
                    ),
                    is_error=True,
                    duration_ms=0,
                    status=SpanStatus.ERROR,
                    payload={"validation_error": True},
                )

        span_ctx = (
            self._trace.start_span(
                EventType.TOOL_CALL,
                name=meta.name,
                payload={"tool_name": meta.name, "args": call.arguments, "source": meta.source},
            )
            if self._trace is not None
            else None
        )
        if span_ctx is not None:
            span_ctx.__enter__()

        start = time.monotonic()
        last_exc: BaseException | None = None
        try:
            attempts = max(1, 1 + (meta.max_retries if meta.idempotent else 0))
            for attempt in range(attempts):
                try:
                    value = await self._invoke(tool, call.arguments, meta.timeout_seconds)
                    duration_ms = int((time.monotonic() - start) * 1000)
                    result = self._finish(
                        meta,
                        call,
                        value,
                        is_error=False,
                        duration_ms=duration_ms,
                        status=SpanStatus.OK,
                        payload={"retry_attempt": attempt, "duration_ms": duration_ms},
                    )
                    # Cache the capped content: a hit must not resurrect the
                    # full result the cap was there to keep out.
                    if cache_key is not None and meta.cache_ttl_seconds:
                        self._cache[cache_key] = _CacheEntry(
                            expires_at=time.monotonic() + meta.cache_ttl_seconds,
                            value=result.content,
                        )
                    return result
                except asyncio.CancelledError:
                    if self._trace is not None:
                        self._trace.record_event(
                            EventType.TOOL_RESULT,
                            name=meta.name,
                            status=SpanStatus.CANCELLED,
                            payload={"tool_name": meta.name, "retry_attempt": attempt},
                        )
                    raise
                except ToolTimeoutError as exc:
                    last_exc = exc
                    return self._finish(
                        meta,
                        call,
                        f"Tool `{meta.name}` timed out after {meta.timeout_seconds}s.",
                        is_error=True,
                        duration_ms=int((time.monotonic() - start) * 1000),
                        status=SpanStatus.TIMEOUT,
                        payload={"retry_attempt": attempt},
                    )
                except RetryableToolError as exc:
                    last_exc = exc
                    if not meta.idempotent or attempt + 1 >= attempts:
                        # Out of retries (or not idempotent → never retried).
                        break
                    # Exponential backoff.
                    await asyncio.sleep(min(2 ** attempt * 0.1, 2.0))
                    if span_ctx is not None:
                        span_ctx.update(retried=attempt + 1)
                    continue
                except Exception as exc:
                    last_exc = exc
                    break  # non-retryable

            # If we fell out of the loop, last_exc is the failure.
            assert last_exc is not None
            err = ErrorInfo.from_exception(
                last_exc,
                tool_name=meta.name,
                tool_args=call.arguments,
                retry_attempt=attempts - 1,
            )
            if span_ctx is not None:
                span_ctx.set_error(err)
            # Exception messages can be arbitrarily large (a dumped response
            # body), so error results are capped like any other — they are
            # still never cleared from the context later.
            return self._finish(
                meta,
                call,
                f"Tool `{meta.name}` failed: {type(last_exc).__name__}: {last_exc}",
                is_error=True,
                duration_ms=int((time.monotonic() - start) * 1000),
                status=SpanStatus.ERROR,
                error=err,
            )
        finally:
            if span_ctx is not None:
                span_ctx.__exit__(None, None, None)

    async def run_many(
        self,
        calls: list[tuple[Tool, ToolCallRequest]],
        *,
        turn_number: int = 0,
    ) -> list[ToolResult]:
        """Dispatch multiple tool calls concurrently. Order of results matches
        input. Concurrency is capped by `max_parallel` when configured."""
        if not calls:
            return []
        sem = asyncio.Semaphore(self._max_parallel) if self._max_parallel else None

        async def _bounded(t: Tool, c: ToolCallRequest) -> ToolResult:
            if sem is None:
                return await self.run(t, c, turn_number=turn_number)
            async with sem:
                return await self.run(t, c, turn_number=turn_number)

        tasks = [asyncio.create_task(_bounded(t, c)) for t, c in calls]
        try:
            return await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            # Build placeholder results so the message-history invariant survives.
            await asyncio.gather(*tasks, return_exceptions=True)
            results: list[ToolResult] = []
            for (tool, call), task in zip(calls, tasks):
                if task.done() and not task.cancelled() and task.exception() is None:
                    results.append(task.result())
                else:
                    results.append(
                        ToolResult(
                            tool_use_id=call.id,
                            name=tool.metadata.name,
                            content="Tool execution cancelled.",
                            is_error=True,
                            duration_ms=None,
                        )
                    )
            raise ToolCancellationError(results) from None

    # --- denial path: produce a tool_result without executing ---
    def deny_result(
        self,
        tool: ToolMetadata,
        call: ToolCallRequest,
        feedback: str,
    ) -> ToolResult:
        return self._finish(
            tool,
            call,
            feedback,
            is_error=True,
            duration_ms=0,
            status=SpanStatus.DENIED,
            payload={"feedback": feedback},
        )

    # ---------------- private helpers ----------------
    def _finish(
        self,
        meta: ToolMetadata,
        call: ToolCallRequest,
        content: Any,
        *,
        is_error: bool,
        duration_ms: int | None,
        status: SpanStatus,
        payload: dict[str, Any] | None = None,
        error: ErrorInfo | None = None,
    ) -> ToolResult:
        """The one place a traced `ToolResult` is built: cap, record, return."""
        content, truncated, original_tokens = cap_result_content(
            content, self.result_cap(meta)
        )
        if self._trace is not None:
            self._trace.record_event(
                EventType.TOOL_RESULT,
                name=meta.name,
                status=status,
                payload={
                    "tool_name": meta.name,
                    **(payload or {}),
                    "truncated": truncated,
                    "original_tokens": original_tokens,
                },
                error=error,
            )
        return ToolResult(
            tool_use_id=call.id,
            name=meta.name,
            content=content,
            is_error=is_error,
            duration_ms=duration_ms,
        )
    async def _invoke(self, tool: Tool, args: dict[str, Any], timeout: float) -> Any:
        async def _call() -> Any:
            if tool.metadata.is_async or inspect.iscoroutinefunction(tool.fn):
                return await tool.fn(**args)
            return await asyncio.to_thread(tool.fn, **args)

        try:
            return await asyncio.wait_for(_call(), timeout=timeout)
        except asyncio.TimeoutError as e:
            raise ToolTimeoutError(f"Tool `{tool.metadata.name}` exceeded {timeout}s") from e

    async def _run_ask_user(
        self,
        meta: ToolMetadata,
        call: ToolCallRequest,
        *,
        turn_number: int,
    ) -> ToolResult:
        if self._interrupt is None:
            return ToolResult(
                tool_use_id=call.id,
                name=meta.name,
                content=(
                    "No interrupt handler is attached; ask_user cannot proceed. "
                    "Make your best guess or stop."
                ),
                is_error=True,
            )
        args = call.arguments or {}
        event = InterruptEvent(
            interrupt_class=InterruptClass.CLARIFICATION,
            tool_name=meta.name,
            tool_args=args,
            question=args.get("question"),
            options=args.get("options"),
            allow_free_text=bool(args.get("allow_free_text", True)),
            urgency=args.get("urgency", "info"),
            session_id=self._session_id,
            turn_number=turn_number,
        )
        response = await self._interrupt.request(event)
        return ToolResult(
            tool_use_id=call.id,
            name=meta.name,
            content=(
                response.answer
                if response.answer is not None
                else (response.feedback or "(no answer)")
            ),
            is_error=not response.approved,
        )

    @staticmethod
    def _cache_key(name: str, args: dict[str, Any]) -> str:
        canonical = json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(f"{name}\x1f{canonical}".encode()).hexdigest()
