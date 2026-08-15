"""InterruptManager — routes ASK + ask_user through the active renderer.

Both interrupt classes (permission, clarification) funnel through here.
Every interrupt is a trace event (raised, answered, denied, timeout).
"""

from __future__ import annotations

import asyncio

from agent_framework.interrupts.models import (
    InterruptEvent,
    InterruptResponse,
)
from agent_framework.interrupts.renderers.base import UIRenderer
from agent_framework.tracing.trace import (
    EventType,
    ExecutionTrace,
    SpanStatus,
)


class InterruptManager:
    def __init__(
        self,
        renderer: UIRenderer,
        *,
        trace: ExecutionTrace | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._renderer = renderer
        self._trace = trace
        self._timeout = timeout_seconds
        # Interrupts are inherently serial: two concurrent prompts would
        # interleave on the same terminal/UI.
        self._lock = asyncio.Lock()

    async def request(self, event: InterruptEvent) -> InterruptResponse:
        async with self._lock:
            return await self._request_inner(event)

    async def _request_inner(self, event: InterruptEvent) -> InterruptResponse:
        span_ctx = None
        if self._trace is not None:
            span_ctx = self._trace.start_span(
                EventType.INTERRUPT,
                name=f"{event.interrupt_class.value}:{event.tool_name}",
                payload={
                    "class": event.interrupt_class.value,
                    "tool_name": event.tool_name,
                    "tool_args": event.tool_args,
                    "permission_category": (
                        event.permission_category.value if event.permission_category else None
                    ),
                    "question": event.question,
                    "urgency": event.urgency,
                },
            )
        try:
            if span_ctx is not None:
                span_ctx.__enter__()
            try:
                if self._timeout is not None:
                    response = await asyncio.wait_for(
                        self._renderer.render(event), timeout=self._timeout
                    )
                else:
                    response = await self._renderer.render(event)
            except asyncio.TimeoutError:
                if span_ctx is not None:
                    span_ctx.set_status(SpanStatus.TIMEOUT)
                return InterruptResponse(
                    approved=False,
                    feedback=f"Interrupt timed out after {self._timeout}s — assume denial.",
                )
            except Exception as exc:
                if span_ctx is not None:
                    span_ctx.set_error(exc)
                return InterruptResponse(
                    approved=False,
                    feedback=f"Interrupt failed: {type(exc).__name__}: {exc}",
                )

            if span_ctx is not None:
                if not response.approved:
                    span_ctx.set_status(SpanStatus.DENIED)
                span_ctx.update(
                    approved=response.approved,
                    answer=response.answer,
                    selected_option=response.selected_option,
                )
            return response
        finally:
            if span_ctx is not None:
                span_ctx.__exit__(None, None, None)
