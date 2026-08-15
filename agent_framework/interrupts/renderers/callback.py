"""CallbackRenderer — accept a user-supplied async callback.

This is the SDK / UI embedding hook. Pass `async def on_interrupt(event)
-> response`, and the loop will deliver every permission ASK and
`ask_user` clarification to it.
"""

from __future__ import annotations

import inspect
from typing import Awaitable, Callable, Union

from agent_framework.interrupts.models import InterruptEvent, InterruptResponse
from agent_framework.interrupts.renderers.base import UIRenderer


Handler = Callable[[InterruptEvent], Union[InterruptResponse, Awaitable[InterruptResponse]]]


class CallbackRenderer(UIRenderer):
    def __init__(self, handler: Handler) -> None:
        if handler is None:
            raise ValueError("CallbackRenderer requires a handler")
        self._handler = handler

    async def render(self, event: InterruptEvent) -> InterruptResponse:
        result = self._handler(event)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, InterruptResponse):
            raise TypeError(
                f"CallbackRenderer handler must return InterruptResponse, got {type(result).__name__}"
            )
        return result
