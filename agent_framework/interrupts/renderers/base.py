"""UIRenderer interface — pluggable interrupt delivery.

The interrupt manager never renders anything itself; it always dispatches
through a `UIRenderer`. Permission ASKs and `ask_user` clarifications
both flow through the same interface.
"""

from __future__ import annotations

import abc

from agent_framework.interrupts.models import InterruptEvent, InterruptResponse


class UIRenderer(abc.ABC):
    @abc.abstractmethod
    async def render(self, event: InterruptEvent) -> InterruptResponse: ...
