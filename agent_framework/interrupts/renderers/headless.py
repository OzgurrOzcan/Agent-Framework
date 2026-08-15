"""HeadlessRenderer — for unattended / CI runs.

Constructed with an explicit policy so the safety/ergonomic tradeoff is
visible in code:

  HeadlessRenderer(policy="auto_deny")              # safest default
  HeadlessRenderer(policy="auto_approve")           # dangerous; sandbox only
  HeadlessRenderer(per_tool={"web_search": True})   # surgical
"""

from __future__ import annotations

from typing import Literal

from agent_framework.interrupts.models import (
    InterruptClass,
    InterruptEvent,
    InterruptResponse,
)
from agent_framework.interrupts.renderers.base import UIRenderer


Policy = Literal["auto_deny", "auto_approve"]


class HeadlessRenderer(UIRenderer):
    def __init__(
        self,
        policy: Policy = "auto_deny",
        per_tool: dict[str, bool] | None = None,
        clarification_default: str = "",
    ) -> None:
        self._policy = policy
        self._per_tool = dict(per_tool or {})
        self._clarification_default = clarification_default

    async def render(self, event: InterruptEvent) -> InterruptResponse:
        if event.interrupt_class == InterruptClass.CLARIFICATION:
            # Models trying to ask the user in headless mode get the configured
            # default, plus structured feedback so they can adapt.
            return InterruptResponse(
                approved=True,
                answer=self._clarification_default or "(no user available; proceed with best guess)",
            )
        # Permission decision.
        if event.tool_name in self._per_tool:
            ok = self._per_tool[event.tool_name]
        else:
            ok = self._policy == "auto_approve"
        if ok:
            return InterruptResponse(approved=True)
        return InterruptResponse(
            approved=False,
            feedback=(
                f"Running headless ({self._policy}); the tool `{event.tool_name}` "
                f"requires interactive approval. Choose a non-interactive approach."
            ),
        )
