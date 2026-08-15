"""PermissionResolver — outer function.

Combines:
  - the RuleEngine's verdict (ALLOW / ASK / DENY)
  - the session ASK_ONCE approval cache
  - dispatch through the InterruptManager when the verdict is ASK

Returns a `FinalDecision`. Behavior is fully determined by category +
rules + cache. There is no mode.
"""

from __future__ import annotations

from typing import Any

from agent_framework.interrupts.manager import InterruptManager
from agent_framework.interrupts.models import (
    InterruptClass,
    InterruptEvent,
)
from agent_framework.permissions.config import PermissionConfig
from agent_framework.permissions.models import (
    FinalDecision,
    PermissionCategory,
    Verdict,
    args_signature,
)
from agent_framework.permissions.rule_engine import RuleEngine
from agent_framework.tools.models import ToolMetadata
from agent_framework.tracing.trace import (
    EventType,
    ExecutionTrace,
    SpanStatus,
)


class PermissionResolver:
    """Per-agent permission orchestrator. One per agent."""

    def __init__(
        self,
        config: PermissionConfig,
        *,
        interrupt_manager: InterruptManager | None = None,
        trace: ExecutionTrace | None = None,
        session_id: str = "",
    ) -> None:
        self.config = config
        self._engine = RuleEngine(config)
        self._interrupt = interrupt_manager
        self._trace = trace
        self._session_id = session_id
        # ASK_ONCE cache: (tool_name, args_signature) → True once approved.
        self._ask_once_cache: dict[tuple[str, str], bool] = {}

    async def resolve(
        self,
        tool: ToolMetadata,
        args: dict[str, Any],
        *,
        turn_number: int = 0,
    ) -> FinalDecision:
        # 1. Inner: rule waterfall.
        decision = self._engine.evaluate(tool, args)

        # 2. Effective category (after overrides).
        category = self.config.category_overrides.get(tool.name, tool.permission)

        # 3. ASK_ONCE cache check — if already approved, short-circuit to ALLOW.
        if decision.verdict == Verdict.ASK and category == PermissionCategory.ASK_ONCE:
            key = (tool.name, args_signature(tool.name, args))
            if self._ask_once_cache.get(key):
                decision.verdict = Verdict.ALLOW
                decision.matched_rule = "ask_once_cache_hit"
                decision.reason += " (cached approval)"

        # 4. Record the rule engine verdict as a trace event.
        if self._trace is not None:
            self._trace.record_event(
                EventType.PERMISSION_VERDICT,
                name=tool.name,
                status=(
                    SpanStatus.OK
                    if decision.verdict == Verdict.ALLOW
                    else SpanStatus.DENIED
                    if decision.verdict == Verdict.DENY
                    else SpanStatus.OK
                ),
                payload={
                    "tool_name": tool.name,
                    "category": category.value,
                    "verdict": decision.verdict.value,
                    "reason": decision.reason,
                    "matched_rule": decision.matched_rule,
                    "escalated_from": (
                        decision.escalated_from.value if decision.escalated_from else None
                    ),
                },
            )

        # 5. If we still need a human, fire an interrupt.
        if decision.verdict == Verdict.ASK:
            if self._interrupt is None:
                # No renderer attached → safest default is deny with feedback.
                decision.verdict = Verdict.DENY
                decision.deny_feedback = (
                    "No interrupt handler is attached to this agent; "
                    "permission-required tools cannot run. "
                    "Choose a tool with category AUTO."
                )
                return decision
            event = InterruptEvent(
                interrupt_class=InterruptClass.PERMISSION,
                tool_name=tool.name,
                tool_args=args,
                permission_category=category,
                urgency=_urgency_for(category),
                session_id=self._session_id,
                turn_number=turn_number,
                metadata={"tool_description": tool.description, "source": tool.source},
            )
            response = await self._interrupt.request(event)
            if response.approved:
                if category == PermissionCategory.ASK_ONCE:
                    self._ask_once_cache[
                        (tool.name, args_signature(tool.name, args))
                    ] = True
                decision.verdict = Verdict.ALLOW
                decision.reason = "approved by user via interrupt"
            else:
                decision.verdict = Verdict.DENY
                decision.deny_feedback = (
                    response.feedback
                    or "The user denied this action. Try a different approach."
                )
                decision.reason = "denied by user via interrupt"

        return decision


def _urgency_for(category: PermissionCategory) -> str:
    if category == PermissionCategory.DENY:
        return "critical"
    if category in (PermissionCategory.ASK, PermissionCategory.ASK_ONCE):
        return "warning"
    return "info"
