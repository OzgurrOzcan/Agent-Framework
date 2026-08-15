"""RuleEngine — pure, stateless inner policy function.

Walks the rule waterfall in strict priority order. First match wins.
Never touches the UI. Returns `Verdict.{ALLOW|ASK|DENY}` only.

Waterfall (deny-first):
  1. DENY rules        → DENY        (always wins)
  2. Argument danger   → escalate (AUTO→ASK, ASK→DENY)
  3. ALLOW rules       → ALLOW       (explicit whitelist)
  4. Tool category     → ALLOW | ASK | DENY by category
  5. Default fallback  → ASK         (never silently allow the unknown)
"""

from __future__ import annotations

import re
from typing import Any

from agent_framework.permissions.config import PermissionConfig
from agent_framework.permissions.models import (
    FinalDecision,
    PermissionCategory,
    Verdict,
)
from agent_framework.tools.models import ToolMetadata


class RuleEngine:
    def __init__(self, config: PermissionConfig) -> None:
        self.config = config
        self._danger_regexes = [re.compile(p, re.IGNORECASE) for p in config.danger_patterns]

    def evaluate(
        self,
        tool: ToolMetadata,
        args: dict[str, Any],
    ) -> FinalDecision:
        # 1. DENY rules — absolute.
        for rule in self.config.deny:
            if rule.matches(tool.name, args):
                return FinalDecision(
                    verdict=Verdict.DENY,
                    reason=f"deny rule matched: tool={rule.tool} pattern={rule.arg_pattern}",
                    matched_rule=f"deny:{rule.tool}",
                    deny_feedback=(
                        f"This action was denied by a deny rule "
                        f"(`{rule.tool}` matched {rule.arg_pattern!r}). "
                        f"Choose a different approach."
                    ),
                )

        # 2. Argument danger scan — escalate one level toward safety.
        category = self.config.category_overrides.get(tool.name, tool.permission)
        escalated_from: PermissionCategory | None = None
        if self._danger_regexes:
            haystack = " ".join(f"{k}={v}" for k, v in args.items())
            for rx in self._danger_regexes:
                if rx.search(haystack):
                    escalated_from = category
                    category = _escalate(category)
                    break

        # 3. Explicit ALLOW rules.
        if escalated_from is None:
            for rule in self.config.allow:
                if rule.matches(tool.name, args):
                    return FinalDecision(
                        verdict=Verdict.ALLOW,
                        reason=f"allow rule matched: tool={rule.tool}",
                        matched_rule=f"allow:{rule.tool}",
                    )

        # 4. Category resolution (after possible escalation).
        verdict = _category_to_verdict(category)
        reason = (
            f"category={category.value}"
            if escalated_from is None
            else f"escalated_from={escalated_from.value}→{category.value} (danger pattern)"
        )

        # 5. Default fallback — never silently allow unknowns. Categories above
        # always resolve, but if a malformed tool ever reaches here, escalate.
        decision = FinalDecision(
            verdict=verdict,
            reason=reason,
            escalated_from=escalated_from,
        )
        if verdict == Verdict.DENY:
            decision.deny_feedback = (
                f"This action was denied (category={category.value})."
                + (
                    f" Escalated from {escalated_from.value} after a dangerous argument pattern."
                    if escalated_from
                    else ""
                )
            )
        return decision


def _escalate(cat: PermissionCategory) -> PermissionCategory:
    if cat == PermissionCategory.AUTO:
        return PermissionCategory.ASK
    if cat in (PermissionCategory.ASK, PermissionCategory.ASK_ONCE):
        return PermissionCategory.DENY
    return cat  # DENY stays DENY


def _category_to_verdict(cat: PermissionCategory) -> Verdict:
    if cat == PermissionCategory.AUTO:
        return Verdict.ALLOW
    if cat in (PermissionCategory.ASK, PermissionCategory.ASK_ONCE):
        return Verdict.ASK
    if cat == PermissionCategory.DENY:
        return Verdict.DENY
    return Verdict.ASK  # unknown → ASK
