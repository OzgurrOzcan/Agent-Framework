"""PermissionConfig — declarative rules attached at agent construction.

Rules let you override categories without editing tool code. They are
plain data, evaluated by the `RuleEngine` in a strict deny-first
waterfall.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _glob_or_eq(pattern: str, value: str) -> bool:
    if pattern == value:
        return True
    if any(ch in pattern for ch in "*?[]"):
        return fnmatch.fnmatchcase(value, pattern)
    return False


@dataclass
class AllowRule:
    """If the tool name (and optional argument regex) match, ALLOW immediately."""

    tool: str
    arg_pattern: str | None = None
    description: str = ""

    def matches(self, tool_name: str, args: dict[str, Any]) -> bool:
        if not _glob_or_eq(self.tool, tool_name):
            return False
        if self.arg_pattern is None:
            return True
        haystack = " ".join(f"{k}={v}" for k, v in args.items())
        return re.search(self.arg_pattern, haystack) is not None


@dataclass
class DenyRule:
    """If the tool name (and optional argument regex) match, DENY immediately."""

    tool: str
    arg_pattern: str | None = None
    description: str = ""

    def matches(self, tool_name: str, args: dict[str, Any]) -> bool:
        if not _glob_or_eq(self.tool, tool_name):
            return False
        if self.arg_pattern is None:
            return True
        haystack = " ".join(f"{k}={v}" for k, v in args.items())
        return re.search(self.arg_pattern, haystack) is not None


@dataclass
class PermissionConfig:
    """All permission configuration for one agent. Plain data."""

    deny: list[DenyRule] = field(default_factory=list)
    allow: list[AllowRule] = field(default_factory=list)
    danger_patterns: list[str] = field(default_factory=list)
    on_ask: Any = None  # `UIRenderer` — injected at construction
    # Per-tool category overrides (override the tool's own declared category).
    category_overrides: dict[str, "Any"] = field(default_factory=dict)

    def is_stricter_than(self, parent: "PermissionConfig") -> bool:
        """Sub-agent config must be a subset of parent — cannot widen permissions.

        Rules:
          1. Every deny rule the parent has must also be present in this config
             (cannot remove deny rules).
          2. This config must not introduce allow rules that the parent does not
             have (cannot add whitelisted paths not present in the parent).

        Comparison is value-based (dataclass equality). Glob patterns are treated
        as opaque values — structural overlap is not checked here.
        """
        parent_deny_set = set(_rule_key(r) for r in parent.deny)
        self_deny_set = set(_rule_key(r) for r in self.deny)
        if not parent_deny_set.issubset(self_deny_set):
            return False

        parent_allow_set = set(_rule_key(r) for r in parent.allow)
        self_allow_set = set(_rule_key(r) for r in self.allow)
        if not self_allow_set.issubset(parent_allow_set):
            return False

        return True


def _rule_key(rule: object) -> tuple:
    """Stable hashable key for a DenyRule or AllowRule."""
    return (type(rule).__name__, getattr(rule, "tool", ""), getattr(rule, "arg_pattern", None))
