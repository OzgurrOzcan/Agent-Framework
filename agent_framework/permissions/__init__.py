"""Permission engine — category-based, mode-free."""

from agent_framework.permissions.models import (
    PermissionCategory,
    Verdict,
    FinalDecision,
)
from agent_framework.permissions.config import (
    PermissionConfig,
    AllowRule,
    DenyRule,
)
from agent_framework.permissions.rule_engine import RuleEngine
from agent_framework.permissions.resolver import PermissionResolver

__all__ = [
    "PermissionCategory",
    "Verdict",
    "FinalDecision",
    "PermissionConfig",
    "AllowRule",
    "DenyRule",
    "RuleEngine",
    "PermissionResolver",
]
