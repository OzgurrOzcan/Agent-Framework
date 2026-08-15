"""Permission core types.

Every tool declares **one** `PermissionCategory` at registration time.
That category is the single source of truth — the rule engine and
resolver compose category, rules, and the ASK_ONCE cache into a
`FinalDecision`.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass
from typing import Any


class PermissionCategory(str, enum.Enum):
    AUTO = "auto"
    ASK = "ask"
    ASK_ONCE = "ask_once"
    DENY = "deny"


class Verdict(str, enum.Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class FinalDecision:
    verdict: Verdict
    reason: str
    matched_rule: str | None = None
    escalated_from: PermissionCategory | None = None
    # When verdict==DENY/denied-via-ASK, we feed this back into the loop as
    # `ToolResult(is_error=True, content=…)`.
    deny_feedback: str | None = None


def args_signature(tool_name: str, args: dict[str, Any]) -> str:
    """Stable signature for the ASK_ONCE cache.

    Sort keys so {a:1, b:2} == {b:2, a:1}. Use SHA-256 to avoid pathological
    collisions across long sessions.
    """
    canonical = json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))
    h = hashlib.sha256(f"{tool_name}\x1f{canonical}".encode()).hexdigest()
    return h[:32]
