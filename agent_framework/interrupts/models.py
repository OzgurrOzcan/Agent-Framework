"""Interrupt event/response data shapes."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_framework.permissions.models import PermissionCategory


class InterruptClass(str, enum.Enum):
    PERMISSION = "permission"
    CLARIFICATION = "clarification"


@dataclass
class InterruptEvent:
    interrupt_class: InterruptClass
    tool_name: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    permission_category: PermissionCategory | None = None
    question: str | None = None
    options: list[str] | None = None
    allow_free_text: bool = True
    urgency: Literal["info", "warning", "critical"] = "info"
    session_id: str = ""
    turn_number: int = 0
    # Free-form metadata for renderers (e.g. tool description, source).
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterruptResponse:
    approved: bool = False
    answer: str | None = None
    selected_option: str | None = None
    # Optional structured feedback the model receives when denied.
    feedback: str | None = None
