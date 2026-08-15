"""LoopPolicy — execution shape configuration.

The agent loop has three shapes (call_once / call_with_tools / run); all
share one engine. The policy is the small set of knobs that selects shape:

  - max_turns: hard ceiling on iterations (any shape)
  - execute_tools: if False, tool_use blocks are returned as data
  - max_tool_rounds: cap on tool execution rounds (None = unlimited)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_framework.providers.base import ToolChoice


@dataclass(frozen=True)
class LoopPolicy:
    max_turns: int = 10
    execute_tools: bool = True
    max_tool_rounds: int | None = None
    # Optional tool-choice constraint. Forced modes ("tool"/"any") apply to
    # the FIRST LLM call only; later turns fall back to auto.
    tool_choice: "ToolChoice | None" = None

    @classmethod
    def once(cls, *, tool_choice: "ToolChoice | None" = None) -> "LoopPolicy":
        """Single LLM call, no tool execution. For pure transforms."""
        return cls(max_turns=1, execute_tools=False, max_tool_rounds=0, tool_choice=tool_choice)

    @classmethod
    def single_round(cls, *, tool_choice: "ToolChoice | None" = None) -> "LoopPolicy":
        """One tool round + a synthesis call. Bounded, predictable."""
        return cls(max_turns=2, execute_tools=True, max_tool_rounds=1, tool_choice=tool_choice)

    @classmethod
    def full(
        cls, max_turns: int = 10, *, tool_choice: "ToolChoice | None" = None
    ) -> "LoopPolicy":
        return cls(
            max_turns=max_turns, execute_tools=True, max_tool_rounds=None,
            tool_choice=tool_choice,
        )


class LoopGuardError(Exception):
    """Raised by the loop when a guardrail (max_turns/stall) fires."""

    def __init__(self, reason: str, *, turn: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.turn = turn


class StallDetectedError(LoopGuardError):
    """Raised when the same tool+args are repeated `stall_threshold` times."""


class BudgetExceededError(LoopGuardError):
    """Raised when cumulative cost exceeds `max_budget_usd`."""

    def __init__(self, reason: str, *, turn: int, spent: float, limit: float) -> None:
        super().__init__(reason, turn=turn)
        self.spent = spent
        self.limit = limit
