"""ShortTermMemory — the live message array for one session.

Owns the append-only conversation buffer and exposes windowing knobs
(`max_messages`, `max_tokens`). Fully serializable.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from agent_framework.core.message import (
    AnyMessage,
    ToolResultMessage,
    estimate_message_tokens,
)


class ShortTermMemory:
    def __init__(
        self,
        *,
        max_messages: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._messages: list[AnyMessage] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens

    # ---- API ----
    def append(self, message: AnyMessage) -> None:
        self._messages.append(message)
        self._maybe_trim()

    def extend(self, messages: Iterable[AnyMessage]) -> None:
        for m in messages:
            self.append(m)

    def clear(self) -> None:
        self._messages = []

    def replace_all(self, messages: Iterable[AnyMessage]) -> None:
        """Atomically replace the whole buffer (used by persistent compaction)."""
        self._messages = list(messages)

    def __iter__(self):
        return iter(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def snapshot(self) -> list[AnyMessage]:
        return list(self._messages)

    @property
    def last(self) -> AnyMessage | None:
        return self._messages[-1] if self._messages else None

    def to_dict_list(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self._messages]

    def to_json(self) -> str:
        return json.dumps(self.to_dict_list(), default=str, indent=2)

    # ---- trimming ----
    def _maybe_trim(self) -> None:
        if self.max_messages is not None and len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]
            self._heal_pairing()
        if self.max_tokens is not None:
            total = sum(estimate_message_tokens(m) for m in self._messages)
            while self._messages and total > self.max_tokens:
                first = self._messages.pop(0)
                total -= estimate_message_tokens(first)
            self._heal_pairing()

    def _heal_pairing(self) -> None:
        """Drop a leading ToolResultMessage if it's now orphaned (its
        corresponding AssistantMessage with tool_calls was trimmed away).
        """
        while self._messages and isinstance(self._messages[0], ToolResultMessage):
            self._messages.pop(0)
