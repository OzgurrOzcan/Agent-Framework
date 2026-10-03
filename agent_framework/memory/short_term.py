"""ShortTermMemory — the live message array of one conversation.

It holds the conversation history in full, plus a `ContextState`: the small
record of how that history is rendered for the model (which old tool results
are cleared, where reasoning is cut, which first message is pinned). The
history itself is never rewritten. It is append-only between trims: only
`drop_head`, `clear` and `replace_all` remove messages, and each of them —
like replacing the state — moves `epoch`, so anything derived from the old
history (the context manager's token anchor) can tell it is stale.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from agent_framework.core.message import (
    AnyMessage,
    ToolResultMessage,
    message_from_dict,
)

# Process-wide, so two histories never share an epoch once either changes —
# not even a history and its `copy()`.
_EPOCHS = itertools.count(1)


@dataclass
class ContextState:
    """How the stored history is rendered into what the model sees.

    Counts rather than message references, so the state stays valid while the
    history grows and serializes as plain data:

      - `pinned`: the conversation's first user message, kept at the head of
        the view after the turn it opened was dropped.
      - `cleared_results`: the oldest N tool results (counted in history
        order, every result included) are in the cleared zone; the clearable
        ones among them are rendered as a one-line note.
      - `reasoning_cut`: reasoning in the first N history messages is left
        out of the view.
    """

    pinned: AnyMessage | None = None
    cleared_results: int = 0
    reasoning_cut: int = 0

    def copy(self) -> ContextState:
        return ContextState(
            pinned=self.pinned,
            cleared_results=self.cleared_results,
            reasoning_cut=self.reasoning_cut,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pinned": self.pinned.to_dict() if self.pinned is not None else None,
            "cleared_results": self.cleared_results,
            "reasoning_cut": self.reasoning_cut,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ContextState:
        data = data or {}
        pinned = data.get("pinned")
        return cls(
            pinned=message_from_dict(pinned) if isinstance(pinned, dict) else None,
            cleared_results=max(0, int(data.get("cleared_results") or 0)),
            reasoning_cut=max(0, int(data.get("reasoning_cut") or 0)),
        )


class ShortTermMemory:
    def __init__(self) -> None:
        self._messages: list[AnyMessage] = []
        self._state = ContextState()
        self._epoch = next(_EPOCHS)

    # ---- the history ----
    def append(self, message: AnyMessage) -> None:
        self._messages.append(message)

    def extend(self, messages: Iterable[AnyMessage]) -> None:
        self._messages.extend(messages)

    def clear(self) -> None:
        """Forget the conversation — messages and context state alike."""
        self.replace_all([])

    def replace_all(
        self, messages: Iterable[AnyMessage], *, state: ContextState | None = None
    ) -> None:
        """Replace the whole history; the state is reset unless one is given.

        A state describes one particular history, so keeping the old one for a
        new history would clear or cut the wrong messages.
        """
        self._messages = list(messages)
        self._state = state.copy() if state is not None else ContextState()
        self._touch()

    def drop_head(self, count: int) -> list[AnyMessage]:
        """Remove the oldest *count* messages and return them.

        The state's counters are shifted so they keep pointing at the same
        messages; the caller is responsible for cutting at a turn boundary.
        """
        if count <= 0:
            return []
        dropped = self._messages[:count]
        self._messages = self._messages[count:]
        results = sum(len(m.content) for m in dropped if isinstance(m, ToolResultMessage))
        self._state.cleared_results = max(0, self._state.cleared_results - results)
        self._state.reasoning_cut = max(0, self._state.reasoning_cut - len(dropped))
        self._touch()
        return dropped

    # ---- the context state ----
    @property
    def state(self) -> ContextState:
        return self._state

    @state.setter
    def state(self, value: ContextState) -> None:
        self._state = value
        self._touch()

    @property
    def epoch(self) -> int:
        """Changes whenever messages are removed or the state is replaced."""
        return self._epoch

    def _touch(self) -> None:
        self._epoch = next(_EPOCHS)

    def copy(self) -> ShortTermMemory:
        """An independent copy (messages are shared, they are never mutated)."""
        other = ShortTermMemory()
        other._messages = list(self._messages)
        other._state = self._state.copy()
        # Same content, so the same epoch: a token anchor stays valid for both
        # until either of them changes.
        other._epoch = self._epoch
        return other

    # ---- read access ----
    def __iter__(self) -> Iterator[AnyMessage]:
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
