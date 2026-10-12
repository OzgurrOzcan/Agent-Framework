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
from dataclasses import dataclass, field
from typing import Any

from agent_framework.core.message import (
    AnyMessage,
    ToolResultMessage,
    message_from_dict,
)

# Process-wide, so two histories never share an epoch once either changes —
# not even a history and its `copy()`.
_EPOCHS = itertools.count(1)

# Metadata key holding a message's sequence number in its session's log
# (see memory.sessions). A message without it has not been logged yet.
LOG_SEQ_KEY = "log_seq"


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
      - `summary`: long-term summary segments shown at the head of the view.
        A snapshot, taken when the conversation starts (and refreshed only
        when a trim rebuilds the prefix anyway), so the view stays stable
        even while other sessions add to the same namespace.

    Turns a trim dropped whose summary failed are not kept here: they are in
    the session's log, and the next summary covers everything in the log
    after the session's `summarized_upto` marker. (Older saved states carry
    a `pending` list; it is ignored.)
    """

    pinned: AnyMessage | None = None
    cleared_results: int = 0
    reasoning_cut: int = 0
    summary: list[str] = field(default_factory=list)

    def copy(self) -> ContextState:
        return ContextState(
            pinned=self.pinned,
            cleared_results=self.cleared_results,
            reasoning_cut=self.reasoning_cut,
            summary=list(self.summary),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pinned": self.pinned.to_dict() if self.pinned is not None else None,
            "cleared_results": self.cleared_results,
            "reasoning_cut": self.reasoning_cut,
            "summary": list(self.summary),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ContextState:
        data = data or {}
        pinned = data.get("pinned")
        return cls(
            pinned=message_from_dict(pinned) if isinstance(pinned, dict) else None,
            cleared_results=max(0, int(data.get("cleared_results") or 0)),
            reasoning_cut=max(0, int(data.get("reasoning_cut") or 0)),
            summary=[str(segment) for segment in data.get("summary") or []],
        )


class ShortTermMemory:
    def __init__(self) -> None:
        self._messages: list[AnyMessage] = []
        self._state = ContextState()
        self._epoch = next(_EPOCHS)
        # With a session log (set by the agent): messages that leave the
        # history before they were logged are held here until the next flush
        # logs them, so the log misses nothing. Without one nothing is held —
        # it would only grow.
        self.track_unlogged = False
        self._unlogged_dropped: list[AnyMessage] = []

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
        self._hold_unlogged(self._messages)
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
        self._hold_unlogged(dropped)
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

    def remove_tail(self, messages: list[AnyMessage]) -> bool:
        """Take *messages* back off the end of the history, if they are its end.

        For a run that left nothing but its own input behind (it was stopped
        before any reply): the conversation continues as if it never ran.
        """
        n = len(messages)
        if n == 0 or len(self._messages) < n:
            return False
        if any(a is not b for a, b in zip(self._messages[-n:], messages)):
            return False
        self._messages = self._messages[:-n]
        self._state.reasoning_cut = min(self._state.reasoning_cut, len(self._messages))
        self._touch()
        return True

    def _hold_unlogged(self, messages: Iterable[AnyMessage]) -> None:
        if self.track_unlogged:
            self._unlogged_dropped.extend(m for m in messages if LOG_SEQ_KEY not in m.metadata)

    def unlogged_dropped(self) -> list[AnyMessage]:
        """Messages a trim removed before they were logged, oldest first."""
        return list(self._unlogged_dropped)

    def forget_logged_dropped(self) -> None:
        """Stop holding dropped messages that have been logged since."""
        self._unlogged_dropped = [
            m for m in self._unlogged_dropped if LOG_SEQ_KEY not in m.metadata
        ]

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
        other.track_unlogged = self.track_unlogged
        other._unlogged_dropped = list(self._unlogged_dropped)
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
