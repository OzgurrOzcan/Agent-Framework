"""The `context_filter` hook: your function decides what is sent.

    def last_three_turns(messages, info):
        ...
        return messages[-n:]

    Agent("claude-opus-5", context_filter=last_three_turns)

The filter runs before every LLM call on the rendered view — after the
framework's own trimming — and may be sync or async. It changes only what
is sent: the stored history is never touched. Messages are handed over as
shallow copies; build changed ones with `dataclasses.replace` rather than
mutating their contents.

Whatever it returns is repaired before it is sent, so a filter cannot
produce a request the provider rejects:

  - leading messages before the first user message are dropped;
  - tool results whose call is not in the preceding assistant message are
    dropped;
  - tool calls left without a result get an "omitted" result;
  - a view that does not end with the user's turn is not sent (the
    unfiltered view is, and the error is traced).

On models that bind reasoning to the conversation prefix (Claude Fable 5.1,
Opus 5.5, Sonnet 5.5) what was already sent must stay exactly as it was
sent. In the middle of a run an edit to it is rejected — the last sent view
plus the new messages goes out instead, and the rejection is traced as an
error; at a run's start an edit costs the reasoning from before the current
turn. A filter that is a pure function of the growing history (dropping or
keeping the same messages on every call) never edits; one that re-appends a
reminder at the end of every call does, because the previous copy moves.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agent_framework.core.message import (
    AnyMessage,
    AssistantMessage,
    Message,
    ToolCallRequest,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)

OMITTED_RESULT = "[result omitted from this context]"


@dataclass(frozen=True)
class ContextInfo:
    """What a context filter knows about the call it is shaping."""

    # 1-based number of this LLM call within the run.
    call: int
    # True for the run's first call — the one point where models that bind
    # reasoning to the prefix accept edits to earlier messages.
    run_start: bool
    # Estimated prompt size of the view handed to the filter.
    estimated_tokens: int
    max_context_tokens: int
    model: str
    reasoning_binds_to_prefix: bool
    # Whether the framework trimmed the history before this call.
    trimmed: bool


ContextFilter = Callable[
    [list[AnyMessage], ContextInfo],
    list[AnyMessage] | Awaitable[list[AnyMessage]],
]


class InvalidFilterOutput(ValueError):
    """The filter returned something that cannot be sent."""


@dataclass
class Repairs:
    """What the repair pass had to fix in a filter's output."""

    leading_dropped: int = 0
    orphan_results_dropped: int = 0
    missing_results_added: int = 0

    def any(self) -> bool:
        return bool(self.leading_dropped or self.orphan_results_dropped or self.missing_results_added)

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


def validate_output(value: Any) -> list[AnyMessage]:
    if not isinstance(value, list) or not all(isinstance(m, Message) for m in value):
        raise InvalidFilterOutput(
            f"a context filter must return a list of messages, got {type(value).__name__}"
        )
    return value


def repair(messages: list[AnyMessage]) -> tuple[list[AnyMessage], Repairs]:
    """Make *messages* a valid conversation: user first, every call answered."""
    repairs = Repairs()
    start = 0
    while start < len(messages) and not _is_user_turn(messages[start]):
        start += 1
    repairs.leading_dropped = start

    out: list[AnyMessage] = []
    expected: list[ToolCallRequest] = []  # calls of the preceding assistant message
    for message in messages[start:]:
        if isinstance(message, ToolResultMessage):
            ids = {call.id for call in expected}
            kept: list[ToolResult] = []
            for result in message.content:
                if result.tool_use_id in ids:
                    kept.append(result)
                    ids.discard(result.tool_use_id)  # one result per call
                else:
                    repairs.orphan_results_dropped += 1
            if expected:
                answered = {r.tool_use_id for r in kept}
                missing = [call for call in expected if call.id not in answered]
                kept.extend(_omitted(call) for call in missing)
                repairs.missing_results_added += len(missing)
                out.append(dataclasses.replace(message, content=kept))
                expected = []
            continue
        if expected:
            out.append(ToolResultMessage(content=[_omitted(call) for call in expected]))
            repairs.missing_results_added += len(expected)
            expected = []
        out.append(message)
        if isinstance(message, AssistantMessage) and message.tool_calls:
            expected = list(message.tool_calls)
    if expected:
        out.append(ToolResultMessage(content=[_omitted(call) for call in expected]))
        repairs.missing_results_added += len(expected)

    if not out:
        raise InvalidFilterOutput("the context filter left no user message to send")
    if isinstance(out[-1], AssistantMessage):
        raise InvalidFilterOutput(
            "the context filter's view ends with an assistant message; it must "
            "end with the user's turn (or tool results)"
        )
    return out, repairs


def fingerprint(message: AnyMessage) -> tuple[Any, ...]:
    """What the provider receives for *message* — timestamps and metadata aside."""
    if isinstance(message, ToolResultMessage):
        content: Any = tuple(
            (r.tool_use_id, r.name, r.as_text(), r.is_error) for r in message.content
        )
    else:
        content = _canonical(message.content)
    if isinstance(message, AssistantMessage):
        calls = tuple((c.id, c.name, _canonical(c.arguments)) for c in message.tool_calls)
        reasoning = tuple(_canonical(r.raw) for r in message.reasoning)
        return (type(message).__name__, content, calls, reasoning)
    return (type(message).__name__, content)


def same_messages(a: list[AnyMessage], b: list[AnyMessage]) -> bool:
    return len(a) == len(b) and all(fingerprint(x) == fingerprint(y) for x, y in zip(a, b))


def without_reasoning_before(messages: list[AnyMessage], cut: int) -> tuple[list[AnyMessage], int]:
    """*messages* with the reasoning of every assistant message before *cut* removed."""
    out: list[AnyMessage] = []
    stripped = 0
    for index, message in enumerate(messages):
        if index < cut and isinstance(message, AssistantMessage) and message.reasoning:
            out.append(dataclasses.replace(message, reasoning=[]))
            stripped += 1
        else:
            out.append(message)
    return out, stripped


def current_turn_start(messages: list[AnyMessage]) -> int:
    """Index of the first message of the last turn in *messages*."""
    start = 0
    for i, message in enumerate(messages):
        if _is_user_turn(message) and (i == 0 or not _is_user_turn(messages[i - 1])):
            start = i
    return start


@dataclass
class FilterOutcome:
    """What applying the filter did, for the trace."""

    changed: bool = False
    repairs: Repairs = field(default_factory=Repairs)
    error: str | None = None
    rejected: bool = False
    reasoning_stripped: int = 0


def _omitted(call: ToolCallRequest) -> ToolResult:
    return ToolResult(tool_use_id=call.id, name=call.name, content=OMITTED_RESULT)


def _is_user_turn(message: AnyMessage) -> bool:
    return isinstance(message, UserMessage) and not isinstance(message, ToolResultMessage)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)
