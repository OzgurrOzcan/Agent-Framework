"""Unified message types — the only durable state of the loop.

Every provider's response is normalized into these types. Tool results are
preserved with their `tool_use_id` so the provider-level invariant (every
`tool_use` is followed by exactly one matching `tool_result`) holds across
success, failure, denial, timeout, and cancellation.
"""

from __future__ import annotations

import enum
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Union


class StopReason(str, enum.Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    # Server-side tool loop paused mid-turn; re-send to continue (Anthropic).
    PAUSE_TURN = "pause_turn"
    # Safety classifiers declined the request (HTTP 200, not an error).
    REFUSAL = "refusal"
    # The model hit the context-window limit (distinct from max_tokens).
    CONTEXT_EXCEEDED = "model_context_window_exceeded"
    ERROR = "error"


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
        )


@dataclass
class ReasoningContent:
    """Opaque reasoning/thinking block — preserved verbatim and round-tripped.

    Anthropic rejects tool-use turns where the prior `thinking` block has been
    stripped, so we keep the raw provider payload here and re-attach it on the
    next request.
    """

    provider: Literal["openai", "anthropic"]
    raw: dict[str, Any]
    text: str | None = None


@dataclass
class ToolCallRequest:
    """Normalized model tool call (OpenAI tool_calls / Anthropic tool_use)."""

    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def new_id(cls) -> str:
        return f"call_{uuid.uuid4().hex[:24]}"


@dataclass
class ToolResult:
    """Normalized tool execution result.

    Every `tool_use` must produce exactly one `ToolResult` with matching id —
    including failures, denials, timeouts, and cancellations. `is_error=True`
    is feedback for the model, not a crash.
    """

    tool_use_id: str
    name: str
    content: Any
    is_error: bool = False
    duration_ms: int | None = None

    def as_text(self) -> str:
        return content_as_text(self.content)


def content_as_text(content: Any) -> str:
    """The text a provider receives for a tool result's *content*.

    Strings pass through; anything else is JSON-encoded (falling back to
    `str()`), exactly as the adapters serialize it.
    """
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, default=str)
    except Exception:
        return str(content)


@dataclass
class Message:
    """Base message type. Concrete kinds: User / Assistant / System / ToolResult-batch."""

    role: Literal["user", "assistant", "system", "tool"]
    content: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Optional provider-side metadata (e.g. cache_control hints).
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "role": self.role,
            "content": _serialize_content(self.content),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class UserMessage(Message):
    role: Literal["user"] = "user"
    content: str | list[Any] = ""


@dataclass
class AssistantMessage(Message):
    role: Literal["assistant"] = "assistant"
    content: str | list[Any] = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    reasoning: list[ReasoningContent] = field(default_factory=list)
    stop_reason: StopReason | None = None
    usage: TokenUsage | None = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["tool_calls"] = [_serialize_content(tc) for tc in self.tool_calls]
        d["reasoning"] = [_serialize_content(r) for r in self.reasoning]
        d["stop_reason"] = self.stop_reason.value if self.stop_reason else None
        if self.usage is not None:
            d["usage"] = {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cache_read_input_tokens": self.usage.cache_read_input_tokens,
                "cache_creation_input_tokens": self.usage.cache_creation_input_tokens,
            }
        return d


@dataclass
class SystemMessage(Message):
    role: Literal["system"] = "system"
    content: str = ""


@dataclass
class ToolResultMessage(Message):
    """User-role message that carries one or more `ToolResult`s.

    Providers expect tool results bundled into a single user turn; we model
    that explicitly so the invariant is auditable.
    """

    role: Literal["user"] = "user"
    content: list[ToolResult] = field(default_factory=list)


AnyMessage = Union[UserMessage, AssistantMessage, SystemMessage, ToolResultMessage]


def _serialize_content(content: Any) -> Any:
    if isinstance(content, (str, int, float, bool, type(None))):
        return content
    if isinstance(content, list):
        return [_serialize_content(x) for x in content]
    if isinstance(content, dict):
        return {k: _serialize_content(v) for k, v in content.items()}
    if isinstance(content, ToolResult):
        return {
            "tool_use_id": content.tool_use_id,
            "name": content.name,
            "content": content.content,
            "is_error": content.is_error,
            "duration_ms": content.duration_ms,
        }
    if isinstance(content, ToolCallRequest):
        return {"id": content.id, "name": content.name, "arguments": content.arguments}
    if isinstance(content, ReasoningContent):
        return {"provider": content.provider, "raw": content.raw, "text": content.text}
    return str(content)


def message_from_dict(raw: dict[str, Any]) -> AnyMessage:
    """Reconstruct a message from its `to_dict()` form (full fidelity).

    Falls back to role-based inference for dicts produced before the
    `type` field existed.
    """
    import warnings
    from datetime import datetime as _dt

    mtype = raw.get("type")
    role = raw.get("role", "user")
    content = raw.get("content")
    metadata = dict(raw.get("metadata") or {})
    ts_raw = raw.get("timestamp")
    try:
        timestamp = _dt.fromisoformat(ts_raw) if ts_raw else _dt.now(timezone.utc)
    except (TypeError, ValueError):
        timestamp = _dt.now(timezone.utc)

    def _tool_results(items: list[Any]) -> list[ToolResult]:
        out = []
        for it in items:
            if isinstance(it, dict) and "tool_use_id" in it:
                out.append(
                    ToolResult(
                        tool_use_id=it["tool_use_id"],
                        name=it.get("name", ""),
                        content=it.get("content"),
                        is_error=bool(it.get("is_error", False)),
                        duration_ms=it.get("duration_ms"),
                    )
                )
        return out

    if mtype is None:
        warnings.warn(
            "message_from_dict: dict has no 'type' field; inferring from role/content."
        )
        if role == "user" and isinstance(content, list) and content and all(
            isinstance(c, dict) and "tool_use_id" in c for c in content
        ):
            mtype = "ToolResultMessage"
        else:
            mtype = {
                "assistant": "AssistantMessage",
                "system": "SystemMessage",
            }.get(role, "UserMessage")

    if mtype == "AssistantMessage":
        tool_calls = [
            ToolCallRequest(
                id=tc["id"], name=tc["name"], arguments=dict(tc.get("arguments") or {})
            )
            for tc in raw.get("tool_calls") or []
            if isinstance(tc, dict)
        ]
        reasoning = [
            ReasoningContent(
                provider=r.get("provider", "anthropic"),
                raw=dict(r.get("raw") or {}),
                text=r.get("text"),
            )
            for r in raw.get("reasoning") or []
            if isinstance(r, dict)
        ]
        sr_raw = raw.get("stop_reason")
        try:
            stop_reason = StopReason(sr_raw) if sr_raw else None
        except ValueError:
            stop_reason = None
        usage_raw = raw.get("usage")
        usage = TokenUsage(**usage_raw) if isinstance(usage_raw, dict) else None
        return AssistantMessage(
            content=content if content is not None else "",
            tool_calls=tool_calls,
            reasoning=reasoning,
            stop_reason=stop_reason,
            usage=usage,
            timestamp=timestamp,
            metadata=metadata,
        )
    if mtype == "ToolResultMessage":
        return ToolResultMessage(
            content=_tool_results(content or []),
            timestamp=timestamp,
            metadata=metadata,
        )
    if mtype == "SystemMessage":
        return SystemMessage(
            content=content or "", timestamp=timestamp, metadata=metadata
        )
    return UserMessage(
        content=content if content is not None else "",
        timestamp=timestamp,
        metadata=metadata,
    )


def estimate_tokens(text: str) -> int:
    """Cheap token estimator — ~4 chars/token. Used for budget calculation only."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(msg: Message) -> int:
    """Estimate tokens for a single normalized message (for budgeting)."""
    total = 4  # role/structure overhead
    content = msg.content
    if isinstance(content, str):
        total += estimate_tokens(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, ToolResult):
                total += estimate_tokens(item.as_text()) + 8
            elif isinstance(item, str):
                total += estimate_tokens(item)
            else:
                total += estimate_tokens(json.dumps(item, default=str))
    if isinstance(msg, AssistantMessage):
        for tc in msg.tool_calls:
            total += 8 + estimate_tokens(tc.name) + estimate_tokens(
                json.dumps(tc.arguments, default=str)
            )
        for r in msg.reasoning:
            if r.text:
                total += estimate_tokens(r.text)
    return total
