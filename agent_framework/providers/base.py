"""BaseProviderAdapter — the interface every provider implements.

An adapter is the only place that knows the wire format of a specific
provider. The rest of the framework only deals in normalized `Message`,
`ToolCallRequest`, `ToolResult` types.
"""

from __future__ import annotations

import abc
import enum
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

from agent_framework.core.message import (
    AnyMessage,
    AssistantMessage,
    StopReason,
    TokenUsage,
)
from agent_framework.tools.models import ToolMetadata


@dataclass(frozen=True)
class ToolChoice:
    """Provider-neutral tool-choice constraint.

    mode:
      - "auto": the model decides (default provider behavior)
      - "any":  the model must call at least one tool
      - "tool": the model must call `tool_name`
      - "none": the model may not call tools
    parallel=False forces at most one tool call per response.
    """

    mode: Literal["auto", "any", "tool", "none"] = "auto"
    tool_name: str | None = None
    parallel: bool = True

    def __post_init__(self) -> None:
        if self.mode == "tool" and not self.tool_name:
            raise ValueError("ToolChoice(mode='tool') requires tool_name")


@dataclass(frozen=True)
class ToolResultClearing:
    """Ask the provider to clear old tool results server-side (if it can).

    Provider-neutral form of Anthropic's context editing: once the prompt
    exceeds `trigger_tokens`, the provider replaces the oldest tool results
    with a placeholder, keeping the newest `keep_results` and clearing only
    if at least `clear_at_least_tokens` are freed. Results of `exclude_tools`
    are never cleared. The provider edits its own copy, so the history sent
    stays unchanged — which is why this is safe mid-run even on models that
    bind reasoning to the prefix.
    """

    trigger_tokens: int
    keep_results: int
    clear_at_least_tokens: int
    exclude_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class AppliedEdit:
    """A server-side context edit the provider reports it applied."""

    type: str
    cleared_tool_uses: int = 0
    cleared_input_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "cleared_tool_uses": self.cleared_tool_uses,
            "cleared_input_tokens": self.cleared_input_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppliedEdit:
        return cls(
            type=str(data.get("type", "")),
            cleared_tool_uses=int(data.get("cleared_tool_uses") or 0),
            cleared_input_tokens=int(data.get("cleared_input_tokens") or 0),
        )


class ContextEditsRejectedError(Exception):
    """The provider rejected the requested server-side context edits.

    Raised before any output was produced, so the caller can resend the same
    request without them.
    """


@dataclass
class NormalizedResponse:
    """Single-shot non-streaming response from a provider, normalized."""

    assistant_message: AssistantMessage
    stop_reason: StopReason
    usage: TokenUsage
    raw: Any = None
    # Server-side context edits applied to this request's prompt.
    applied_edits: list[AppliedEdit] = field(default_factory=list)


class StreamChunkType(str, enum.Enum):
    TEXT_DELTA = "text_delta"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_COMPLETE = "tool_use_complete"
    REASONING_DELTA = "reasoning_delta"
    USAGE = "usage"
    STOP = "stop"
    ERROR = "error"


@dataclass
class StreamChunk:
    type: StreamChunkType
    payload: dict[str, Any] = field(default_factory=dict)


class BaseProviderAdapter(abc.ABC):
    """Each provider implements:

      - `call()` for non-streaming requests
      - `stream()` for SSE streaming
      - `denormalize_messages()` to convert framework messages → provider wire format
      - `tool_schema()` to render `ToolMetadata` → provider tool spec

    Optional override:
      - `structured()` — schema-constrained JSON generation using the
        provider's native mechanism. The default implementation is a
        prompt-based fallback that works with any provider.

    Notes on parameters:
      - `temperature=None` means "omit the parameter" (current-generation
        Anthropic models reject sampling parameters with a 400).
      - `enable_caching` lets adapters apply provider-side prompt caching
        (no-op for providers that cache automatically).
      - `context_edits` asks for server-side tool-result clearing. It is only
        ever passed to an adapter whose `supports_tool_result_clearing` is
        True, so adapters without the feature need not accept it. Such an
        adapter reports what was applied in `NormalizedResponse.applied_edits`
        (and in the `applied_edits` key of the USAGE/STOP stream chunks), and
        raises `ContextEditsRejectedError` when the provider refuses the edits.
    """

    name: str = "base"
    supports_tool_result_clearing: bool = False

    @abc.abstractmethod
    async def call(
        self,
        *,
        system: str,
        messages: list[AnyMessage],
        tools: list[ToolMetadata],
        model: str,
        max_tokens: int,
        temperature: float | None = None,
        tool_choice: ToolChoice | None = None,
        enable_caching: bool = True,
        extra: dict[str, Any] | None = None,
        context_edits: ToolResultClearing | None = None,
    ) -> NormalizedResponse: ...

    @abc.abstractmethod
    def stream(
        self,
        *,
        system: str,
        messages: list[AnyMessage],
        tools: list[ToolMetadata],
        model: str,
        max_tokens: int,
        temperature: float | None = None,
        tool_choice: ToolChoice | None = None,
        enable_caching: bool = True,
        extra: dict[str, Any] | None = None,
        context_edits: ToolResultClearing | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    @abc.abstractmethod
    def tool_schema(self, meta: ToolMetadata) -> dict[str, Any]: ...

    @abc.abstractmethod
    def denormalize_messages(self, messages: list[AnyMessage]) -> list[dict[str, Any]]: ...

    # ---------- structured output (overridable) ----------
    async def structured(
        self,
        *,
        system: str,
        messages: list[AnyMessage],
        schema: dict[str, Any],
        model: str,
        max_tokens: int,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Generate a single JSON object conforming to `schema`; returns raw text.

        Default implementation is prompt-based (works everywhere). Adapters
        override this with the provider's native structured-output mechanism.
        """
        schema_str = json.dumps(schema, indent=2)
        aug_system = (
            (system.rstrip() + "\n\n") if system else ""
        ) + (
            "Respond with a single valid JSON object that conforms to this schema:\n"
            f"```json\n{schema_str}\n```\n"
            "Output ONLY the JSON — no explanation, no markdown wrapper, no extra text."
        )
        resp = await self.call(
            system=aug_system,
            messages=messages,
            tools=[],
            model=model,
            max_tokens=max_tokens,
            temperature=None,
            extra=extra,
        )
        content = resp.assistant_message.content
        return content if isinstance(content, str) else ""
