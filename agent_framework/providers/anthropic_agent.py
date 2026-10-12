"""AnthropicAdapter — wire format normalization for the Anthropic API.

Key invariants:
  - `thinking` blocks are preserved opaquely in `AssistantMessage.reasoning`
    and re-sent on the next request (Anthropic rejects tool-use turns
    where the prior thinking has been stripped).
  - tool_use blocks become `ToolCallRequest`s.
  - tool_result blocks are accepted on input via `ToolResultMessage`.
  - Sampling parameters are only sent when explicitly set — current-generation
    Anthropic models reject `temperature`/`top_p`/`top_k` with a 400.
  - Prompt caching: one breakpoint on the system prompt and one incremental
    breakpoint on the last stable conversation message (preamble messages —
    those with `metadata["system"]` set — stay after the breakpoint).
  - Server-side tool-result clearing (context editing) goes through
    `client.beta.messages.*` with the context-management beta — only when
    `context_edits` is passed; every other request uses the regular API.
"""

from __future__ import annotations

import contextlib
from typing import Any, AsyncIterator

from agent_framework.core.message import (
    AnyMessage,
    AssistantMessage,
    Message,
    ReasoningContent,
    StopReason,
    SystemMessage,
    TokenUsage,
    ToolCallRequest,
    ToolResultMessage,
    UserMessage,
)
from agent_framework.providers.base import (
    AppliedEdit,
    BaseProviderAdapter,
    ContextEditsRejectedError,
    NormalizedResponse,
    StreamChunk,
    StreamChunkType,
    ToolChoice,
    ToolResultClearing,
)
from agent_framework.providers.streaming import StreamingToolAccumulator
from agent_framework.tools.models import ToolMetadata

CONTEXT_MANAGEMENT_BETA = "context-management-2025-06-27"
CLEAR_TOOL_USES_EDIT = "clear_tool_uses_20250919"


class AnthropicAdapter(BaseProviderAdapter):
    name = "anthropic"
    supports_tool_result_clearing = True

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        client_kwargs: dict[str, Any] | None = None,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError("The `anthropic` package is required.") from e
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if client_kwargs:
            kwargs.update(client_kwargs)
        self._client = AsyncAnthropic(**kwargs)

    # ---------- tool schema ----------
    def tool_schema(self, meta: ToolMetadata) -> dict[str, Any]:
        return {
            "name": meta.name,
            "description": meta.description,
            "input_schema": meta.parameters_schema,
        }

    # ---------- request denormalization ----------
    def denormalize_messages(self, messages: list[AnyMessage]) -> list[dict[str, Any]]:
        out, _ = self._denormalize_with_origins(messages)
        return out

    def _denormalize_with_origins(
        self, messages: list[AnyMessage]
    ) -> tuple[list[dict[str, Any]], list[AnyMessage]]:
        out: list[dict[str, Any]] = []
        origins: list[AnyMessage] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # System prompts are passed separately on the Anthropic API.
                continue
            if isinstance(msg, ToolResultMessage):
                blocks = []
                for tr in msg.content:
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tr.tool_use_id,
                            "content": tr.as_text(),
                            "is_error": tr.is_error,
                        }
                    )
                out.append({"role": "user", "content": blocks})
                origins.append(msg)
            elif isinstance(msg, UserMessage):
                out.append({"role": "user", "content": _user_content_blocks(msg.content)})
                origins.append(msg)
            elif isinstance(msg, AssistantMessage):
                blocks = []
                for r in msg.reasoning:
                    if r.provider == "anthropic":
                        blocks.append(r.raw)
                text = msg.content if isinstance(msg.content, str) else ""
                if text:
                    blocks.append({"type": "text", "text": text})
                for tc in msg.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                if not blocks:
                    blocks = [{"type": "text", "text": ""}]
                out.append({"role": "assistant", "content": blocks})
                origins.append(msg)
        return out, origins

    def _build_request(
        self,
        *,
        system: str,
        messages: list[AnyMessage],
        tools: list[ToolMetadata],
        model: str,
        max_tokens: int,
        temperature: float | None,
        tool_choice: ToolChoice | None,
        enable_caching: bool,
        extra: dict[str, Any] | None,
        context_edits: ToolResultClearing | None = None,
    ) -> dict[str, Any]:
        wire, origins = self._denormalize_with_origins(messages)
        if enable_caching:
            _apply_incremental_cache(wire, origins)
        req: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": wire,
        }
        if temperature is not None:
            req["temperature"] = temperature
        blocks = _system_blocks(system, cache=enable_caching)
        if blocks:
            req["system"] = blocks
        if tools:
            req["tools"] = [self.tool_schema(t) for t in tools]
        if tool_choice is not None:
            req["tool_choice"] = _tool_choice_param(tool_choice)
        if extra:
            req.update(extra)
        if context_edits is not None:
            req["context_management"] = {"edits": [_clear_tool_uses_edit(context_edits)]}
            betas = list(req.get("betas") or [])
            if CONTEXT_MANAGEMENT_BETA not in betas:
                betas.append(CONTEXT_MANAGEMENT_BETA)
            req["betas"] = betas
        return req

    def _messages_api(self, context_edits: ToolResultClearing | None) -> Any:
        """The beta API for context editing; the regular one otherwise."""
        return self._client.beta.messages if context_edits is not None else self._client.messages

    # ---------- non-streaming call ----------
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
    ) -> NormalizedResponse:
        req = self._build_request(
            system=system, messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            tool_choice=tool_choice, enable_caching=enable_caching, extra=extra,
            context_edits=context_edits,
        )
        try:
            raw = await self._messages_api(context_edits).create(**req)
        except Exception as exc:
            if context_edits is not None and _rejects_context_edits(exc):
                raise ContextEditsRejectedError(str(exc)) from exc
            raise
        return self._normalize(raw)

    # ---------- structured output (native) ----------
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
        req: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": self.denormalize_messages(messages),
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        blocks = _system_blocks(system, cache=False)
        if blocks:
            req["system"] = blocks
        if extra:
            req.update(extra)
        raw = await self._client.messages.create(**req)
        normalized = self._normalize(raw)
        content = normalized.assistant_message.content
        return content if isinstance(content, str) else ""

    # ---------- streaming ----------
    async def stream(
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
    ) -> AsyncIterator[StreamChunk]:
        req = self._build_request(
            system=system, messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            tool_choice=tool_choice, enable_caching=enable_caching, extra=extra,
            context_edits=context_edits,
        )

        acc = StreamingToolAccumulator()
        text_parts: list[str] = []
        reasoning_blocks: list[ReasoningContent] = []
        stop_reason: StopReason = StopReason.END_TURN
        usage = TokenUsage()
        applied_edits: list[AppliedEdit] = []

        async with contextlib.AsyncExitStack() as stack:
            # The request is sent when the stream is entered, so a rejection
            # of the edits surfaces here — before any chunk was produced.
            try:
                stream = await stack.enter_async_context(
                    self._messages_api(context_edits).stream(**req)
                )
            except Exception as exc:
                if context_edits is not None and _rejects_context_edits(exc):
                    raise ContextEditsRejectedError(str(exc)) from exc
                raise
            current_block_index: dict[int, str] = {}  # idx → "tool_use:<id>" / "text" / "thinking"
            current_tool_ids: dict[int, str] = {}
            async for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        acc.start_tool(block.id, block.name)
                        current_block_index[event.index] = f"tool_use:{block.id}"
                        current_tool_ids[event.index] = block.id
                        yield StreamChunk(
                            StreamChunkType.TOOL_USE_START,
                            payload={"id": block.id, "name": block.name},
                        )
                    elif block.type == "text":
                        current_block_index[event.index] = "text"
                    elif block.type == "thinking":
                        current_block_index[event.index] = "thinking"
                elif etype == "content_block_delta":
                    delta = event.delta
                    dtype = getattr(delta, "type", None)
                    if dtype == "text_delta":
                        text_parts.append(delta.text)
                        yield StreamChunk(StreamChunkType.TEXT_DELTA, payload={"text": delta.text})
                    elif dtype == "input_json_delta":
                        tool_id = current_tool_ids.get(event.index, "")
                        if tool_id:
                            acc.delta_tool(tool_id, delta.partial_json)
                            yield StreamChunk(
                                StreamChunkType.TOOL_USE_DELTA,
                                payload={"id": tool_id, "partial_json": delta.partial_json},
                            )
                    elif dtype == "thinking_delta":
                        yield StreamChunk(
                            StreamChunkType.REASONING_DELTA, payload={"text": delta.thinking}
                        )
                elif etype == "content_block_stop":
                    tool_id = current_tool_ids.get(event.index)
                    if tool_id:
                        req_done = acc.finish_tool(tool_id)
                        if req_done is not None:
                            yield StreamChunk(
                                StreamChunkType.TOOL_USE_COMPLETE,
                                payload={
                                    "id": req_done.id,
                                    "name": req_done.name,
                                    "arguments": req_done.arguments,
                                },
                            )
                elif etype == "message_delta":
                    sr = getattr(event.delta, "stop_reason", None)
                    if sr:
                        stop_reason = _to_stop_reason(sr)
                    edits = _applied_edits(event)
                    if edits:
                        applied_edits = edits
                    u = getattr(event, "usage", None)
                    if u is not None:
                        usage = TokenUsage(
                            input_tokens=getattr(u, "input_tokens", 0) or 0,
                            output_tokens=getattr(u, "output_tokens", 0) or 0,
                            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0)
                            or 0,
                        )

            # Get the fully assembled final message from the SDK helper.
            try:
                final_msg = await stream.get_final_message()
                # Capture thinking blocks for round-trip.
                for block in getattr(final_msg, "content", []):
                    if getattr(block, "type", None) == "thinking":
                        reasoning_blocks.append(
                            ReasoningContent(
                                provider="anthropic",
                                raw=_block_to_dict(block),
                                text=getattr(block, "thinking", None),
                            )
                        )
                u = getattr(final_msg, "usage", None)
                if u is not None:
                    usage = TokenUsage(
                        input_tokens=getattr(u, "input_tokens", 0) or 0,
                        output_tokens=getattr(u, "output_tokens", 0) or 0,
                        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                    )
                sr = getattr(final_msg, "stop_reason", None)
                if sr:
                    stop_reason = _to_stop_reason(sr)
                applied_edits = _applied_edits(final_msg) or applied_edits
            except Exception:
                pass

        tool_calls = acc.all_complete()
        edits_payload = [e.to_dict() for e in applied_edits]
        yield StreamChunk(
            StreamChunkType.USAGE,
            payload={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                "applied_edits": edits_payload,
            },
        )
        yield StreamChunk(
            StreamChunkType.STOP,
            payload={
                "stop_reason": stop_reason.value,
                "text": "".join(text_parts),
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls
                ],
                "reasoning": [r.raw for r in reasoning_blocks],
                "applied_edits": edits_payload,
            },
        )

    # ---------- normalization ----------
    def _normalize(self, raw: Any) -> NormalizedResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        reasoning: list[ReasoningContent] = []
        for block in getattr(raw, "content", []):
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCallRequest(
                        id=block.id,
                        name=block.name,
                        arguments=dict(getattr(block, "input", {}) or {}),
                    )
                )
            elif btype == "thinking":
                reasoning.append(
                    ReasoningContent(
                        provider="anthropic",
                        raw=_block_to_dict(block),
                        text=getattr(block, "thinking", None),
                    )
                )
        u = getattr(raw, "usage", None)
        usage = TokenUsage(
            input_tokens=getattr(u, "input_tokens", 0) or 0 if u else 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0 if u else 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0 if u else 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0 if u else 0,
        )
        stop = _to_stop_reason(getattr(raw, "stop_reason", "end_turn"))
        msg = AssistantMessage(
            content="".join(text_parts),
            tool_calls=tool_calls,
            reasoning=reasoning,
            stop_reason=stop,
            usage=usage,
        )
        return NormalizedResponse(
            assistant_message=msg,
            stop_reason=stop,
            usage=usage,
            raw=raw,
            applied_edits=_applied_edits(raw),
        )


def _clear_tool_uses_edit(clearing: ToolResultClearing) -> dict[str, Any]:
    edit: dict[str, Any] = {
        "type": CLEAR_TOOL_USES_EDIT,
        "trigger": {"type": "input_tokens", "value": clearing.trigger_tokens},
        "keep": {"type": "tool_uses", "value": clearing.keep_results},
        "clear_at_least": {"type": "input_tokens", "value": clearing.clear_at_least_tokens},
    }
    if clearing.exclude_tools:
        edit["exclude_tools"] = list(clearing.exclude_tools)
    return edit


def _rejects_context_edits(exc: BaseException) -> bool:
    """A 400 that names context editing: the edits, not the request, are refused.

    A gateway that does not forward the beta, an account without access, or a
    model without the feature all answer this way; any other 400 is a real
    error and propagates.
    """
    if getattr(exc, "status_code", None) != 400:
        return False
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("context_management", "context-management", "clear_tool_uses")
    )


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _applied_edits(obj: Any) -> list[AppliedEdit]:
    """`context_management.applied_edits` of a message or delta, normalized."""
    management = _field(obj, "context_management")
    edits = _field(management, "applied_edits") if management is not None else None
    out: list[AppliedEdit] = []
    for edit in edits or []:
        out.append(
            AppliedEdit(
                type=str(_field(edit, "type") or ""),
                cleared_tool_uses=int(_field(edit, "cleared_tool_uses") or 0),
                cleared_input_tokens=int(_field(edit, "cleared_input_tokens") or 0),
            )
        )
    return out


def _system_blocks(system: str, *, cache: bool = True) -> list[dict[str, Any]]:
    """System content, optionally with cache-control on the trailing block."""
    if not system:
        return []
    block: dict[str, Any] = {"type": "text", "text": system}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def _tool_choice_param(tc: ToolChoice) -> dict[str, Any]:
    if tc.mode == "tool":
        out: dict[str, Any] = {"type": "tool", "name": tc.tool_name}
    else:
        out = {"type": tc.mode}
    if not tc.parallel and tc.mode != "none":
        out["disable_parallel_tool_use"] = True
    return out


def _is_preamble(msg: AnyMessage) -> bool:
    """Dynamic per-turn context (memory facts, summaries) — regenerated every
    turn, so it must stay *after* the incremental cache breakpoint."""
    return bool(isinstance(msg, Message) and msg.metadata.get("system"))


def _apply_incremental_cache(
    wire: list[dict[str, Any]], origins: list[AnyMessage]
) -> None:
    """Place a cache breakpoint on the last stable (non-preamble) message.

    The conversation prefix up to that message is byte-identical on the next
    turn, so Anthropic serves it from cache; volatile preamble appended after
    the breakpoint is re-processed at full price but is small.
    """
    for i in range(len(wire) - 1, -1, -1):
        if _is_preamble(origins[i]):
            continue
        entry = wire[i]
        content = entry.get("content")
        if isinstance(content, str):
            entry["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
        elif isinstance(content, list) and content and isinstance(content[-1], dict):
            content[-1]["cache_control"] = {"type": "ephemeral"}
        return


def _user_content_blocks(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Pass through pre-built blocks (e.g. tool_result lists from ToolResultMessage).
        return content
    return str(content)


def _to_stop_reason(value: Any) -> StopReason:
    mapping = {
        "end_turn": StopReason.END_TURN,
        "tool_use": StopReason.TOOL_USE,
        "max_tokens": StopReason.MAX_TOKENS,
        "stop_sequence": StopReason.STOP_SEQUENCE,
        "pause_turn": StopReason.PAUSE_TURN,
        "refusal": StopReason.REFUSAL,
        "model_context_window_exceeded": StopReason.CONTEXT_EXCEEDED,
    }
    return mapping.get(str(value), StopReason.END_TURN)


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Best-effort conversion of an Anthropic content block to a dict for round-tripping."""
    if hasattr(block, "model_dump"):
        try:
            return block.model_dump()
        except Exception:
            pass
    if hasattr(block, "__dict__"):
        return {k: v for k, v in block.__dict__.items() if not k.startswith("_")}
    return {"type": getattr(block, "type", "unknown")}
