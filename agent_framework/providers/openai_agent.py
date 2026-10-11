"""OpenAIAdapter — wire format normalization for OpenAI Chat Completions.

Key mappings:
  - `tool_calls` array → list[ToolCallRequest]
  - `role: tool` messages → ToolResultMessage
  - `finish_reason: tool_calls` → StopReason.TOOL_USE
  - `usage.prompt_tokens_details.cached_tokens` → TokenUsage.cache_read_input_tokens
    (input_tokens is normalized to the *uncached* remainder so cost math is
    provider-neutral, matching Anthropic semantics).
  - Sampling parameters are only sent when explicitly set (reasoning models
    reject non-default temperature).
  - `reasoning_content` (the de-facto field for exposed reasoning on
    compatible endpoints: Kimi, DeepSeek, vLLM) → ReasoningContent, and back
    onto the assistant message when the history is sent again. OpenAI itself
    never returns it, so requests to OpenAI do not change.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from agent_framework.core.message import (
    AnyMessage,
    AssistantMessage,
    ReasoningContent,
    StopReason,
    SystemMessage,
    TokenUsage,
    ToolCallRequest,
    ToolResultMessage,
    UserMessage,
)
from agent_framework.providers.base import (
    BaseProviderAdapter,
    NormalizedResponse,
    StreamChunk,
    StreamChunkType,
    ToolChoice,
    ToolResultClearing,
)
from agent_framework.providers.streaming import StreamingToolAccumulator
from agent_framework.tools.models import ToolMetadata


def _is_tools_vs_reasoning_effort_error(exc: BaseException) -> bool:
    """Detect Chat Completions' "function tools + reasoning_effort" rejection.

    Newer reasoning models apply a reasoning effort server-side and then refuse
    function tools on /v1/chat/completions unless the effort is explicitly
    'none'. Which models behave this way is not derivable from the id — plain
    `gpt-5` accepts tools normally, and older o-series models reject 'none' as a
    value — so the model's own error is the only reliable signal.
    """
    message = str(exc)
    return "reasoning_effort" in message and "tools" in message


class OpenAIAdapter(BaseProviderAdapter):
    name = "openai"

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
        # Models that turned out to reject `function tools + reasoning_effort`.
        # Populated lazily from the API's own error, then reused for the process.
        self._tools_need_effort_none: set[str] = set()
        if client is not None:
            self._client = client
            return
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("The `openai` package is required.") from e
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if client_kwargs:
            kwargs.update(client_kwargs)
        self._client = AsyncOpenAI(**kwargs)

    # ---------- tool schema ----------
    def tool_schema(self, meta: ToolMetadata) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": meta.name,
                "description": meta.description,
                "parameters": meta.parameters_schema,
            },
        }

    # ---------- request denormalization ----------
    def denormalize_messages(self, messages: list[AnyMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                out.append({"role": "system", "content": msg.content})
            elif isinstance(msg, ToolResultMessage):
                # Each tool_result becomes its own role=tool message.
                for tr in msg.content:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.tool_use_id,
                            "content": tr.as_text(),
                        }
                    )
            elif isinstance(msg, UserMessage):
                content = msg.content if isinstance(msg.content, str) else _stringify(msg.content)
                out.append({"role": "user", "content": content})
            elif isinstance(msg, AssistantMessage):
                rec: dict[str, Any] = {"role": "assistant"}
                text = msg.content if isinstance(msg.content, str) else ""
                rec["content"] = text or None
                if msg.tool_calls:
                    rec["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, default=str),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                # OpenAI keeps reasoning server-side and never returns it;
                # compatible endpoints that expose it (Kimi, DeepSeek) expect
                # it back on every historical assistant message — Kimi rejects
                # a tool-call message without it. Only blocks this adapter
                # produced are forwarded: an Anthropic thinking block has no
                # meaning here.
                reasoning = _reasoning_content(msg)
                if reasoning is not None:
                    rec["reasoning_content"] = reasoning
                out.append(rec)
        return out

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
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        all_messages = [SystemMessage(content=system)] + messages if system else list(messages)
        req: dict[str, Any] = {
            "model": model,
            "messages": self.denormalize_messages(all_messages),
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:
            req["temperature"] = temperature
        if tools:
            req["tools"] = [self.tool_schema(t) for t in tools]
            # Learned once per model from the API's own 400 — see
            # `_is_tools_vs_reasoning_effort_error`. `extra` is applied last, so
            # request_extra can always override this.
            if model in self._tools_need_effort_none:
                req["reasoning_effort"] = "none"
        if tool_choice is not None:
            tc_param, parallel = _tool_choice_param(tool_choice)
            req["tool_choice"] = tc_param
            if not parallel:
                req["parallel_tool_calls"] = False
        if extra:
            req.update(extra)
        return req

    async def _create(self, req: dict[str, Any]) -> Any:
        """`chat.completions.create` that learns the reasoning_effort quirk once.

        On the "function tools + reasoning_effort" rejection the model is
        remembered, `reasoning_effort='none'` is added and the call is retried.
        Every later request for that model carries the flag from the start, so
        the extra round-trip is paid at most once per model per process.
        """
        try:
            return await self._client.chat.completions.create(**req)
        except Exception as exc:
            already_set = "reasoning_effort" in req
            if already_set or not req.get("tools") or not _is_tools_vs_reasoning_effort_error(exc):
                raise
            self._tools_need_effort_none.add(req["model"])
            req["reasoning_effort"] = "none"
            return await self._client.chat.completions.create(**req)

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
        enable_caching: bool = True,  # no-op: OpenAI caches automatically
        extra: dict[str, Any] | None = None,
        # Never passed: supports_tool_result_clearing is False here.
        context_edits: ToolResultClearing | None = None,
    ) -> NormalizedResponse:
        req = self._build_request(
            system=system, messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            tool_choice=tool_choice, extra=extra,
        )
        raw = await self._create(req)
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
        all_messages = [SystemMessage(content=system)] + messages if system else list(messages)
        req: dict[str, Any] = {
            "model": model,
            "messages": self.denormalize_messages(all_messages),
            "max_completion_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if extra:
            req.update(extra)
        raw = await self._client.chat.completions.create(**req)
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
        enable_caching: bool = True,  # no-op: OpenAI caches automatically
        extra: dict[str, Any] | None = None,
        # Never passed: supports_tool_result_clearing is False here.
        context_edits: ToolResultClearing | None = None,
    ) -> AsyncIterator[StreamChunk]:
        req = self._build_request(
            system=system, messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            tool_choice=tool_choice, extra=extra,
        )
        req["stream"] = True
        req["stream_options"] = {"include_usage": True}

        acc = StreamingToolAccumulator()
        text_parts: list[str] = []
        stop_reason: StopReason = StopReason.END_TURN
        usage = TokenUsage()
        # Map index → tool_call_id for proper aggregation.
        index_to_id: dict[int, str] = {}
        index_to_name: dict[int, str] = {}

        stream = await self._create(req)
        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                u = getattr(chunk, "usage", None)
                if u is not None:
                    usage = _normalize_usage(u)
                continue
            delta = choice.delta
            if getattr(delta, "content", None):
                text_parts.append(delta.content)
                yield StreamChunk(StreamChunkType.TEXT_DELTA, payload={"text": delta.content})
            tcs = getattr(delta, "tool_calls", None) or []
            for tc in tcs:
                idx = tc.index
                if tc.id and idx not in index_to_id:
                    index_to_id[idx] = tc.id
                    name = (tc.function.name if tc.function else "") or ""
                    index_to_name[idx] = name
                    acc.start_tool(tc.id, name)
                    yield StreamChunk(
                        StreamChunkType.TOOL_USE_START, payload={"id": tc.id, "name": name}
                    )
                tool_id = index_to_id.get(idx, "")
                if tool_id and tc.function:
                    if tc.function.name and not index_to_name.get(idx):
                        index_to_name[idx] = tc.function.name
                        acc.start_tool(tool_id, tc.function.name)
                    args_chunk = tc.function.arguments or ""
                    if args_chunk:
                        acc.delta_tool(tool_id, args_chunk)
                        yield StreamChunk(
                            StreamChunkType.TOOL_USE_DELTA,
                            payload={"id": tool_id, "partial_json": args_chunk},
                        )
            fin = getattr(choice, "finish_reason", None)
            if fin:
                stop_reason = _to_stop_reason(fin)

        tool_calls = acc.all_complete()
        for tc in tool_calls:
            yield StreamChunk(
                StreamChunkType.TOOL_USE_COMPLETE,
                payload={"id": tc.id, "name": tc.name, "arguments": tc.arguments},
            )
        yield StreamChunk(
            StreamChunkType.USAGE,
            payload={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
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
            },
        )

    # ---------- normalization ----------
    def _normalize(self, raw: Any) -> NormalizedResponse:
        choice = raw.choices[0]
        msg = choice.message
        text = getattr(msg, "content", "") or ""
        tool_calls: list[ToolCallRequest] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except Exception:
                args = {}
            tool_calls.append(ToolCallRequest(id=tc.id, name=tc.function.name, arguments=args))
        usage = _normalize_usage(getattr(raw, "usage", None))
        stop = _to_stop_reason(getattr(choice, "finish_reason", "stop"))
        # The SDK keeps fields it does not model, so a compatible endpoint's
        # `reasoning_content` is reachable; OpenAI's own replies have none.
        reasoning = _reasoning_block(getattr(msg, "reasoning_content", None))
        return NormalizedResponse(
            assistant_message=AssistantMessage(
                content=text,
                tool_calls=tool_calls,
                reasoning=[reasoning] if reasoning else [],
                stop_reason=stop,
                usage=usage,
            ),
            stop_reason=stop,
            usage=usage,
            raw=raw,
        )


def _reasoning_block(reasoning_content: Any) -> ReasoningContent | None:
    """The opaque block for a reply's `reasoning_content`, if it carried one."""
    if not isinstance(reasoning_content, str) or not reasoning_content:
        return None
    return ReasoningContent(
        provider="openai",
        raw={"reasoning_content": reasoning_content},
        text=reasoning_content,
    )


def _reasoning_content(msg: AssistantMessage) -> str | None:
    """What to send back as `reasoning_content`: this adapter's blocks, joined."""
    parts = [
        r.raw["reasoning_content"]
        for r in msg.reasoning
        if r.provider == "openai" and isinstance(r.raw.get("reasoning_content"), str)
    ]
    return "".join(parts) if parts else None


def _normalize_usage(u: Any) -> TokenUsage:
    """Normalize OpenAI usage to provider-neutral semantics.

    OpenAI's `prompt_tokens` *includes* cached tokens; the framework's
    `input_tokens` means the uncached remainder (Anthropic semantics), so
    cached tokens are subtracted out and reported separately.
    """
    if u is None:
        return TokenUsage()
    prompt = getattr(u, "prompt_tokens", 0) or 0
    completion = getattr(u, "completion_tokens", 0) or 0
    details = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    cached = min(cached, prompt)
    return TokenUsage(
        input_tokens=prompt - cached,
        output_tokens=completion,
        cache_read_input_tokens=cached,
    )


def _tool_choice_param(tc: ToolChoice) -> tuple[Any, bool]:
    if tc.mode == "tool":
        return {"type": "function", "function": {"name": tc.tool_name}}, tc.parallel
    mapping = {"auto": "auto", "any": "required", "none": "none"}
    return mapping[tc.mode], tc.parallel


def _to_stop_reason(value: Any) -> StopReason:
    mapping = {
        "stop": StopReason.END_TURN,
        "tool_calls": StopReason.TOOL_USE,
        "length": StopReason.MAX_TOKENS,
        "function_call": StopReason.TOOL_USE,
        "content_filter": StopReason.REFUSAL,
    }
    return mapping.get(str(value), StopReason.END_TURN)


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, default=str)
    except Exception:
        return str(content)
