"""MockLLMProvider — scripted responses for unit testing without API calls.

Each `ScriptedTurn` describes what the mock returns when called. Turns
are consumed in order; after the last one, every call returns a plain
end-turn text response.

Usage:
    mock = MockLLMProvider([
        ScriptedTurn(text="Let me search for that.", tool_calls=[
            {"id": "call_1", "name": "web_search", "arguments": {"query": "test"}}
        ]),
        ScriptedTurn(text="Here is what I found."),
    ])

    # Plug into an agent by replacing its provider:
    agent._provider = mock
    agent._loop.provider = mock
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agent_framework.core.message import (
    AnyMessage,
    AssistantMessage,
    ReasoningContent,
    StopReason,
    TokenUsage,
    ToolCallRequest,
)
from agent_framework.providers.base import (
    BaseProviderAdapter,
    NormalizedResponse,
    StreamChunk,
    StreamChunkType,
    ToolChoice,
)
from agent_framework.tools.models import ToolMetadata


@dataclass
class ScriptedTurn:
    """One scripted LLM response."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(input_tokens=10, output_tokens=20))
    stop_reason: StopReason | None = None
    # Raw reasoning blocks (e.g. Anthropic thinking dicts) to attach.
    reasoning: list[dict[str, Any]] = field(default_factory=list)
    # Raise this exception instead of returning a response.
    raise_exc: Exception | None = None


class MockLLMProvider(BaseProviderAdapter):
    name = "mock"

    def __init__(self, script: list[ScriptedTurn] | None = None) -> None:
        self._script = list(script or [])
        self._call_count = 0
        self.call_log: list[dict[str, Any]] = []
        self.structured_calls: list[dict[str, Any]] = []
        # Scripted raw-text responses for `structured()`.
        self.structured_script: list[str] = []

    def add_turn(self, turn: ScriptedTurn) -> None:
        self._script.append(turn)

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
    ) -> NormalizedResponse:
        turn = self._next_turn()
        self.call_log.append({
            "system": system[:100],
            "messages": list(messages),
            "message_count": len(messages),
            "tool_count": len(tools),
            "temperature": temperature,
            "tool_choice": tool_choice,
            "enable_caching": enable_caching,
        })
        if turn.raise_exc is not None:
            raise turn.raise_exc
        tool_calls = [
            ToolCallRequest(id=tc["id"], name=tc["name"], arguments=tc.get("arguments", {}))
            for tc in turn.tool_calls
        ]
        stop_reason = turn.stop_reason or (
            StopReason.TOOL_USE if tool_calls else StopReason.END_TURN
        )
        msg = AssistantMessage(
            content=turn.text,
            tool_calls=tool_calls,
            reasoning=[
                ReasoningContent(provider="anthropic", raw=r, text=r.get("thinking"))
                for r in turn.reasoning
            ],
            stop_reason=stop_reason,
            usage=turn.usage,
        )
        return NormalizedResponse(
            assistant_message=msg,
            stop_reason=stop_reason,
            usage=turn.usage,
        )

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
    ) -> AsyncIterator[StreamChunk]:
        resp = await self.call(
            system=system, messages=messages, tools=tools,
            model=model, max_tokens=max_tokens, temperature=temperature,
            tool_choice=tool_choice, enable_caching=enable_caching, extra=extra,
        )
        if resp.assistant_message.content:
            yield StreamChunk(
                StreamChunkType.TEXT_DELTA,
                payload={"text": resp.assistant_message.content},
            )
        for tc in resp.assistant_message.tool_calls:
            yield StreamChunk(
                StreamChunkType.TOOL_USE_COMPLETE,
                payload={"id": tc.id, "name": tc.name, "arguments": tc.arguments},
            )
        yield StreamChunk(
            StreamChunkType.USAGE,
            payload={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        )
        yield StreamChunk(
            StreamChunkType.STOP,
            payload={
                "stop_reason": resp.stop_reason.value,
                "text": resp.assistant_message.content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in resp.assistant_message.tool_calls
                ],
                "reasoning": [r.raw for r in resp.assistant_message.reasoning],
            },
        )

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
        self.structured_calls.append({"schema": schema, "messages": list(messages)})
        if self.structured_script:
            return self.structured_script.pop(0)
        # Default: echo an empty object.
        return json.dumps({})

    def tool_schema(self, meta: ToolMetadata) -> dict[str, Any]:
        return {"name": meta.name, "description": meta.description}

    def denormalize_messages(self, messages: list[AnyMessage]) -> list[dict[str, Any]]:
        return [{"role": getattr(m, "role", "user"), "content": str(getattr(m, "content", ""))}
                for m in messages]

    def _next_turn(self) -> ScriptedTurn:
        if self._call_count < len(self._script):
            turn = self._script[self._call_count]
        else:
            turn = ScriptedTurn(text="Done.")
        self._call_count += 1
        return turn
