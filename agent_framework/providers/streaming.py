"""StreamingToolAccumulator — buffers partial tool_use JSON deltas.

Tools are dispatched only after their input JSON has fully arrived. Text
deltas are forwarded to the caller in real time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agent_framework.core.message import ToolCallRequest


@dataclass
class _PartialCall:
    id: str
    name: str
    raw_json: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    complete: bool = False


class StreamingToolAccumulator:
    """Tracks in-progress tool calls during a streamed response.

    Provider adapters call:
      - start_tool(id, name)
      - delta_tool(id, partial_json_text)
      - finish_tool(id) → returns the completed ToolCallRequest (or None)

    Final flush via `all_complete()` returns every accumulated tool call.
    """

    def __init__(self) -> None:
        self._partials: dict[str, _PartialCall] = {}
        self._order: list[str] = []

    def start_tool(self, tool_id: str, name: str) -> None:
        if tool_id not in self._partials:
            self._partials[tool_id] = _PartialCall(id=tool_id, name=name)
            self._order.append(tool_id)
        else:
            self._partials[tool_id].name = name

    def delta_tool(self, tool_id: str, json_chunk: str) -> None:
        p = self._partials.get(tool_id)
        if p is None:
            self.start_tool(tool_id, name="")
            p = self._partials[tool_id]
        p.raw_json += json_chunk

    def finish_tool(self, tool_id: str) -> ToolCallRequest | None:
        p = self._partials.get(tool_id)
        if p is None:
            return None
        if p.complete:
            return ToolCallRequest(id=p.id, name=p.name, arguments=p.arguments)
        try:
            p.arguments = json.loads(p.raw_json or "{}")
        except Exception:
            p.arguments = {}
        p.complete = True
        return ToolCallRequest(id=p.id, name=p.name, arguments=p.arguments)

    def all_complete(self) -> list[ToolCallRequest]:
        out: list[ToolCallRequest] = []
        for tool_id in self._order:
            req = self.finish_tool(tool_id)
            if req is not None:
                out.append(req)
        return out
