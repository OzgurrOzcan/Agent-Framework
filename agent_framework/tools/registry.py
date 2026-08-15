"""ToolRegistry — composable collection of `Tool` objects.

You can pass either `list[Tool]` or `ToolRegistry` to `BaseAgent.create()`.
The registry adds composition operators (filter, +) for advanced cases.
"""

from __future__ import annotations

import warnings
from typing import Callable, Iterable, Iterator

from agent_framework.permissions.models import PermissionCategory
from agent_framework.tools.models import Tool


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        if tools:
            for t in tools:
                self.add(t)

    # --- mutation ---
    def add(self, *tools: Tool) -> "ToolRegistry":
        for t in tools:
            if not isinstance(t, Tool):
                raise TypeError(
                    f"ToolRegistry.add() requires Tool, got {type(t).__name__}. "
                    f"Did you forget @tool(...)?"
                )
            name = t.metadata.name
            existing = self._tools.get(name)
            if existing is not None and existing is not t:
                warnings.warn(
                    f"Tool '{name}' is being overwritten in the registry; "
                    f"use replace() to silence this warning."
                )
            self._tools[name] = t
        return self

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def replace(self, tool: Tool) -> None:
        """Insert or overwrite a tool without the `add()` overwrite warning."""
        self._tools[tool.metadata.name] = tool

    # --- query ---
    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def as_list(self) -> list[Tool]:
        return list(self._tools.values())

    # --- composition ---
    def __add__(self, other: "ToolRegistry | Iterable[Tool]") -> "ToolRegistry":
        out = ToolRegistry(self._tools.values())
        if isinstance(other, ToolRegistry):
            for t in other:
                out.add(t)
        else:
            for t in other:
                out.add(t)
        return out

    def filter(
        self,
        *,
        permission: PermissionCategory | None = None,
        tags: Iterable[str] | None = None,
        predicate: Callable[[Tool], bool] | None = None,
    ) -> "ToolRegistry":
        out = ToolRegistry()
        tags_set = set(tags or ())
        for t in self:
            if permission is not None and t.metadata.permission != permission:
                continue
            if tags_set and not (set(t.metadata.tags) & tags_set):
                continue
            if predicate is not None and not predicate(t):
                continue
            out.add(t)
        return out


def to_registry(value: ToolRegistry | Iterable[Tool] | None) -> ToolRegistry:
    """Normalize `tools=` parameter into a ToolRegistry."""
    if value is None:
        return ToolRegistry()
    if isinstance(value, ToolRegistry):
        return value
    return ToolRegistry(value)
