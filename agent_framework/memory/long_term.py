"""LongTermMemory — persistent across sessions.

Two non-vector mechanisms behind one interface:
  1. Rolling summary — LLM-compressed narrative of past sessions.
  2. Structured fact store — typed key-value/tag store with keyword search.

Retrieval is by exact key, tag match, keyword/substring, and recency —
never by embedding similarity.
"""

from __future__ import annotations

from typing import Iterable

from agent_framework.memory.backends.base import Fact, LongTermBackend


class LongTermMemory:
    def __init__(self, *, backend: LongTermBackend, namespace: str = "default") -> None:
        self.backend = backend
        self.namespace = namespace

    # ---- summary ----
    async def get_summary(self) -> str:
        return await self.backend.get_summary(self.namespace)

    async def set_summary(self, summary: str) -> None:
        await self.backend.set_summary(self.namespace, summary)

    async def append_to_summary(self, addition: str, *, max_chars: int = 12_000) -> None:
        current = await self.get_summary()
        combined = (current + "\n\n" + addition).strip() if current else addition
        if len(combined) > max_chars:
            combined = combined[-max_chars:]
        await self.set_summary(combined)

    # ---- fact store ----
    async def remember(
        self,
        key: str,
        value: str,
        *,
        tags: Iterable[str] = (),
    ) -> Fact:
        fact = Fact(key=key, value=value, tags=tuple(tags), namespace=self.namespace)
        await self.backend.put_fact(fact)
        return fact

    async def recall(self, key: str) -> Fact | None:
        return await self.backend.get_fact(key, namespace=self.namespace)

    async def forget(self, key: str) -> None:
        await self.backend.delete_fact(key, namespace=self.namespace)

    async def search(
        self,
        *,
        query: str | None = None,
        tags: Iterable[str] | None = None,
        limit: int = 20,
    ) -> list[Fact]:
        return await self.backend.search_facts(
            query=query, tags=tags, namespace=self.namespace, limit=limit
        )

    async def all(self, limit: int = 200) -> list[Fact]:
        return await self.backend.list_facts(namespace=self.namespace, limit=limit)
