"""InMemoryBackend — for tests and ephemeral runs."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Iterable

from agent_framework.memory.backends.base import Fact, LongTermBackend


class InMemoryBackend(LongTermBackend):
    def __init__(self) -> None:
        self._facts: dict[tuple[str, str], Fact] = {}
        self._summaries: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def put_fact(self, fact: Fact) -> None:
        async with self._lock:
            now = datetime.now(timezone.utc)
            existing = self._facts.get((fact.namespace, fact.key))
            self._facts[(fact.namespace, fact.key)] = Fact(
                key=fact.key,
                value=fact.value,
                tags=tuple(fact.tags),
                namespace=fact.namespace,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )

    async def get_fact(self, key: str, namespace: str = "default") -> Fact | None:
        async with self._lock:
            return self._facts.get((namespace, key))

    async def delete_fact(self, key: str, namespace: str = "default") -> None:
        async with self._lock:
            self._facts.pop((namespace, key), None)

    async def search_facts(
        self,
        *,
        query: str | None = None,
        tags: Iterable[str] | None = None,
        namespace: str = "default",
        limit: int = 20,
    ) -> list[Fact]:
        async with self._lock:
            tags_set = set(tags or ())
            results: list[Fact] = []
            ql = (query or "").lower()
            for (ns, _), fact in self._facts.items():
                if ns != namespace:
                    continue
                if tags_set and not tags_set.issubset(set(fact.tags)):
                    continue
                if ql:
                    hay = f"{fact.key} {fact.value} {' '.join(fact.tags)}".lower()
                    if ql not in hay:
                        continue
                results.append(fact)
            results.sort(key=lambda f: f.updated_at or datetime.min, reverse=True)
            return results[:limit]

    async def list_facts(self, namespace: str = "default", limit: int = 200) -> list[Fact]:
        return await self.search_facts(namespace=namespace, limit=limit)

    async def set_summary(self, namespace: str, summary: str) -> None:
        async with self._lock:
            self._summaries[namespace] = summary

    async def get_summary(self, namespace: str) -> str:
        async with self._lock:
            return self._summaries.get(namespace, "")
