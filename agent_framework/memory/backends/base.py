"""LongTermBackend ABC — swappable persistence for long-term memory."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


@dataclass
class Fact:
    key: str
    value: str
    tags: tuple[str, ...] = ()
    namespace: str = "default"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class LongTermBackend(abc.ABC):
    """Pure persistence interface for `LongTermMemory`."""

    @abc.abstractmethod
    async def put_fact(self, fact: Fact) -> None: ...

    @abc.abstractmethod
    async def get_fact(self, key: str, namespace: str = "default") -> Fact | None: ...

    @abc.abstractmethod
    async def delete_fact(self, key: str, namespace: str = "default") -> None: ...

    @abc.abstractmethod
    async def search_facts(
        self,
        *,
        query: str | None = None,
        tags: Iterable[str] | None = None,
        namespace: str = "default",
        limit: int = 20,
    ) -> list[Fact]: ...

    @abc.abstractmethod
    async def list_facts(self, namespace: str = "default", limit: int = 200) -> list[Fact]: ...

    async def match_facts(
        self, words: Iterable[str], *, namespace: str = "default", limit: int = 20
    ) -> list[Fact] | None:
        """The facts containing any of *words*, best match first — or None
        when this backend cannot search, in which case the manager matches
        the listed facts in Python. `SQLiteBackend` answers from its FTS5
        index: over every fact, with diacritics folded."""
        return None

    @abc.abstractmethod
    async def set_summary(self, namespace: str, summary: str) -> None: ...

    @abc.abstractmethod
    async def get_summary(self, namespace: str) -> str: ...

    async def close(self) -> None:
        return None
