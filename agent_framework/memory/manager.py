"""MemoryManager — coordinates short-term + long-term memory.

Owns the working buffer (`ShortTermMemory`), decides when to fold content
into the long-term rolling summary, and supplies the `<relevant_context>`
block for context assembly (Step 4 of the compaction pipeline) via
keyword/entity/recency match.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from agent_framework.memory.long_term import LongTermMemory
from agent_framework.memory.short_term import ShortTermMemory


_WORD_RX = re.compile(r"[A-Za-z][A-Za-z0-9_\-/.]{2,}")


class MemoryManager:
    def __init__(
        self,
        *,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
        recent_query_limit: int = 8,
    ) -> None:
        self.short_term = short_term or ShortTermMemory()
        self.long_term = long_term
        self.recent_query_limit = recent_query_limit

    # ---- relevant context (Step 4 of context assembly) ----
    async def relevant_facts(self, current_user_input: str) -> list[dict[str, Any]]:
        if self.long_term is None or not current_user_input:
            return []
        # Pull keywords/entities — alphanumeric tokens length ≥ 3.
        tokens = {w.lower() for w in _WORD_RX.findall(current_user_input)}
        if not tokens:
            return []
        # Score by overlap with key+value+tags. Cheap, sufficient.
        all_facts = await self.long_term.search(limit=200)
        scored: list[tuple[int, dict[str, Any]]] = []
        for f in all_facts:
            hay = f"{f.key} {f.value} {' '.join(f.tags)}".lower()
            score = sum(1 for t in tokens if t in hay)
            if score == 0:
                continue
            scored.append(
                (
                    score,
                    {"key": f.key, "value": f.value, "tags": list(f.tags)},
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[: self.recent_query_limit]]

    async def rolling_summary(self) -> str:
        if self.long_term is None:
            return ""
        return await self.long_term.get_summary()

    async def remember(
        self, key: str, value: str, *, tags: Iterable[str] = ()
    ) -> None:
        if self.long_term is not None:
            await self.long_term.remember(key, value, tags=tags)

    async def fold_into_summary(self, addition: str) -> None:
        if self.long_term is not None and addition.strip():
            await self.long_term.append_to_summary(addition)
