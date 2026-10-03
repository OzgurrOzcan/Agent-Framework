"""MemoryManager — coordinates short-term + long-term memory.

Owns the working buffer (`ShortTermMemory`) and finds the long-term facts
relevant to a user message, which the context manager stores in the history
once per run.

Facts are matched by whole words in any script: "faturalar" finds a fact
that says "faturalar", "art" does not find "party", and the words inside a
key ("user_name", "deployTarget") count as words.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from agent_framework.memory.long_term import LongTermMemory
from agent_framework.memory.short_term import ShortTermMemory

# Letters and digits of any script; underscores separate words in keys.
_WORD_RX = re.compile(r"[^\W_]+")
# A lower-case letter or digit followed by an upper-case one: deployTarget.
_CAMEL_RX = re.compile(r"(?<=[^\W_])(?=[A-Z])")
# Shorter words ("is", "ve", "de") match far too much to mean anything.
_MIN_WORD = 3


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
        """The facts sharing the most whole words with *current_user_input*.

        Ties go to the most recently updated fact (the search returns facts
        newest first and the sort is stable).
        """
        if self.long_term is None or not current_user_input:
            return []
        query = words(current_user_input)
        if not query:
            return []
        all_facts = await self.long_term.search(limit=200)
        scored: list[tuple[int, dict[str, Any]]] = []
        for f in all_facts:
            hay = words(f.key, split_case=True) | words(f.value) | words(" ".join(f.tags))
            score = len(query & hay)
            if score == 0:
                continue
            scored.append((score, {"key": f.key, "value": f.value, "tags": list(f.tags)}))
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


def words(text: str, *, split_case: bool = False) -> set[str]:
    """The comparable words of *text*: Unicode-aware, case-folded, >= 3 chars.

    NFKC folds compatibility forms (full-width letters, ligatures) and
    casefold() handles scripts where lower() falls short. The combining dot
    that casefold() leaves on a dotted capital I ("İ" → "i̇") is removed, so
    "İstanbul" and "istanbul" are one word. *split_case* also splits
    camelCase, for keys such as "deployTarget".
    """
    if split_case:
        text = _CAMEL_RX.sub(" ", text)
    folded = unicodedata.normalize("NFKC", text).casefold().replace("\u0307", "")
    return {w for w in _WORD_RX.findall(folded) if len(w) >= _MIN_WORD}
