"""ContextManager — assemble + compact context before every LLM call.

Cache boundary discipline:
  - The system prompt is byte-stable for the entire session (top of prefix,
    cached on Anthropic).
  - Everything dynamic (memory injection, reminders) is appended as
    user-role messages at the END of the message list, after the incremental
    cache breakpoint, so the stable conversation prefix stays cacheable.

Compaction strategies, cheapest-first:
  1. Tool-result truncation       (free, recomputed per turn)
  2. Sliding window               (free, recomputed per turn)
  3. Selective summary            (1 LLM call: summarize oldest 30%)
  4. Full compaction              (1 LLM call: summarize all but last K)

LLM-driven steps (3, 4) are expensive, so their result is *persisted*: the
returned `AssembledContext.persist` list replaces short-term memory, meaning
the summary is computed once, not on every subsequent turn.

Token estimates are calibrated against the provider's real usage numbers via
`note_actual_tokens()` (EWMA), because the chars/4 heuristic drifts badly on
non-English text and code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from agent_framework.core.config import CompactionConfig
from agent_framework.core.message import (
    AnyMessage,
    AssistantMessage,
    ToolResult,
    ToolResultMessage,
    UserMessage,
    estimate_message_tokens,
    estimate_tokens,
)
from agent_framework.tracing.trace import (
    EventType,
    ExecutionTrace,
)

if TYPE_CHECKING:
    from agent_framework.memory.manager import MemoryManager
    from agent_framework.providers.base import BaseProviderAdapter


@dataclass
class AssembledContext:
    system_prompt: str
    messages: list[AnyMessage]
    estimated_tokens: int
    compaction_applied: list[str]
    # When set, the caller must replace short-term memory with this list —
    # it contains the persisted result of an LLM-driven compaction step.
    persist: list[AnyMessage] | None = None


class ContextManager:
    def __init__(
        self,
        *,
        system_prompt: str,
        context_window: int,
        response_buffer_tokens: int,
        compaction_buffer_tokens: int,
        compaction: CompactionConfig,
        memory: "MemoryManager | None" = None,
        trace: ExecutionTrace | None = None,
        summarizer: "BaseProviderAdapter | None" = None,
        summarizer_model: str = "",
    ) -> None:
        self._system_prompt = system_prompt
        self._context_window = context_window
        self._response_buffer = response_buffer_tokens
        self._compaction_buffer = compaction_buffer_tokens
        self._cfg = compaction
        self._memory = memory
        self._trace = trace
        self._summarizer = summarizer
        self._summarizer_model = summarizer_model
        # EWMA correction factor: actual_tokens / estimated_tokens.
        self._calibration = 1.0
        self._last_estimate_total = 0

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def calibration(self) -> float:
        return self._calibration

    def budget(self) -> int:
        """Total tokens available for conversation messages."""
        sys_tokens = estimate_tokens(self._system_prompt)
        return max(
            1,
            self._context_window - sys_tokens - self._response_buffer - self._compaction_buffer,
        )

    def note_actual_tokens(self, actual_input_tokens: int) -> None:
        """Calibrate the estimator with the provider-reported context size.

        Pass the *full* context size (input + cache_read + cache_creation
        tokens). Called by the loop after every LLM response.
        """
        if actual_input_tokens <= 0 or self._last_estimate_total <= 0:
            return
        ratio = actual_input_tokens / self._last_estimate_total
        self._calibration = min(4.0, max(0.5, 0.8 * self._calibration + 0.2 * ratio))

    def _calibrated(self, raw_tokens: int) -> int:
        return int(raw_tokens * self._calibration)

    async def assemble(
        self,
        history: list[AnyMessage],
        *,
        current_user_input: str | None = None,
    ) -> AssembledContext:
        applied: list[str] = []
        working: list[AnyMessage] = list(history)
        summary_msgs: list[AnyMessage] = []
        persist_needed = False

        budget = self.budget()
        current = sum(estimate_message_tokens(m) for m in working)

        # Memory-augmented dynamic preamble. Appended at the END of the
        # message list so the stable conversation prefix stays cacheable.
        preamble: list[AnyMessage] = []
        if self._memory is not None and current_user_input is not None:
            facts = await self._memory.relevant_facts(current_user_input)
            if facts:
                preamble.append(
                    UserMessage(
                        content=_relevant_context_block(facts),
                        metadata={"system": "relevant_context"},
                    )
                )
            summary = await self._memory.rolling_summary()
            if summary:
                preamble.append(
                    UserMessage(
                        content=f"<long_term_summary>\n{summary}\n</long_term_summary>",
                        metadata={"system": "long_term_summary"},
                    )
                )
                if self._trace is not None:
                    self._trace.record_event(
                        EventType.MEMORY_READ,
                        name="rolling_summary",
                        payload={"chars": len(summary)},
                    )

        threshold = int(budget * self._cfg.trigger_ratio)
        if self._cfg.enabled and self._calibrated(current) > threshold:
            # Step 1 — truncate verbose tool results (free, not persisted).
            working, reclaimed = _truncate_tool_results(
                working, self._cfg.tool_result_truncate_threshold
            )
            if reclaimed > 0:
                applied.append(f"truncate_tool_results(-{reclaimed} tok)")
                if self._trace is not None:
                    self._trace.record_event(
                        EventType.COMPACTION,
                        name="tool_result_truncation",
                        payload={"reclaimed_tokens": reclaimed},
                    )
            current = sum(estimate_message_tokens(m) for m in working)

        if self._cfg.enabled and self._calibrated(current) > threshold:
            # Step 2 — sliding window (free, not persisted).
            working, reclaimed = _sliding_window(working, self._cfg.sliding_window_min_turns)
            if reclaimed > 0:
                applied.append(f"sliding_window(-{reclaimed} tok)")
                if self._trace is not None:
                    self._trace.record_event(
                        EventType.COMPACTION,
                        name="sliding_window",
                        payload={"reclaimed_tokens": reclaimed},
                    )
            current = sum(estimate_message_tokens(m) for m in working)

        if self._cfg.enabled and self._calibrated(current) > threshold and self._summarizer is not None:
            # Step 3 — selective summary (1 LLM call; persisted). The cutoff
            # is sized to land comfortably below the trigger threshold so the
            # next turn does not immediately re-summarize.
            target_raw = int(threshold * 0.6 / max(self._calibration, 1e-9))
            working, reclaimed, summary = await self._selective_summary(
                working, current_tokens=current, target_tokens=target_raw
            )
            if reclaimed > 0:
                applied.append(f"selective_summary(-{reclaimed} tok)")
                if self._trace is not None:
                    self._trace.record_event(
                        EventType.COMPACTION,
                        name="selective_summary",
                        payload={"reclaimed_tokens": reclaimed},
                    )
                if summary:
                    summary_msgs.append(_summary_message(summary))
                persist_needed = True
            current = sum(estimate_message_tokens(m) for m in working)

        if self._cfg.enabled and self._calibrated(current) > budget and self._summarizer is not None:
            # Step 4 — full compaction (1 LLM call; persisted).
            working, reclaimed, summary = await self._full_compaction(working)
            if reclaimed > 0:
                applied.append(f"full_compaction(-{reclaimed} tok)")
                if self._trace is not None:
                    self._trace.record_event(
                        EventType.COMPACTION,
                        name="full_compaction",
                        payload={"reclaimed_tokens": reclaimed},
                    )
                if summary:
                    summary_msgs.append(_summary_message(summary))
                persist_needed = True
            current = sum(estimate_message_tokens(m) for m in working)

        persist: list[AnyMessage] | None = None
        if persist_needed:
            persist = summary_msgs + working

        assembled: list[AnyMessage] = (summary_msgs + working) + preamble
        estimated = sum(estimate_message_tokens(m) for m in assembled)
        self._last_estimate_total = estimated + estimate_tokens(self._system_prompt)
        return AssembledContext(
            system_prompt=self._system_prompt,
            messages=assembled,
            estimated_tokens=estimated,
            compaction_applied=applied,
            persist=persist,
        )

    # ---------- LLM-call-driven strategies ----------
    async def _selective_summary(
        self,
        messages: list[AnyMessage],
        *,
        current_tokens: int,
        target_tokens: int,
    ) -> tuple[list[AnyMessage], int, str]:
        if not self._summarizer:
            return messages, 0, ""
        # Summarize at least the oldest 30%, and as much more as needed to
        # bring the estimate down to `target_tokens` (keeping ≥ 2 messages).
        need = max(0, current_tokens - target_tokens)
        acc = 0
        cutoff = 0
        for i, m in enumerate(messages[:-2]):
            acc += estimate_message_tokens(m)
            cutoff = i + 1
            if acc >= need:
                break
        cutoff = max(cutoff, max(1, int(len(messages) * 0.30)))
        cutoff = min(cutoff, max(0, len(messages) - 2))
        # The kept suffix must itself be a valid conversation start.
        cutoff = _safe_forward_cut(messages, cutoff)
        if cutoff <= 0:
            return messages, 0, ""
        to_summarize = messages[:cutoff]
        keep = messages[cutoff:]
        before_tokens = sum(estimate_message_tokens(m) for m in to_summarize)
        summary = await self._summarize_batch(to_summarize)
        after_tokens = estimate_tokens(summary)
        reclaimed = max(0, before_tokens - after_tokens)
        return keep, reclaimed, summary

    async def _full_compaction(
        self, messages: list[AnyMessage]
    ) -> tuple[list[AnyMessage], int, str]:
        if not self._summarizer:
            return messages, 0, ""
        keep_n = self._cfg.full_compaction_min_recent_turns * 2  # turn ≈ 2 messages
        keep_n = max(2, keep_n)
        if len(messages) <= keep_n:
            to_summarize = list(messages)
            keep: list[AnyMessage] = []
        else:
            cutoff = _safe_forward_cut(messages, len(messages) - keep_n)
            if cutoff <= 0:
                to_summarize = list(messages)
                keep = []
            else:
                to_summarize = messages[:cutoff]
                keep = messages[cutoff:]
        before_tokens = sum(estimate_message_tokens(m) for m in to_summarize)
        summary = await self._summarize_batch(to_summarize)
        after_tokens = estimate_tokens(summary)
        reclaimed = max(0, before_tokens - after_tokens)
        return keep, reclaimed, summary

    async def _summarize_batch(self, messages: list[AnyMessage]) -> str:
        if not messages or self._summarizer is None or not self._summarizer_model:
            return ""
        flattened = _flatten_for_summary(messages)
        prompt = (
            "Summarize the following agent conversation into a structured, terse digest. "
            "Preserve concrete facts (names, ids, decisions, errors, file paths, numbers). "
            "Use under "
            f"{self._cfg.summary_max_tokens} tokens. Output plain prose, no preamble."
        )
        try:
            resp = await self._summarizer.call(
                system=prompt,
                messages=[UserMessage(content=flattened)],
                tools=[],
                model=self._summarizer_model,
                max_tokens=self._cfg.summary_max_tokens,
                temperature=None,
            )
            return resp.assistant_message.content if isinstance(
                resp.assistant_message.content, str
            ) else ""
        except Exception:
            return ""


def _summary_message(summary: str) -> UserMessage:
    """Persisted compaction summary — part of stable history (cacheable),
    so it deliberately does NOT carry the volatile `system` metadata key."""
    return UserMessage(
        content=f"<summary>\n{summary}\n</summary>",
        metadata={"compaction": "summary"},
    )


# -------- pure compaction helpers --------
def _truncate_tool_results(
    messages: list[AnyMessage], threshold_tokens: int
) -> tuple[list[AnyMessage], int]:
    out: list[AnyMessage] = []
    reclaimed = 0
    for m in messages:
        if isinstance(m, ToolResultMessage):
            new_results: list[ToolResult] = []
            for tr in m.content:
                text = tr.as_text()
                if estimate_tokens(text) > threshold_tokens:
                    truncated = text[: threshold_tokens * 4]
                    note = (
                        f"\n[result truncated — original ≈ {estimate_tokens(text)} tokens, "
                        f"showing first {threshold_tokens}]"
                    )
                    reclaimed += estimate_tokens(text) - threshold_tokens
                    new_results.append(
                        ToolResult(
                            tool_use_id=tr.tool_use_id,
                            name=tr.name,
                            content=truncated + note,
                            is_error=tr.is_error,
                            duration_ms=tr.duration_ms,
                        )
                    )
                else:
                    new_results.append(tr)
            out.append(ToolResultMessage(content=new_results, metadata=m.metadata))
        else:
            out.append(m)
    return out, reclaimed


def _is_safe_cut_point(messages: list[AnyMessage], cutoff: int) -> bool:
    """A cut at `cutoff` (dropping messages[:cutoff]) is safe iff the kept
    list starts with a plain UserMessage. That single condition guarantees
    both provider invariants:
      - the conversation starts with a user turn (Anthropic requirement);
      - no assistant(tool_calls) → ToolResultMessage pair is split, because
        a ToolResultMessage always immediately follows its assistant message
        and is not a plain UserMessage.
    """
    cur = messages[cutoff]
    return isinstance(cur, UserMessage) and not isinstance(cur, ToolResultMessage)


def _safe_forward_cut(messages: list[AnyMessage], desired_cutoff: int) -> int:
    """Walk the cutoff backwards until it lands on a safe point (or 0)."""
    cutoff = min(desired_cutoff, len(messages) - 1)
    while cutoff > 0 and not _is_safe_cut_point(messages, cutoff):
        cutoff -= 1
    return cutoff


def _sliding_window(
    messages: list[AnyMessage], min_messages: int
) -> tuple[list[AnyMessage], int]:
    """Drop oldest messages, keeping at least `min_messages` most recent.

    The kept list must start with a plain UserMessage (see
    `_is_safe_cut_point`); if no safe split exists, everything is kept.
    """
    if len(messages) <= min_messages:
        return messages, 0

    cutoff = _safe_forward_cut(messages, len(messages) - min_messages)
    if cutoff <= 0:
        return messages, 0

    dropped = messages[:cutoff]
    reclaimed = sum(estimate_message_tokens(m) for m in dropped)
    return messages[cutoff:], reclaimed


def _flatten_for_summary(messages: list[AnyMessage]) -> str:
    lines: list[str] = []
    for m in messages:
        if isinstance(m, ToolResultMessage):
            for tr in m.content:
                tag = "tool-error" if tr.is_error else "tool"
                lines.append(f"[{tag}:{tr.name}] {tr.as_text()[:400]}")
        elif isinstance(m, UserMessage):
            content = m.content if isinstance(m.content, str) else json.dumps(m.content, default=str)
            lines.append(f"[user] {content}")
        elif isinstance(m, AssistantMessage):
            text = m.content if isinstance(m.content, str) else ""
            if text:
                lines.append(f"[assistant] {text}")
            for tc in m.tool_calls:
                lines.append(f"[assistant→tool] {tc.name}({json.dumps(tc.arguments, default=str)})")
    return "\n".join(lines)


def _relevant_context_block(facts: list[dict[str, Any]]) -> str:
    body = "\n".join(f"- {f.get('key', '')}: {f.get('value', '')}" for f in facts)
    return f"<relevant_context>\n{body}\n</relevant_context>"
