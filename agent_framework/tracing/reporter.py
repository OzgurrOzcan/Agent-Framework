"""TraceReporter — human-readable post-mortems and failure summaries.

`TraceReporter.summarize(trace)` produces a structured report containing:
  - total turns, tools called, failures with counts and most-common error
  - where the loop terminated and why
  - total cost and token breakdown
  - chronological list of error spans with tracebacks
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent_framework.tracing.trace import (
    EventType,
    ExecutionTrace,
    SpanStatus,
    TraceEvent,
)


@dataclass
class ToolFailureSummary:
    tool_name: str
    total_calls: int
    failures: int
    most_common_error: str | None
    last_traceback: str | None


@dataclass
class TraceReport:
    session_id: str
    agent_id: str
    total_turns: int
    total_llm_calls: int
    total_tool_calls: int
    successful_tools: int
    failed_tools: int
    denied_tools: int
    timed_out_tools: int
    termination_reason: str
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    tool_failures: list[ToolFailureSummary]
    error_spans: list[TraceEvent]
    compaction_events: list[TraceEvent]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_text(self) -> str:
        lines: list[str] = [
            "=== Agent Session Report ===",
            f"Session:    {self.session_id}",
            f"Agent:      {self.agent_id}",
            f"Generated:  {self.generated_at.isoformat()}",
            "",
            "--- Execution ---",
            f"Turns:      {self.total_turns}",
            f"LLM calls:  {self.total_llm_calls}",
            f"Tool calls: {self.total_tool_calls}  "
            f"(ok={self.successful_tools}, "
            f"fail={self.failed_tools}, "
            f"denied={self.denied_tools}, "
            f"timeout={self.timed_out_tools})",
            f"Terminated: {self.termination_reason}",
            "",
            "--- Cost ---",
            f"Total cost:   ${self.total_cost_usd:.6f}",
            f"Input tokens: {self.total_input_tokens:,}",
            f"Output tokens:{self.total_output_tokens:,}",
        ]
        if self.compaction_events:
            lines.append("")
            lines.append(f"--- Compaction ({len(self.compaction_events)} event(s)) ---")
            for ev in self.compaction_events:
                strategy = ev.payload.get("strategy", ev.name)
                reclaimed = ev.payload.get("reclaimed_tokens", "?")
                lines.append(f"  turn={ev.turn_number}  {strategy}  reclaimed={reclaimed} tok")
        if self.tool_failures:
            lines.append("")
            lines.append("--- Tool Failures ---")
            for tf in self.tool_failures:
                lines.append(
                    f"  {tf.tool_name}: {tf.failures}/{tf.total_calls} failed"
                    + (f"  [{tf.most_common_error}]" if tf.most_common_error else "")
                )
                if tf.last_traceback:
                    for tb_line in tf.last_traceback.strip().split("\n")[-6:]:
                        lines.append(f"    {tb_line}")
        if self.error_spans:
            lines.append("")
            lines.append(f"--- Error Spans ({len(self.error_spans)}) ---")
            for ev in self.error_spans:
                lines.append(
                    f"  [{ev.event_type.value}] {ev.name}  "
                    f"status={ev.status.value}  "
                    f"turn={ev.turn_number}"
                )
                if ev.error:
                    lines.append(f"    {ev.error.exception_type}: {ev.error.message}")
                    if ev.error.traceback:
                        for tb_line in ev.error.traceback.strip().split("\n")[-4:]:
                            lines.append(f"    {tb_line}")
        lines.append("")
        lines.append("=== End of Report ===")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "generated_at": self.generated_at.isoformat(),
            "total_turns": self.total_turns,
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "successful_tools": self.successful_tools,
            "failed_tools": self.failed_tools,
            "denied_tools": self.denied_tools,
            "timed_out_tools": self.timed_out_tools,
            "termination_reason": self.termination_reason,
            "total_cost_usd": self.total_cost_usd,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "tool_failures": [
                {
                    "tool_name": tf.tool_name,
                    "total_calls": tf.total_calls,
                    "failures": tf.failures,
                    "most_common_error": tf.most_common_error,
                }
                for tf in self.tool_failures
            ],
            "error_count": len(self.error_spans),
        }


class TraceReporter:
    @staticmethod
    def summarize(trace: ExecutionTrace) -> TraceReport:
        spans = trace.all_spans()

        total_turns = trace.current_turn
        llm_calls = [s for s in spans if s.event_type == EventType.LLM_CALL]
        tool_calls = [s for s in spans if s.event_type == EventType.TOOL_CALL]
        tool_results = [s for s in spans if s.event_type == EventType.TOOL_RESULT]
        error_spans = [s for s in spans if s.status != SpanStatus.OK]
        compaction_events = [s for s in spans if s.event_type == EventType.COMPACTION]

        ok = sum(1 for s in tool_results if s.status == SpanStatus.OK)
        fail = sum(1 for s in tool_results if s.status == SpanStatus.ERROR)
        denied = sum(1 for s in tool_results if s.status == SpanStatus.DENIED)
        timed = sum(1 for s in tool_results if s.status == SpanStatus.TIMEOUT)

        # Termination reason.
        guard_events = [s for s in spans if s.event_type == EventType.LOOP_GUARD]
        if guard_events:
            last = max(guard_events, key=lambda s: s.timestamp)
            termination = last.name or "loop_guard"
        elif not tool_calls:
            termination = "natural (no tools)"
        else:
            termination = "natural (end_turn)"

        # Cost.
        cost_summary = trace.cost_summary()

        # Per-tool failure breakdown.
        tool_call_counts: dict[str, int] = defaultdict(int)
        tool_fail_errors: dict[str, list[str]] = defaultdict(list)
        tool_last_tb: dict[str, str] = {}
        for s in tool_results:
            name = s.name or s.payload.get("tool_name", "?")
            tool_call_counts[name] += 1
            if s.status not in (SpanStatus.OK,):
                if s.error:
                    tool_fail_errors[name].append(s.error.exception_type)
                    tool_last_tb[name] = s.error.traceback
                else:
                    tool_fail_errors[name].append(s.status.value)

        tool_failures: list[ToolFailureSummary] = []
        for name, failures in tool_fail_errors.items():
            most_common = max(set(failures), key=failures.count) if failures else None
            tool_failures.append(
                ToolFailureSummary(
                    tool_name=name,
                    total_calls=tool_call_counts[name],
                    failures=len(failures),
                    most_common_error=most_common,
                    last_traceback=tool_last_tb.get(name),
                )
            )

        return TraceReport(
            session_id=trace.session_id,
            agent_id=trace.agent_id,
            total_turns=total_turns,
            total_llm_calls=len(llm_calls),
            total_tool_calls=len(tool_results),
            successful_tools=ok,
            failed_tools=fail,
            denied_tools=denied,
            timed_out_tools=timed,
            termination_reason=termination,
            total_cost_usd=cost_summary.get("total_cost_usd", 0.0),
            total_input_tokens=cost_summary.get("total_input_tokens", 0),
            total_output_tokens=cost_summary.get("total_output_tokens", 0),
            tool_failures=tool_failures,
            error_spans=error_spans,
            compaction_events=compaction_events,
        )
