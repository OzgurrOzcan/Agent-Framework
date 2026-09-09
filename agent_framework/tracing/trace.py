"""ExecutionTrace — append-only span tree per session.

Every unit of work is a span with explicit status (`ok | error | denied |
timeout | stalled | cancelled`). Spans nest via `parent_span_id`. The
session forms the root.

Design principle: failures are first-class queryable data, not strings
buried in logs. `where_it_failed()` walks back to the first error and
returns its ancestor chain.
"""

from __future__ import annotations

import contextvars
import enum
import os
import re
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from agent_framework.core.message import TokenUsage


class EventType(str, enum.Enum):
    # How the agent was built: one per agent, carrying its config snapshot.
    SESSION = "session"
    # How one run was shaped: one per run()/run_stream()/complete(), carrying
    # the effective LoopPolicy. Config defaults and per-call overrides can
    # disagree, so the run is the only place the effective values are known.
    RUN = "run"
    TURN = "turn"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PERMISSION_VERDICT = "permission_verdict"
    INTERRUPT = "interrupt"
    COMPACTION = "compaction"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    LOOP_GUARD = "loop_guard"
    ERROR = "error"


class SpanStatus(str, enum.Enum):
    OK = "ok"
    ERROR = "error"
    DENIED = "denied"
    TIMEOUT = "timeout"
    STALLED = "stalled"
    CANCELLED = "cancelled"


@dataclass
class ErrorInfo:
    exception_type: str
    message: str
    traceback: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    retry_attempt: int = 0

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        retry_attempt: int = 0,
    ) -> "ErrorInfo":
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return cls(
            exception_type=type(exc).__name__,
            message=str(exc),
            traceback=tb,
            tool_name=tool_name,
            tool_args=tool_args,
            retry_attempt=retry_attempt,
        )


@dataclass
class TraceEvent:
    span_id: str
    parent_span_id: str | None
    event_type: EventType
    status: SpanStatus
    timestamp: datetime
    duration_ms: int | None = None
    agent_id: str = ""
    session_id: str = ""
    turn_number: int = 0
    token_usage: TokenUsage | None = None
    estimated_cost_usd: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    error: ErrorInfo | None = None
    name: str = ""  # human-readable label (e.g. tool name)
    _ended: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "event_type": self.event_type.value,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "estimated_cost_usd": self.estimated_cost_usd,
            "name": self.name,
            "payload": _scrub(self.payload),
        }
        if self.token_usage is not None:
            d["token_usage"] = asdict(self.token_usage)
        if self.error is not None:
            d["error"] = asdict(self.error)
        return d


def _scrub(obj: Any) -> Any:
    """Make payloads JSON-serializable (best effort)."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {str(k): _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(x) for x in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, enum.Enum):
        return obj.value
    return str(obj)


# ---------------------------------------------------------------------------
# Payload secret scrubbing
# ---------------------------------------------------------------------------

_SENSITIVE_KEY_RE = re.compile(
    r"password|passwd|token|secret|api_key|apikey|authorization|credential",
    re.IGNORECASE,
)


def _default_redactor(key: str, value: Any) -> Any:
    """Replace values whose key looks sensitive with ``"[REDACTED]"``.

    Plain numeric/bool values are never redacted: counters such as
    ``ctx_tokens`` or ``reclaimed_tokens`` match the key pattern but
    cannot be secrets.
    """
    if isinstance(value, (bool, int, float)):
        return value
    if _SENSITIVE_KEY_RE.search(key):
        return "[REDACTED]"
    return value


def redact_payload(
    payload: dict[str, Any],
    redactor: Callable[[str, Any], Any] | None = None,
) -> dict[str, Any]:
    """Return a copy of *payload* with sensitive values redacted.

    *redactor* is called as ``redactor(key, value)`` for every key/value pair,
    including nested dicts and dicts inside lists; whatever it returns is
    written in place of the original value. The default redactor replaces
    values whose key contains ``password``, ``passwd``, ``token``, ``secret``,
    ``api_key``, ``apikey``, ``authorization`` or ``credential``
    (case-insensitive) with ``"[REDACTED]"``; plain numeric/bool values
    (token counters etc.) are left intact. Pass ``redactor=lambda k, v: v``
    to disable redaction entirely.
    """
    return _redact_value(payload, redactor or _default_redactor)


def _redact_value(value: Any, redactor: Callable[[str, Any], Any]) -> Any:
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for k, v in value.items():
            new_v = redactor(str(k), v)
            if new_v is v and isinstance(v, (dict, list, tuple)):
                new_v = _redact_value(v, redactor)
            out[k] = new_v
        return out
    if isinstance(value, (list, tuple)):
        return [_redact_value(x, redactor) for x in value]
    return value


# Active span context for nesting (works across threads via contextvars).
_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agent_framework_current_span", default=None
)


@dataclass
class SpanContext:
    """Handle returned by `ExecutionTrace.start_span()`. Use as a context manager."""

    trace: "ExecutionTrace"
    event: TraceEvent
    _token: Any = None
    _start_ns: int = 0

    def __enter__(self) -> "SpanContext":
        self._token = _current_span.set(self.event.span_id)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None and not self.event._ended:
            self.set_error(exc)
        self.end()
        if self._token is not None:
            try:
                _current_span.reset(self._token)
            except ValueError:
                pass

    # --- mutation helpers (always go through the trace so subscribers fire) ---
    def update(self, **payload: Any) -> None:
        self.event.payload.update(self.trace._redact(payload))

    def set_status(self, status: SpanStatus) -> None:
        self.event.status = status

    def set_usage(self, usage: TokenUsage, cost_usd: float | None = None) -> None:
        self.event.token_usage = usage
        if cost_usd is not None:
            self.event.estimated_cost_usd = cost_usd

    def set_error(
        self,
        exc: BaseException | ErrorInfo,
        *,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        retry_attempt: int = 0,
        status: SpanStatus = SpanStatus.ERROR,
    ) -> None:
        if isinstance(exc, ErrorInfo):
            self.event.error = exc
        else:
            self.event.error = ErrorInfo.from_exception(
                exc, tool_name=tool_name, tool_args=tool_args, retry_attempt=retry_attempt
            )
        self.event.status = status

    def end(self) -> None:
        if self.event._ended:
            return
        self.event._ended = True
        elapsed_ms = (_now_ns() - self._start_ns) // 1_000_000
        self.event.duration_ms = max(0, int(elapsed_ms))
        self.trace._notify(self.event)


def _now_ns() -> int:
    import time
    return time.monotonic_ns()


class ExecutionTrace:
    """Append-only span tree for one session.

    Thread-safe. Spans are immutable once `end()` is called (status/payload
    can still be inspected; mutation methods are no-ops after end).

    Payloads are scrubbed before they are stored: values whose key looks
    sensitive (password, token, secret, api_key, ...) are replaced with
    ``"[REDACTED]"``, so `to_dict()` and every exporter see clean data.
    Pass a custom ``redactor`` (called as ``redactor(key, value)``; the
    return value is what gets stored) to change this, or
    ``redactor=lambda k, v: v`` to disable redaction.
    """

    def __init__(
        self,
        *,
        agent_id: str = "",
        session_id: str | None = None,
        redactor: Callable[[str, Any], Any] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:16]}"
        self._spans: list[TraceEvent] = []
        self._subscribers: list[Callable[[TraceEvent], None]] = []
        self._lock = threading.RLock()
        self._current_turn: int = 0
        self._redactor = redactor

    def _redact(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply this trace's redactor (or the default) to *payload*."""
        return redact_payload(payload, self._redactor)

    # ---------- span creation ----------
    def start_span(
        self,
        event_type: EventType,
        *,
        name: str = "",
        parent_span_id: str | None = None,
        turn_number: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> SpanContext:
        parent = parent_span_id or _current_span.get()
        span = TraceEvent(
            span_id=f"sp_{uuid.uuid4().hex[:16]}",
            parent_span_id=parent,
            event_type=event_type,
            status=SpanStatus.OK,
            timestamp=datetime.now(timezone.utc),
            agent_id=self.agent_id,
            session_id=self.session_id,
            turn_number=turn_number if turn_number is not None else self._current_turn,
            payload=self._redact(payload or {}),
            name=name,
        )
        with self._lock:
            self._spans.append(span)
        return SpanContext(trace=self, event=span, _start_ns=_now_ns())

    def record_event(
        self,
        event_type: EventType,
        *,
        name: str = "",
        status: SpanStatus = SpanStatus.OK,
        parent_span_id: str | None = None,
        turn_number: int | None = None,
        payload: dict[str, Any] | None = None,
        error: ErrorInfo | None = None,
        token_usage: TokenUsage | None = None,
        estimated_cost_usd: float | None = None,
    ) -> TraceEvent:
        """Record a zero-duration event (no nested work).

        `turn_number` defaults to the trace's current turn when omitted.
        """
        parent = parent_span_id or _current_span.get()
        ev = TraceEvent(
            span_id=f"sp_{uuid.uuid4().hex[:16]}",
            parent_span_id=parent,
            event_type=event_type,
            status=status,
            timestamp=datetime.now(timezone.utc),
            duration_ms=0,
            agent_id=self.agent_id,
            session_id=self.session_id,
            turn_number=turn_number if turn_number is not None else self._current_turn,
            payload=self._redact(payload or {}),
            error=error,
            name=name,
            token_usage=token_usage,
            estimated_cost_usd=estimated_cost_usd,
            _ended=True,
        )
        with self._lock:
            self._spans.append(ev)
        self._notify(ev)
        return ev

    def bump_turn(self) -> int:
        with self._lock:
            self._current_turn += 1
            return self._current_turn

    @property
    def current_turn(self) -> int:
        return self._current_turn

    # ---------- subscription / streaming ----------
    def subscribe(self, callback: Callable[[TraceEvent], None]) -> Callable[[], None]:
        """Attach a callback fired whenever a span ends. Returns an unsubscribe fn."""
        with self._lock:
            self._subscribers.append(callback)

        def _unsub() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(callback)
                except ValueError:
                    pass

        return _unsub

    def _notify(self, ev: TraceEvent) -> None:
        # Capture subscribers under lock, fire outside.
        with self._lock:
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(ev)
            except Exception:
                # Subscriber failures must never break the loop.
                pass
        if os.environ.get("AGENT_DEBUG") == "1":
            _debug_print(ev)

    # ---------- query API ----------
    def __iter__(self) -> Iterator[TraceEvent]:
        with self._lock:
            return iter(list(self._spans))

    def all_spans(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._spans)

    def failures(self) -> list[TraceEvent]:
        return [s for s in self.all_spans() if s.status != SpanStatus.OK]

    def tool_stats(self) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for s in self.all_spans():
            if s.event_type != EventType.TOOL_RESULT:
                continue
            name = s.name or s.payload.get("tool_name", "?")
            row = stats.setdefault(
                name,
                {"calls": 0, "ok": 0, "errors": 0, "denied": 0, "timeouts": 0, "latencies": []},
            )
            row["calls"] += 1
            if s.status == SpanStatus.OK:
                row["ok"] += 1
            elif s.status == SpanStatus.DENIED:
                row["denied"] += 1
            elif s.status == SpanStatus.TIMEOUT:
                row["timeouts"] += 1
            else:
                row["errors"] += 1
            if s.duration_ms is not None:
                row["latencies"].append(s.duration_ms)
        for row in stats.values():
            lats = sorted(row.pop("latencies"))
            row["p50_ms"] = lats[len(lats) // 2] if lats else None
            row["p95_ms"] = lats[int(len(lats) * 0.95)] if lats else None
            row["success_rate"] = row["ok"] / row["calls"] if row["calls"] else 0.0
        return stats

    def timeline(self) -> list[TraceEvent]:
        return sorted(self.all_spans(), key=lambda s: s.timestamp)

    def cost_summary(self) -> dict[str, Any]:
        total_cost = 0.0
        total_in = 0
        total_out = 0
        per_model: dict[str, dict[str, Any]] = {}
        per_turn: dict[int, float] = {}
        for s in self.all_spans():
            if s.event_type != EventType.LLM_CALL:
                continue
            if s.estimated_cost_usd is not None:
                total_cost += s.estimated_cost_usd
                per_turn[s.turn_number] = per_turn.get(s.turn_number, 0.0) + s.estimated_cost_usd
            if s.token_usage is not None:
                total_in += s.token_usage.input_tokens
                total_out += s.token_usage.output_tokens
                model = s.payload.get("model", "?")
                row = per_model.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
                row["input_tokens"] += s.token_usage.input_tokens
                row["output_tokens"] += s.token_usage.output_tokens
                row["cost_usd"] += s.estimated_cost_usd or 0.0
        return {
            "total_cost_usd": total_cost,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "per_model": per_model,
            "per_turn_cost_usd": per_turn,
        }

    def where_it_failed(self) -> list[TraceEvent] | None:
        """First failure span + its ancestor chain (root → failure)."""
        failures = self.failures()
        if not failures:
            return None
        target = min(failures, key=lambda s: s.timestamp)
        chain: list[TraceEvent] = [target]
        by_id = {s.span_id: s for s in self.all_spans()}
        cur = target
        while cur.parent_span_id:
            parent = by_id.get(cur.parent_span_id)
            if parent is None:
                break
            chain.append(parent)
            cur = parent
        return list(reversed(chain))


def _debug_print(ev: TraceEvent) -> None:
    """AGENT_DEBUG=1 → color-coded stdout pretty-print on span end."""
    try:
        from rich.console import Console
        from rich.text import Text
        c = Console(stderr=True)
        color = {
            SpanStatus.OK: "green",
            SpanStatus.ERROR: "red bold",
            SpanStatus.DENIED: "yellow",
            SpanStatus.TIMEOUT: "red",
            SpanStatus.STALLED: "yellow",
            SpanStatus.CANCELLED: "magenta",
        }.get(ev.status, "white")
        line = Text()
        line.append(f"[{ev.event_type.value}] ", style="dim")
        line.append(ev.name or "", style="bold")
        line.append(f"  ({ev.status.value}", style=color)
        if ev.duration_ms is not None:
            line.append(f", {ev.duration_ms}ms", style="dim")
        line.append(")", style=color)
        c.print(line)
        if ev.error is not None:
            c.print(ev.error.traceback, style="red")
    except Exception:
        # Fall back to stderr.
        print(
            f"[{ev.event_type.value}] {ev.name} ({ev.status.value}, {ev.duration_ms}ms)",
            file=sys.stderr,
        )
        if ev.error is not None:
            print(ev.error.traceback, file=sys.stderr)
