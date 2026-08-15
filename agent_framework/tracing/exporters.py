"""Trace exporters — JSON, JSONL, SQLite, and optional OpenTelemetry.

Usage:
    exporter = JSONLExporter("runs/trace.jsonl")
    trace.subscribe(exporter.on_span)

Or flush a completed trace:
    SQLiteExporter("traces.db").export(trace)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from agent_framework.tracing.trace import ExecutionTrace, SpanStatus, TraceEvent


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    parent_span_id TEXT,
    event_type TEXT NOT NULL,
    status TEXT NOT NULL,
    name TEXT,
    timestamp TEXT NOT NULL,
    duration_ms INTEGER,
    agent_id TEXT,
    session_id TEXT,
    turn_number INTEGER,
    estimated_cost_usd REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    payload TEXT,
    error_type TEXT,
    error_message TEXT,
    error_traceback TEXT,
    tool_name TEXT,
    tool_args TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_session ON spans(session_id);
CREATE INDEX IF NOT EXISTS idx_spans_event_type ON spans(event_type);
CREATE INDEX IF NOT EXISTS idx_spans_status ON spans(status);
CREATE INDEX IF NOT EXISTS idx_spans_timestamp ON spans(timestamp);
"""


class JSONExporter:
    """Write the entire trace as a single JSON file when flushed."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def export(self, trace: ExecutionTrace) -> None:
        data = {
            "session_id": trace.session_id,
            "agent_id": trace.agent_id,
            "spans": [s.to_dict() for s in trace.timeline()],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


class JSONLExporter:
    """Append one JSON line per span as it completes — suitable for streaming."""

    def __init__(self, path: str | Path, *, append: bool = True) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._mode = "a" if append else "w"
        self._lock = threading.Lock()

    def on_span(self, event: TraceEvent) -> None:
        line = json.dumps(event.to_dict(), default=str)
        with self._lock:
            with self._path.open(self._mode, encoding="utf-8") as f:
                f.write(line + "\n")

    def export(self, trace: ExecutionTrace) -> None:
        with self._lock, self._path.open(self._mode, encoding="utf-8") as f:
            for s in trace.timeline():
                f.write(json.dumps(s.to_dict(), default=str) + "\n")


class SQLiteExporter:
    """Queryable history across sessions.

    Example query:
        SELECT * FROM spans
        WHERE event_type='tool_result' AND status='error'
        AND timestamp > datetime('now', '-7 days');
    """

    def __init__(self, path: str | Path = "agent_traces.db") -> None:
        self._path = str(path)
        self._lock = threading.RLock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SQLITE_SCHEMA)
            conn.commit()

    def on_span(self, event: TraceEvent) -> None:
        with self._lock, self._connect() as conn:
            _insert_span(conn, event)
            conn.commit()

    def export(self, trace: ExecutionTrace) -> None:
        spans = trace.timeline()
        with self._lock, self._connect() as conn:
            for s in spans:
                _insert_span(conn, s)
            conn.commit()


def _insert_span(conn: sqlite3.Connection, s: TraceEvent) -> None:
    usage = s.token_usage
    err = s.error
    conn.execute(
        """
        INSERT OR REPLACE INTO spans(
            span_id, parent_span_id, event_type, status, name, timestamp,
            duration_ms, agent_id, session_id, turn_number, estimated_cost_usd,
            input_tokens, output_tokens, payload,
            error_type, error_message, error_traceback, tool_name, tool_args
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            s.span_id,
            s.parent_span_id,
            s.event_type.value,
            s.status.value,
            s.name,
            s.timestamp.isoformat(),
            s.duration_ms,
            s.agent_id,
            s.session_id,
            s.turn_number,
            s.estimated_cost_usd,
            usage.input_tokens if usage else None,
            usage.output_tokens if usage else None,
            json.dumps(s.payload, default=str),
            err.exception_type if err else None,
            err.message if err else None,
            err.traceback if err else None,
            err.tool_name if err else None,
            json.dumps(err.tool_args, default=str) if (err and err.tool_args) else None,
        ),
    )


class OTelExporter:
    """Optional OpenTelemetry-compatible span exporter.

    Requires `opentelemetry-api` and `opentelemetry-sdk`.
    Wire it up with your existing OTel provider before using.
    """

    def __init__(self) -> None:
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.trace import Status, StatusCode
            self._tracer = otel_trace.get_tracer("agent_framework")
            self._StatusCode = StatusCode
            self._Status = Status
        except ImportError as e:
            raise ImportError(
                "opentelemetry-api and opentelemetry-sdk are required for OTelExporter. "
                "Install with `pip install opentelemetry-api opentelemetry-sdk`."
            ) from e

    def on_span(self, event: TraceEvent) -> None:
        with self._tracer.start_as_current_span(
            name=event.name or event.event_type.value,
            attributes={
                "session_id": event.session_id,
                "agent_id": event.agent_id,
                "turn": event.turn_number,
                "event_type": event.event_type.value,
                "status": event.status.value,
                **{f"payload.{k}": str(v) for k, v in (event.payload or {}).items()},
            },
        ) as span:
            if event.status in (SpanStatus.ERROR, SpanStatus.TIMEOUT):
                span.set_status(
                    self._Status(self._StatusCode.ERROR, event.error.message if event.error else "")
                )
            if event.estimated_cost_usd is not None:
                span.set_attribute("cost_usd", event.estimated_cost_usd)
            if event.token_usage is not None:
                span.set_attribute("input_tokens", event.token_usage.input_tokens)
                span.set_attribute("output_tokens", event.token_usage.output_tokens)
