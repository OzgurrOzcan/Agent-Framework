"""Tests for tracing: spans, cost tracker, reporter."""
import pytest
from agent_framework.tracing.trace import (
    ErrorInfo,
    EventType,
    ExecutionTrace,
    SpanStatus,
)
from agent_framework.tracing.cost import CostTracker, ModelPricing, PricingTable
from agent_framework.tracing.reporter import TraceReporter
from agent_framework.core.message import TokenUsage


def test_trace_start_and_end_span():
    trace = ExecutionTrace(agent_id="test")
    with trace.start_span(EventType.TURN, name="turn1") as span:
        span.update(extra="data")
    spans = trace.all_spans()
    assert len(spans) == 1
    assert spans[0].event_type == EventType.TURN
    assert spans[0].status == SpanStatus.OK
    assert spans[0].duration_ms is not None
    assert spans[0].payload.get("extra") == "data"


def test_trace_error_span():
    trace = ExecutionTrace()
    try:
        with trace.start_span(EventType.TOOL_CALL, name="fail_tool"):
            raise ValueError("test error")
    except ValueError:
        pass
    spans = trace.all_spans()
    assert spans[0].status == SpanStatus.ERROR
    assert spans[0].error is not None
    assert spans[0].error.exception_type == "ValueError"


def test_trace_record_event():
    trace = ExecutionTrace()
    ev = trace.record_event(
        EventType.PERMISSION_VERDICT,
        name="write_file",
        status=SpanStatus.OK,
        payload={"verdict": "ALLOW"},
    )
    assert ev.event_type == EventType.PERMISSION_VERDICT
    assert ev.payload["verdict"] == "ALLOW"


def test_record_event_explicit_turn_number():
    trace = ExecutionTrace()
    ev = trace.record_event(EventType.LOOP_GUARD, name="stall", turn_number=5)
    assert ev.turn_number == 5


def test_record_event_defaults_to_current_turn():
    trace = ExecutionTrace()
    trace.bump_turn()
    trace.bump_turn()
    ev = trace.record_event(EventType.TURN, name="t")
    assert ev.turn_number == 2


def test_record_event_redacts_sensitive_payload_keys():
    trace = ExecutionTrace()
    ev = trace.record_event(
        EventType.TOOL_CALL,
        name="search",
        payload={"args": {"api_key": "sk-123", "query": "x"}},
    )
    assert ev.payload["args"]["api_key"] == "[REDACTED]"
    assert ev.payload["args"]["query"] == "x"


def test_redaction_nested_list_of_dicts_and_to_dict():
    trace = ExecutionTrace()
    ev = trace.record_event(
        EventType.TOOL_CALL,
        name="batch",
        payload={
            "items": [
                {"Password": "hunter2", "user": "alice"},
                {"nested": {"Authorization": "Bearer abc"}},
            ],
            "note": "keep me",
        },
    )
    # Redacted at record time...
    assert ev.payload["items"][0]["Password"] == "[REDACTED]"
    assert ev.payload["items"][0]["user"] == "alice"
    assert ev.payload["items"][1]["nested"]["Authorization"] == "[REDACTED]"
    assert ev.payload["note"] == "keep me"
    # ...so serialized output is clean too.
    d = ev.to_dict()
    assert d["payload"]["items"][0]["Password"] == "[REDACTED]"
    assert d["payload"]["items"][1]["nested"]["Authorization"] == "[REDACTED]"
    assert d["payload"]["note"] == "keep me"


def test_custom_redactor_override():
    # Identity redactor disables scrubbing entirely.
    trace = ExecutionTrace(redactor=lambda k, v: v)
    ev = trace.record_event(EventType.TOOL_CALL, payload={"api_key": "sk-123"})
    assert ev.payload["api_key"] == "sk-123"

    # Custom rule replaces the default one.
    masked = ExecutionTrace(redactor=lambda k, v: "***" if k == "query" else v)
    ev2 = masked.record_event(EventType.TOOL_CALL, payload={"query": "x", "other": 1})
    assert ev2.payload["query"] == "***"
    assert ev2.payload["other"] == 1


def test_redaction_keeps_numeric_token_counters():
    # Keys like ctx_tokens/reclaimed_tokens match the "token" pattern but
    # carry counters, not secrets — they must survive redaction.
    trace = ExecutionTrace()
    ev = trace.record_event(
        EventType.COMPACTION,
        payload={"reclaimed_tokens": 1200, "ctx_tokens": 900, "auth_token": "abc"},
    )
    assert ev.payload["reclaimed_tokens"] == 1200
    assert ev.payload["ctx_tokens"] == 900
    assert ev.payload["auth_token"] == "[REDACTED]"


def test_span_payload_and_update_redacted():
    trace = ExecutionTrace()
    with trace.start_span(
        EventType.TOOL_CALL, name="t", payload={"password": "hunter2"}
    ) as span:
        span.update(secret_token="abc", plain="ok")
    ev = trace.all_spans()[0]
    assert ev.payload["password"] == "[REDACTED]"
    assert ev.payload["secret_token"] == "[REDACTED]"
    assert ev.payload["plain"] == "ok"


def test_trace_failures_query():
    trace = ExecutionTrace()
    trace.record_event(EventType.TOOL_RESULT, name="ok_tool", status=SpanStatus.OK)
    trace.record_event(EventType.TOOL_RESULT, name="bad_tool", status=SpanStatus.ERROR)
    failures = trace.failures()
    assert len(failures) == 1
    assert failures[0].name == "bad_tool"


def test_trace_cost_summary():
    trace = ExecutionTrace()
    with trace.start_span(EventType.LLM_CALL, name="gpt-4o") as span:
        span.set_usage(
            TokenUsage(input_tokens=1000, output_tokens=500),
            cost_usd=0.01,
        )
        span.update(model="gpt-4o")
    summary = trace.cost_summary()
    assert summary["total_cost_usd"] == pytest.approx(0.01)
    assert summary["total_input_tokens"] == 1000


def test_trace_subscribe():
    received = []
    trace = ExecutionTrace()
    unsub = trace.subscribe(lambda ev: received.append(ev.name))
    trace.record_event(EventType.TURN, name="turn_1")
    trace.record_event(EventType.TOOL_CALL, name="tool_1")
    assert received == ["turn_1", "tool_1"]
    unsub()
    trace.record_event(EventType.TURN, name="turn_2")
    assert "turn_2" not in received


def test_trace_where_it_failed():
    trace = ExecutionTrace()
    with trace.start_span(EventType.TURN, name="t1"):
        trace.record_event(EventType.LLM_CALL, name="llm", status=SpanStatus.OK)
        trace.record_event(
            EventType.TOOL_RESULT, name="failing_tool",
            status=SpanStatus.ERROR,
            error=ErrorInfo(exception_type="RuntimeError", message="oops", traceback=""),
        )
    chain = trace.where_it_failed()
    assert chain is not None
    assert any(s.name == "failing_tool" for s in chain)


def test_cost_tracker():
    pricing = PricingTable({"gpt-4o": ModelPricing(input_per_mtok=2.5, output_per_mtok=10.0)})
    tracker = CostTracker(pricing=pricing)
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
    cost = tracker.record("gpt-4o", usage)
    assert cost == pytest.approx(7.5)  # 2.5 + 10*0.5
    total, _ = tracker.snapshot()
    assert total == pytest.approx(7.5)


def test_cost_tracker_prefix_match():
    pricing = PricingTable({"claude-opus-4-7": ModelPricing(input_per_mtok=15.0, output_per_mtok=75.0)})
    tracker = CostTracker(pricing=pricing)
    usage = TokenUsage(input_tokens=100_000, output_tokens=100_000)
    cost = tracker.record("claude-opus-4-7-20261201", usage)
    assert cost == pytest.approx(9.0)  # (15 + 75) * 0.1


def test_trace_reporter_basic():
    trace = ExecutionTrace(agent_id="agent1", session_id="sess1")
    trace.bump_turn()
    with trace.start_span(EventType.LLM_CALL, name="claude") as span:
        span.set_usage(TokenUsage(input_tokens=500, output_tokens=200), cost_usd=0.005)
        span.update(model="claude-opus-4-7")
    trace.record_event(EventType.TOOL_RESULT, name="web_search", status=SpanStatus.OK)
    trace.record_event(EventType.TOOL_RESULT, name="write_file", status=SpanStatus.ERROR,
                       error=ErrorInfo("PermissionError", "denied", "tb..."))

    report = TraceReporter.summarize(trace)
    assert report.session_id == "sess1"
    assert report.total_llm_calls == 1
    assert report.successful_tools == 1
    assert report.failed_tools == 1
    assert len(report.tool_failures) == 1
    assert report.tool_failures[0].tool_name == "write_file"
    text = report.to_text()
    assert "write_file" in text
    assert "PermissionError" in text
