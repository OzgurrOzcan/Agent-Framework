# Observability

Tracing is not an add-on — the loop, executor, permission resolver, and context manager
all write spans. Values are redacted before they are stored, so keys and tokens don't
leak into your trace store.

```python
from agent_framework.tracing import TraceReporter, SQLiteExporter

result = await agent.run("...")
trace = agent.get_trace()

print(TraceReporter.summarize(trace).to_text())   # human-readable run summary
print(trace.cost_summary())                       # tokens + USD, per model
print(trace.tool_stats())                         # calls, errors, per tool
print(trace.where_it_failed())                    # the failing span chain

SQLiteExporter("traces.db").export(trace)         # or JSON / JSONL / OTel
```

Span types: `session`, `run`, `turn`, `llm_call`, `tool_call`, `tool_result`,
`permission_verdict`, `interrupt`, `compaction`, `memory_read`, `memory_write`,
`loop_guard`, `error`.

Every `llm_call` records what it sent — messages, the token estimate, cleared results,
cache hits (see [Context and memory](context-and-memory.md)). `compaction` events record
each `trim`, each server-side `native_clear`, each long-term `summary` (with its tokens
and cost) and each `context_filter` change or repair; `memory_read` / `memory_write`
record facts, the summary and every session load and save. A prompt stopped before it
was sent is a `loop_guard:context_exceeded`.

The trace explains its own behaviour, because settings are recorded alongside the spans
they caused. `session` carries the config the agent was built with — the *resolved*
context window and context limits, the memory settings and session id, the sampling
parameters, and whether the model was priced at all (an unpriced model records `$0.00`
and silently disables `max_budget_usd`, which is otherwise indistinguishable from a
genuinely cheap run).
`run` carries the effective policy of one run, since `run(max_turns=3)` overrides the
config per call: without it, a trace cannot separate "the caller capped this run" from
"the deployed config changed", nor tell a `max_tool_rounds` cap or `complete()` apart
from a permission wall — all three look like `tool_use` with no `tool_call` span.

Like every other span, `session` records settings, not content: the system prompt is
fingerprinted rather than copied (enough to tell whether two runs used the same prompt),
and `base_url` is stripped of userinfo and query strings, where credentials hide.

`where_it_failed()` returns the chain from the root down to the *cause* — for a run
stopped by its turn limit, `[run, loop_guard:max_turns]` — skipping spans that failed only
because a child did. Cancelling a run, or closing its stream early, marks its spans
`cancelled` rather than `error`, and the agent stays usable.

For a live UI, subscribe once — that single hook is all a renderer needs:

```python
unsubscribe = trace.subscribe(lambda event: print(event.name, event.duration_ms))
```
