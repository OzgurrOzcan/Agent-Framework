<p align="center">
  <img src="docs/assets/banner.png"
       alt="Agent Framework — build AI agents once, run them on any provider"
       width="100%">
</p>

<p align="center">
  A provider-agnostic Python framework for production LLM agents — with a tool-calling
  loop you can read, permissions you can audit, and a trace that tells you where it broke.
</p>

<p align="center">
  <a href="https://github.com/OzgurrOzcan/Agent-Framework/actions/workflows/ci.yml">
    <img src="https://github.com/OzgurrOzcan/Agent-Framework/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue" alt="Python 3.11 | 3.12">
  <img src="https://img.shields.io/badge/tests-202%20passing-brightgreen" alt="202 tests passing">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/typed-py.typed-informational" alt="Typed">
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#what-you-get">Features</a> ·
  <a href="#why-not-langchain-or-crewai">Why not LangChain?</a> ·
  <a href="#api-surface">API</a> ·
  <a href="#examples">Examples</a>
</p>

---

## The problem

You write your tool schemas against one provider's wire format, wire up MCP, handle its
streaming events, and ship. Then a cheaper model lands somewhere else — or a client
demands their own vendor, or a local model has to sit behind an OpenAI-compatible
endpoint. Now you rewrite the schemas, the message normalization, the tool-call plumbing
and the streaming handler. New providers keep arriving, so this is not a one-time tax —
it is a tax on every agent you build.

And the parts that actually decide whether an agent is *production-ready* are the ones
nobody wants to write twice: what stops a runaway loop, which tool may touch what, what
the last run cost, and — when it breaks at 3am — which turn it broke on.

**Agent Framework** owns exactly those parts, in ~6,500 lines of plain Python you can
read in an afternoon. Switching provider is one line:

```diff
- provider=ProviderKind.ANTHROPIC, model="claude-opus-5"
+ provider=ProviderKind.OPENAI,    model="gpt-5"
```

Nothing else changes — not your tools, not your permissions, not your traces.

---

## What you get

| | |
|---|---|
| **One loop, any provider** | Anthropic and OpenAI adapters behind a normalized message model; any OpenAI-compatible endpoint via `base_url`. Golden tests pin both wire formats. |
| **Four verbs, not forty** | `run()`, `complete()`, `run_stream()`, `to_json()`. Run shape comes from parameters, not from a new class per pattern. |
| **Can't run away** | Four independent stop layers: natural · `max_turns` · stall detection · USD budget guard. |
| **Deny-first permissions** | Four categories per tool, overridable by glob and argument-regex rules without touching tool code. Sub-agents can never widen a parent's permissions. |
| **Human in the loop** | `ASK` tools route to a pluggable renderer — terminal, callback (web UI), or headless (CI) — and `ask_user` lets the agent ask mid-run. |
| **MCP, first class** | Any stdio or HTTP server in three lines. Tools are namespaced `server__tool` and pass through the same executor, permissions and tracing as native ones. |
| **Context that holds** | Token budgeting, tool-result truncation, sliding window, LLM compaction with a persistent summary, Anthropic prompt caching. |
| **Traces, not log lines** | A span tree with automatic secret redaction: `where_it_failed()`, `tool_stats()`, `cost_summary()`, and JSON / JSONL / SQLite / OTel export. |
| **Real cost tracking** | Per-model pricing including cache-read and cache-write tiers, wired straight into the budget guard. |
| **Safe by construction** | Arguments validated against the schema before your function runs. Per-tool timeouts, retries only for `idempotent` tools, and the `tool_use ↔ tool_result` invariant preserved even on cancellation. |

---

## Why not LangChain or CrewAI?

Those frameworks are excellent for getting to a demo quickly, and for a prototype they
are often the right call. The trouble starts when the prototype has to run in
production and something goes wrong — because in production, something always goes wrong.

**1. Abstractions you can't see through.** [Octomind ran LangChain in production for
over a year before removing it](https://octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents):
as requirements got more sophisticated they had to reach into LangChain's internals to
change low-level behaviour, and because the framework deliberately abstracts those
details away, that was often not possible. The pattern is
[well documented](https://www.mindstudio.ai/blog/llm-frameworks-replaced-by-agent-sdks) —
an error thrown three levels deep in a chain tells you almost nothing about the real
failure, and it is hard to see what was even sent to the model. LCEL's pipe operator
routes execution through internal `invoke()` machinery with no natural place to insert
logging, so engineers end up
[adding print statements inside framework source](https://logic.inc/resources/langchain-production-issues)
just to follow execution.

**2. Multi-agent opacity.** [CrewAI offers no native way](https://atlan.com/know/ai-agent/what-is-crewai/)
to isolate whether a failure came from the model, the prompt, or the data an agent read.
When a three-agent crew produces wrong output you need to know which agent deviated,
which tool call returned unexpected data, and how many tokens each step consumed — and
the open-source version gives limited built-in observability, so teams fall back to
print statements and third-party tools.

**3. Cost you find out about later.** Each agent in a crew makes its own model calls.
Practitioners report hierarchical crews making 50–100+ calls for a single task, with
bills 5–10× a single-agent approach — and default logging too noisy to diagnose it.

**4. Abstractions age badly in a field this young.** New concepts land monthly, which
makes it genuinely hard to design abstractions that survive. The cost shows up as
[API churn](https://www.upgrad.com/blog/why-are-developers-quitting-langchain/): minor
version bumps breaking production, and a package split across `langchain`,
`langchain-core`, `langchain-community` and per-integration packages that turns
migrations into projects.

### What this framework does instead

| | LangChain / CrewAI | Agent Framework |
|---|---|---|
| **Debuggability** | Errors surface deep inside framework internals | Span tree with turn, tool, args, tokens, cost, and `where_it_failed()` |
| **Cost visibility** | Bolt-on, usually a third-party service | Built in — per model, per turn, and a hard USD budget guard |
| **Runaway protection** | Mostly your problem | Four independent termination layers |
| **Provider switching** | Integration package per provider | One enum value |
| **Dependency weight** | Large transitive tree | 5 runtime deps: `anthropic`, `openai`, `rich`, `jsonschema`, `pydantic` |
| **Reading the source** | Large, layered, DSL-heavy | ~6,500 lines of plain Python. No DSL, no pipe operator, no magic |

**And it is yours to change.** This is MIT-licensed and small enough to actually read in
an afternoon. Clone it, add the provider your client insists on, swap the permission
engine for your company's policy service, replace the compaction strategy, teach the
cost tracker your negotiated rates. There is no plugin API to learn and no abstraction
to fight — the extension point is the source code, and it is written to be read. When
your requirements stop matching the framework's assumptions, you edit the framework
instead of working around it.

> **When you probably shouldn't use this.** If you want a large ecosystem of prebuilt
> integrations, retrievers, and vector-store adapters, LangChain has years of head start
> and this project does not compete there. If you need a visual builder or hosted
> orchestration, look elsewhere. This is for teams who want a small, legible core they
> own and can debug.

---

## Install

```bash
pip install -e ".[dev]"      # framework + test tooling
pip install -e ".[mcp]"      # optional: MCP server support
```

Requires Python 3.11+. API keys are read from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`,
or passed explicitly via `ProviderConfig(api_key=...)`.

```bash
pytest agent_framework/tests/ -q      # 202 tests, fully offline against mocked providers
```

No key is required to run the test suite or most of the examples.

---

## Quickstart

```python
import asyncio
from agent_framework import (
    BaseAgent, AgentConfig, ProviderConfig, ProviderKind,
    tool, PermissionCategory,
)

@tool(description="Add two numbers", permission=PermissionCategory.AUTO)
async def add(a: int, b: int) -> int:
    return a + b

async def main():
    agent = BaseAgent.create(
        AgentConfig(
            provider=ProviderConfig(
                provider=ProviderKind.ANTHROPIC,   # or ProviderKind.OPENAI
                model="claude-opus-5",
            ),
            system_prompt="You are a helpful assistant.",
            max_turns=6,
            max_budget_usd=0.50,                   # hard stop, enforced by the loop
        ),
        tools=[add],
    )

    result = await agent.run("What is 2 + 3?")
    print(result.final_message)
    print(agent.cost.snapshot())                   # (usd, TokenUsage)

asyncio.run(main())
```

`run()` never raises for operational failures. A budget stop, a turn-limit stop, or a
provider error comes back as `result.stop_reason == StopReason.ERROR` with the exception
on `result.error`. Only `asyncio.CancelledError` propagates.

---

## Core concepts

### Tools

`@tool` turns a plain function into a tool. The JSON Schema is generated from type
hints; parameter descriptions come from `Annotated[...]` metadata or a Google-style
docstring `Args:` section. Sync and async functions both work — sync tools are run in a
thread automatically.

```python
from typing import Annotated

@tool(
    description="Look up current stock for a SKU.",
    permission=PermissionCategory.AUTO,
    timeout_seconds=10,
    idempotent=True,          # only idempotent tools are retried
    max_retries=2,
    tags=("catalog",),
)
async def get_stock(sku: Annotated[str, "Product SKU, e.g. LAP-14"]) -> dict:
    ...
```

Arguments are validated against that schema before your function runs. A malformed call
becomes an error `ToolResult` the model can see and correct — it never reaches your code.

### Permissions

Every tool has exactly one category. Rules layer on top of it in a strict deny-first
waterfall, so you can tighten a third-party or MCP tool without editing it.

```python
from agent_framework import PermissionConfig, DenyRule, AllowRule, TerminalRenderer

permissions = PermissionConfig(
    deny=[DenyRule(tool="shell", arg_pattern=r"rm\s+-rf")],
    allow=[AllowRule(tool="read_*")],           # glob on tool names
    on_ask=TerminalRenderer(),                  # who answers ASK prompts
)

agent = BaseAgent.create(config, tools=[...], permissions=permissions)
```

| Category | Behaviour |
|---|---|
| `AUTO` | Runs without asking |
| `ASK` | Prompts the renderer every time |
| `ASK_ONCE` | Prompts once per unique tool + argument signature, then remembers |
| `DENY` | Never runs; the model gets structured feedback explaining why |

Sub-agents pass `parent_permissions=` and are rejected at construction if they try to
remove a deny rule or introduce an allow rule the parent doesn't have.

### MCP

```python
from agent_framework import (
    BaseAgent, ExecutionTrace, MCPServerRegistry, MCPConnectionConfig,
)

trace = ExecutionTrace()
registry = MCPServerRegistry(trace=trace).add(
    "fs",
    MCPConnectionConfig(
        transport="stdio",
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/data"],
    ),
)

agent = await BaseAgent.acreate(config, mcp=registry, trace=trace)   # async factory for MCP
try:
    result = await agent.run("Summarise the newest file in /data.")
finally:
    await registry.close_all()                          # connections are yours to close
```

Tools arrive namespaced (`fs__read_text_file`), so two servers exposing the same tool
name never collide. Use `acreate()` from async code — `create(mcp=...)` raises inside a
running event loop rather than silently discovering zero tools. Always pass a `trace`:
discovery swallows connection errors and reports them as trace events.

### Observability

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

Span types: `session`, `turn`, `llm_call`, `tool_call`, `tool_result`,
`permission_verdict`, `interrupt`, `compaction`, `memory_read`, `memory_write`,
`loop_guard`, `error`.

For a live UI, subscribe once — that single hook is all `examples/_ui.py` uses to render
its terminal view:

```python
unsubscribe = trace.subscribe(lambda event: print(event.name, event.duration_ms))
```

### Structured output

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    number: str
    total: float

invoice = await agent.to_json("Extract the invoice.", schema=Invoice)
```

Uses each provider's native mechanism (Anthropic `output_config.format`, OpenAI strict
`json_schema`), with a prompt-based fallback for adapters that don't override it. The
result is re-validated client-side against your original schema — including constraints
provider-side decoders can't enforce — and the call **never touches the agent's
conversation history**.

### Streaming

```python
from agent_framework.providers.base import StreamChunkType

async for chunk in agent.run_stream("Explain the plan."):
    if chunk.type is StreamChunkType.TEXT_DELTA:
        print(chunk.payload["text"], end="", flush=True)
```

Chunk types: `text_delta`, `tool_use_start`, `tool_use_delta`, `tool_use_complete`,
`reasoning_delta`, `usage`, `stop`, `error`. Tools execute mid-stream under the same
permission engine as `run()`.

### Memory and checkpoints

Short-term memory is the working message buffer. Long-term memory is optional, backed by
SQLite or in-memory, and surfaces matching facts into a `<relevant_context>` block during
context assembly.

```python
cid = agent.checkpoint()          # full fidelity: tool calls, results, reasoning, cost
await agent.run("try something risky")
agent.restore(cid)                # exact rollback
```

---

## API surface

```python
# construction
agent = BaseAgent.create(config, tools=[...], permissions=..., memory=..., trace=..., pricing=...)
agent = await BaseAgent.acreate(config, mcp=registry, tools=[...])      # required for MCP

# four verbs — all async
result = await agent.run(prompt, max_turns=..., max_tool_rounds=..., tool_choice=...)
result = await agent.complete(prompt)                  # one call; tools advertised, NOT executed
async for chunk in agent.run_stream(prompt): ...
obj    = await agent.to_json(prompt, schema=PydanticModel)   # or a jsonschema dict

# result
result.final_message   # str
result.stop_reason     # StopReason
result.state.turn      # int
result.tool_calls      # list[ToolCallRequest] — mainly for complete()
result.error           # Exception | None

# inspection
agent.get_history() · agent.clear_history() · agent.get_trace()
agent.tools · agent.memory · agent.permissions · agent.cost
agent.checkpoint() · agent.restore(cid)
```

`max_tool_rounds=1` gives the classic "one round of tools, then synthesise" pattern.
`tool_choice` constrains only the first model call; later turns fall back to auto.

---

## Architecture

```mermaid
flowchart TD
    U[Your code] --> A[BaseAgent]
    A --> L[AgentLoop]

    L --> CM[ContextManager<br/>compaction · caching · token budget]
    L --> P[PermissionResolver<br/>deny-first waterfall]
    L --> E[ToolExecutor<br/>validate · timeout · retry]
    L --> AD[Provider adapter]

    P -.ASK.-> R[UIRenderer<br/>terminal · callback · headless]
    E --> NT[Native @tool]
    E --> MT[MCP tools<br/>server__tool]
    AD --> AN[Anthropic]
    AD --> OA[OpenAI / compatible]

    L --> T[(ExecutionTrace<br/>spans · cost · redaction)]
    CM --> T
    P --> T
    E --> T
    T --> X[JSON · JSONL · SQLite · OTel]
```

Everything above the adapter boundary is provider-independent. Adding a provider means
implementing one class: `call()`, `stream()`, `denormalize_messages()`, `tool_schema()`,
and optionally `structured()`.

---

## Examples

Runnable end-to-end scripts in [`examples/`](examples/). With no API key present they
fall back to **mock mode** — the loop, executor, permission engine, MCP layer, and
tracing all still run for real, so you can verify your pipeline before spending a cent.

| Script | Shows | Needs a key |
|---|---|---|
| `00_check_setup.py` | Environment check — what's ready, what's missing | no |
| `01_tool_calling.py` | `@tool` schema generation, the `run()` loop, `complete()`, error feedback | no |
| `02_mcp_integration.py` | A real MCP server driving a browser, end to end | no |
| `03_structured_output.py` | `to_json()` with Pydantic and raw jsonschema | no |
| `04_interactive_chat.py` | Permission prompts, `ask_user`, terminal UI | yes |

```bash
python examples/00_check_setup.py
python examples/01_tool_calling.py --mock     # force mock even with a key present
```

See [`examples/README.md`](examples/README.md) for the full walkthrough.

---

## Project status

`0.1.0` — the core is stable and covered by **202 tests** that run fully offline, including
golden-request tests that pin both providers' wire formats. CI runs lint and the full
suite on Python 3.11 and 3.12.

The public API is small on purpose and intended to stay that way. Contributions are
welcome — especially new provider adapters, since that is the whole point.

- [Architecture analysis & design decisions](agent_framework/docs/framework_analiz_raporu.md)
- [Improvement plan & changelog](agent_framework/docs/framework_iyilestirme_plani.md)

## License

MIT — see [LICENSE](LICENSE).
