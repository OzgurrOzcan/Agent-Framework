# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — BREAKING: context and memory

An agent is now **stateless by default** — every `run()`, `run_stream()` and
`complete()` starts from an empty history — and memory is one setting:

```diff
-agent = Agent("claude-opus-5")              # remembered every earlier run
+agent = Agent("claude-opus-5")              # every run starts fresh
+chat = Agent("claude-opus-5", memory=MemoryConfig())          # the conversation, in RAM
+desk = Agent(
+    "claude-opus-5",
+    memory=MemoryConfig(store="support.db", namespace="customer-42"),
+    session_id="ticket-4412",                # long-term: saved and resumable
+)
```

| Removed or changed | Replacement |
|---|---|
| History kept across runs by default | `memory=MemoryConfig()`; without it each run starts fresh |
| `CompactionConfig`, `AgentConfig.compaction` | `AgentConfig.max_context_tokens` and `max_tool_result_tokens`, both automatic by default |
| `AgentConfig.response_buffer_tokens` | removed: the prompt may use `context_window - max_tokens_per_response` |
| `AgentConfig.compaction_buffer_tokens` | removed: trimming makes no LLM call, so it needs no headroom |
| `ShortTermMemory(max_messages=, max_tokens=)` | removed (it trimmed silently on every append); the token limit decides |
| `Agent(memory=MemoryManager(...))` | `Agent(memory=MemoryConfig(...), memory_backend=<LongTermBackend>)` |
| `agent.clear_history()` | `await agent.clear_history()` (it saves the emptied conversation) |
| `LLM_CALL` payload `ctx_tokens`, `compaction` | `estimated_tokens`, `trimmed` and the rest of the context report |
| `ContextManager(context_window=, response_buffer_tokens=, compaction=, summarizer=...)`, `assemble(history, current_user_input=)`, `AssembledContext.persist` | `ContextManager(max_context_tokens=, max_tool_result_tokens=, ...)`, `assemble(short_term, call=)` |
| Memory facts and the rolling summary appended at the end of every call | facts stored once per run before the user's message; the summary at the head of the view |
| Tool results unbounded until compaction cut them to 500 tokens | capped when produced, at `max_tool_result_tokens` (head and tail kept) |

`AgentConfig.from_dict()` rejects a removed field with its replacement in the
message, so a stale config file fails at load time.

Why: the old pipeline truncated, windowed and re-summarized on every call. Each
of those rewrote content the provider had already seen, which invalidated the
prompt cache on every call — and, on models that bind reasoning to the
conversation prefix, turned the next request into a 400. Memory facts were
regenerated at the end of every prompt for the same reason. And remembering by
default meant every unrelated `run()` carried, and paid for, the ones before it.

Other behaviour changes:

- **Trimming is deterministic and rare.** Nothing already sent changes between
  trims. When the prompt (system + tool definitions + messages) exceeds
  `max_context_tokens`, one trim — no LLM call — brings it to 60 % of the
  limit: old tool results become a one-line note (the newest five stay, as do
  errors, small results, `ask_user` and tools with `clear_old_results=False`),
  then whole old turns go, the first user message pinned and the newest three
  turns kept. Reasoning before the current turn is left out after a trim.
- **Reasoning stays valid on prefix-bound models.** On Claude Fable 5.1, Opus
  5.5 and Sonnet 5.5 (and, to be safe, any Claude model the catalog does not
  know yet) the history is trimmed only at the start of a run; the catalog
  records this per model.
- **A prompt that cannot fit is never sent.** The run ends with
  `StopReason.CONTEXT_EXCEEDED` before the request, and the history stays
  valid; with memory, the next run's first call trims it.
- **Anthropic clears old tool results server-side in long runs**
  (`clear_tool_uses_20250919`, requested automatically against the vendor API,
  off for a custom `base_url`, switched off for the session after a rejection).
- **Tool calls without results are answered.** A remembered conversation in
  which `complete()` returned tool calls unexecuted no longer fails on the next
  run: they get a "not executed" result first.
- `Agent(session_id=...)` defaults to a fresh id per agent, never the trace's,
  and is what permission prompts and `ask_user` report.

### Changed — BREAKING: agent construction

Agent construction is one call, and the model id is the only required input.
There is no compatibility shim: the project is pre-1.0 and unpublished, so
callers migrate rather than carry two config shapes.

```diff
-from agent_framework import BaseAgent, AgentConfig, ProviderConfig, ProviderKind
-
-agent = BaseAgent.create(
-    AgentConfig(
-        provider=ProviderConfig(
-            provider=ProviderKind.ANTHROPIC,
-            model="claude-opus-5",
-        ),
-        system_prompt="You are a helpful assistant.",
-        max_turns=6,
-    ),
-    tools=[add],
-)
+from agent_framework import Agent
+
+agent = Agent(
+    "claude-opus-5",
+    system_prompt="You are a helpful assistant.",
+    max_turns=6,
+    tools=[add],
+)
```

| Removed | Replacement |
|---|---|
| `BaseAgent` | `Agent` |
| `BaseAgent.create(config, ...)` | `Agent(model_or_config, ...)` |
| `BaseAgent.acreate(config, mcp=...)` | `await Agent.connect(model_or_config, mcp=...)` — note it is strict, raising `MCPDiscoveryError` where `acreate` silently skipped a broken server — or `Agent(..., mcp=...)` for tolerant, lazy discovery |
| `AnthropicAgent`, `OpenAIAgent` | `Agent` (they were empty subclasses with no references) |
| `ProviderKind` | the model id (`"claude-opus-5"`), or an explicit prefix (`"openai:llama-3.3-70b"`) |
| `ProviderConfig` | flat `AgentConfig.api_key` / `.base_url`, plus `TransportConfig` for timeouts, retries and SDK client kwargs |
| `ProviderConfig.extra` | `TransportConfig.client_kwargs` (per client) |
| `AgentConfig.provider_extra` | `AgentConfig.request_extra` (per request) |
| `agent_framework.core.base_agent` | `agent_framework.core.agent` |
| `config.model_context_window()` | `providers.catalog.model_context_window()` |

Why: in the old expression the word `provider` named three different things —
`AgentConfig.provider` was a `ProviderConfig` whose own `.provider` was a
`ProviderKind` — and the enum was redundant with the model name, which already
implies the provider. `ProviderConfig` held no field `AgentConfig` could not
own and was read in exactly one place.

Other behaviour changes:

- **`AgentConfig` is frozen and validated at construction.** `max_turns=0`,
  `temperature=3`, a negative budget, or buffers larger than the context window
  used to construct fine and misbehave several turns later. Derive variants with
  `config.replace(...)`; normalization is idempotent, so every model id —
  including `"openai:<local-model>"` — survives `replace`, `to_dict`/`from_dict`
  and `Agent(config, override=...)`.
- **MCP discovery happens where you choose.** `Agent(..., mcp=...)` no longer
  raises inside a running event loop: it defers discovery to the first run, where
  a failing server is recorded in the trace and skipped. `await
  Agent.connect(...)` discovers at start-up and strictly, raising
  `MCPDiscoveryError` if any server cannot be reached, so a misconfigured
  deployment crashes at boot instead of serving a silently reduced toolset.
- **A registry passed as `tools=` is no longer mutated.** It is copied, so
  registering `ask_user` (and MCP tools) cannot leak between agents sharing one
  registry.
- **`MCPServerRegistry.discover_all_sync()`** is kept as a standalone utility
  for sync callers (`tools=registry.discover_all_sync()`); its error message now
  points at `Agent.connect`.
- **Self-hosted endpoints need no credential.** A custom `base_url` makes the
  API key optional, since a local runtime accepts any token; an explicit
  `api_key` and the provider's environment variable still win, so an
  authenticating gateway is unaffected. Previously every local agent had to
  invent `api_key="not-needed"`.

### Added

- **`MemoryConfig`** (`store`, `namespace`, `summarize`, `summary_model`) and
  `AgentConfig.memory`, also a flat `Agent(memory=...)` override. Long-term
  fields without a store or backend are rejected rather than ignored.
- **Long-term memory with no database code.** `MemoryConfig(store="app.db")`
  builds SQLite-backed memory (one connection, WAL, `":memory:"` works) that
  `await agent.memory.close()` closes; `Agent(memory_backend=...)` takes any
  `LongTermBackend` instead.
- **Sessions.** With a store, every run saves the conversation — a full log of
  every message plus the current history and context state — and an agent with
  the same `session_id` resumes it; `Agent.connect()` loads it eagerly. A custom
  backend keeps sessions by implementing `SessionStore`.
- **Long-term summaries.** Turns a trim removes are summarized into the
  namespace's summary (one call each, segments merged over 8,000 tokens); a new
  conversation starts with it; an idle session's unsummarized tail is
  summarized once (claimed atomically); `await agent.memory.summarize()` does it
  on demand. Calls are cost-tracked and traced.
- **`context_filter=fn`** on `Agent(...)`: your function shapes what is sent
  before every call, never what is stored; its output is repaired into a valid
  conversation, and on prefix-bound models an edit of what was already sent
  mid-run is rejected and traced.
- **`run(dry_run=True)`** shows what the next call would send (trimmed if it
  would be) without calling the model or changing anything, and
  **`AgentResult.context`** (`ContextReport`) reports what the last call sent:
  message counts, the token estimate, cleared results, dropped reasoning, cache
  hits. Every `llm_call` span records the same.
- `@tool(max_result_tokens=..., clear_old_results=...)`,
  `ToolExecutor(default_max_result_tokens=...)`, `StopReason.DRY_RUN`,
  `ProviderSpec.binds_reasoning_to_prefix`, and the provider-neutral
  `ToolResultClearing` / `context_edits=` adapter parameter.
- **Provider catalog** (`agent_framework.providers.catalog`). One
  `ProviderSpec` per provider is the single source of truth for the adapter
  factory, the API-key environment variable, the model-name prefixes the
  provider owns, and its context windows. `register_provider()` opens the set to
  third parties, so adding a provider is one record plus one adapter class
  instead of edits in four places.
- **Model-id resolution.** A bare name is matched by longest owned prefix
  (`"claude-opus-5"` → anthropic); an explicit provider — the
  `"provider:model"` prefix or `AgentConfig(provider=...)` — always wins, and is
  how OpenAI-compatible endpoints are addressed (`"openai:llama-3.3-70b"`,
  `"openai:llama3.2:3b"` for an Ollama tag). Anything that cannot resolve to
  exactly one provider raises `ProviderResolutionError` with the fix in the
  message — never a silent default, and never inferred from which API keys
  happen to be set. Prefixes are broad, so a newly released model works without
  a framework release.
- **Eight flat overrides on `Agent(...)`** — `system_prompt`, `max_turns`,
  `max_budget_usd`, `temperature`, `api_key`, `base_url`, `context_window`,
  `memory` — which override a passed `AgentConfig` without mutating it, making a
  sub-agent a one-liner. `context_window` pairs with `base_url`: for a self-hosted model it
  is required rather than exotic. A test asserts the list cannot drift from the
  dataclass fields.
- **`AgentConfig.to_dict()` / `from_dict()` / `from_env()`** for config files,
  12-factor deploys, and trace provenance. The API key is never serialized;
  `from_dict` rejects unknown keys so a typo fails at load time.
- **`MCPServerRegistry.discover_all(strict=True)`** and **`MCPDiscoveryError`**,
  which names every server that failed; the default stays tolerant.
- **`EventType.SESSION` spans** (the enum member existed but was never emitted):
  one per agent, carrying the resolved config, the agent id, tool names, MCP
  servers and discovery mode, whether pricing resolved, and the framework
  version. It records settings, not content: the system prompt is fingerprinted
  rather than copied, and `base_url` is stripped of credential-bearing parts.
- **`EventType.RUN` spans**: one per `run()` / `run_stream()` / `complete()`,
  carrying the effective `LoopPolicy` (`max_turns`, `max_tool_rounds`,
  `execute_tools`, `tool_choice`), the run's own turns, tool rounds and cost, and
  its duration, with the turns nested underneath. Per-call overrides are
  recoverable from nowhere else.

### Fixed

- **Memory facts are matched by whole words in any script.** Matching used
  ASCII substrings: a Turkish query ("faturalar", "İstanbul") matched nothing,
  while "art" matched "party".
- **`None` is no longer redacted in traces.** A setting left automatic
  (`max_context_tokens=None`) reached the session span as `"[REDACTED]"`
  because its key contains "token".
- **The SQLite backend no longer loses an in-memory database** — it opened a
  new connection per operation — and its connection can now be closed.
- **MCP servers over HTTP connect again.** Every `transport="http"` connection
  failed with `TypeError: streamable_http_client() got an unexpected keyword
  argument 'headers'`: the SDK renamed its streamable HTTP client and changed
  how it is called in the same step, and the framework called the new name the
  old way. This affected every SDK release since the rename — mcp 1.30 as well
  as 2.0 — though stdio servers were unaffected. `auth_headers` now travel on
  an HTTP client built by the SDK's own factory, and both SDK generations are
  supported.
- **A failed MCP handshake is no longer mistaken for a connection.** The client
  published its session before the handshake finished, so a server that
  accepted the connection but failed or timed out the handshake left it
  reporting `connected=True`; the retry then "succeeded" without handshaking,
  and the transport — for stdio, the server subprocess — was never torn down.
  Nothing is published until the handshake completes, and a failure closes
  everything it opened.
- **MCP connection errors name their cause.** The SDK's transports report even
  a single failure as a nested `ExceptionGroup`, so errors read "unhandled
  errors in a TaskGroup (1 sub-exception)". The one real error — a 401, a
  refused connection — is now surfaced; genuine multi-error groups are kept.
- **A failed MCP boot is reported 2s sooner.** The retry loop also slept after
  its final attempt, where no retry can follow.
- **Closing a stream early no longer bricks the agent.** `run_stream()` never
  closed the loop's own generator, so after a consumer stopped early — a "stop
  generating" button — the run-in-progress flag stayed set and every later
  `run()` raised. An early close now ends the run cleanly.
- **Cancellation is recorded as `cancelled`, not `error`.** A span interrupted
  by task cancellation or an early stream close used to be marked as failed.
- **`where_it_failed()` names the cause, not a wrapper.** It chose the failure
  with the earliest timestamp, and a span that failed only because a child did
  always starts first. It now returns the chain from the root down to the first
  root-cause failure — e.g. `[run, loop_guard:max_turns]`.
- A `ToolRegistry` passed to more than one agent no longer accumulates
  `ask_user` and MCP tools from each of them.
- **An uncatalogued model no longer budgets its context silently wrong.** It
  used to inherit the provider's 200k fallback, so a locally served 8k model
  never trimmed and overflowed at the endpoint. Construction now warns and
  names `context_window=` as the fix.

## [0.1.0]

Initial public release.
