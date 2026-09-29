# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — BREAKING

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
- **Seven flat overrides on `Agent(...)`** — `system_prompt`, `max_turns`,
  `max_budget_usd`, `temperature`, `api_key`, `base_url`, `context_window` —
  which override a passed `AgentConfig` without mutating it, making a sub-agent a
  one-liner. `context_window` pairs with `base_url`: for a self-hosted model it
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
  never triggered compaction and overflowed at the endpoint. Construction now
  warns and names `context_window=` as the fix.

## [0.1.0]

Initial public release.
