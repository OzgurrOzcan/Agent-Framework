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
| `BaseAgent.acreate(config, mcp=...)` | `await Agent.connect(model_or_config, mcp=...)`, or `Agent(..., mcp=...)` for lazy discovery |
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
  `config.replace(...)`.
- **`Agent(..., mcp=...)` no longer raises inside a running event loop.** It
  defers discovery to the first run. Use `await Agent.connect(...)` to discover
  at start-up so a misconfigured server fails at boot instead of silently
  serving a reduced toolset.
- **A registry passed as `tools=` is no longer mutated.** It is copied, so
  registering `ask_user` (and MCP tools) cannot leak between agents sharing one
  registry.
- **`MCPServerRegistry.discover_all_sync()`** is kept as a standalone utility
  for sync callers (`tools=registry.discover_all_sync()`); its error message now
  points at `Agent.connect`.

### Added

- **Provider catalog** (`agent_framework.providers.catalog`). One
  `ProviderSpec` per provider is the single source of truth for the adapter
  factory, the API-key environment variable, the model-name prefixes the
  provider owns, and its context windows. `register_provider()` opens the set to
  third parties, so adding a provider is one record plus one adapter class
  instead of edits in four places.
- **Model-id resolution.** A bare name is matched by longest owned prefix
  (`"claude-opus-5"` → anthropic); `"provider:model"` always wins and is how
  OpenAI-compatible endpoints are addressed (`"openai:llama-3.3-70b"`). Anything
  that cannot resolve to exactly one provider raises `ProviderResolutionError`
  with the fix in the message — never a silent default, and never inferred from
  which API keys happen to be set. Prefixes are broad, so a newly released model
  works without a framework release.
- **Six flat overrides on `Agent(...)`** — `system_prompt`, `max_turns`,
  `max_budget_usd`, `temperature`, `api_key`, `base_url` — which override a
  passed `AgentConfig` without mutating it, making a sub-agent a one-liner. A
  test asserts they cannot drift from the dataclass fields.
- **`AgentConfig.to_dict()` / `from_dict()` / `from_env()`** for config files,
  12-factor deploys, and trace provenance. The API key is never serialized;
  `from_dict` rejects unknown keys so a typo fails at load time.
- **`EventType.SESSION` spans** (the enum member existed but was never emitted):
  one per agent, carrying the resolved config, tool names, MCP discovery mode,
  whether pricing resolved, and the framework version.
- **`EventType.RUN` spans**: one per `run()` / `run_stream()` / `complete()`,
  carrying the effective `LoopPolicy` (`max_turns`, `max_tool_rounds`,
  `execute_tools`, `tool_choice`), the outcome and the whole-run duration, with
  the turns nested underneath. Per-call overrides are recoverable from nowhere
  else, and `where_it_failed()` now names which run broke.

- **Self-hosted endpoints need no credential.** A custom `base_url` makes the
  API key optional, since a local runtime accepts any token; an explicit
  `api_key` and the provider's environment variable still win, so an
  authenticating gateway is unaffected. Previously every local agent had to
  invent `api_key="not-needed"`.
- **`context_window` is a flat `Agent(...)` override**, pairing with `base_url`:
  for a self-hosted model it is required rather than exotic.

### Fixed

- A `ToolRegistry` passed to more than one agent no longer accumulates
  `ask_user` and MCP tools from each of them.
- **An uncatalogued model no longer budgets its context silently wrong.** It
  used to inherit the provider's 200k fallback, so a locally served 8k model
  never triggered compaction and overflowed at the endpoint. Construction now
  warns and names `context_window=` as the fix.

## [0.1.0]

Initial public release.
