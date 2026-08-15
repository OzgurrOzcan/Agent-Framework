# Agent Framework

A production-grade, provider-agnostic Python agent framework. Write your agent
once — the framework owns the tool-calling loop, provider wire formats,
permissions, MCP integration, memory, and tracing.

## Highlights

- **One loop, two providers** — normalized message model over Anthropic and
  OpenAI adapters; golden-request tests pin both wire formats.
- **Small execution surface** — `complete()` (single model call, tool calls
  returned as data), `run()` / `run_stream()` (the agent loop, shaped by
  `max_turns`, `max_tool_rounds`, `tool_choice`), `to_json()` (native
  structured output, validated client-side, never touches history).
- **Safety rails** — four termination layers (natural / max-turns / stall /
  budget), deny-first permission engine with human-in-the-loop interrupts,
  argument validation before every tool call, `tool_use ↔ tool_result`
  invariant preserved even on cancellation.
- **MCP layer** — plug any MCP server (stdio / streamable HTTP) into an agent
  in three lines; discovered tools are namespaced (`server__tool`) and flow
  through the same executor, permissions, and tracing as native tools.
- **Context management** — persistent LLM compaction, safe sliding windows,
  incremental Anthropic prompt caching, usage-calibrated token estimation.
- **Observability** — span-tree tracing with secret redaction, cost tracking
  with a pluggable pricing table, JSON/JSONL/SQLite/OTel exporters.

## Quick start

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
    agent = BaseAgent.create(AgentConfig(
        provider=ProviderConfig(
            provider=ProviderKind.ANTHROPIC,   # or ProviderKind.OPENAI
            model="claude-opus-5",             # API key via ANTHROPIC_API_KEY env var
        ),
        system_prompt="You are a helpful assistant.",
    ), tools=[add])

    result = await agent.run("What is 2 + 3?")
    print(result.final_message)

asyncio.run(main())
```

## Install

```bash
pip install -e ".[dev]"      # framework + test tooling
pip install -e ".[mcp]"      # optional: MCP server support
pytest agent_framework/tests/ -q
```

API keys are read from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` environment
variables (or passed via `ProviderConfig(api_key=...)`). Nothing in this
repository contains or requires a committed key; tests run fully offline
against mocked providers.

## Docs

- [Architecture analysis & design decisions](agent_framework/docs/framework_analiz_raporu.md)
- [Improvement plan & changelog](agent_framework/docs/framework_iyilestirme_plani.md)
