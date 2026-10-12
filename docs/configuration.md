# Configuration

How an agent is constructed, where its settings live, and what `run()` returns when
something goes wrong.

## `Agent(...)` and `AgentConfig`

The model id is the only thing you must supply. `Agent(...)` takes the handful of
settings you actually reach for; everything else lives on `AgentConfig`, which is
frozen, validated at construction, and serializable:

```python
from agent_framework import Agent, AgentConfig

config = AgentConfig(
    model="claude-opus-5",
    system_prompt="You are a helpful assistant.",
    max_context_tokens=60_000,           # trim the prompt when it reaches this size
    request_extra={"thinking": {"type": "adaptive"}},
)

agent      = Agent(config, tools=[search_products])
summariser = Agent(config, system_prompt="Summarise.", max_turns=2)   # override, no mutation
```

An override is applied with `dataclasses.replace`, so the config you passed is never
mutated and one config can seed a whole family of agents. An unknown model id is
rejected at construction with the fix in the message, so a typo can never reach a
provider.

Config can also come from the environment, for the same image deployed with different
settings per stage — `AGENT_MODEL`, `AGENT_MAX_TURNS`, `AGENT_MAX_BUDGET_USD` and the
other flat fields, all validated on the way in:

```python
agent = Agent(AgentConfig.from_env())            # or from_env(prefix="MYAPP_")
```

`run()` never raises for operational failures. A budget stop, a turn-limit stop, or a
provider error comes back as `result.stop_reason == StopReason.ERROR` with the exception
on `result.error`; a prompt that would not fit the context window is stopped *before* it
is sent, as `StopReason.CONTEXT_EXCEEDED`. Only `asyncio.CancelledError` propagates.

## API keys

Keys are read from the environment variable the model's provider declares
(`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`), or passed explicitly via
`Agent(..., api_key=...)`. A custom `base_url` makes the key optional; see
[Local and self-hosted models](local-models.md).

`temperature=None` means "omit the parameter". Current-generation Anthropic models reject
explicit sampling parameters, so `None` is a meaningful value rather than "unset".

`AgentConfig.request_extra` holds per-request keyword arguments for the provider SDK.
Fields the SDK does not know must go inside `request_extra={"extra_body": {...}}`.
