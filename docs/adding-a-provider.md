# Adding a provider

Everything above the adapter boundary is provider-independent and operates on the
normalized message model in `core/message.py`. The loop, the executor, permissions,
memory and tracing never see provider-shaped data.

Adding a provider means implementing one class and registering one record.

## The adapter

Subclass `BaseProviderAdapter` and implement:

| Method | Purpose |
|---|---|
| `call()` | one non-streaming request, returning a normalized assistant message and usage |
| `stream()` | the same request as a stream of `StreamChunk`s |
| `denormalize_messages()` | the normalized history, in the provider's wire format |
| `tool_schema()` | one `Tool` as the provider's tool definition |
| `structured()` | optional: native structured output for `to_json()`; a prompt-based fallback is used without it |

Never leak provider-shaped data upward. If the provider returns reasoning in its own
block, the adapter keeps it opaque and sends it back verbatim.

## The record

from agent_framework import ProviderSpec, register_provider

register_provider(ProviderSpec(
    name="bedrock",
    env_key="BEDROCK_API_KEY",
    model_prefixes=("bedrock.",),          # so "bedrock.foo-1" resolves without a prefix
    build=lambda **kw: MyBedrockAdapter(**kw),
    context_windows=(("bedrock.", 200_000),),
))

agent = Agent("bedrock:anthropic.claude-opus-5")
```

That one record is the single source of truth for the adapter, the API-key variable, the
model names the provider owns, and its context windows — there is no enum to extend and
nothing else in the framework to touch.

Model-id resolution is a table lookup. A bare name resolves to the provider that owns
its longest matching prefix; `"provider:model"` always wins; an ambiguous or unknown
name raises `ProviderResolutionError` at construction. Resolution never consults the
environment, so which model runs cannot depend on ambient state.

`tests/unit/test_adapters_golden.py` pins the exact request dicts both shipped adapters
produce. Add the same kind of golden test for a new adapter.
