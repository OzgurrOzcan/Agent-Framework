# Local and self-hosted models

Any server that speaks the OpenAI wire format is a first-class target: vLLM, llama.cpp's
`llama-server`, Ollama's `/v1` endpoint, LM Studio, or a corporate gateway. Name the
adapter explicitly and point `base_url` at it — the `openai:` prefix selects the wire
format, not the vendor.

```python
agent = Agent(
    "openai:llama-3.3-70b",                  # adapter := openai; "llama-3.3-70b" hits the wire
    base_url="http://localhost:8000/v1",
    context_window=32_768,                   # your server's real window — see below
    system_prompt="You are a helpful assistant.",
    tools=[search_products],
)
```

No API key is needed. A custom `base_url` makes the credential optional, because a local
runtime accepts any token and the vendor's key would be the wrong one anyway; an explicit
`api_key=` or the provider's environment variable still wins, so an authenticating gateway
works unchanged.

Everything above the adapter is untouched by the switch: the tool loop, permissions,
`ask_user`, MCP servers, memory, context trimming, checkpoints and the trace all behave
exactly as they do against a vendor API.

**Three things differ, and the framework tells you about two of them.**

*Set `context_window` yourself.* The catalog knows the windows of models it ships with;
yours it cannot. Left unset, the model inherits the provider's 200k fallback — and a model
actually served with an 8k window would then never trigger a trim and simply overflow
at the endpoint. Constructing an uncatalogued model without a window warns, naming the fix.

*The response shares the window.* The prompt may use what `max_tokens_per_response`
(4096 by default) leaves, and the automatic limits follow from that: on an 8k window the
prompt is trimmed at 4096 tokens and one tool result is capped at 819. A window no larger
than the response share fails at construction, naming the setting to lower. To give the
prompt more room, shrink the response share, and set the trim point yourself if you like:

```python
from agent_framework import Agent, AgentConfig

agent = Agent(AgentConfig(
    model="openai:qwen2.5-7b-instruct",
    base_url="http://localhost:8000/v1",
    context_window=8192,
    max_tokens_per_response=1024,
    max_context_tokens=6_000,     # must stay within 8192 - 1024
))
```

*Cost tracking is inert.* An unpriced model records `$0.00`, which for local inference is
honest, but it also means `max_budget_usd` enforces nothing — bound those runs with
`max_turns` instead, or register your own cost-per-token in the `PricingTable` if you
account for GPU time. The session span records `pricing_resolved: false` so a trace never
lies about which guard was armed.

Two compatibility notes. Requests carry `max_completion_tokens`, the current Chat
Completions field. A server that only understands the older `max_tokens` either rejects
the request or ignores the field, and then applies no response cap at all;
`request_extra={"max_tokens": 1024}` adds the older field alongside, which covers servers
that ignore unknown fields but not ones that reject them. And `to_json()` uses the
provider's native structured-output mode, which self-hosted runtimes support to varying
degrees — `request_extra` does not apply to it, so on a runtime without that support use
`complete()` and parse the text yourself.

**Hosted endpoints that expose their reasoning.** Kimi, DeepSeek and a vLLM-served
thinking model speak the same wire format but return the model's reasoning as
`reasoning_content` — and expect it back on every earlier assistant message: Kimi rejects
a tool-call message without it. The adapter keeps the field as an opaque block and sends
it back verbatim, so the tool loop, memory and a context trim all work; a trim still
clears old results and drops whole turns, but leaves the reasoning of the turns it keeps
in place. Kimi is recognized from its model name; for an endpoint the name cannot reveal,
say so with `keep_reasoning_in_history=True`.

```python
agent = Agent(AgentConfig(
    model="openai:kimi-k3",
    base_url="https://api.moonshot.ai/v1",
    api_key="<your Moonshot key>",      # explicit: OPENAI_API_KEY would be the wrong key
    context_window=1_000_000,
    # Non-OpenAI request fields go inside extra_body: the SDK rejects unknown
    # keyword arguments. Kimi K2.6's thinking switch, for instance:
    # request_extra={"extra_body": {"thinking": {"type": "disabled"}}},
))
```

Two Kimi-specific notes: leave `temperature` unset (each Kimi model accepts one fixed
value and rejects any other), and `tool_choice="any"` — sent as `required` — is accepted
by K3 only.
