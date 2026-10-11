"""Golden request tests — adapters get a fake SDK client injected and the
exact wire-format request dict is asserted. No network, no API keys."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_framework.core.message import (
    AssistantMessage,
    ReasoningContent,
    StopReason,
    ToolCallRequest,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from agent_framework.providers.anthropic_agent import AnthropicAdapter, _to_stop_reason as anth_stop
from agent_framework.providers.base import (
    AppliedEdit,
    ContextEditsRejectedError,
    StreamChunkType,
    ToolChoice,
    ToolResultClearing,
)
from agent_framework.providers.openai_agent import OpenAIAdapter, _to_stop_reason as oai_stop
from agent_framework.tools.models import ToolMetadata


# ---------------- fakes ----------------

def _anthropic_raw(text="ok", stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
        stop_reason=stop_reason,
    )


def _openai_raw(text="ok", finish_reason="stop", prompt=1000, cached=0):
    details = SimpleNamespace(cached_tokens=cached) if cached else None
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=5,
            prompt_tokens_details=details,
        ),
    )


class _FakeMessagesAPI:
    def __init__(self, response):
        self._response = response
        self.last_request: dict | None = None

    async def create(self, **kwargs):
        self.last_request = kwargs
        return self._response


class FakeAnthropicClient:
    def __init__(self, response=None):
        self.messages = _FakeMessagesAPI(response or _anthropic_raw())


class FakeOpenAIClient:
    def __init__(self, response=None):
        api = _FakeMessagesAPI(response or _openai_raw())
        self.chat = SimpleNamespace(completions=api)
        self._api = api


@pytest.fixture
def anth():
    client = FakeAnthropicClient()
    return AnthropicAdapter(api_key="test-key", client=client), client


@pytest.fixture
def oai():
    client = FakeOpenAIClient()
    return OpenAIAdapter(api_key="test-key", client=client), client


_TOOL = ToolMetadata(
    name="search",
    description="Search things",
    parameters_schema={"type": "object", "properties": {"q": {"type": "string"}},
                       "required": ["q"], "additionalProperties": False},
)


# ---------------- basic request shape ----------------

async def test_anthropic_basic_request_shape(anth):
    adapter, client = anth
    await adapter.call(
        system="sys", messages=[UserMessage(content="hi")], tools=[_TOOL],
        model="claude-opus-5", max_tokens=128,
    )
    req = client.messages.last_request
    assert req["model"] == "claude-opus-5"
    assert req["max_tokens"] == 128
    assert "temperature" not in req  # None → omitted (400 on current models)
    assert req["system"][0]["text"] == "sys"
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["tools"][0]["input_schema"] == _TOOL.parameters_schema


async def test_openai_basic_request_shape(oai):
    adapter, client = oai
    await adapter.call(
        system="sys", messages=[UserMessage(content="hi")], tools=[_TOOL],
        model="gpt-5", max_tokens=128,
    )
    req = client._api.last_request
    assert req["model"] == "gpt-5"
    assert req["max_completion_tokens"] == 128
    assert "temperature" not in req
    assert req["messages"][0] == {"role": "system", "content": "sys"}
    assert req["tools"][0]["function"]["parameters"] == _TOOL.parameters_schema


_EFFORT_400 = (
    "Error code: 400 - Function tools with reasoning_effort are not supported for "
    "gpt-5.6-luna in /v1/chat/completions. To use function tools, use /v1/responses "
    "or set reasoning_effort to 'none'."
)


class _EffortRejectingAPI:
    """Rejects `tools` + implicit reasoning_effort once, like Chat Completions."""

    def __init__(self, client):
        self._client = client
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(dict(kwargs))
        if kwargs.get("tools") and "reasoning_effort" not in kwargs:
            raise RuntimeError(_EFFORT_400)
        return self._client._api._response  # reuse the fake's canned response


@pytest.fixture()
def oai_effort(oai):
    adapter, client = oai
    api = _EffortRejectingAPI(client)
    client.chat.completions = api
    return adapter, api


async def test_openai_retries_once_with_effort_none_then_remembers(oai_effort):
    """The 400 is recovered from, and the retry is not paid again."""
    adapter, api = oai_effort
    await adapter.call(
        system="", messages=[UserMessage(content="hi")], tools=[_TOOL],
        model="gpt-5.6-luna", max_tokens=64,
    )
    # First attempt without the flag, second with it.
    assert len(api.requests) == 2
    assert "reasoning_effort" not in api.requests[0]
    assert api.requests[1]["reasoning_effort"] == "none"

    # Second call for the same model sends it up front — no wasted round-trip.
    await adapter.call(
        system="", messages=[UserMessage(content="hi")], tools=[_TOOL],
        model="gpt-5.6-luna", max_tokens=64,
    )
    assert len(api.requests) == 3
    assert api.requests[2]["reasoning_effort"] == "none"


async def test_openai_effort_workaround_is_scoped(oai_effort):
    """Untouched models and tool-less calls keep their normal request shape."""
    adapter, api = oai_effort
    await adapter.call(
        system="", messages=[UserMessage(content="hi")], tools=[_TOOL],
        model="gpt-5.6-luna", max_tokens=64,
    )
    api.requests.clear()
    # Different model → learned flag must not leak across models.
    await adapter.call(
        system="", messages=[UserMessage(content="hi")], tools=[],
        model="gpt-4o", max_tokens=64,
    )
    assert "reasoning_effort" not in api.requests[0]


async def test_openai_unrelated_400_is_not_swallowed(oai):
    adapter, client = oai

    class _AlwaysFails:
        async def create(self, **kwargs):
            raise RuntimeError("Error code: 400 - unknown parameter 'banana'")

    client.chat.completions = _AlwaysFails()
    with pytest.raises(RuntimeError, match="banana"):
        await adapter.call(
            system="", messages=[UserMessage(content="hi")], tools=[_TOOL],
            model="gpt-5", max_tokens=64,
        )


async def test_openai_request_extra_overrides_reasoning_effort(oai):
    adapter, client = oai
    await adapter.call(
        system="", messages=[UserMessage(content="hi")], tools=[_TOOL],
        model="gpt-5", max_tokens=64, extra={"reasoning_effort": "low"},
    )
    assert client._api.last_request["reasoning_effort"] == "low"


async def test_temperature_sent_when_explicit(anth, oai):
    a_adapter, a_client = anth
    o_adapter, o_client = oai
    await a_adapter.call(system="", messages=[UserMessage(content="x")], tools=[],
                         model="claude-sonnet-4-6", max_tokens=10, temperature=0.3)
    await o_adapter.call(system="", messages=[UserMessage(content="x")], tools=[],
                         model="gpt-4o", max_tokens=10, temperature=0.3)
    assert a_client.messages.last_request["temperature"] == 0.3
    assert o_client._api.last_request["temperature"] == 0.3


# ---------------- denormalization golden ----------------

def _history():
    return [
        UserMessage(content="find it"),
        AssistantMessage(
            content="searching",
            tool_calls=[ToolCallRequest(id="c1", name="search", arguments={"q": "x"})],
            reasoning=[ReasoningContent(
                provider="anthropic",
                raw={"type": "thinking", "thinking": "hmm", "signature": "sig"},
            )],
        ),
        ToolResultMessage(content=[
            ToolResult(tool_use_id="c1", name="search", content="result!", is_error=False),
        ]),
    ]


def test_anthropic_denormalize_golden():
    adapter = AnthropicAdapter(api_key="k", client=FakeAnthropicClient())
    wire = adapter.denormalize_messages(_history())
    assert wire == [
        {"role": "user", "content": "find it"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            {"type": "text", "text": "searching"},
            {"type": "tool_use", "id": "c1", "name": "search", "input": {"q": "x"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "result!",
             "is_error": False},
        ]},
    ]


def test_openai_denormalize_golden():
    adapter = OpenAIAdapter(api_key="k", client=FakeOpenAIClient())
    wire = adapter.denormalize_messages(_history())
    assert wire == [
        {"role": "user", "content": "find it"},
        {"role": "assistant", "content": "searching", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "search", "arguments": json.dumps({"q": "x"})}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "result!"},
    ]


# ---------------- reasoning_content round-trip (Kimi, DeepSeek, vLLM) ----------------

def _openai_reasoning_history(reasoning_content="think first"):
    return [
        UserMessage(content="find it"),
        AssistantMessage(
            content=None,
            tool_calls=[ToolCallRequest(id="c1", name="search", arguments={"q": "x"})],
            reasoning=[ReasoningContent(
                provider="openai",
                raw={"reasoning_content": reasoning_content},
                text=reasoning_content,
            )],
        ),
        ToolResultMessage(content=[
            ToolResult(tool_use_id="c1", name="search", content="result!", is_error=False),
        ]),
    ]


def test_openai_denormalize_sends_reasoning_content_back():
    """A compatible endpoint that exposed its reasoning gets it back verbatim —
    Kimi rejects a historical tool-call message without it."""
    adapter = OpenAIAdapter(api_key="k", client=FakeOpenAIClient())
    wire = adapter.denormalize_messages(_openai_reasoning_history())
    assert wire[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "search", "arguments": json.dumps({"q": "x"})}},
        ],
        "reasoning_content": "think first",
    }


def test_openai_denormalize_omits_reasoning_content_when_absent():
    """Without a block the key is not on the wire at all — a request to
    OpenAI itself must not change by a byte."""
    adapter = OpenAIAdapter(api_key="k", client=FakeOpenAIClient())
    history = _openai_reasoning_history()
    history[1] = AssistantMessage(content=None, tool_calls=history[1].tool_calls)
    wire = adapter.denormalize_messages(history)
    assert "reasoning_content" not in wire[1]


def test_openai_denormalize_ignores_foreign_reasoning():
    """An Anthropic thinking block has no `reasoning_content` meaning here;
    the OpenAI golden above already pins that it is dropped, this pins it
    stays dropped when the two kinds are mixed."""
    adapter = OpenAIAdapter(api_key="k", client=FakeOpenAIClient())
    history = _openai_reasoning_history()
    history[1] = AssistantMessage(
        content=None,
        tool_calls=history[1].tool_calls,
        reasoning=[
            ReasoningContent(provider="anthropic",
                             raw={"type": "thinking", "thinking": "hmm", "signature": "s"}),
            *history[1].reasoning,
        ],
    )
    wire = adapter.denormalize_messages(history)
    assert wire[1]["reasoning_content"] == "think first"


async def test_openai_normalize_captures_reasoning_content(oai):
    adapter, client = oai
    raw = _openai_raw(text="done")
    raw.choices[0].message.reasoning_content = "let me think"
    client._api._response = raw
    resp = await adapter.call(system="", messages=[UserMessage(content="x")], tools=[],
                              model="m", max_tokens=10)
    [block] = resp.assistant_message.reasoning
    assert block.provider == "openai"
    assert block.raw == {"reasoning_content": "let me think"}
    assert block.text == "let me think"
    assert resp.assistant_message.content == "done"


@pytest.mark.parametrize("value", [None, ""])
async def test_openai_normalize_without_reasoning_content(oai, value):
    """OpenAI's own replies carry no such field (the fake has none); an empty
    string from a non-thinking call is not a block either."""
    adapter, client = oai
    raw = _openai_raw(text="done")
    if value is not None:
        raw.choices[0].message.reasoning_content = value
    client._api._response = raw
    resp = await adapter.call(system="", messages=[UserMessage(content="x")], tools=[],
                              model="m", max_tokens=10)
    assert resp.assistant_message.reasoning == []


async def test_openai_reasoning_content_survives_a_full_round_trip(oai):
    """Reply → AssistantMessage → next request: what came in goes back out."""
    adapter, client = oai
    raw = _openai_raw(text=None, finish_reason="tool_calls")
    raw.choices[0].message.reasoning_content = "need the tool"
    raw.choices[0].message.tool_calls = [SimpleNamespace(
        id="c1", function=SimpleNamespace(name="search", arguments=json.dumps({"q": "x"})),
    )]
    client._api._response = raw
    resp = await adapter.call(system="", messages=[UserMessage(content="x")], tools=[_TOOL],
                              model="m", max_tokens=10)
    history = [UserMessage(content="x"), resp.assistant_message, ToolResultMessage(content=[
        ToolResult(tool_use_id="c1", name="search", content="r", is_error=False),
    ])]
    await adapter.call(system="", messages=history, tools=[_TOOL], model="m", max_tokens=10)
    sent = client._api.last_request["messages"]
    assert sent[1]["reasoning_content"] == "need the tool"
    assert sent[1]["tool_calls"][0]["id"] == "c1"


# ---------------- tool_choice matrix ----------------

@pytest.mark.parametrize("tc,expected", [
    (ToolChoice(mode="auto"), {"type": "auto"}),
    (ToolChoice(mode="any"), {"type": "any"}),
    (ToolChoice(mode="tool", tool_name="search"), {"type": "tool", "name": "search"}),
    (ToolChoice(mode="none"), {"type": "none"}),
    (ToolChoice(mode="auto", parallel=False),
     {"type": "auto", "disable_parallel_tool_use": True}),
])
async def test_anthropic_tool_choice(anth, tc, expected):
    adapter, client = anth
    await adapter.call(system="", messages=[UserMessage(content="x")], tools=[_TOOL],
                       model="m", max_tokens=10, tool_choice=tc)
    assert client.messages.last_request["tool_choice"] == expected


@pytest.mark.parametrize("tc,expected,parallel_flag", [
    (ToolChoice(mode="auto"), "auto", None),
    (ToolChoice(mode="any"), "required", None),
    (ToolChoice(mode="tool", tool_name="search"),
     {"type": "function", "function": {"name": "search"}}, None),
    (ToolChoice(mode="none"), "none", None),
    (ToolChoice(mode="auto", parallel=False), "auto", False),
])
async def test_openai_tool_choice(oai, tc, expected, parallel_flag):
    adapter, client = oai
    await adapter.call(system="", messages=[UserMessage(content="x")], tools=[_TOOL],
                       model="m", max_tokens=10, tool_choice=tc)
    req = client._api.last_request
    assert req["tool_choice"] == expected
    assert req.get("parallel_tool_calls") is parallel_flag or (
        parallel_flag is None and "parallel_tool_calls" not in req
    )


async def test_tool_choice_omitted_by_default(anth):
    adapter, client = anth
    await adapter.call(system="", messages=[UserMessage(content="x")], tools=[_TOOL],
                       model="m", max_tokens=10)
    assert "tool_choice" not in client.messages.last_request


def test_forced_tool_choice_requires_name():
    with pytest.raises(ValueError):
        ToolChoice(mode="tool")


# ---------------- incremental prompt caching (Anthropic) ----------------

async def test_anthropic_incremental_cache_on_last_stable_message(anth):
    adapter, client = anth
    msgs = _history()
    await adapter.call(system="sys", messages=msgs, tools=[], model="m", max_tokens=10)
    wire = client.messages.last_request["messages"]
    # Breakpoint on the LAST message's last block only.
    assert wire[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in wire[0].get("content", [{}])[0] if isinstance(
        wire[0]["content"], list) else True
    for entry in wire[:-1]:
        content = entry["content"]
        if isinstance(content, list):
            assert all("cache_control" not in b for b in content if isinstance(b, dict))


async def test_anthropic_cache_skips_volatile_preamble(anth):
    adapter, client = anth
    msgs = _history() + [
        UserMessage(content="<relevant_context>...</relevant_context>",
                    metadata={"system": "relevant_context"}),
    ]
    await adapter.call(system="sys", messages=msgs, tools=[], model="m", max_tokens=10)
    wire = client.messages.last_request["messages"]
    # Preamble (last wire msg) must NOT carry the breakpoint; the tool-result
    # message before it must.
    last = wire[-1]["content"]
    if isinstance(last, list):
        assert all("cache_control" not in b for b in last if isinstance(b, dict))
    assert wire[-2]["content"][-1]["cache_control"] == {"type": "ephemeral"}


async def test_anthropic_caching_disabled(anth):
    adapter, client = anth
    await adapter.call(system="sys", messages=_history(), tools=[], model="m",
                       max_tokens=10, enable_caching=False)
    req = client.messages.last_request
    assert "cache_control" not in req["system"][0]
    for entry in req["messages"]:
        content = entry["content"]
        if isinstance(content, list):
            assert all("cache_control" not in b for b in content if isinstance(b, dict))


async def test_anthropic_cache_converts_string_content(anth):
    adapter, client = anth
    await adapter.call(system="", messages=[UserMessage(content="hello")], tools=[],
                       model="m", max_tokens=10)
    wire = client.messages.last_request["messages"]
    assert wire[0]["content"] == [
        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
    ]


# ---------------- structured output requests ----------------

async def test_anthropic_structured_request(anth):
    adapter, client = anth
    client.messages._response = _anthropic_raw(text='{"a": 1}')
    schema = {"type": "object", "properties": {"a": {"type": "integer"}},
              "required": ["a"], "additionalProperties": False}
    out = await adapter.structured(system="", messages=[UserMessage(content="go")],
                                   schema=schema, model="m", max_tokens=10)
    req = client.messages.last_request
    assert req["output_config"] == {"format": {"type": "json_schema", "schema": schema}}
    assert "tools" not in req and "temperature" not in req
    assert out == '{"a": 1}'


async def test_openai_structured_request(oai):
    adapter, client = oai
    client._api._response = _openai_raw(text='{"a": 1}')
    schema = {"type": "object", "properties": {"a": {"type": "integer"}},
              "required": ["a"], "additionalProperties": False}
    out = await adapter.structured(system="", messages=[UserMessage(content="go")],
                                   schema=schema, model="m", max_tokens=10)
    rf = client._api.last_request["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == schema
    assert out == '{"a": 1}'


# ---------------- usage normalization ----------------

async def test_openai_cached_tokens_normalized(oai):
    adapter, client = oai
    client._api._response = _openai_raw(prompt=1000, cached=800)
    resp = await adapter.call(system="", messages=[UserMessage(content="x")], tools=[],
                              model="m", max_tokens=10)
    assert resp.usage.input_tokens == 200          # uncached remainder
    assert resp.usage.cache_read_input_tokens == 800


async def test_openai_usage_without_details(oai):
    adapter, client = oai
    client._api._response = _openai_raw(prompt=1000, cached=0)
    resp = await adapter.call(system="", messages=[UserMessage(content="x")], tools=[],
                              model="m", max_tokens=10)
    assert resp.usage.input_tokens == 1000
    assert resp.usage.cache_read_input_tokens == 0


# ---------------- stop-reason mapping ----------------

@pytest.mark.parametrize("value,expected", [
    ("pause_turn", StopReason.PAUSE_TURN),
    ("refusal", StopReason.REFUSAL),
    ("model_context_window_exceeded", StopReason.CONTEXT_EXCEEDED),
    ("end_turn", StopReason.END_TURN),
    ("totally_unknown", StopReason.END_TURN),
])
def test_anthropic_stop_reason_map(value, expected):
    assert anth_stop(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("content_filter", StopReason.REFUSAL),
    ("tool_calls", StopReason.TOOL_USE),
    ("length", StopReason.MAX_TOKENS),
    ("stop", StopReason.END_TURN),
])
def test_openai_stop_reason_map(value, expected):
    assert oai_stop(value) == expected


# ---------------- client construction kwargs ----------------

def test_injected_client_skips_sdk_construction():
    sentinel = FakeAnthropicClient()
    adapter = AnthropicAdapter(api_key="ignored", client=sentinel)
    assert adapter._client is sentinel
    sentinel2 = FakeOpenAIClient()
    adapter2 = OpenAIAdapter(api_key="ignored", client=sentinel2)
    assert adapter2._client is sentinel2


def test_sdk_client_receives_retry_and_extra_kwargs(monkeypatch):
    captured: dict = {}

    class FakeSDKClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import anthropic as anthropic_sdk
    monkeypatch.setattr(anthropic_sdk, "AsyncAnthropic", FakeSDKClient)
    AnthropicAdapter(api_key="test-key", timeout=33.0, max_retries=5,
                     client_kwargs={"default_headers": {"x": "1"}})
    assert captured["api_key"] == "test-key"
    assert captured["timeout"] == 33.0
    assert captured["max_retries"] == 5
    assert captured["default_headers"] == {"x": "1"}


# ---------------- server-side tool-result clearing (context editing) ----------------

_CLEARING = ToolResultClearing(
    trigger_tokens=150_000,
    keep_results=5,
    clear_at_least_tokens=15_000,
    exclude_tools=("ask_user", "read_spec"),
)


def _anthropic_raw_with_edits(cleared_uses=3, cleared_tokens=42_000):
    raw = _anthropic_raw()
    raw.context_management = SimpleNamespace(
        applied_edits=[
            SimpleNamespace(
                type="clear_tool_uses_20250919",
                cleared_tool_uses=cleared_uses,
                cleared_input_tokens=cleared_tokens,
            )
        ]
    )
    return raw


class _FakeAnthropicStream:
    """`client.messages.stream(...)`: an async context manager of events."""

    def __init__(self, events, final):
        self._events, self._final = events, final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._final


class _FakeStreamingAPI(_FakeMessagesAPI):
    def __init__(self, response, events):
        super().__init__(response)
        self._events = events

    def stream(self, **kwargs):
        self.last_request = kwargs
        return _FakeAnthropicStream(self._events, self._response)


class FakeBetaAnthropicClient:
    """Both the regular and the beta Messages API, recorded separately."""

    def __init__(self, response=None, events=()):
        response = response or _anthropic_raw()
        self.messages = _FakeStreamingAPI(response, list(events))
        self.beta = SimpleNamespace(messages=_FakeStreamingAPI(response, list(events)))


async def test_golden_request_with_tool_result_clearing():
    client = FakeBetaAnthropicClient()
    adapter = AnthropicAdapter(api_key="test-key", client=client)
    await adapter.call(
        system="sys", messages=[UserMessage(content="hi")], tools=[_TOOL],
        model="claude-opus-5-5", max_tokens=128, context_edits=_CLEARING,
    )
    req = client.beta.messages.last_request
    assert req["betas"] == ["context-management-2025-06-27"]
    assert req["context_management"] == {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 150_000},
                "keep": {"type": "tool_uses", "value": 5},
                "clear_at_least": {"type": "input_tokens", "value": 15_000},
                "exclude_tools": ["ask_user", "read_spec"],
            }
        ]
    }
    # Everything else is the regular request, unchanged.
    plain = FakeBetaAnthropicClient()
    await AnthropicAdapter(api_key="k", client=plain).call(
        system="sys", messages=[UserMessage(content="hi")], tools=[_TOOL],
        model="claude-opus-5-5", max_tokens=128,
    )
    expected = dict(plain.messages.last_request)
    assert {k: v for k, v in req.items() if k not in ("betas", "context_management")} == expected


async def test_beta_client_only_when_clearing_is_requested():
    client = FakeBetaAnthropicClient()
    adapter = AnthropicAdapter(api_key="test-key", client=client)
    await adapter.call(
        system="", messages=[UserMessage(content="hi")], tools=[],
        model="claude-opus-5-5", max_tokens=64,
    )
    assert client.messages.last_request is not None
    assert client.beta.messages.last_request is None
    assert "betas" not in client.messages.last_request

    await adapter.call(
        system="", messages=[UserMessage(content="hi")], tools=[],
        model="claude-opus-5-5", max_tokens=64, context_edits=_CLEARING,
    )
    assert client.beta.messages.last_request is not None


async def test_caller_betas_are_kept_alongside_context_management():
    client = FakeBetaAnthropicClient()
    await AnthropicAdapter(api_key="k", client=client).call(
        system="", messages=[UserMessage(content="hi")], tools=[],
        model="claude-opus-5-5", max_tokens=64, context_edits=_CLEARING,
        extra={"betas": ["fast-mode-2026-02-01"]},
    )
    assert client.beta.messages.last_request["betas"] == [
        "fast-mode-2026-02-01",
        "context-management-2025-06-27",
    ]


async def test_applied_edits_are_normalized():
    raw = _anthropic_raw_with_edits()
    expected = [AppliedEdit("clear_tool_uses_20250919", 3, 42_000)]

    client = FakeBetaAnthropicClient(response=raw)
    adapter = AnthropicAdapter(api_key="k", client=client)
    response = await adapter.call(
        system="", messages=[UserMessage(content="hi")], tools=[],
        model="claude-opus-5-5", max_tokens=64, context_edits=_CLEARING,
    )
    assert response.applied_edits == expected

    events = [
        SimpleNamespace(
            type="content_block_start", index=0, content_block=SimpleNamespace(type="text")
        ),
        SimpleNamespace(
            type="content_block_delta", index=0,
            delta=SimpleNamespace(type="text_delta", text="ok"),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=raw.usage,
            context_management=raw.context_management,
        ),
    ]
    client = FakeBetaAnthropicClient(response=raw, events=events)
    adapter = AnthropicAdapter(api_key="k", client=client)
    chunks = [
        c
        async for c in adapter.stream(
            system="", messages=[UserMessage(content="hi")], tools=[],
            model="claude-opus-5-5", max_tokens=64, context_edits=_CLEARING,
        )
    ]
    usage = next(c for c in chunks if c.type is StreamChunkType.USAGE)
    stop = next(c for c in chunks if c.type is StreamChunkType.STOP)
    assert [AppliedEdit.from_dict(e) for e in usage.payload["applied_edits"]] == expected
    assert stop.payload["applied_edits"] == usage.payload["applied_edits"]


class _BadRequest(Exception):
    status_code = 400


@pytest.mark.parametrize("streaming", [False, True])
async def test_a_rejection_of_the_edits_is_normalized(streaming):
    client = FakeBetaAnthropicClient()

    def reject(**kwargs):
        raise _BadRequest("Error code: 400 - context_management: Extra inputs are not permitted")

    async def reject_async(**kwargs):
        reject()

    client.beta.messages.create = reject_async
    client.beta.messages.stream = reject
    adapter = AnthropicAdapter(api_key="k", client=client)
    request = {
        "system": "", "messages": [UserMessage(content="hi")], "tools": [],
        "model": "claude-opus-5-5", "max_tokens": 64, "context_edits": _CLEARING,
    }
    with pytest.raises(ContextEditsRejectedError):
        if streaming:
            async for _ in adapter.stream(**request):
                pass
        else:
            await adapter.call(**request)


async def test_an_unrelated_400_is_not_mistaken_for_a_rejection():
    client = FakeBetaAnthropicClient()

    async def fail(**kwargs):
        raise _BadRequest("Error code: 400 - max_tokens: must be positive")

    client.beta.messages.create = fail
    with pytest.raises(_BadRequest):
        await AnthropicAdapter(api_key="k", client=client).call(
            system="", messages=[UserMessage(content="hi")], tools=[],
            model="claude-opus-5-5", max_tokens=64, context_edits=_CLEARING,
        )
