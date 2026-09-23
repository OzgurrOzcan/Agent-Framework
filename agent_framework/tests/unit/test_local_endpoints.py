"""Self-hosted and OpenAI-compatible endpoints.

Any server speaking the OpenAI wire format — vLLM, llama.cpp, Ollama, a
gateway — is reached by naming the provider explicitly and pointing `base_url`
at it. Two things differ from a vendor API and both are handled here: such a
server usually needs no credential, and its context window is not in the
catalog.

The end-to-end test runs against a real HTTP server on localhost, so the suite
stays offline while still exercising the actual openai SDK transport.
"""

from __future__ import annotations

import json
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.config import LOCAL_API_KEY_PLACEHOLDER, AgentConfig
from agent_framework.permissions.models import PermissionCategory
from agent_framework.tools.decorator import tool

LOCAL_MODEL = "openai:llama-3.3-70b"
LOCAL_WINDOW = 32_768


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch):
    """A self-hosted endpoint must not depend on vendor keys being present."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def local_config(**kw) -> AgentConfig:
    kw.setdefault("model", LOCAL_MODEL)
    kw.setdefault("base_url", "http://127.0.0.1:1/v1")
    kw.setdefault("context_window", LOCAL_WINDOW)
    return AgentConfig(**kw)


# ---------------- credentials ----------------

def test_base_url_makes_the_api_key_optional():
    """Otherwise every local model needs an invented placeholder key."""
    assert local_config().resolved_api_key() == LOCAL_API_KEY_PLACEHOLDER


def test_an_explicit_key_still_wins():
    assert local_config(api_key="sk-gateway").resolved_api_key() == "sk-gateway"


def test_the_environment_still_wins_over_the_placeholder(monkeypatch):
    """A gateway that does authenticate keeps working from the env var."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert local_config().resolved_api_key() == "sk-from-env"


def test_without_a_base_url_a_key_is_still_required():
    """The vendor's own API is unchanged: no silent placeholder."""
    with pytest.raises(RuntimeError, match="No API key"):
        AgentConfig(model="gpt-5").resolved_api_key()


def test_the_missing_key_error_mentions_the_self_hosted_escape():
    with pytest.raises(RuntimeError, match="base_url"):
        AgentConfig(model="gpt-5").resolved_api_key()


def test_an_agent_builds_against_a_local_endpoint_without_any_key():
    agent = Agent(LOCAL_MODEL, base_url="http://127.0.0.1:1/v1", context_window=LOCAL_WINDOW)
    assert agent.config.provider == "openai"
    assert agent.config.model == "llama-3.3-70b"


# ---------------- context window ----------------

def test_an_unknown_model_warns_that_its_window_was_guessed():
    """A 8k model budgeted as 200k never compacts and overflows at the server."""
    with pytest.warns(UserWarning, match="Unknown context window"):
        AgentConfig(model=LOCAL_MODEL, base_url="http://127.0.0.1:1/v1")


def test_setting_the_window_silences_the_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        local_config()


def test_a_catalogued_model_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AgentConfig(model="claude-opus-5", api_key="k")
        AgentConfig(model="gpt-4o", api_key="k")


def test_context_window_is_settable_from_the_flat_surface():
    """It pairs with base_url, so it must not require a config object."""
    agent = Agent(
        LOCAL_MODEL, base_url="http://127.0.0.1:1/v1", context_window=LOCAL_WINDOW
    )
    assert agent.config.resolved_context_window == LOCAL_WINDOW


def test_a_small_window_rejects_the_default_buffers():
    """8k models need smaller buffers; the error says exactly that."""
    with pytest.raises(ValueError, match="less than the context window"):
        local_config(context_window=8192)


def test_a_small_window_works_once_the_buffers_fit():
    config = local_config(
        context_window=8192, response_buffer_tokens=1024, compaction_buffer_tokens=512
    )
    assert config.resolved_context_window == 8192


# ---------------- end to end against a real local server ----------------

class _Handler(BaseHTTPRequestHandler):
    """Minimal /v1/chat/completions: one tool call, then a final answer."""

    # Deliberately class-level: the handler is instantiated per request.
    received: ClassVar[list[dict]] = []

    def log_message(self, *args):  # keep pytest output clean
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.received.append(
            {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        )
        if len(_Handler.received) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_stock", "arguments": '{"sku": "LAP-14"}'},
                    }
                ],
            }
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": "LAP-14: 7 in stock."}
            finish = "stop"
        raw = json.dumps(
            {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "local"),
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 18, "total_tokens": 138},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def local_server():
    """An OpenAI-compatible server on an ephemeral localhost port."""
    _Handler.received = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", _Handler
    finally:
        server.shutdown()
        server.server_close()


@tool(description="Stock for a SKU.", permission=PermissionCategory.AUTO)
async def get_stock(sku: str) -> dict:
    return {"sku": sku, "quantity": 7}


async def test_the_full_loop_runs_against_a_local_server(local_server):
    """Tool calling, results fed back, natural stop — no vendor API involved."""
    base_url, handler = local_server
    agent = Agent(
        LOCAL_MODEL,
        base_url=base_url,
        context_window=LOCAL_WINDOW,
        system_prompt="You are a stock assistant.",
        tools=[get_stock],
        max_turns=4,
    )
    result = await agent.run("Is LAP-14 in stock?")

    assert result.error is None
    assert result.final_message == "LAP-14: 7 in stock."
    assert (result.state.turn, result.state.tool_rounds) == (2, 1)
    # Two round trips: the tool request, then the synthesis.
    assert len(handler.received) == 2


async def test_the_local_server_receives_the_bare_model_name(local_server):
    """The "openai:" prefix selects the adapter; it must not reach the wire."""
    base_url, handler = local_server
    agent = Agent(LOCAL_MODEL, base_url=base_url, context_window=LOCAL_WINDOW,
                  tools=[get_stock])
    await agent.run("go")
    assert handler.received[0]["body"]["model"] == "llama-3.3-70b"


async def test_tools_are_advertised_in_openai_wire_format(local_server):
    base_url, handler = local_server
    agent = Agent(LOCAL_MODEL, base_url=base_url, context_window=LOCAL_WINDOW,
                  tools=[get_stock])
    await agent.run("go")
    tools = handler.received[0]["body"]["tools"]
    assert {t["function"]["name"] for t in tools} == {"get_stock", "ask_user"}
    assert tools[0]["type"] == "function"


async def test_the_placeholder_credential_is_sent_as_a_bearer_token(local_server):
    """The header must exist even when its value is meaningless."""
    base_url, handler = local_server
    agent = Agent(LOCAL_MODEL, base_url=base_url, context_window=LOCAL_WINDOW)
    await agent.run("go")
    assert handler.received[0]["auth"] == f"Bearer {LOCAL_API_KEY_PLACEHOLDER}"


async def test_an_unpriced_local_model_reports_zero_cost(local_server):
    """Local inference is free, so 0.0 is right — but the guard is inactive."""
    base_url, _ = local_server
    agent = Agent(LOCAL_MODEL, base_url=base_url, context_window=LOCAL_WINDOW)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        await agent.run("go")
    usd, usage = agent.cost.snapshot()
    assert usd == 0.0
    assert usage.input_tokens > 0
    assert agent.get_trace().all_spans()[0].payload["pricing_resolved"] is False
