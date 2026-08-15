"""Tests for core/message.py"""
from agent_framework.core.message import (
    AssistantMessage,
    StopReason,
    ToolCallRequest,
    ToolResult,
    ToolResultMessage,
    TokenUsage,
    UserMessage,
    estimate_message_tokens,
    estimate_tokens,
)


def test_token_usage_add():
    a = TokenUsage(input_tokens=10, output_tokens=20)
    b = TokenUsage(input_tokens=5, output_tokens=8)
    c = a + b
    assert c.input_tokens == 15
    assert c.output_tokens == 28
    assert c.total == 43


def test_tool_result_as_text_dict():
    tr = ToolResult(tool_use_id="id1", name="search", content={"results": ["a", "b"]})
    text = tr.as_text()
    assert "results" in text


def test_tool_result_as_text_str():
    tr = ToolResult(tool_use_id="id1", name="calc", content="42")
    assert tr.as_text() == "42"


def test_user_message_to_dict():
    msg = UserMessage(content="hello")
    d = msg.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "hello"


def test_assistant_message_with_tool_calls():
    tc = ToolCallRequest(id="call_1", name="search", arguments={"q": "test"})
    msg = AssistantMessage(
        content="Searching...",
        tool_calls=[tc],
        stop_reason=StopReason.TOOL_USE,
    )
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "search"


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_rough():
    text = "a" * 100
    # ~4 chars/token
    assert 20 <= estimate_tokens(text) <= 30


def test_estimate_message_tokens_user():
    msg = UserMessage(content="Hello world, how are you today?")
    tokens = estimate_message_tokens(msg)
    assert tokens > 0


def test_tool_result_message_tokens():
    tr = ToolResult(tool_use_id="id1", name="tool", content="short result")
    msg = ToolResultMessage(content=[tr])
    tokens = estimate_message_tokens(msg)
    assert tokens > 0
