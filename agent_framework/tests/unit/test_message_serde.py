"""Message (de)serialization round-trip tests (T3.1)."""

from __future__ import annotations

import pytest

from agent_framework.core.message import (
    AssistantMessage,
    ReasoningContent,
    StopReason,
    SystemMessage,
    TokenUsage,
    ToolCallRequest,
    ToolResult,
    ToolResultMessage,
    UserMessage,
    message_from_dict,
)


def test_user_message_roundtrip():
    m = UserMessage(content="hello", metadata={"k": "v"})
    r = message_from_dict(m.to_dict())
    assert isinstance(r, UserMessage)
    assert r.content == "hello"
    assert r.metadata == {"k": "v"}
    assert r.timestamp == m.timestamp


def test_system_message_roundtrip():
    m = SystemMessage(content="rules")
    r = message_from_dict(m.to_dict())
    assert isinstance(r, SystemMessage)
    assert r.content == "rules"


def test_assistant_message_full_roundtrip():
    m = AssistantMessage(
        content="working",
        tool_calls=[ToolCallRequest(id="c1", name="search", arguments={"q": "x"})],
        reasoning=[ReasoningContent(
            provider="anthropic",
            raw={"type": "thinking", "thinking": "hm", "signature": "s"},
            text="hm",
        )],
        stop_reason=StopReason.TOOL_USE,
        usage=TokenUsage(input_tokens=10, output_tokens=20,
                         cache_read_input_tokens=5),
    )
    r = message_from_dict(m.to_dict())
    assert isinstance(r, AssistantMessage)
    assert r.content == "working"
    assert r.tool_calls == m.tool_calls
    assert r.reasoning[0].raw == m.reasoning[0].raw
    assert r.stop_reason == StopReason.TOOL_USE
    assert r.usage == m.usage


def test_tool_result_message_roundtrip():
    m = ToolResultMessage(content=[
        ToolResult(tool_use_id="c1", name="search", content="found",
                   is_error=False, duration_ms=12),
        ToolResult(tool_use_id="c2", name="fetch", content="fail",
                   is_error=True),
    ])
    r = message_from_dict(m.to_dict())
    assert isinstance(r, ToolResultMessage)
    assert len(r.content) == 2
    assert r.content[0].tool_use_id == "c1"
    assert r.content[0].duration_ms == 12
    assert r.content[1].is_error is True


def test_legacy_dict_without_type_falls_back_with_warning():
    with pytest.warns(UserWarning, match="no 'type'"):
        r = message_from_dict({"role": "assistant", "content": "hi"})
    assert isinstance(r, AssistantMessage)
    # tool-result heuristic
    with pytest.warns(UserWarning):
        r2 = message_from_dict({
            "role": "user",
            "content": [{"tool_use_id": "a", "name": "t", "content": "r",
                         "is_error": False}],
        })
    assert isinstance(r2, ToolResultMessage)


def test_checkpoint_restore_full_fidelity():
    from agent_framework.core.base_agent import BaseAgent
    from agent_framework.core.config import AgentConfig, ProviderConfig, ProviderKind

    agent = BaseAgent.create(AgentConfig(
        provider=ProviderConfig(
            provider=ProviderKind.ANTHROPIC, model="claude-opus-5", api_key="test-key",
        ),
    ))
    agent.memory.short_term.append(UserMessage(content="q"))
    agent.memory.short_term.append(AssistantMessage(
        content="a",
        tool_calls=[ToolCallRequest(id="c1", name="t", arguments={"x": 1})],
    ))
    agent.memory.short_term.append(ToolResultMessage(content=[
        ToolResult(tool_use_id="c1", name="t", content="r"),
    ]))

    cid = agent.checkpoint()
    original = [m.to_dict() for m in agent.get_history()]
    agent.clear_history()
    agent.memory.short_term.append(UserMessage(content="unrelated"))

    agent.restore(cid)
    restored = [m.to_dict() for m in agent.get_history()]
    assert restored == original
    # Restored history must be a valid provider conversation.
    assert isinstance(agent.get_history()[2], ToolResultMessage)
