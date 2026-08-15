"""Tests for ContextManager and compaction helpers."""
import pytest
from agent_framework.core.config import CompactionConfig
from agent_framework.core.context_manager import (
    ContextManager,
    _sliding_window,
    _truncate_tool_results,
)
from agent_framework.core.message import (
    AssistantMessage,
    ToolCallRequest,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)


def _make_tool_pair(call_id: str, result_text: str):
    """Create an AssistantMessage + ToolResultMessage pair."""
    tc = ToolCallRequest(id=call_id, name="tool", arguments={})
    assistant = AssistantMessage(content="", tool_calls=[tc])
    result = ToolResultMessage(content=[
        ToolResult(tool_use_id=call_id, name="tool", content=result_text)
    ])
    return assistant, result


# --- _sliding_window ---

def test_sliding_window_basic():
    msgs = [UserMessage(content=f"msg {i}") for i in range(10)]
    kept, reclaimed = _sliding_window(msgs, min_messages=4)
    assert len(kept) == 4
    assert reclaimed > 0


def test_sliding_window_no_drop_when_at_min():
    msgs = [UserMessage(content=f"msg {i}") for i in range(3)]
    kept, reclaimed = _sliding_window(msgs, min_messages=3)
    assert len(kept) == 3
    assert reclaimed == 0


def test_sliding_window_preserves_tool_pair():
    """Cutoff must not land between assistant+tool_result pair."""
    user = UserMessage(content="question")
    assistant, result = _make_tool_pair("c1", "answer")
    # 3 messages: [user, assistant(tool_calls), tool_result]
    # min_messages=2 → initial cutoff=1 (after user).
    # But messages[1] = AssistantMessage with tool_calls and messages[2]=ToolResultMessage
    # → cutoff must back up to 0 → keep everything.
    msgs = [user, assistant, result]
    kept, reclaimed = _sliding_window(msgs, min_messages=2)
    assert len(kept) == 3  # can't safely drop any
    assert reclaimed == 0


def test_sliding_window_safe_split_before_pair():
    """Two user messages followed by a tool pair — can drop first user message."""
    u1 = UserMessage(content="first question")
    u2 = UserMessage(content="second question")
    assistant, result = _make_tool_pair("c1", "answer")
    msgs = [u1, u2, assistant, result]
    # min_messages=3 → initial cutoff=1
    # messages[0]=user, messages[1]=user → safe split after index 0.
    kept, reclaimed = _sliding_window(msgs, min_messages=3)
    assert len(kept) == 3
    assert kept[0].content == "second question"


# --- _truncate_tool_results ---

def test_truncate_large_tool_result():
    large_content = "x" * 10_000  # ~2500 tokens
    tr = ToolResult(tool_use_id="id1", name="tool", content=large_content)
    msg = ToolResultMessage(content=[tr])
    out, reclaimed = _truncate_tool_results([msg], threshold_tokens=100)
    assert reclaimed > 0
    result_text = out[0].content[0].as_text()
    assert "truncated" in result_text


def test_truncate_small_tool_result_unchanged():
    small_content = "short result"
    tr = ToolResult(tool_use_id="id1", name="tool", content=small_content)
    msg = ToolResultMessage(content=[tr])
    out, reclaimed = _truncate_tool_results([msg], threshold_tokens=500)
    assert reclaimed == 0
    assert out[0].content[0].content == small_content


def test_truncate_preserves_non_tool_messages():
    user = UserMessage(content="hello")
    out, reclaimed = _truncate_tool_results([user], threshold_tokens=100)
    assert len(out) == 1
    assert isinstance(out[0], UserMessage)


# --- ContextManager.assemble ---

@pytest.mark.asyncio
async def test_assemble_returns_system_prompt():
    ctx = ContextManager(
        system_prompt="You are helpful.",
        context_window=10000,
        response_buffer_tokens=256,
        compaction_buffer_tokens=128,
        compaction=CompactionConfig(enabled=False),
    )
    result = await ctx.assemble([UserMessage(content="hello")])
    assert result.system_prompt == "You are helpful."
    assert len(result.messages) >= 1


@pytest.mark.asyncio
async def test_assemble_no_compaction_when_disabled():
    ctx = ContextManager(
        system_prompt="",
        context_window=10000,
        response_buffer_tokens=256,
        compaction_buffer_tokens=128,
        compaction=CompactionConfig(enabled=False),
    )
    msgs = [UserMessage(content=f"message {i}") for i in range(20)]
    result = await ctx.assemble(msgs)
    assert result.compaction_applied == []
