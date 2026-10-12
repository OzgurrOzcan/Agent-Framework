"""ContextManager: token calibration against the provider's reported usage."""

from __future__ import annotations

from agent_framework.core.context_manager import ContextManager
from agent_framework.core.message import UserMessage
from agent_framework.memory.manager import MemoryManager
from agent_framework.memory.short_term import ShortTermMemory


def _mgr(memory: MemoryManager | None = None) -> ContextManager:
    return ContextManager(
        system_prompt="sys",
        max_context_tokens=10_000,
        max_tool_result_tokens=1_000,
        memory=memory,
    )


def _memory(*messages) -> ShortTermMemory:
    memory = ShortTermMemory()
    memory.extend(messages)
    return memory


# ---------------- calibration ----------------

async def test_calibration_adjusts_toward_actual():
    mgr = _mgr()
    await mgr.assemble(_memory(UserMessage(content="hello world " * 50)))
    assert mgr.calibration == 1.0
    est = mgr._pending.raw_total
    # Provider reports 3x the estimate → factor moves up.
    mgr.note_actual_tokens(est * 3)
    assert mgr.calibration > 1.0
    first = mgr.calibration
    mgr.note_actual_tokens(est * 3)
    assert mgr.calibration > first
    # Factor is clamped.
    for _ in range(50):
        mgr.note_actual_tokens(est * 9)
    assert mgr.calibration == 4.0


async def test_implausible_usage_is_ignored():
    """A proxy reporting placeholder usage must not anchor every later estimate."""
    mgr = _mgr()
    memory = _memory(UserMessage(content="hello world " * 500))
    await mgr.assemble(memory)
    est = mgr._pending.raw_total
    mgr.note_actual_tokens(10)  # far too small to describe this prompt
    mgr.note_actual_tokens(est * 50)  # far too large
    assert mgr.calibration == 1.0
    assert mgr._anchor is None


async def test_calibration_ignores_zero_usage():
    mgr = _mgr()
    await mgr.assemble(_memory(UserMessage(content="hi")))
    mgr.note_actual_tokens(0)
    assert mgr.calibration == 1.0
