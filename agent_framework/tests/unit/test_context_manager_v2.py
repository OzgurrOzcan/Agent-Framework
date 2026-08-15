"""ContextManager v2 tests: persistent compaction, safe cuts, preamble
placement, token calibration."""

from __future__ import annotations


from agent_framework.core.config import CompactionConfig
from agent_framework.core.context_manager import (
    ContextManager,
    _sliding_window,
)
from agent_framework.core.message import (
    AssistantMessage,
    ToolCallRequest,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from agent_framework.memory.manager import MemoryManager
from agent_framework.memory.long_term import LongTermMemory
from agent_framework.memory.backends.in_memory import InMemoryBackend
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn


def _mgr(**overrides) -> ContextManager:
    defaults = dict(
        system_prompt="sys",
        context_window=10_000,
        response_buffer_tokens=100,
        compaction_buffer_tokens=100,
        compaction=CompactionConfig(enabled=False),
    )
    defaults.update(overrides)
    return ContextManager(**defaults)


def _bulky_history(n_pairs: int = 6) -> list:
    out = []
    for i in range(n_pairs):
        out.append(UserMessage(content=f"q{i} " + "x" * 400))
        out.append(AssistantMessage(content=f"a{i} " + "y" * 400))
    return out


# ---------------- persistent compaction (T1.3) ----------------

async def test_summary_computed_once_then_persisted():
    summarizer = MockLLMProvider([ScriptedTurn(text="SUMMARY-1")])
    mgr = _mgr(
        context_window=3000,
        compaction=CompactionConfig(
            enabled=True, trigger_ratio=0.3, tool_result_truncate_threshold=100_000,
            sliding_window_min_turns=50, full_compaction_min_recent_turns=1,
            summary_max_tokens=40,
        ),
        summarizer=summarizer,
        summarizer_model="m",
    )
    history = _bulky_history()
    first = await mgr.assemble(history, current_user_input="q")
    assert first.persist is not None
    # The adaptive cutoff shrinks below the trigger threshold in ONE call.
    calls_after_first = summarizer._call_count
    assert calls_after_first == 1
    assert isinstance(first.persist[0], UserMessage)
    assert "SUMMARY-1" in str(first.persist[0].content)
    # Persisted summary is stable history — must NOT carry the volatile
    # "system" metadata key (it would be excluded from prompt caching).
    assert "system" not in first.persist[0].metadata

    # Second turn works on the persisted history → no new summary call.
    second = await mgr.assemble(list(first.persist), current_user_input="q2")
    assert summarizer._call_count == calls_after_first, (
        "summary must not be recomputed every turn"
    )
    assert second.persist is None


async def test_no_persist_when_compaction_disabled():
    mgr = _mgr()
    out = await mgr.assemble(_bulky_history(), current_user_input="q")
    assert out.persist is None


# ---------------- safe cut points (T1.4) ----------------

def test_sliding_window_never_starts_with_assistant():
    msgs = [
        UserMessage(content="u0"),
        AssistantMessage(content="a0"),
        UserMessage(content="u1"),
        AssistantMessage(content="a1"),
    ]
    # Desired cutoff (min=2) lands exactly on u1 — a safe cut.
    kept, reclaimed = _sliding_window(msgs, min_messages=2)
    assert [m.content for m in kept] == ["u1", "a1"]
    assert reclaimed > 0
    # Desired cutoff (min=3) would land on a0 (assistant) — walked back to 0,
    # everything kept rather than starting the history with an assistant turn.
    kept3, reclaimed3 = _sliding_window(msgs, min_messages=3)
    assert len(kept3) == 4
    assert reclaimed3 == 0


def test_sliding_window_keeps_all_when_no_safe_cut():
    msgs = [
        UserMessage(content="u"),
        AssistantMessage(content="a", tool_calls=[
            ToolCallRequest(id="t", name="x", arguments={})]),
        ToolResultMessage(content=[ToolResult(tool_use_id="t", name="x", content="r")]),
    ]
    kept, reclaimed = _sliding_window(msgs, min_messages=2)
    assert len(kept) == 3
    assert reclaimed == 0


def test_sliding_window_invariants_random_histories():
    import random
    rng = random.Random(42)
    for _ in range(50):
        msgs: list = []
        for i in range(rng.randint(4, 16)):
            msgs.append(UserMessage(content=f"u{i} " + "z" * rng.randint(10, 80)))
            if rng.random() < 0.5:
                tc = ToolCallRequest(id=f"t{i}", name="tool", arguments={"i": i})
                msgs.append(AssistantMessage(content="", tool_calls=[tc]))
                msgs.append(ToolResultMessage(content=[
                    ToolResult(tool_use_id=f"t{i}", name="tool", content="r")]))
            else:
                msgs.append(AssistantMessage(content=f"a{i}"))
        for min_msgs in (2, 4, 6):
            kept, _ = _sliding_window(msgs, min_msgs)
            # invariant 1: starts with a plain user message
            assert isinstance(kept[0], UserMessage)
            assert not isinstance(kept[0], ToolResultMessage)
            # invariant 2: every tool_use has its result and vice versa
            use_ids = {tc.id for m in kept if isinstance(m, AssistantMessage)
                       for tc in m.tool_calls}
            result_ids = {r.tool_use_id for m in kept
                          if isinstance(m, ToolResultMessage) for r in m.content}
            assert use_ids == result_ids


# ---------------- preamble placement (T2.4) ----------------

async def test_preamble_is_appended_after_history():
    backend = InMemoryBackend()
    lt = LongTermMemory(backend=backend)
    await lt.remember("deploy_env", "production cluster eu-1", tags=["env"])
    memory = MemoryManager(long_term=lt)
    mgr = _mgr(memory=memory)

    history = [UserMessage(content="first"), AssistantMessage(content="hello")]
    out = await mgr.assemble(history, current_user_input="deploy_env status?")
    assert out.messages[0].content == "first"          # stable prefix first
    last = out.messages[-1]
    assert last.metadata.get("system") == "relevant_context"
    assert "deploy_env" in str(last.content)


# ---------------- calibration (T3.4) ----------------

async def test_calibration_adjusts_toward_actual():
    mgr = _mgr()
    await mgr.assemble([UserMessage(content="hello world " * 50)],
                       current_user_input="q")
    assert mgr.calibration == 1.0
    est = mgr._last_estimate_total
    # Provider reports 3x the estimate → factor moves up.
    mgr.note_actual_tokens(est * 3)
    assert mgr.calibration > 1.0
    first = mgr.calibration
    mgr.note_actual_tokens(est * 3)
    assert mgr.calibration > first
    # Factor is clamped.
    for _ in range(50):
        mgr.note_actual_tokens(est * 100)
    assert mgr.calibration <= 4.0


async def test_calibration_ignores_zero_usage():
    mgr = _mgr()
    await mgr.assemble([UserMessage(content="hi")], current_user_input="q")
    mgr.note_actual_tokens(0)
    assert mgr.calibration == 1.0
