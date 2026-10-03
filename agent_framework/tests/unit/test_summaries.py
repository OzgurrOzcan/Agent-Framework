"""Long-term summaries: what a conversation leaves for the next one.

The mock provider is also the summarizer, so its script runs in call order:
a trim's summary call comes before the call it trims for.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.config import AgentConfig, MemoryConfig
from agent_framework.core.message import AssistantMessage, StopReason, UserMessage
from agent_framework.memory.summaries import SUMMARY_BUDGET_TOKENS
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn
from agent_framework.tracing.trace import EventType, SpanStatus

SUMMARIZER = "You write the long-term memory"


def make_agent(db, script=(), *, session_id="s1", max_context_tokens=None, **memory):
    mock = MockLLMProvider(list(script))
    agent = Agent(
        AgentConfig(
            model="claude-haiku-4-5",
            api_key="k",
            max_context_tokens=max_context_tokens,
            max_tool_result_tokens=300 if max_context_tokens else None,
        ),
        provider=mock,
        memory=MemoryConfig(store=str(db), **memory),
        session_id=session_id,
    )
    return agent, mock


def fill(agent: Agent, turns: range, chars: int = 2_000) -> None:
    for i in turns:
        agent.memory.short_term.extend(
            [UserMessage(content=f"question {i}"), AssistantMessage(content=f"answer {i} " + "x" * chars)]
        )


def summary_calls(mock: MockLLMProvider) -> list[dict]:
    return [c for c in mock.call_log if c["system"].startswith(SUMMARIZER)]


def events(agent: Agent, name: str):
    return [e for e in agent.get_trace().all_spans() if e.name == name]


async def segments(agent: Agent) -> list[str]:
    assert agent._summarizer is not None
    return await agent._summarizer.segments()


# ---------------- summaries at trims ----------------

async def test_dropped_turns_are_summarized_once_per_trim(tmp_path):
    agent, mock = make_agent(
        tmp_path / "m.db",
        [ScriptedTurn(text="SUMMARY-1"), ScriptedTurn(text="answer")],
        max_context_tokens=3_000,
    )
    fill(agent, range(8))
    await agent.run("question 8")

    [call] = summary_calls(mock)
    excerpt = call["messages"][0].content
    assert "question 0" in excerpt and "question 8" not in excerpt  # only what was dropped
    assert len(mock.call_log) == 2
    assert await segments(agent) == ["SUMMARY-1"]
    [event] = events(agent, "summary")
    assert event.payload["source"] == "trim"
    assert event.token_usage is not None
    # The summary is shown at the head of the trimmed view.
    assert "SUMMARY-1" in mock.call_log[1]["messages"][0].content
    assert agent._session.summarized_upto > 0
    await agent.memory.close()


async def test_summarize_false_makes_no_calls(tmp_path):
    agent, mock = make_agent(
        tmp_path / "m.db", [ScriptedTurn(text="answer")], max_context_tokens=3_000, summarize=False
    )
    fill(agent, range(8))
    await agent.run("question 8")

    assert summary_calls(mock) == []
    assert len(mock.call_log) == 1
    assert await segments(agent) == []
    await agent.memory.close()


async def test_failed_summary_retries_at_the_next_trim(tmp_path):
    agent, mock = make_agent(
        tmp_path / "m.db",
        [
            ScriptedTurn(raise_exc=RuntimeError("overloaded")),  # first summary fails
            ScriptedTurn(text="answer 8"),
            ScriptedTurn(text="SUMMARY-RETRY"),
            ScriptedTurn(text="answer 16"),
        ],
        max_context_tokens=3_000,
    )
    fill(agent, range(8))
    await agent.run("question 8")
    pending = agent.memory.short_term.state.pending
    assert pending and events(agent, "summary_failed")
    assert await segments(agent) == []

    fill(agent, range(9, 16))
    await agent.run("question 16")

    [_, retry] = summary_calls(mock)
    assert "question 0" in retry["messages"][0].content  # the failed turns came along
    assert await segments(agent) == ["SUMMARY-RETRY"]
    assert agent.memory.short_term.state.pending == []
    await agent.memory.close()


async def test_segments_are_appended_not_rewritten(tmp_path):
    agent, _ = make_agent(
        tmp_path / "m.db",
        [
            ScriptedTurn(text="FIRST"),
            ScriptedTurn(text="a"),
            ScriptedTurn(text="SECOND"),
            ScriptedTurn(text="b"),
        ],
        max_context_tokens=3_000,
    )
    fill(agent, range(8))
    await agent.run("question 8")
    fill(agent, range(9, 16))
    await agent.run("question 16")

    assert await segments(agent) == ["FIRST", "SECOND"]
    await agent.memory.close()


async def test_segments_merge_over_the_limit(tmp_path):
    db = tmp_path / "m.db"
    agent, mock = make_agent(
        db,
        [ScriptedTurn(text="NEW"), ScriptedTurn(text="MERGED"), ScriptedTurn(text="answer")],
        max_context_tokens=3_000,
    )
    store = agent.memory.long_term.backend
    big = "y" * (SUMMARY_BUDGET_TOKENS * 4 // 2)  # two of these fill the budget
    await store.append_summary("default", "OLD-1 " + big)
    await store.append_summary("default", "OLD-2 " + big)
    fill(agent, range(8))
    await agent.run("question 8")

    merge = [c for c in mock.call_log if c["system"].startswith("You maintain the long-term")]
    assert len(merge) == 1 and "OLD-1" in merge[0]["messages"][0].content
    assert await segments(agent) == ["MERGED", "NEW"]
    assert events(agent, "summary_merge")
    await agent.memory.close()


# ---------------- new sessions ----------------

async def test_new_session_starts_with_the_summary(tmp_path):
    db = tmp_path / "m.db"
    first, _ = make_agent(
        db, [ScriptedTurn(text="The user is Alice."), ScriptedTurn(text="a")], max_context_tokens=3_000
    )
    fill(first, range(8))
    await first.run("question 8")
    await first.memory.close()

    second, mock = make_agent(db, [ScriptedTurn(text="hi Alice")], session_id="s2")
    await second.run("hello")
    head = mock.call_log[0]["messages"][0]
    assert head.metadata.get("context") == "summary"
    assert "The user is Alice." in head.content
    assert summary_calls(mock) == []  # the first session is not idle yet
    await second.memory.close()


def _make_idle(db, session_id: str) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE session_state SET updated_at='2000-01-01T00:00:00+00:00' WHERE session_id=?",
        (session_id,),
    )
    conn.commit()
    conn.close()


async def test_idle_session_tail_is_summarized_once(tmp_path):
    """Two new sessions start at once; the idle session is summarized once."""
    db = tmp_path / "m.db"
    short, _ = make_agent(db, [ScriptedTurn(text="noted")], session_id="old")
    await short.run("my favourite colour is green")
    await short.memory.close()
    _make_idle(db, "old")

    script = [ScriptedTurn(text="IDLE-SUMMARY"), ScriptedTurn(text="hi")]
    b, mock_b = make_agent(db, script, session_id="b")
    c, mock_c = make_agent(db, list(script), session_id="c")
    await asyncio.gather(b.run("hello"), c.run("hello"))

    calls = summary_calls(mock_b) + summary_calls(mock_c)
    assert len(calls) == 1
    assert "green" in calls[0]["messages"][0].content
    assert await segments(b) == ["IDLE-SUMMARY"]
    for agent in (b, c):
        await agent.memory.close()


async def test_live_session_tail_is_not_summarized(tmp_path):
    db = tmp_path / "m.db"
    live, _ = make_agent(db, [ScriptedTurn(text="noted")], session_id="live")
    await live.run("my favourite colour is green")

    newcomer, mock = make_agent(db, [ScriptedTurn(text="hi")], session_id="new")
    await newcomer.run("hello")
    assert summary_calls(mock) == []
    for agent in (live, newcomer):
        await agent.memory.close()


async def test_a_failed_idle_summary_releases_its_claim(tmp_path):
    db = tmp_path / "m.db"
    short, _ = make_agent(db, [ScriptedTurn(text="noted")], session_id="old")
    await short.run("remember the eu-1 cluster")
    await short.memory.close()
    _make_idle(db, "old")

    failing, _ = make_agent(
        db, [ScriptedTurn(raise_exc=RuntimeError("down")), ScriptedTurn(text="hi")], session_id="b"
    )
    await failing.run("hello")
    assert events(failing, "summary_failed")
    await failing.memory.close()

    later, mock = make_agent(db, [ScriptedTurn(text="IDLE"), ScriptedTurn(text="hi")], session_id="c")
    await later.run("hello")
    assert len(summary_calls(mock)) == 1  # the tail was still there to summarize
    await later.memory.close()


# ---------------- on demand ----------------

async def test_explicit_summarize_covers_a_short_conversation(tmp_path):
    db = tmp_path / "m.db"
    agent, mock = make_agent(
        db, [ScriptedTurn(text="noted"), ScriptedTurn(text="ok"), ScriptedTurn(text="SHORT-SUMMARY")]
    )
    await agent.run("my name is Alice")
    await agent.run("I work on billing")

    assert await agent.memory.summarize() is True
    [call] = summary_calls(mock)
    assert "Alice" in call["messages"][0].content and "billing" in call["messages"][0].content
    assert await agent.memory.summarize() is False  # nothing new since
    assert len(summary_calls(mock)) == 1
    await agent.memory.close()

    later, later_mock = make_agent(db, [ScriptedTurn(text="hi")], session_id="s2")
    await later.run("hello")
    assert "SHORT-SUMMARY" in later_mock.call_log[0]["messages"][0].content
    await later.memory.close()


async def test_summarize_without_a_store_does_nothing():
    agent = Agent("claude-haiku-4-5", api_key="k", provider=MockLLMProvider([]))
    assert await agent.memory.summarize() is False


# ---------------- the summary model ----------------

async def test_summary_model_is_used_and_validated(tmp_path):
    agent, mock = make_agent(
        tmp_path / "m.db",
        [ScriptedTurn(text="S"), ScriptedTurn(text="answer")],
        max_context_tokens=3_000,
        summary_model="anthropic:claude-haiku-4-5",
    )
    fill(agent, range(8))
    await agent.run("question 8")
    [call] = summary_calls(mock)
    assert call["model"] == "claude-haiku-4-5"
    assert mock.call_log[-1]["model"] == "claude-haiku-4-5"  # the agent's own model too
    await agent.memory.close()

    with pytest.raises(ValueError, match="same provider"):
        Agent(
            "claude-opus-5-5",
            api_key="k",
            memory=MemoryConfig(store=str(tmp_path / "x.db"), summary_model="gpt-5"),
        )


async def test_summaries_never_run_on_a_dry_run(tmp_path):
    agent, mock = make_agent(tmp_path / "m.db", [], max_context_tokens=3_000)
    fill(agent, range(8))
    preview = await agent.run("question 8", dry_run=True)
    assert preview.context.trimmed is True
    assert mock.call_log == []
    assert await segments(agent) == []
    failures = [e for e in agent.get_trace().all_spans() if e.status is SpanStatus.ERROR]
    assert failures == []
    assert [e for e in agent.get_trace().all_spans() if e.event_type is EventType.MEMORY_WRITE] == []
    await agent.memory.close()


# ---------------- the summary store contract (both backends) ----------------

@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    from agent_framework.memory.backends.in_memory import InMemoryBackend
    from agent_framework.memory.backends.sqlite import SQLiteBackend

    return InMemoryBackend() if request.param == "memory" else SQLiteBackend(tmp_path / "s.db")


async def _idle_session(store, session_id="old", last_seq=3):
    from agent_framework.memory.sessions import SessionRecord

    await store.save_session("ns", session_id, SessionRecord(history=[], state={}, last_seq=last_seq), [])


async def test_an_idle_tail_is_claimed_once(store):
    await _idle_session(store)
    later = "9999-01-01T00:00:00+00:00"  # everything counts as idle before this
    first = await store.claim_idle_session("ns", idle_before=later, exclude_session="me")
    second = await store.claim_idle_session("ns", idle_before=later, exclude_session="me")

    assert (first.session_id, first.after_seq, first.upto_seq) == ("old", 0, 3)
    assert second is None
    # Released after a failed summary: claimable again.
    await store.release_claim("ns", first)
    assert await store.claim_idle_session("ns", idle_before=later, exclude_session="me") is not None
    await store.close()


async def test_a_session_is_not_idle_before_its_time_and_never_its_own(store):
    await _idle_session(store)
    assert await store.claim_idle_session("ns", idle_before="2000-01-01", exclude_session="me") is None
    later = "9999-01-01T00:00:00+00:00"
    assert await store.claim_idle_session("ns", idle_before=later, exclude_session="old") is None
    await store.close()


async def test_merging_is_compare_and_set(store):
    await store.append_summary("ns", "A")
    await store.append_summary("ns", "B")
    await store.append_summary("ns", "C")
    [a, b, _] = await store.summary_segments("ns")

    assert await store.merge_summaries("ns", a.seq, b.seq, "AB") is True
    assert await store.merge_summaries("ns", a.seq, b.seq, "AB again") is False  # already merged
    assert [s.text for s in await store.summary_segments("ns")] == ["AB", "C"]
    await store.close()


async def test_appending_advances_the_session_marker_with_it(store):
    await _idle_session(store, session_id="s", last_seq=10)
    await store.append_summary("ns", "S", session_id="s", summarized_upto=7)
    assert (await store.load_session("ns", "s")).summarized_upto == 7
    await store.close()


async def test_an_old_single_summary_becomes_the_first_segment(store):
    await store.set_summary("ns", "written before segments existed")
    assert [s.text for s in await store.summary_segments("ns")] == ["written before segments existed"]
    await store.append_summary("ns", "new")
    assert [s.text for s in await store.summary_segments("ns")] == [
        "written before segments existed",
        "new",
    ]
    await store.close()


# ---------------- review regressions ----------------

async def test_a_long_lived_agent_never_summarizes_turns_twice(tmp_path):
    """Another session summarized this one's tail while it sat idle; the
    agent object stayed alive and must not cover the same turns again."""
    db = tmp_path / "m.db"
    a, mock_a = make_agent(db, [ScriptedTurn(text="a1"), ScriptedTurn(text="a2")], session_id="A")
    await a.run("my name is Ada")
    _make_idle(db, "A")
    b, _ = make_agent(db, [ScriptedTurn(text="IDLE-OF-A"), ScriptedTurn(text="hi")], session_id="B")
    await b.run("hello")

    mock_a.add_turn(ScriptedTurn(text="EXPLICIT"))
    await a.run("I live in Oslo")
    await a.memory.summarize()

    [call] = summary_calls(mock_a)
    assert "Ada" not in call["messages"][0].content and "Oslo" in call["messages"][0].content
    assert await segments(a) == ["IDLE-OF-A", "EXPLICIT"]
    for agent in (a, b):
        await agent.memory.close()


async def test_a_conflicting_append_is_discarded_not_duplicated(tmp_path):
    store_path = tmp_path / "m.db"
    agent, _ = make_agent(store_path, [ScriptedTurn(text="answer")])
    await agent.run("q")
    store = agent.memory.long_term.backend
    # Another process advanced the marker after this agent last read it.
    await store.append_summary("default", "OTHER", session_id="s1", summarized_upto=2)
    ok = await store.append_summary(
        "default", "MINE", session_id="s1", expected_upto=0, summarized_upto=2
    )
    assert ok is False
    assert await segments(agent) == ["OTHER"]
    await agent.memory.close()


class _UnreadableSessions:
    """Mix-in: the session row cannot be read."""

    async def load_session(self, namespace, session_id):
        raise OSError("unreadable")


async def test_a_session_that_could_not_be_loaded_is_never_summarized():
    from agent_framework.memory.backends.in_memory import InMemoryBackend
    from agent_framework.memory.sessions import LogEntry, SessionRecord

    class Unreadable(_UnreadableSessions, InMemoryBackend):
        pass

    backend = Unreadable()
    await InMemoryBackend.save_session(
        backend, "default", "s", SessionRecord(history=[], state={}, last_seq=1),
        [LogEntry(1, UserMessage(content="already summarized").to_dict())],
    )
    mock = MockLLMProvider([ScriptedTurn(text="ok"), ScriptedTurn(text="SUMMARY")])
    agent = Agent(
        AgentConfig(model="claude-haiku-4-5", api_key="k"),
        provider=mock,
        memory=MemoryConfig(),
        memory_backend=backend,
        session_id="s",
    )
    await agent.run("q")
    assert await agent.memory.summarize() is False
    assert summary_calls(mock) == []
    assert await backend.summary_segments("default") == []


async def test_the_summary_budget_follows_the_window(tmp_path):
    """On an 8k model a summary written for a large window must not fill the
    prompt of every new conversation."""
    mock = MockLLMProvider([ScriptedTurn(text="hi")])
    agent = Agent(
        AgentConfig(
            model="openai:local-8k",
            base_url="http://127.0.0.1:1/v1",
            context_window=8_192,
        ),
        provider=mock,
        memory=MemoryConfig(store=str(tmp_path / "m.db")),
    )
    budget = agent._summarizer.budget_tokens
    assert budget == agent.config.resolved_max_context_tokens // 5
    await agent.memory.long_term.backend.append_summary("default", "z" * 32_000)  # ≈8k tokens

    result = await agent.run("hello")
    assert result.stop_reason is not StopReason.CONTEXT_EXCEEDED
    head = mock.call_log[0]["messages"][0]
    assert head.metadata.get("context") == "summary"
    assert len(head.content) <= budget * 4 + 200
    await agent.memory.close()


async def test_a_claim_is_released_when_the_tail_cannot_be_read(tmp_path):
    db = tmp_path / "m.db"
    old, _ = make_agent(db, [ScriptedTurn(text="noted")], session_id="old")
    await old.run("remember the eu-1 cluster")
    await old.memory.close()
    _make_idle(db, "old")

    failing, _ = make_agent(db, [ScriptedTurn(text="hi")], session_id="b")
    backend = failing.memory.long_term.backend

    async def broken_log(*args, **kwargs):
        raise OSError("disk read error")

    backend.session_log = broken_log
    await failing.run("hello")
    assert events(failing, "summary_failed")
    await failing.memory.close()

    later, mock = make_agent(db, [ScriptedTurn(text="IDLE"), ScriptedTurn(text="hi")], session_id="c")
    await later.run("hello")
    assert len(summary_calls(mock)) == 1  # released: still there to summarize
    await later.memory.close()


@pytest.mark.parametrize("broken", ["search_facts", "summary_segments"])
async def test_an_unreadable_store_costs_memory_context_not_the_run(tmp_path, broken):
    agent, _ = make_agent(tmp_path / "m.db", [ScriptedTurn(text="ok")])
    backend = agent.memory.long_term.backend

    async def fail(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    setattr(backend, broken, fail)
    result = await agent.run("hello")

    assert result.error is None and result.final_message == "ok"
    assert [e for e in agent.get_trace().all_spans() if e.name.startswith("memory_read_failed")]
    await agent.memory.close()
