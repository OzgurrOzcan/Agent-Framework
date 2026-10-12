"""Idle-session summaries as leases, oldest first, and store housekeeping.

A claim on an idle tail is a lease with a token, not a marker move: the
marker advances only in the transaction that writes the summary, so a
holder that crashes mid-call just lets the lease expire. Tails are claimed
oldest first and a new conversation works through a bounded number of them;
`memory.summarize_idle()` drains the rest from a background job.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import timedelta

import pytest

from agent_framework import tool
from agent_framework.core.agent import Agent
from agent_framework.core.config import AgentConfig, MemoryConfig
from agent_framework.core.message import UserMessage
from agent_framework.memory.backends.in_memory import InMemoryBackend
from agent_framework.memory.backends.sqlite import SQLiteBackend
from agent_framework.memory.sessions import IdleTail, SessionRecord
from agent_framework.memory.short_term import ContextState
from agent_framework.permissions.models import PermissionCategory
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn
from agent_framework.tracing.trace import EventType

SUMMARIZER = "You write the long-term memory"
FAR = "9999-01-01T00:00:00+00:00"
PAST = "2000-01-01T00:00:00+00:00"


# ---------------- the store contract (both backends) ----------------

@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    return InMemoryBackend() if request.param == "memory" else SQLiteBackend(tmp_path / "s.db")


async def _set_updated_at(store, session_id: str, when: str, namespace="ns") -> None:
    if isinstance(store, InMemoryBackend):
        store._sessions[(namespace, session_id)].updated_at = when
    else:
        await store._call(
            lambda conn: conn.execute(
                "UPDATE session_state SET updated_at=? WHERE namespace=? AND session_id=?",
                (when, namespace, session_id),
            )
        )


async def _session(store, session_id="old", *, last_seq=3, summarized_upto=0, updated_at=None):
    await store.save_session(
        "ns",
        session_id,
        SessionRecord(history=[], state={}, last_seq=last_seq, summarized_upto=summarized_upto),
        [],
    )
    if updated_at is not None:
        await _set_updated_at(store, session_id, updated_at)


async def _claim(store, *, lease_until=FAR, exclude="me"):
    return await store.claim_idle_session(
        "ns", idle_before=FAR, exclude_session=exclude, lease_until=lease_until
    )


async def _marker(store, session_id="old") -> int:
    return (await store.load_session("ns", session_id)).summarized_upto


async def test_a_claim_is_a_lease_and_the_marker_moves_only_with_the_summary(store):
    await _session(store)
    tail = await _claim(store)
    assert tail.token and (tail.session_id, tail.after_seq, tail.upto_seq) == ("old", 0, 3)
    assert await _marker(store) == 0  # claimed, not summarized
    assert await _claim(store) is None  # held

    ok = await store.append_summary(
        "ns", "S", session_id="old", expected_upto=0, claim_token=tail.token, summarized_upto=3
    )
    assert ok is True
    assert await _marker(store) == 3
    assert [s.text for s in await store.summary_segments("ns")] == ["S"]
    assert await _claim(store) is None  # nothing left to summarize
    await store.close()


async def test_an_expired_lease_is_claimed_again_and_the_dead_holder_is_refused(store):
    await _session(store)
    dead = await _claim(store, lease_until=PAST)  # a holder that crashed: its lease ran out
    live = await _claim(store)
    assert live is not None and live.token != dead.token

    refused = await store.append_summary(
        "ns", "LATE", session_id="old", expected_upto=0, claim_token=dead.token, summarized_upto=3
    )
    assert refused is False and await _marker(store) == 0
    assert await store.summary_segments("ns") == []
    assert await store.append_summary(
        "ns", "S", session_id="old", expected_upto=0, claim_token=live.token, summarized_upto=3
    )
    assert await _marker(store) == 3
    await store.close()


async def test_a_claim_is_released_only_by_its_holder(store):
    await _session(store)
    tail = await _claim(store)
    await store.release_claim("ns", IdleTail("old", 0, 3, token="not-mine"))
    assert await _claim(store) is None  # still held
    await store.release_claim("ns", tail)
    again = await _claim(store)
    assert again is not None and again.token != tail.token
    await store.close()


async def test_the_sessions_own_summary_waits_for_an_active_lease(store):
    """The live agent of a claimed session must not cover the same turns."""
    await _session(store, "s")
    tail = await _claim(store)
    assert tail.session_id == "s"
    assert (
        await store.append_summary("ns", "MINE", session_id="s", expected_upto=0, summarized_upto=2)
        is False
    )
    await store.release_claim("ns", tail)
    assert (
        await store.append_summary("ns", "MINE", session_id="s", expected_upto=0, summarized_upto=2)
        is True
    )
    assert await _marker(store, "s") == 2
    await store.close()


async def test_a_holder_whose_lease_expired_cannot_write_once_the_marker_moved(store):
    await _session(store, "s")
    expired = await _claim(store, lease_until=PAST)
    # The session's own agent summarizes meanwhile (an expired lease does not block it).
    assert await store.append_summary("ns", "OWN", session_id="s", expected_upto=0, summarized_upto=2)
    # The token still matches (nobody re-claimed), but the marker moved: refused.
    assert (
        await store.append_summary(
            "ns", "STALE", session_id="s", expected_upto=0, claim_token=expired.token, summarized_upto=3
        )
        is False
    )
    assert [s.text for s in await store.summary_segments("ns")] == ["OWN"]
    await store.close()


async def test_the_oldest_idle_session_is_claimed_first(store):
    await _session(store, "b", updated_at="2002-01-01T00:00:00+00:00")
    await _session(store, "c", updated_at="2003-01-01T00:00:00+00:00")
    await _session(store, "a", updated_at="2001-01-01T00:00:00+00:00")
    order = [(await _claim(store)).session_id for _ in range(3)]
    assert order == ["a", "b", "c"]
    assert await _claim(store) is None
    await store.close()


async def test_a_fully_summarized_or_active_session_is_never_claimed(store):
    await _session(store, "done", summarized_upto=3)
    await _session(store, "busy", updated_at=FAR)  # updated after the cut-off
    await _session(store, "mine")
    assert await _claim(store, exclude="mine") is None
    await store.close()


async def test_idle_backlog_counts_what_is_still_unsummarized(store):
    for name in ("a", "b", "c"):
        await _session(store, name, updated_at="2001-01-01T00:00:00+00:00")
    await _session(store, "done", summarized_upto=3, updated_at="2001-01-01T00:00:00+00:00")
    await _session(store, "fresh", updated_at=FAR)
    cutoff = "2010-01-01T00:00:00+00:00"
    assert await store.idle_backlog("ns", idle_before=cutoff, exclude_session="me") == 3
    assert await store.idle_backlog("ns", idle_before=cutoff, exclude_session="a") == 2
    tail = await _claim(store)  # a claim does not change the backlog
    assert await store.idle_backlog("ns", idle_before=cutoff, exclude_session="me") == 3
    await store.append_summary(
        "ns", "S", session_id=tail.session_id, expected_upto=0, claim_token=tail.token, summarized_upto=3
    )
    assert await store.idle_backlog("ns", idle_before=cutoff, exclude_session="me") == 2
    await store.close()


# ---------------- housekeeping (both backends) ----------------

async def test_list_delete_and_prune_sessions(store):
    old = "2001-01-01T00:00:00+00:00"
    await _session(store, "summarized-old", summarized_upto=3, updated_at=old)
    await _session(store, "unsummarized-old", updated_at=old)
    await _session(store, "fresh", summarized_upto=3)

    listed = await store.list_sessions("ns")
    assert [s.session_id for s in listed] == ["fresh", "summarized-old", "unsummarized-old"]
    assert listed[0].summarized and not listed[2].summarized
    idle = await store.list_sessions("ns", idle_before="2010-01-01T00:00:00+00:00")
    assert sorted(s.session_id for s in idle) == ["summarized-old", "unsummarized-old"]
    assert await store.list_sessions("other") == []

    assert await store.prune_sessions("ns", idle_before="2010-01-01T00:00:00+00:00") == 1
    assert sorted(s.session_id for s in await store.list_sessions("ns")) == ["fresh", "unsummarized-old"]
    assert await store.prune_sessions("ns", idle_before="2010-01-01T00:00:00+00:00", summarized_only=False) == 1
    assert [s.session_id for s in await store.list_sessions("ns")] == ["fresh"]

    await store.save_session("ns", "fresh", SessionRecord(history=[], state={}, last_seq=1),
                             [_log(1, "x")])
    assert await store.delete_session("ns", "fresh") is True
    assert await store.delete_session("ns", "fresh") is False
    assert await store.list_sessions("ns") == [] and await store.session_log("ns", "fresh") == []
    await store.close()


def _log(seq, text):
    from agent_framework.memory.sessions import LogEntry

    return LogEntry(seq=seq, message=UserMessage(content=text).to_dict())


# ---------------- the agent ----------------

def make_agent(db, script=(), *, session_id="s1", **memory):
    mock = MockLLMProvider(list(script))
    agent = Agent(
        AgentConfig(model="claude-haiku-4-5", api_key="k"),
        provider=mock,
        memory=MemoryConfig(store=str(db), **memory),
        session_id=session_id,
    )
    return agent, mock


def summary_calls(mock: MockLLMProvider) -> list[dict]:
    return [c for c in mock.call_log if c["system"].startswith(SUMMARIZER)]


def events(agent: Agent, name: str, event_type: EventType | None = None):
    return [
        e
        for e in agent.get_trace().all_spans()
        if e.name == name and (event_type is None or e.event_type is event_type)
    ]


def summaries_written(agent: Agent) -> list[str]:
    """The session each summary call covered, in order (the COMPACTION
    events; a MEMORY_READ event is also named "summary")."""
    return [e.payload["session_id"] for e in events(agent, "summary", EventType.COMPACTION)]


async def segments(agent: Agent) -> list[str]:
    return await agent._summarizer.segments()


def _idle(db, session_id: str, at: str = PAST) -> None:
    conn = sqlite3.connect(db)
    conn.execute("UPDATE session_state SET updated_at=? WHERE session_id=?", (at, session_id))
    conn.commit()
    conn.close()


async def _seed(db, session_id: str, text: str, *, idle_at: str = PAST, summarize: bool = False):
    """A finished conversation, idle since *idle_at*. It summarizes nothing
    itself (idle_summaries_per_start=0), so the backlog is exactly the seeds."""
    agent, _ = make_agent(db, [ScriptedTurn(text="noted"), ScriptedTurn(text=f"SUM-{session_id}")],
                          session_id=session_id, idle_summaries_per_start=0)
    await agent.run(text)
    if summarize:
        await agent.memory.summarize()
    await agent.memory.close()
    _idle(db, session_id, idle_at)


async def _seed_three(db):
    await _seed(db, "A", "alpha", idle_at="2001-01-01T00:00:00+00:00")
    await _seed(db, "B", "beta", idle_at="2002-01-01T00:00:00+00:00")
    await _seed(db, "C", "gamma", idle_at="2003-01-01T00:00:00+00:00")


async def test_a_new_conversation_summarizes_idle_sessions_oldest_first(tmp_path):
    db = tmp_path / "m.db"
    await _seed_three(db)

    fresh, mock = make_agent(
        db, [ScriptedTurn(text="SUM-A"), ScriptedTurn(text="SUM-B"), ScriptedTurn(text="hi")], session_id="new"
    )
    await fresh.run("hello")

    calls = summary_calls(mock)
    assert len(calls) == 2  # idle_summaries_per_start defaults to 2
    assert "alpha" in calls[0]["messages"][0].content and "beta" in calls[1]["messages"][0].content
    assert await segments(fresh) == ["SUM-A", "SUM-B"]
    assert summaries_written(fresh) == ["A", "B"]
    [backlog] = events(fresh, "idle_backlog")
    assert backlog.payload == {"namespace": "default", "summarized": 2, "remaining": 1}
    head = mock.call_log[-1]["messages"][0]
    assert "SUM-A" in head.content and "SUM-B" in head.content  # both head the new conversation
    await fresh.memory.close()

    # C waits for the next conversation; nothing is summarized twice.
    later, mock = make_agent(db, [ScriptedTurn(text="SUM-C"), ScriptedTurn(text="hi")], session_id="new2")
    await later.run("hello")
    [call] = summary_calls(mock)
    assert "gamma" in call["messages"][0].content
    assert await segments(later) == ["SUM-A", "SUM-B", "SUM-C"]
    await later.memory.close()


@pytest.mark.parametrize("per_start", [0, 1, 3])
async def test_idle_summaries_per_start_is_a_knob(tmp_path, per_start):
    db = tmp_path / "m.db"
    await _seed_three(db)
    script = [ScriptedTurn(text=f"SUM-{i}") for i in range(per_start)] + [ScriptedTurn(text="hi")]
    agent, mock = make_agent(db, script, session_id="new", idle_summaries_per_start=per_start)
    await agent.run("hello")
    assert len(summary_calls(mock)) == per_start
    assert len(mock.call_log) == per_start + 1
    await agent.memory.close()


async def test_summarize_idle_drains_the_backlog_from_a_background_job(tmp_path):
    db = tmp_path / "m.db"
    await _seed_three(db)
    worker, mock = make_agent(
        db, [ScriptedTurn(text=f"SUM-{s}") for s in "ABC"], session_id="worker", idle_summaries_per_start=0
    )
    assert await worker.memory.summarize_idle(limit=10) == 3
    assert await segments(worker) == ["SUM-A", "SUM-B", "SUM-C"]
    assert summaries_written(worker) == ["A", "B", "C"]
    assert events(worker, "idle_backlog")[-1].payload["remaining"] == 0

    assert await worker.memory.summarize_idle(limit=10) == 0  # nothing left: no model call
    assert len(mock.call_log) == 3
    # The worker never ran a conversation, so it saved no session of its own.
    assert await worker.memory.long_term.backend.load_session("default", "worker") is None
    await worker.memory.close()


async def test_summarize_idle_honours_its_limit(tmp_path):
    db = tmp_path / "m.db"
    await _seed_three(db)
    worker, mock = make_agent(db, [ScriptedTurn(text="SUM-A"), ScriptedTurn(text="SUM-B")],
                              session_id="worker", idle_summaries_per_start=0)
    assert await worker.memory.summarize_idle(limit=2) == 2
    assert len(mock.call_log) == 2 and await segments(worker) == ["SUM-A", "SUM-B"]
    await worker.memory.close()


async def test_summarize_idle_stops_at_the_first_failure_and_keeps_the_tail(tmp_path):
    db = tmp_path / "m.db"
    await _seed_three(db)
    worker, mock = make_agent(
        db, [ScriptedTurn(raise_exc=RuntimeError("provider down"))], session_id="worker",
        idle_summaries_per_start=0,
    )
    assert await worker.memory.summarize_idle(limit=10) == 0
    assert len(mock.call_log) == 1  # a provider that is down is not asked ten times
    assert events(worker, "summary_failed")
    assert events(worker, "idle_backlog")[-1].payload == {"namespace": "default", "summarized": 0, "remaining": 3}

    for text in ("SUM-A", "SUM-B", "SUM-C"):
        mock.add_turn(ScriptedTurn(text=text))
    assert await worker.memory.summarize_idle(limit=10) == 3  # the lease was released: nothing lost
    assert await segments(worker) == ["SUM-A", "SUM-B", "SUM-C"]
    await worker.memory.close()


async def test_summarize_idle_without_a_store_does_nothing():
    agent = Agent("claude-haiku-4-5", api_key="k", provider=MockLLMProvider([]), memory=MemoryConfig())
    assert await agent.memory.summarize_idle() == 0
    stateless = Agent("claude-haiku-4-5", api_key="k", provider=MockLLMProvider([]))
    assert await stateless.memory.summarize_idle() == 0


async def test_a_tail_held_by_a_live_holder_is_left_alone_and_a_dead_holders_lease_expires(tmp_path):
    db = tmp_path / "m.db"
    await _seed(db, "A", "alpha", idle_at="2001-01-01T00:00:00+00:00")
    await _seed(db, "B", "beta", idle_at="2002-01-01T00:00:00+00:00")
    backend = SQLiteBackend(db)
    held = await backend.claim_idle_session("default", idle_before=FAR, exclude_session="", lease_until=FAR)
    dead = await backend.claim_idle_session("default", idle_before=FAR, exclude_session="", lease_until=PAST)
    assert (held.session_id, dead.session_id) == ("A", "B")  # oldest first
    await backend.close()

    fresh, mock = make_agent(db, [ScriptedTurn(text="SUM-B"), ScriptedTurn(text="hi")], session_id="new")
    await fresh.run("hello")
    [call] = summary_calls(mock)
    assert "beta" in call["messages"][0].content  # B's dead lease expired; A's live one is respected
    assert await segments(fresh) == ["SUM-B"]
    await fresh.memory.close()


async def test_a_lease_lost_during_the_call_discards_the_summary(tmp_path):
    """The lease ran out while the model was answering and another process
    claimed the tail: the late summary is dropped, never written twice."""
    db = tmp_path / "m.db"
    await _seed(db, "A", "alpha")

    class SlowSummarizer(MockLLMProvider):
        async def call(self, **kwargs):
            if kwargs["system"].startswith(SUMMARIZER):
                other = SQLiteBackend(db)  # another process takes the tail meanwhile
                await other._call(
                    lambda conn: conn.execute("UPDATE session_state SET claim_until=? WHERE session_id='A'", (PAST,))
                )
                taken = await other.claim_idle_session("default", idle_before=FAR, exclude_session="", lease_until=FAR)
                assert taken is not None and taken.session_id == "A"
                await other.close()
            return await super().call(**kwargs)

    mock = SlowSummarizer([ScriptedTurn(text="LATE-SUMMARY"), ScriptedTurn(text="hi")])
    agent = Agent(AgentConfig(model="claude-haiku-4-5", api_key="k"), provider=mock,
                  memory=MemoryConfig(store=str(db)), session_id="new")
    result = await agent.run("hello")
    assert result.error is None and result.final_message == "hi"
    [failed] = events(agent, "summary_failed")
    assert "lease" in failed.error.message
    assert await segments(agent) == []
    assert (await agent.memory.long_term.backend.load_session("default", "A")).summarized_upto == 0
    await agent.memory.close()


async def test_a_cancelled_idle_summary_releases_its_lease(tmp_path):
    db = tmp_path / "m.db"
    await _seed(db, "A", "alpha")
    cancelled, _ = make_agent(db, [ScriptedTurn(raise_exc=asyncio.CancelledError())], session_id="new")
    with pytest.raises(asyncio.CancelledError):
        await cancelled.run("hello")
    await cancelled.memory.close()

    later, mock = make_agent(db, [ScriptedTurn(text="SUM-A"), ScriptedTurn(text="hi")], session_id="new2")
    await later.run("hello")
    assert len(summary_calls(mock)) == 1 and await segments(later) == ["SUM-A"]
    await later.memory.close()


async def test_a_trim_summary_waits_while_another_session_holds_the_tail(tmp_path):
    """The live agent trims while its idle tail is leased elsewhere: its own
    summary is refused, the marker stays, and the next trim covers the turns."""
    db = tmp_path / "m.db"
    agent, mock = make_agent(db, [ScriptedTurn(text="first")], session_id="live")
    await agent.run("my name is Ada")
    backend = agent.memory.long_term.backend
    _idle(db, "live")
    held = await backend.claim_idle_session("default", idle_before=FAR, exclude_session="", lease_until=FAR)
    assert held.session_id == "live"

    # Force a trim on the next run: a tiny window with old turns to drop.
    small = Agent(
        AgentConfig(model="claude-haiku-4-5", api_key="k", max_context_tokens=3_000, max_tool_result_tokens=300),
        provider=MockLLMProvider([ScriptedTurn(text="TRIM-SUMMARY"), ScriptedTurn(text="answer")]),
        memory=MemoryConfig(store=str(db)), session_id="live",
    )
    await small._ensure_session()
    for i in range(8):
        small.memory.short_term.extend([UserMessage(content=f"q{i}"), UserMessage(content="x" * 8_000)])
    await small.run("q8")
    [failed] = events(small, "summary_failed")
    assert "another session" in failed.error.message
    assert await backend.summary_segments("default") == []
    assert (await backend.load_session("default", "live")).summarized_upto == 0
    for a in (agent, small):
        await a.memory.close()


async def test_the_idle_threshold_is_configurable(tmp_path):
    db = tmp_path / "m.db"
    from datetime import UTC, datetime

    await _seed(db, "A", "alpha", idle_at=(datetime.now(UTC) - timedelta(minutes=10)).isoformat())
    default, mock = make_agent(db, [ScriptedTurn(text="hi")], session_id="n1")
    await default.run("hello")
    assert summary_calls(mock) == []  # ten minutes is not idle at the default of thirty
    await default.memory.close()

    quick, mock = make_agent(db, [ScriptedTurn(text="SUM-A"), ScriptedTurn(text="hi")],
                             session_id="n2", idle_after_minutes=5)
    await quick.run("hello")
    assert len(summary_calls(mock)) == 1
    await quick.memory.close()


# ---------------- retention ----------------

async def test_retention_prunes_only_fully_summarized_idle_sessions(tmp_path):
    db = tmp_path / "m.db"
    await _seed(db, "remembered", "alpha", summarize=True)  # fully in the summary, idle for years
    await _seed(db, "forgotten-not", "beta")  # idle for years, but its log is not summarized
    await _seed(db, "recent", "gamma", idle_at=FAR, summarize=True)

    agent, mock = make_agent(
        db, [ScriptedTurn(text="hi")], session_id="new", retention_days=30, idle_summaries_per_start=0
    )
    await agent.run("hello")
    remaining = sorted(s.session_id for s in await agent.memory.sessions())
    assert remaining == ["forgotten-not", "new", "recent"]
    [pruned] = events(agent, "prune_sessions")
    assert pruned.payload == {"namespace": "default", "pruned": 1}
    assert await agent.memory.long_term.backend.session_log("default", "remembered") == []
    await agent.memory.close()

    # Without retention nothing is ever deleted.
    keep, _ = make_agent(db, [ScriptedTurn(text="hi")], session_id="new3", idle_summaries_per_start=0)
    await keep.run("hello")
    assert events(keep, "prune_sessions") == []
    await keep.memory.close()


async def test_housekeeping_through_the_memory_api(tmp_path):
    db = tmp_path / "m.db"
    await _seed(db, "old-done", "alpha", summarize=True)
    await _seed(db, "old-open", "beta")
    agent, _ = make_agent(db, [ScriptedTurn(text="hi")], session_id="me", idle_summaries_per_start=0)
    await agent.run("hello")

    assert [s.session_id for s in await agent.memory.sessions()] == ["me", "old-done", "old-open"]
    idle = await agent.memory.sessions(idle_for=timedelta(days=365))
    assert sorted(s.session_id for s in idle) == ["old-done", "old-open"]
    assert [s.session_id for s in await agent.memory.sessions(limit=1)] == ["me"]

    assert await agent.memory.prune_sessions(idle_for=timedelta(days=30)) == 1  # only the summarized one
    assert await agent.memory.prune_sessions(idle_for=timedelta(days=30), summarized_only=False) == 1
    assert [s.session_id for s in await agent.memory.sessions()] == ["me"]
    assert await agent.memory.delete_session("me") is True
    assert await agent.memory.delete_session("me") is False
    assert await agent.memory.sessions() == []
    writes = [e.name for e in agent.get_trace().all_spans() if e.event_type is EventType.MEMORY_WRITE]
    assert writes.count("prune_sessions") == 2 and "delete_session" in writes
    await agent.memory.close()

    ram = Agent("claude-haiku-4-5", api_key="k", provider=MockLLMProvider([]), memory=MemoryConfig())
    assert await ram.memory.sessions() == []
    assert await ram.memory.delete_session("x") is False
    assert await ram.memory.prune_sessions(idle_for=timedelta(days=1)) == 0


# ---------------- one summary, one source ----------------

async def test_rolling_summary_and_fold_read_and_write_the_segments(tmp_path):
    db = tmp_path / "m.db"
    agent, mock = make_agent(db, [ScriptedTurn(text="noted"), ScriptedTurn(text="THE-SUMMARY")])
    await agent.run("my name is Ada")
    assert await agent.memory.rolling_summary() == ""
    assert await agent.memory.summarize() is True
    assert await agent.memory.rolling_summary() == "THE-SUMMARY"

    await agent.memory.fold_into_summary("  Ada prefers the phone.  ")
    assert await agent.memory.rolling_summary() == "THE-SUMMARY\n\nAda prefers the phone."
    assert await segments(agent) == ["THE-SUMMARY", "Ada prefers the phone."]
    assert await agent.memory.long_term.backend.get_summary("default") == ""  # not a second store
    await agent.memory.fold_into_summary("   ")  # nothing to add
    assert len(await segments(agent)) == 2
    await agent.memory.close()

    fresh, mock = make_agent(db, [ScriptedTurn(text="hi")], session_id="s2")
    await fresh.run("hello")
    assert "Ada prefers the phone." in mock.call_log[0]["messages"][0].content
    await fresh.memory.close()


async def test_rolling_summary_of_a_facts_only_backend_is_its_single_text():
    from agent_framework.memory.long_term import LongTermMemory
    from agent_framework.memory.manager import MemoryManager
    from agent_framework.tests.unit.test_sessions import FactsOnly

    manager = MemoryManager(long_term=LongTermMemory(backend=FactsOnly()))
    await manager.fold_into_summary("one")
    await manager.fold_into_summary("two")
    assert await manager.rolling_summary() == "one\n\ntwo"


# ---------------- guards ----------------

async def test_summarize_during_a_run_is_refused(tmp_path):
    db = tmp_path / "m.db"
    agent, mock = make_agent(db, [], session_id="s")

    @tool(description="Tries to summarize mid-run.", permission=PermissionCategory.AUTO)
    async def summarize_now() -> str:
        return str(await agent.memory.summarize())

    agent._tools.add(summarize_now)
    mock.add_turn(ScriptedTurn(tool_calls=[{"id": "c1", "name": "summarize_now", "arguments": {}}]))
    mock.add_turn(ScriptedTurn(text="done"))
    result = await agent.run("go")
    assert result.error is None
    [tool_result] = [m for m in agent.get_history() if m.role == "user" and hasattr(m, "content") and isinstance(m.content, list)]
    assert tool_result.content[0].is_error and "between runs" in tool_result.content[0].as_text()
    assert summary_calls(mock) == [] and await segments(agent) == []
    await agent.memory.close()


def test_an_old_saved_state_with_pending_turns_still_loads():
    state = ContextState.from_dict(
        {"summary": ["s"], "cleared_results": 1, "pending": [UserMessage(content="x").to_dict()]}
    )
    assert state.summary == ["s"] and state.cleared_results == 1
    assert "pending" not in state.to_dict()


@pytest.mark.parametrize(
    "bad",
    [
        {"idle_after_minutes": 0},
        {"idle_after_minutes": 1.5},
        {"idle_after_minutes": True},
        {"idle_summaries_per_start": -1},
        {"idle_summaries_per_start": "2"},
        {"retention_days": 0},
        {"retention_days": False},
    ],
)
def test_memory_config_rejects_bad_housekeeping_values(bad):
    [name] = bad
    with pytest.raises(ValueError, match=f"MemoryConfig.{name}"):
        MemoryConfig(store="x.db", **bad)


def test_memory_config_round_trips_the_new_fields():
    config = AgentConfig(
        model="claude-haiku-4-5",
        memory=MemoryConfig(store="x.db", idle_after_minutes=5, idle_summaries_per_start=0, retention_days=90),
    )
    again = AgentConfig.from_dict(config.to_dict())
    assert again.memory == config.memory
    assert config.memory.long_term_settings() == [
        "idle_after_minutes",
        "idle_summaries_per_start",
        "retention_days",
    ]
