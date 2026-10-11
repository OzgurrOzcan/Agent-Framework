"""One live agent per session.

Loading a session takes an owner token; every save presents it. A second
agent that opens the same session (the same ticket landing on another
server) takes it over, and the first agent's next save is refused and
recorded — from then on it raises `SessionTakenOverError` before any model
call, instead of the two silently overwriting each other's conversation or
one of them silently never saving again.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from agent_framework import SessionTakenOverError
from agent_framework.core.agent import Agent
from agent_framework.core.config import AgentConfig, MemoryConfig
from agent_framework.core.message import UserMessage
from agent_framework.memory.backends.in_memory import InMemoryBackend
from agent_framework.memory.backends.sqlite import SQLiteBackend
from agent_framework.memory.sessions import LogEntry, SessionRecord
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn
from agent_framework.tracing.trace import EventType, SpanStatus


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    return InMemoryBackend() if request.param == "memory" else SQLiteBackend(tmp_path / "s.db")


def _record(texts: list[str]) -> SessionRecord:
    return SessionRecord(
        history=[UserMessage(content=t).to_dict() for t in texts], state={}, last_seq=len(texts)
    )


def _entry(seq: int, text: str) -> LogEntry:
    return LogEntry(seq=seq, message=UserMessage(content=text).to_dict())


async def _history(store, session_id="s") -> list[str]:
    record = await store.load_session("ns", session_id)
    return [m["content"] for m in record.history]


# ---------------- the store contract (both backends) ----------------

async def test_the_first_save_establishes_the_owner(store):
    await store.save_session("ns", "s", _record(["a"]), [_entry(1, "a")], owner="A")
    assert (await store.load_session("ns", "s")).owner == "A"
    await store.save_session("ns", "s", _record(["a", "b"]), [_entry(2, "b")], owner="A")

    with pytest.raises(SessionTakenOverError, match="owned by another agent"):
        await store.save_session("ns", "s", _record(["a", "b", "x"]), [_entry(3, "x")], owner="B")

    # Nothing of B's reached the store: neither the state nor the log.
    loaded = await store.load_session("ns", "s")
    assert (loaded.owner, loaded.last_seq) == ("A", 2)
    assert await _history(store) == ["a", "b"]
    assert [e.seq for e in await store.session_log("ns", "s")] == [1, 2]
    await store.close()


async def test_taking_the_session_moves_ownership(store):
    await store.save_session("ns", "s", _record(["a"]), [_entry(1, "a")], owner="A")
    await store.take_session("ns", "s", "B")

    with pytest.raises(SessionTakenOverError):
        await store.save_session("ns", "s", _record(["a", "a2"]), [_entry(2, "a2")], owner="A")
    await store.save_session("ns", "s", _record(["a", "b2"]), [_entry(2, "b2")], owner="B")
    assert (await store.load_session("ns", "s")).owner == "B"
    assert await _history(store) == ["a", "b2"]
    await store.close()


async def test_taking_an_unsaved_session_is_a_no_op(store):
    await store.take_session("ns", "nothing", "A")
    assert await store.load_session("ns", "nothing") is None
    # The first save still establishes whoever makes it.
    await store.save_session("ns", "nothing", _record(["a"]), [_entry(1, "a")], owner="B")
    assert (await store.load_session("ns", "nothing")).owner == "B"
    await store.close()


async def test_an_anonymous_save_neither_checks_nor_changes_the_owner(store):
    """Direct store use (tooling, tests) passes no owner and is always accepted."""
    await store.save_session("ns", "s", _record(["a"]), [_entry(1, "a")], owner="A")
    await store.save_session("ns", "s", _record(["a", "t"]), [_entry(2, "t")])
    loaded = await store.load_session("ns", "s")
    assert loaded.owner == "A" and await _history(store) == ["a", "t"]
    # And a session saved without an owner belongs to the first agent to claim it.
    await store.save_session("ns", "free", _record(["f"]), [_entry(1, "f")])
    await store.save_session("ns", "free", _record(["f", "g"]), [_entry(2, "g")], owner="C")
    assert (await store.load_session("ns", "free")).owner == "C"
    await store.close()


async def test_an_old_database_gains_the_owner_and_lease_columns(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE session_state (
            namespace TEXT NOT NULL, session_id TEXT NOT NULL, history TEXT NOT NULL,
            state TEXT NOT NULL, last_seq INTEGER NOT NULL DEFAULT 0,
            fingerprint TEXT NOT NULL DEFAULT '', summarized_upto INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY (namespace, session_id));
        INSERT INTO session_state VALUES ('ns', 's', '[{"type": "UserMessage", "role": "user",
            "content": "kept", "metadata": {}}]', '{}', 1, '', 0, '2026-01-01', '2026-01-01');
        """
    )
    conn.commit()
    conn.close()

    backend = SQLiteBackend(path)
    columns = {
        row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(session_state)")
    }
    assert {"owner", "owner_since", "claim_token", "claim_until"} <= columns
    loaded = await backend.load_session("ns", "s")
    assert loaded.owner == "" and loaded.history[0]["content"] == "kept"  # old data intact
    await backend.save_session("ns", "s", _record(["kept", "new"]), [_entry(2, "new")], owner="A")
    assert (await backend.load_session("ns", "s")).owner == "A"
    far = "9999-01-01T00:00:00+00:00"
    tail = await backend.claim_idle_session("ns", idle_before=far, exclude_session="", lease_until=far)
    assert tail is not None and tail.token
    await backend.close()


# ---------------- the agent ----------------

def make_agent(db, script=(), *, session_id="ticket-1"):
    mock = MockLLMProvider(list(script))
    agent = Agent(
        AgentConfig(model="claude-haiku-4-5", api_key="k"),
        provider=mock,
        memory=MemoryConfig(store=str(db)),
        session_id=session_id,
    )
    return agent, mock


def errors(agent: Agent, name: str):
    return [
        e
        for e in agent.get_trace().all_spans()
        if e.event_type is EventType.ERROR and e.name == name
    ]


def texts(messages) -> list[str]:
    return [str(m.content) for m in messages]


async def test_two_live_agents_on_one_session_the_loser_fails_loudly(tmp_path):
    db = tmp_path / "m.db"
    x, mock_x = make_agent(db, [ScriptedTurn(text="x1"), ScriptedTurn(text="x2")])
    y, mock_y = make_agent(db, [ScriptedTurn(text="y1"), ScriptedTurn(text="y2")])

    await x.run("from x")  # x owns the session
    await y.run("from y")  # the same ticket on another server: y takes it over
    assert texts(y.get_history()) == ["from x", "x1", "from y", "y1"]

    result = await x.run("x again")
    assert result.final_message == "x2" and result.error is None  # the run itself completed
    [refused] = errors(x, "session_taken_over")  # but its save was refused, and recorded
    assert refused.status is SpanStatus.ERROR and "another agent" in refused.error.message
    assert errors(x, "session_save_failed") == []

    # From here on x refuses to continue — before any model call.
    calls = len(mock_x.call_log)
    with pytest.raises(SessionTakenOverError, match="ticket-1"):
        await x.run("x once more")
    with pytest.raises(SessionTakenOverError):
        await x.complete("x once more")
    with pytest.raises(SessionTakenOverError):
        async for _ in x.run_stream("x once more"):
            pass
    with pytest.raises(SessionTakenOverError):
        await x.memory.summarize()
    with pytest.raises(SessionTakenOverError):
        await x.clear_history()
    assert len(mock_x.call_log) == calls

    # y is unaffected, and the store holds exactly y's conversation.
    await y.run("y again")
    assert errors(y, "session_taken_over") == []
    backend = y.memory.long_term.backend
    stored = await backend.load_session("default", "ticket-1")
    assert [m["content"] for m in stored.history] == ["from x", "x1", "from y", "y1", "y again", "y2"]
    assert [e.seq for e in await backend.session_log("default", "ticket-1")] == [1, 2, 3, 4, 5, 6]
    for agent in (x, y):
        await agent.memory.close()


async def test_a_fresh_agent_takes_the_session_back(tmp_path):
    db = tmp_path / "m.db"
    y, _ = make_agent(db, [ScriptedTurn(text="y1"), ScriptedTurn(text="y2")])
    await y.run("from y")

    z, _ = make_agent(db, [ScriptedTurn(text="z1")])
    result = await z.run("from z")
    assert result.error is None
    assert texts(z.get_history()) == ["from y", "y1", "from z", "z1"]  # resumed y's conversation

    await y.run("y again")  # y's save is refused now
    assert errors(y, "session_taken_over")
    with pytest.raises(SessionTakenOverError):
        await y.run("y once more")
    assert [m["content"] for m in (await z.memory.long_term.backend.load_session("default", "ticket-1")).history] == [
        "from y", "y1", "from z", "z1"
    ]
    for agent in (y, z):
        await agent.memory.close()


async def test_a_sequential_handoff_between_processes_is_fine(tmp_path):
    """The common case: one agent closes, another opens the same session later."""
    db = tmp_path / "m.db"
    first, _ = make_agent(db, [ScriptedTurn(text="one")])
    await first.run("q1")
    await first.memory.close()

    second, _ = make_agent(db, [ScriptedTurn(text="two")])
    result = await second.run("q2")
    assert result.error is None
    assert texts(second.get_history()) == ["q1", "one", "q2", "two"]
    assert errors(second, "session_taken_over") == [] and errors(second, "session_save_failed") == []
    assert (await second.memory.long_term.backend.load_session("default", "ticket-1")).owner.startswith(
        second.agent_id
    )
    await second.memory.close()


async def test_connect_takes_ownership_when_it_loads(tmp_path):
    db = tmp_path / "m.db"
    first, _ = make_agent(db, [ScriptedTurn(text="one"), ScriptedTurn(text="two")])
    await first.run("q1")

    second = await Agent.connect(
        AgentConfig(model="claude-haiku-4-5", api_key="k"),
        mcp=_NoServers(),
        provider=MockLLMProvider([ScriptedTurn(text="2")]),
        memory=MemoryConfig(store=str(db)),
        session_id="ticket-1",
    )
    assert texts(second.get_history()) == ["q1", "one"]
    await first.run("q2")  # refused: connect() loaded the session, so it owns it
    assert errors(first, "session_taken_over")
    await second.run("q2")
    assert errors(second, "session_taken_over") == []
    for agent in (first, second):
        await agent.memory.close()


class _NoServers:
    """An MCP registry with nothing to discover."""

    async def discover_all(self, strict=False):
        return []

    def names(self):
        return []


async def test_two_new_agents_racing_for_a_new_session(tmp_path):
    """Neither loaded anything: the first to save owns the session, the
    other is refused — never two interleaved conversations under one id."""
    db = tmp_path / "m.db"
    a, _ = make_agent(db, [ScriptedTurn(text="a1")], session_id="fresh")
    b, _ = make_agent(db, [ScriptedTurn(text="b1")], session_id="fresh")
    await asyncio.gather(a.run("from a"), b.run("from b"))

    refused = [agent for agent in (a, b) if errors(agent, "session_taken_over")]
    owner = [agent for agent in (a, b) if not errors(agent, "session_taken_over")]
    assert len(refused) == 1 and len(owner) == 1
    stored = await a.memory.long_term.backend.load_session("default", "fresh")
    assert [m["content"] for m in stored.history] == texts(owner[0].get_history())
    assert stored.owner.startswith(owner[0].agent_id)
    for agent in (a, b):
        await agent.memory.close()


async def test_clear_history_before_the_first_run_takes_ownership(tmp_path):
    db = tmp_path / "m.db"
    first, _ = make_agent(db, [ScriptedTurn(text="one"), ScriptedTurn(text="two")])
    await first.run("q1")

    second, _ = make_agent(db, [ScriptedTurn(text="fresh")])
    await second.clear_history()  # continues the numbering, owns the session now
    await second.run("q2")
    assert texts(second.get_history()) == ["q2", "fresh"]
    await first.run("q2 from first")
    assert errors(first, "session_taken_over")
    for agent in (first, second):
        await agent.memory.close()


async def test_a_refused_save_takes_its_numbers_back(tmp_path):
    """Messages the refused save stamped are not left looking logged."""
    db = tmp_path / "m.db"
    x, _ = make_agent(db, [ScriptedTurn(text="x1"), ScriptedTurn(text="x2")])
    y, _ = make_agent(db, [ScriptedTurn(text="y1")])
    await x.run("from x")
    await y.run("from y")
    await x.run("x again")
    assert errors(x, "session_taken_over")
    unlogged = [m for m in x.get_history() if "log_seq" not in m.metadata]
    assert texts(unlogged) == ["x again", "x2"]
    for agent in (x, y):
        await agent.memory.close()
