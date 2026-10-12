"""Save and resume conversations by `session_id`.

With a long-term store, every run/complete/run_stream saves the conversation
(in `finally`), and an agent with the same `session_id` continues it. The
store keeps a full log besides the current history, so nothing a trim drops
is lost.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.config import AgentConfig, MemoryConfig
from agent_framework.core.message import (
    AssistantMessage,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from agent_framework.memory.backends.base import Fact, LongTermBackend
from agent_framework.memory.backends.in_memory import InMemoryBackend
from agent_framework.memory.backends.sqlite import SQLiteBackend
from agent_framework.memory.sessions import LogEntry, SessionRecord
from agent_framework.memory.short_term import ContextState
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn
from agent_framework.tracing.trace import EventType, SpanStatus


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    backend = InMemoryBackend() if request.param == "memory" else SQLiteBackend(tmp_path / "s.db")
    yield backend


def _record(history, state=None, last_seq=0, **kw) -> SessionRecord:
    return SessionRecord(history=history, state=state or {}, last_seq=last_seq, **kw)


def _entry(seq: int, text: str) -> LogEntry:
    return LogEntry(seq=seq, message=UserMessage(content=text).to_dict())


# ---------------- the store contract (both backends) ----------------

async def test_log_entries_come_back_in_order(store):
    await store.save_session("ns", "s1", _record([], last_seq=2), [_entry(1, "a"), _entry(2, "b")])
    await store.save_session("ns", "s1", _record([], last_seq=3), [_entry(3, "c")])

    log = await store.session_log("ns", "s1")
    assert [(e.seq, e.message["content"]) for e in log] == [(1, "a"), (2, "b"), (3, "c")]
    assert [e.seq for e in await store.session_log("ns", "s1", after_seq=2)] == [3]
    await store.close()


async def test_history_and_state_round_trip(store):
    history = [UserMessage(content="q").to_dict(), AssistantMessage(content="a").to_dict()]
    state = ContextState(pinned=UserMessage(content="first"), cleared_results=2).to_dict()
    await store.save_session("ns", "s1", _record(history, state, last_seq=2, fingerprint="fp"), [])

    loaded = await store.load_session("ns", "s1")
    assert loaded.history == history and loaded.state == state
    assert (loaded.last_seq, loaded.fingerprint) == (2, "fp")
    assert loaded.created_at and loaded.updated_at
    await store.close()


async def test_sessions_are_scoped_by_namespace(store):
    await store.save_session("a", "s1", _record([{"content": "in a"}]), [_entry(1, "a")])
    assert await store.load_session("b", "s1") is None
    assert await store.session_log("b", "s1") == []
    await store.close()


async def test_non_json_content_is_stored_as_text(store):
    odd = ToolResultMessage(
        content=[ToolResult(tool_use_id="c", name="t", content={"at": datetime(2026, 1, 2, tzinfo=UTC)})]
    )
    await store.save_session("ns", "s1", _record([odd.to_dict()]), [LogEntry(1, odd.to_dict())])

    loaded = await store.load_session("ns", "s1")
    assert loaded.history[0]["content"][0]["content"]["at"].startswith("2026-01-02")
    await store.close()


async def test_nothing_aliases_the_stored_data(store):
    history = [{"content": "original"}]
    await store.save_session("ns", "s1", _record(history), [])
    history[0]["content"] = "changed after saving"
    loaded = await store.load_session("ns", "s1")
    loaded.history[0]["content"] = "changed after loading"

    assert (await store.load_session("ns", "s1")).history == [{"content": "original"}]
    await store.close()


async def test_counters_never_move_backwards(store):
    await store.save_session("ns", "s1", _record([], last_seq=5, summarized_upto=4), [])
    # A writer holding older values (another process summarized meanwhile).
    await store.save_session("ns", "s1", _record([], last_seq=3, summarized_upto=1), [])
    loaded = await store.load_session("ns", "s1")
    assert (loaded.last_seq, loaded.summarized_upto) == (5, 4)
    await store.close()


async def test_old_database_gains_session_tables(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE facts (namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY (namespace, key));
        CREATE TABLE summaries (namespace TEXT PRIMARY KEY, summary TEXT NOT NULL,
            updated_at TEXT NOT NULL);
        INSERT INTO facts VALUES ('default', 'k', 'v', '[]', '2026-01-01', '2026-01-01');
        """
    )
    conn.commit()
    conn.close()

    backend = SQLiteBackend(path)
    assert (await backend.get_fact("k")).value == "v"  # old data intact
    await backend.save_session("default", "s", _record([]), [_entry(1, "x")])
    assert len(await backend.session_log("default", "s")) == 1
    await backend.close()


# ---------------- the agent ----------------

def make_agent(store_path, script=(), *, session_id="ticket-1", **kwargs):
    mock = MockLLMProvider(list(script))
    config = AgentConfig(model="claude-haiku-4-5", api_key="k", **kwargs.pop("config", {}))
    agent = Agent(
        config,
        provider=mock,
        memory=MemoryConfig(store=str(store_path)),
        session_id=session_id,
        **kwargs,
    )
    return agent, mock


def contents(messages):
    return [str(m.content) for m in messages]


def events(agent, event_type, name=None):
    return [
        e
        for e in agent.get_trace().all_spans()
        if e.event_type is event_type and (name is None or e.name == name)
    ]


async def test_same_session_id_resumes_history_and_state(tmp_path):
    db = tmp_path / "m.db"
    first, _ = make_agent(db, [ScriptedTurn(text="one"), ScriptedTurn(text="two")])
    await first.run("q1")
    # As a trim would leave it; saved by the next run.
    first.memory.short_term.state = ContextState(cleared_results=1, reasoning_cut=2)
    await first.run("q2")
    await first.memory.close()

    second, mock = make_agent(db, [ScriptedTurn(text="three")])
    await second.run("q3")

    assert contents(mock.call_log[0]["messages"])[:4] == ["q1", "one", "q2", "two"]
    assert (second.memory.short_term.state.cleared_results, second.memory.short_term.state.reasoning_cut) == (1, 2)
    [loaded] = events(second, EventType.MEMORY_READ, "session")
    assert loaded.payload["resumed"] is True and loaded.payload["messages"] == 4
    await second.memory.close()


async def test_connect_loads_the_session_before_the_first_run(tmp_path):
    db = tmp_path / "m.db"
    first, _ = make_agent(db, [ScriptedTurn(text="one")])
    await first.run("q1")
    await first.memory.close()

    mock = MockLLMProvider([])
    from agent_framework.mcp.registry import MCPServerRegistry

    second = await Agent.connect(
        AgentConfig(model="claude-haiku-4-5", api_key="k"),
        mcp=MCPServerRegistry(),
        provider=mock,
        memory=MemoryConfig(store=str(db)),
        session_id="ticket-1",
    )
    assert contents(second.get_history()) == ["q1", "one"]
    await second.memory.close()


async def test_new_session_id_starts_fresh(tmp_path):
    db = tmp_path / "m.db"
    first, _ = make_agent(db, [ScriptedTurn(text="one")])
    await first.run("q1")
    await first.memory.close()

    other, mock = make_agent(db, [ScriptedTurn(text="hi")], session_id="ticket-2")
    await other.run("hello")
    assert contents(mock.call_log[0]["messages"]) == ["hello"]
    await other.memory.close()


async def test_run_complete_and_closed_stream_flush(tmp_path):
    db = tmp_path / "m.db"
    agent, _ = make_agent(db, [ScriptedTurn(text="one"), ScriptedTurn(text="two"), ScriptedTurn(text="three")])
    await agent.run("q1")
    await agent.complete("q2")
    stream = agent.run_stream("q3")
    async for _ in stream:
        break  # the consumer stops early
    await stream.aclose()
    await agent.memory.close()

    resumed, _ = make_agent(db)
    await resumed._ensure_session()
    assert contents(resumed.get_history())[:4] == ["q1", "one", "q2", "two"]
    assert "q3" in contents(resumed.get_history())
    assert len(events(agent, EventType.MEMORY_WRITE, "session")) == 3
    await resumed.memory.close()


async def test_clear_history_persists_an_empty_state(tmp_path):
    db = tmp_path / "m.db"
    agent, _ = make_agent(db, [ScriptedTurn(text="one")])
    await agent.run("q1")
    await agent.clear_history()
    await agent.memory.close()

    resumed, mock = make_agent(db, [ScriptedTurn(text="fresh")])
    await resumed.run("hello again")
    assert contents(mock.call_log[0]["messages"]) == ["hello again"]
    await resumed.memory.close()


async def test_clear_history_before_the_first_run_never_loads_the_old_one(tmp_path):
    db = tmp_path / "m.db"
    agent, _ = make_agent(db, [ScriptedTurn(text="one")])
    await agent.run("q1")
    await agent.memory.close()

    resumed, mock = make_agent(db, [ScriptedTurn(text="fresh")])
    await resumed.clear_history()
    await resumed.run("hello")
    assert contents(mock.call_log[0]["messages"]) == ["hello"]
    await resumed.memory.close()


class FailingSaves(InMemoryBackend):
    async def save_session(self, *args, **kwargs):
        raise OSError("disk full")


async def test_flush_failure_is_traced_not_raised():
    agent = Agent(
        "claude-haiku-4-5",
        api_key="k",
        provider=MockLLMProvider([ScriptedTurn(text="ok")]),
        memory=MemoryConfig(),
        memory_backend=FailingSaves(),
    )
    result = await agent.run("q")

    assert result.final_message == "ok" and result.error is None
    [failure] = events(agent, EventType.ERROR, "session_save_failed")
    assert failure.status is SpanStatus.ERROR
    assert "disk full" in failure.error.message


class FailingLoads(InMemoryBackend):
    async def load_session(self, namespace, session_id):
        raise OSError("unreadable")


async def test_a_load_failure_never_overwrites_the_stored_session():
    backend = FailingLoads()
    await InMemoryBackend.save_session(backend, "default", "s", _record([{"content": "kept"}]), [])
    agent = Agent(
        "claude-haiku-4-5",
        api_key="k",
        provider=MockLLMProvider([ScriptedTurn(text="ok")]),
        memory=MemoryConfig(),
        memory_backend=backend,
        session_id="s",
    )
    result = await agent.run("q")

    assert result.error is None
    assert events(agent, EventType.ERROR, "session_load_failed")
    assert not events(agent, EventType.MEMORY_WRITE, "session")
    assert (await InMemoryBackend.load_session(backend, "default", "s")).history == [{"content": "kept"}]


async def test_stateless_agent_never_flushes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stateless = Agent("claude-haiku-4-5", api_key="k", provider=MockLLMProvider([]))
    in_ram = Agent("claude-haiku-4-5", api_key="k", provider=MockLLMProvider([]), memory=MemoryConfig())
    await stateless.run("q")
    await in_ram.run("q")

    for agent in (stateless, in_ram):
        assert events(agent, EventType.MEMORY_WRITE) == []
        assert events(agent, EventType.MEMORY_READ, "session") == []
    assert list(tmp_path.iterdir()) == []


async def test_dry_run_does_not_save(tmp_path):
    agent, _ = make_agent(tmp_path / "m.db")
    await agent.run("q", dry_run=True)
    assert events(agent, EventType.MEMORY_WRITE) == []
    await agent.memory.close()


async def test_the_log_keeps_messages_a_trim_dropped(tmp_path):
    db = tmp_path / "m.db"
    agent, _ = make_agent(
        db, [ScriptedTurn(text="ok")], config={"max_context_tokens": 3_000, "max_tool_result_tokens": 300}
    )
    for i in range(8):
        agent.memory.short_term.extend(
            [UserMessage(content=f"q{i}"), AssistantMessage(content=f"a{i} " + "x" * 2_000)]
        )
    await agent.run("q8")
    assert len(agent.get_history()) < 18  # the trim dropped old turns ...

    log = await agent.memory.long_term.backend.session_log("default", "ticket-1")
    texts = [str(e.message["content"]) for e in log]
    assert texts[:2] == ["q0", "a0 " + "x" * 2_000]  # ... the log kept them
    assert len(texts) == 18 and [e.seq for e in log] == list(range(1, 19))
    await agent.memory.close()


async def test_resume_with_a_changed_prompt_drops_old_reasoning(tmp_path):
    """Reasoning is bound to the system prompt it was produced with."""
    db = tmp_path / "m.db"
    first, _ = make_agent(
        db,
        [ScriptedTurn(text="one", reasoning=[{"type": "thinking", "signature": "s"}])],
        config={"system_prompt": "v1"},
    )
    await first.run("q1")
    await first.memory.close()

    same, mock_same = make_agent(db, [ScriptedTurn(text="x")], config={"system_prompt": "v1"})
    await same.run("q2")
    assert mock_same.call_log[0]["messages"][1].reasoning  # unchanged prompt: kept
    await same.memory.close()

    changed, mock_changed = make_agent(db, [ScriptedTurn(text="y")], config={"system_prompt": "v2"})
    await changed.run("q3")
    sent = mock_changed.call_log[0]["messages"]
    assert not any(isinstance(m, AssistantMessage) and m.reasoning for m in sent)
    [loaded] = events(changed, EventType.MEMORY_READ, "session")
    assert loaded.payload["reasoning_dropped"] is True
    await changed.memory.close()


class FactsOnly(LongTermBackend):
    """A custom backend that keeps facts but not sessions."""

    def __init__(self):
        self.facts = InMemoryBackend()

    async def put_fact(self, fact: Fact) -> None:
        await self.facts.put_fact(fact)

    async def get_fact(self, key, namespace="default"):
        return await self.facts.get_fact(key, namespace)

    async def delete_fact(self, key, namespace="default"):
        await self.facts.delete_fact(key, namespace)

    async def search_facts(self, **kwargs):
        return await self.facts.search_facts(**kwargs)

    async def list_facts(self, namespace="default", limit=200):
        return await self.facts.list_facts(namespace, limit)

    async def set_summary(self, namespace, summary):
        await self.facts.set_summary(namespace, summary)

    async def get_summary(self, namespace):
        return await self.facts.get_summary(namespace)


def test_a_backend_without_sessions_warns():
    with pytest.warns(UserWarning, match="SessionStore"):
        Agent("claude-haiku-4-5", api_key="k", memory=MemoryConfig(), memory_backend=FactsOnly())


# ---------------- review regressions: numbering, cancellation, failures ----------------

async def _log_contents(agent, session_id="ticket-1"):
    log = await agent.memory.long_term.backend.session_log("default", session_id)
    return [e.seq for e in log], [str(e.message["content"]) for e in log]


async def test_clearing_before_the_first_run_continues_the_numbering(tmp_path):
    """The stored log already holds 1..N; a fresh numbering would collide."""
    db = tmp_path / "m.db"
    first, _ = make_agent(db, [ScriptedTurn(text="one")])
    await first.run("q1")
    await first.memory.close()

    second, _ = make_agent(db, [ScriptedTurn(text="two")])
    await second.clear_history()
    await second.run("q2")

    assert events(second, EventType.ERROR, "session_save_failed") == []
    seqs, texts = await _log_contents(second)
    assert seqs == [1, 2, 3, 4] and texts[2:] == ["q2", "two"]
    await second.memory.close()


async def test_a_save_interrupted_by_cancellation_keeps_its_numbers(tmp_path):
    """The SQLite write runs on in its thread after the run is cancelled; it
    committed, so its numbers must stay — or every later save collides."""
    import asyncio
    import time

    agent, _ = make_agent(tmp_path / "m.db", [ScriptedTurn(text=t) for t in ("1", "2", "3")])
    await agent.run("q1")
    backend = agent.memory.long_term.backend
    original = backend._run
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()

    def slow(work, *, immediate=False):
        if work.__qualname__.startswith("SQLiteBackend.save_session"):
            loop.call_soon_threadsafe(entered.set)
            time.sleep(0.2)
        return original(work, immediate=immediate)

    backend._run = slow
    task = asyncio.create_task(agent.run("q2"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    backend._run = original
    await agent.run("q3")

    assert events(agent, EventType.ERROR, "session_save_failed") == []
    seqs, texts = await _log_contents(agent)
    assert seqs == list(range(1, 7))
    assert texts == ["q1", "1", "q2", "2", "q3", "3"]
    await agent.memory.close()


async def test_concurrent_saves_do_not_reuse_numbers(tmp_path):
    import asyncio

    agent, _ = make_agent(tmp_path / "m.db", [ScriptedTurn(text="one")])
    await agent.run("q1")
    agent.memory.short_term.append(UserMessage(content="extra"))
    await asyncio.gather(agent._flush_session(), agent._flush_session())

    assert events(agent, EventType.ERROR, "session_save_failed") == []
    seqs, _ = await _log_contents(agent)
    assert seqs == [1, 2, 3]
    await agent.memory.close()


async def test_messages_cleared_after_a_failed_save_still_reach_the_log():
    class FlakySaves(InMemoryBackend):
        fail = True

        async def save_session(self, *args, **kwargs):
            if self.fail:
                raise OSError("disk full")
            await super().save_session(*args, **kwargs)

    backend = FlakySaves()
    agent = Agent(
        "claude-haiku-4-5",
        api_key="k",
        provider=MockLLMProvider([ScriptedTurn(text="one"), ScriptedTurn(text="two")]),
        memory=MemoryConfig(),
        memory_backend=backend,
        session_id="s",
    )
    await agent.run("q1")  # the save fails: nothing logged
    backend.fail = False
    await agent.clear_history()
    await agent.run("q2")

    log = await backend.session_log("default", "s")
    assert [str(e.message["content"]) for e in log] == ["q1", "one", "q2", "two"]


def test_ram_only_memory_holds_nothing_for_a_log():
    """Without a log, messages a trim drops must not pile up anywhere."""
    agent = Agent("claude-haiku-4-5", api_key="k", memory=MemoryConfig())
    memory = agent.memory.short_term
    assert memory.track_unlogged is False
    memory.extend([UserMessage(content="q"), AssistantMessage(content="a")] * 3)
    memory.drop_head(2)
    memory.clear()
    assert memory.unlogged_dropped() == []
