"""Long-term memory from `MemoryConfig.store` — no database code for users.

A store path builds a SQLite-backed long-term memory the agent owns and
closes; `memory_backend=` swaps in any other backend; a stateless agent never
opens a store at all.
"""

from __future__ import annotations

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.config import MemoryConfig
from agent_framework.memory.backends.in_memory import InMemoryBackend
from agent_framework.memory.backends.sqlite import SQLiteBackend
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn
from agent_framework.tracing.trace import EventType


def make_agent(script=(), **kwargs) -> tuple[Agent, MockLLMProvider]:
    mock = MockLLMProvider(list(script))
    return Agent("claude-opus-5-5", api_key="k", provider=mock, **kwargs), mock


def memory_events(agent: Agent):
    return [
        e
        for e in agent.get_trace().all_spans()
        if e.event_type in (EventType.MEMORY_READ, EventType.MEMORY_WRITE)
    ]


def facts_sent(mock: MockLLMProvider, call: int = 0) -> list[str]:
    return [
        str(m.content)
        for m in mock.call_log[call]["messages"]
        if m.metadata.get("context") == "relevant_facts"
    ]


async def test_store_builds_sqlite_memory(tmp_path):
    store = str(tmp_path / "support.db")
    first, _ = make_agent(memory=MemoryConfig(store=store, namespace="customer-42"))
    backend = first.memory.long_term.backend
    assert isinstance(backend, SQLiteBackend) and backend.path == store

    await first.memory.remember("plan", "the customer is on the enterprise plan")
    [write] = [e for e in memory_events(first) if e.event_type is EventType.MEMORY_WRITE]
    assert write.payload == {"namespace": "customer-42", "key": "plan"}
    await first.memory.close()

    # Another agent — another process, in production — on the same store.
    second, mock = make_agent(
        [ScriptedTurn(text="ok")], memory=MemoryConfig(store=store, namespace="customer-42")
    )
    await second.run("which plan is the customer on?")
    assert "enterprise plan" in facts_sent(mock)[0]
    assert any(e.name == "relevant_facts" for e in memory_events(second))
    await second.memory.close()


async def test_namespaces_are_separate(tmp_path):
    store = str(tmp_path / "m.db")
    a, _ = make_agent(memory=MemoryConfig(store=store, namespace="a"))
    b, mock = make_agent([ScriptedTurn(text="ok")], memory=MemoryConfig(store=store, namespace="b"))
    await a.memory.remember("plan", "enterprise plan")
    await b.run("which plan?")
    assert facts_sent(mock) == []
    await a.memory.close()
    await b.memory.close()


class RecordingBackend(InMemoryBackend):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


async def test_memory_backend_is_the_escape_hatch():
    backend = RecordingBackend()
    agent, mock = make_agent(
        [ScriptedTurn(text="ok")],
        memory=MemoryConfig(namespace="team"),
        memory_backend=backend,
    )
    await agent.memory.remember("deploy_target", "the eu-1 cluster")
    assert (await backend.get_fact("deploy_target", namespace="team")).value == "the eu-1 cluster"

    await agent.run("what is the deploy_target?")
    assert "eu-1" in facts_sent(mock)[0]
    # The backend is the caller's: the agent does not close it.
    await agent.memory.close()
    assert backend.closed is False


async def test_stateless_agent_never_touches_the_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent, _ = make_agent([ScriptedTurn(text="ok")])
    await agent.run("hello")
    await agent.memory.remember("k", "v")  # nowhere to remember it: a no-op

    assert agent.memory.long_term is None
    assert list(tmp_path.iterdir()) == []
    assert memory_events(agent) == []
    await agent.memory.close()


async def test_store_accepts_an_in_memory_sqlite_path():
    """One connection for the backend's lifetime keeps ':memory:' alive."""
    agent, mock = make_agent([ScriptedTurn(text="ok")], memory=MemoryConfig(store=":memory:"))
    await agent.memory.remember("color", "the favourite color is green")
    await agent.run("what is my favourite color?")
    assert "green" in facts_sent(mock)[0]
    await agent.memory.close()


async def test_close_is_idempotent(tmp_path):
    agent, _ = make_agent(memory=MemoryConfig(store=str(tmp_path / "m.db")))
    await agent.memory.close()
    await agent.memory.close()
    assert agent.memory.long_term.backend.closed
    with pytest.raises(RuntimeError, match="closed"):
        await agent.memory.remember("k", "v")


async def test_to_json_does_not_load_memory(tmp_path):
    agent, mock = make_agent(memory=MemoryConfig(store=str(tmp_path / "m.db")))
    await agent.memory.remember("secret_plan", "never in structured output")
    before = len(memory_events(agent))
    mock.structured_script.append('{"answer": "x"}')

    await agent.to_json(
        "what is the secret_plan?",
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )

    [call] = mock.structured_calls
    assert [str(m.content) for m in call["messages"]] == ["what is the secret_plan?"]
    assert len(memory_events(agent)) == before  # nothing read, nothing written
    assert agent.get_history() == []
    await agent.memory.close()
