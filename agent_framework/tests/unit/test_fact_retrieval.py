"""Fact retrieval: which `remember()`ed facts a message brings into the context.

With SQLite the matching runs in the database (FTS5): every fact is
searched, diacritics are folded, the best match comes first. A fact whose
value changed since it was shown is shown again, saying what it replaces.
"""

from __future__ import annotations

import sqlite3

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.config import MemoryConfig
from agent_framework.memory.backends.in_memory import InMemoryBackend
from agent_framework.memory.backends.sqlite import SQLiteBackend
from agent_framework.memory.long_term import LongTermMemory
from agent_framework.memory.manager import MemoryManager, words
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn
from agent_framework.tracing.trace import EventType


def make_agent(db, script=()) -> tuple[Agent, MockLLMProvider]:
    mock = MockLLMProvider(list(script))
    return Agent("claude-opus-5-5", api_key="k", provider=mock, memory=MemoryConfig(store=str(db))), mock


def injected(messages) -> list[str]:
    return [str(m.content) for m in messages if m.metadata.get("context") == "relevant_facts"]


async def keys(agent: Agent, query: str) -> list[str]:
    return [f["key"] for f in await agent.memory.relevant_facts(query)]


# ---------------- an updated fact ----------------

async def test_an_updated_fact_is_shown_again_with_what_it_replaces(tmp_path):
    agent, mock = make_agent(tmp_path / "m.db", [ScriptedTurn(text=str(i)) for i in range(4)])
    await agent.memory.remember("deploy_target", "the staging cluster")
    await agent.run("where do we deploy?")
    assert injected(mock.call_log[0]["messages"]) == [
        "<relevant_context>\n- deploy_target: the staging cluster\n</relevant_context>"
    ]

    await agent.memory.remember("deploy_target", "the production cluster")
    await agent.run("deploy target again?")
    blocks = injected(agent.get_history())
    assert len(blocks) == 2  # the old block stays: nothing already sent is edited
    assert blocks[1] == (
        "<relevant_context>\n- deploy_target: the production cluster "
        "(updated; replaces: the staging cluster)\n</relevant_context>"
    )
    [first, second] = [
        e for e in agent.get_trace().all_spans()
        if e.event_type is EventType.MEMORY_READ and e.name == "relevant_facts"
    ]
    assert (first.payload["updated"], second.payload["updated"]) == (0, 1)

    await agent.run("and the deploy target once more?")  # the value is current: not repeated
    assert len(injected(agent.get_history())) == 2
    await agent.memory.remember("deploy_target", "the staging cluster")  # back to the first value
    await agent.run("deploy target?")
    assert injected(agent.get_history())[-1].endswith(
        "- deploy_target: the staging cluster (updated; replaces: the production cluster)\n</relevant_context>"
    )
    await agent.memory.close()


# ---------------- SQLite: matching in the database ----------------

async def test_sqlite_matches_regardless_of_diacritics(tmp_path):
    agent, _ = make_agent(tmp_path / "m.db")
    assert agent.memory.long_term.backend.full_text_search
    await agent.memory.remember("iletisim_tercihi", "Müşteri telefonla aranmak istiyor.")
    await agent.memory.remember("şube", "Merkez ofis İstanbul'da.")

    assert await keys(agent, "iletişim tercihi ne?") == ["iletisim_tercihi"]
    assert await keys(agent, "musteri nasil aranmak istiyor") == ["iletisim_tercihi"]
    assert await keys(agent, "sube nerede") == ["şube"]
    assert await keys(agent, "ISTANBUL ofisi") == ["şube"]
    assert await keys(agent, "something unrelated") == []
    await agent.memory.close()


async def test_sqlite_searches_every_fact_not_only_the_newest(tmp_path):
    agent, _ = make_agent(tmp_path / "m.db")
    await agent.memory.remember("oldest_fact", "the zebra lives in the savanna")
    for i in range(1_200):
        await agent.memory.remember(f"filler_{i}", f"filler value number {i}")
    assert await keys(agent, "where does the zebra live?") == ["oldest_fact"]
    await agent.memory.close()


async def test_capitalized_short_words_and_key_words_match(tmp_path):
    agent, _ = make_agent(tmp_path / "m.db")
    await agent.memory.remember("db_host", "postgres on pg-1")
    await agent.memory.remember("ci", "GitHub Actions, workflow build.yml")

    assert await keys(agent, "which DB do we use?") == ["db_host"]
    assert await keys(agent, "which db do we use?") == []  # two lower-case letters are noise
    assert await keys(agent, "CI status?") == ["ci"]
    assert await keys(agent, "is it ok?") == []
    await agent.memory.close()


async def test_sqlite_ranks_the_best_match_first(tmp_path):
    agent, _ = make_agent(tmp_path / "m.db")
    await agent.memory.remember("a", "python version is 3.12")
    await agent.memory.remember("b", "python packaging uses uv and python version pins")
    await agent.memory.remember("c", "the office plant is a fern")
    assert await keys(agent, "which python version and packaging tool?") == ["b", "a"]
    assert await keys(agent, "deployTarget or deploy target?") == []
    await agent.memory.remember("deployTarget", "eu-1 cluster")
    assert await keys(agent, "which target do we deploy to") == ["deployTarget"]
    await agent.memory.close()


async def test_tags_match_too(tmp_path):
    agent, _ = make_agent(tmp_path / "m.db")
    await agent.memory.long_term.remember("plan", "enterprise", tags=["billing", "contract"])
    assert await keys(agent, "a billing question") == ["plan"]
    await agent.memory.close()


async def test_forgotten_and_rewritten_facts_update_the_index(tmp_path):
    agent, _ = make_agent(tmp_path / "m.db")
    await agent.memory.remember("colour", "green")
    assert await keys(agent, "green?") == ["colour"]
    await agent.memory.remember("colour", "blue")
    assert await keys(agent, "green?") == [] and await keys(agent, "blue?") == ["colour"]
    await agent.memory.forget("colour")
    assert await keys(agent, "blue?") == []
    await agent.memory.close()


async def test_the_index_is_built_for_an_existing_database(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE facts (namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY (namespace, key));
        INSERT INTO facts VALUES ('default', 'iletisim_tercihi', 'telefonla aranmak istiyor',
            '["musteri"]', '2026-01-01', '2026-01-01');
        """
    )
    conn.commit()
    conn.close()

    agent, _ = make_agent(path)
    assert await keys(agent, "iletişim?") == ["iletisim_tercihi"]
    assert await keys(agent, "müşteri") == ["iletisim_tercihi"]  # tags were indexed too
    await agent.memory.close()
    # Reopening does not rebuild or duplicate the index.
    again, _ = make_agent(path)
    assert await keys(agent if False else again, "iletişim?") == ["iletisim_tercihi"]
    rows = sqlite3.connect(path).execute("SELECT COUNT(*) FROM facts_fts").fetchone()[0]
    assert rows == 1
    await again.memory.close()


async def test_without_fts_the_manager_matches_in_python(tmp_path):
    agent, mock = make_agent(tmp_path / "m.db", [ScriptedTurn(text="ok")])
    backend = agent.memory.long_term.backend
    await agent.memory.remember("plan", "the enterprise plan")
    backend._fts = False  # a SQLite build without the module
    assert await backend.match_facts({"plan"}) is None
    assert await keys(agent, "which plan?") == ["plan"]
    await agent.run("which plan?")
    assert "enterprise" in injected(mock.call_log[0]["messages"])[0]
    await agent.memory.close()


async def test_the_in_memory_backend_matches_in_python():
    backend = InMemoryBackend()
    assert await backend.match_facts({"x"}) is None
    manager = MemoryManager(long_term=LongTermMemory(backend=backend))
    await manager.remember("db_host", "pg-1")
    assert [f["key"] for f in await manager.relevant_facts("which DB?")] == ["db_host"]
    assert await manager.relevant_facts("which db?") == []


def test_the_word_rules():
    assert words("which DB do we use?") == {"which", "use", "db"}
    assert words("is it ok") == set()
    assert words("db_host", split_case=True) == {"db", "host"}
    assert words("deployTarget", split_case=True) == {"deploy", "target"}
    assert words("deployTarget") == {"deploytarget"}
    assert words("İstanbul'da k8s") == {"istanbul", "k8s"}
    assert words("V2 api") == {"v2", "api"}


@pytest.mark.parametrize("backend_name", ["memory", "sqlite"])
async def test_retrieval_is_scoped_to_the_namespace(tmp_path, backend_name):
    backend = InMemoryBackend() if backend_name == "memory" else SQLiteBackend(tmp_path / "s.db")
    a = MemoryManager(long_term=LongTermMemory(backend=backend, namespace="a"))
    b = MemoryManager(long_term=LongTermMemory(backend=backend, namespace="b"))
    await a.remember("plan", "enterprise plan")
    assert [f["key"] for f in await a.relevant_facts("plan?")] == ["plan"]
    assert await b.relevant_facts("plan?") == []
    await backend.close()
