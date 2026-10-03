"""Tests for memory backends and manager."""
import pytest
from agent_framework.memory.backends.in_memory import InMemoryBackend
from agent_framework.memory.long_term import LongTermMemory
from agent_framework.memory.manager import MemoryManager
from agent_framework.memory.short_term import ShortTermMemory
from agent_framework.core.message import UserMessage, AssistantMessage


@pytest.mark.asyncio
async def test_in_memory_put_get():
    backend = InMemoryBackend()
    lt = LongTermMemory(backend=backend)
    await lt.remember("user_name", "Alice", tags=["user"])
    fact = await lt.recall("user_name")
    assert fact is not None
    assert fact.value == "Alice"
    assert "user" in fact.tags


@pytest.mark.asyncio
async def test_in_memory_search():
    backend = InMemoryBackend()
    lt = LongTermMemory(backend=backend)
    await lt.remember("project_x", "A machine learning project", tags=["project"])
    await lt.remember("project_y", "A web development project", tags=["project"])
    results = await lt.search(query="machine learning")
    assert len(results) == 1
    assert results[0].key == "project_x"


@pytest.mark.asyncio
async def test_in_memory_summary():
    backend = InMemoryBackend()
    lt = LongTermMemory(backend=backend)
    await lt.set_summary("User is a Python developer.")
    s = await lt.get_summary()
    assert s == "User is a Python developer."


@pytest.mark.asyncio
async def test_in_memory_forget():
    backend = InMemoryBackend()
    lt = LongTermMemory(backend=backend)
    await lt.remember("temp_key", "temp_value")
    await lt.forget("temp_key")
    fact = await lt.recall("temp_key")
    assert fact is None


def test_short_term_append_and_snapshot():
    st = ShortTermMemory()
    st.append(UserMessage(content="Hello"))
    st.append(AssistantMessage(content="Hi!"))
    snapshot = st.snapshot()
    assert len(snapshot) == 2
    assert snapshot[0].role == "user"


@pytest.mark.asyncio
async def test_memory_manager_relevant_facts():
    backend = InMemoryBackend()
    lt = LongTermMemory(backend=backend)
    await lt.remember("python_version", "3.12", tags=["environment"])
    await lt.remember("user_goal", "build an AI agent", tags=["goal"])
    mm = MemoryManager(long_term=lt)
    facts = await mm.relevant_facts("I want to build a Python agent")
    keys = [f["key"] for f in facts]
    assert "python_version" in keys or "user_goal" in keys


# ---------------- whole-word, Unicode-aware fact matching ----------------

async def _manager_with(*facts: tuple[str, str]) -> MemoryManager:
    lt = LongTermMemory(backend=InMemoryBackend())
    for key, value in facts:
        await lt.remember(key, value)
    return MemoryManager(long_term=lt)


async def test_turkish_query_finds_the_matching_fact():
    mm = await _manager_with(
        ("odeme_gunu", "Bekleyen faturalar her ayın 5'inde ödenir."),
        ("sube", "Merkez ofis İstanbul'da."),
    )
    found = await mm.relevant_facts("FATURALAR ne zaman ödeniyor?")
    assert [f["key"] for f in found] == ["odeme_gunu"]
    # A dotted capital I folds to the "i" users type.
    found = await mm.relevant_facts("istanbul ofisi nerede")
    assert [f["key"] for f in found] == ["sube"]


async def test_whole_word_not_substring():
    mm = await _manager_with(("event", "the party planning starts at noon"))
    assert await mm.relevant_facts("show me some art") == []
    assert [f["key"] for f in await mm.relevant_facts("party time?")] == ["event"]


async def test_words_inside_keys_match():
    mm = await _manager_with(("user_name", "Alice"), ("deployTarget", "eu-1 cluster"))
    assert [f["key"] for f in await mm.relevant_facts("what is my name?")] == ["user_name"]
    assert [f["key"] for f in await mm.relevant_facts("which target do we deploy to")] == [
        "deployTarget"
    ]


async def test_more_shared_words_rank_higher():
    mm = await _manager_with(
        ("a", "python version is 3.12"),
        ("b", "python packaging uses uv and python version pins"),
    )
    found = await mm.relevant_facts("which python version and packaging tool?")
    assert [f["key"] for f in found] == ["b", "a"]
