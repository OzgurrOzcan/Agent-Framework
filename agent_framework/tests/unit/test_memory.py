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
