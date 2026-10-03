"""Memory: short-term (the conversation) and long-term (facts, sessions, summaries).

Agents are stateless unless `MemoryConfig` turns memory on; see
`agent_framework.core.config.MemoryConfig`. A custom long-term backend
implements `LongTermBackend` (facts) and, to keep conversations and their
summaries, `SessionStore`.
"""

from agent_framework.memory.short_term import ContextState, ShortTermMemory
from agent_framework.memory.long_term import LongTermMemory, Fact
from agent_framework.memory.manager import MemoryManager
from agent_framework.memory.backends.base import LongTermBackend
from agent_framework.memory.backends.sqlite import SQLiteBackend
from agent_framework.memory.backends.in_memory import InMemoryBackend
from agent_framework.memory.sessions import (
    IdleTail,
    LogEntry,
    SessionRecord,
    SessionStore,
    SummarySegment,
)

__all__ = [
    "ContextState",
    "ShortTermMemory",
    "LongTermMemory",
    "Fact",
    "MemoryManager",
    "LongTermBackend",
    "SQLiteBackend",
    "InMemoryBackend",
    "SessionStore",
    "SessionRecord",
    "LogEntry",
    "SummarySegment",
    "IdleTail",
]
