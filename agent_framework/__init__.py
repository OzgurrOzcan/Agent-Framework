"""Production-grade Python agent framework.

Public API entry points:

    from agent_framework import Agent, AgentConfig, tool, PermissionCategory
    from agent_framework import PermissionConfig, AllowRule, DenyRule
    from agent_framework import ToolRegistry, MemoryManager
    from agent_framework import TerminalRenderer, CallbackRenderer, HeadlessRenderer

The model id names its own provider, so the common case needs no provider
argument at all:

    agent = Agent("claude-opus-5", system_prompt="...", tools=[my_tool])
"""

from agent_framework.core.config import AgentConfig, CompactionConfig, TransportConfig
from agent_framework.core.message import (
    Message,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ToolCallRequest,
    ToolResult,
    ReasoningContent,
    StopReason,
    message_from_dict,
)
from agent_framework.providers.base import ToolChoice
from agent_framework.providers.catalog import (
    ProviderResolutionError,
    ProviderSpec,
    register_provider,
)
from agent_framework.core.agent import Agent
from agent_framework.tools.decorator import tool
from agent_framework.tools.registry import ToolRegistry
from agent_framework.tools.models import ToolMetadata, RetryableToolError
from agent_framework.permissions.models import PermissionCategory, Verdict
from agent_framework.permissions.config import (
    PermissionConfig,
    AllowRule,
    DenyRule,
)
from agent_framework.interrupts.renderers.terminal import TerminalRenderer
from agent_framework.interrupts.renderers.callback import CallbackRenderer
from agent_framework.interrupts.renderers.headless import HeadlessRenderer
from agent_framework.interrupts.models import InterruptEvent, InterruptResponse
from agent_framework.memory.manager import MemoryManager
from agent_framework.memory.short_term import ShortTermMemory
from agent_framework.memory.long_term import LongTermMemory
from agent_framework.memory.backends.sqlite import SQLiteBackend
from agent_framework.mcp.registry import MCPServerRegistry, MCPConnectionConfig
from agent_framework.tracing.trace import ExecutionTrace, TraceEvent, ErrorInfo
from agent_framework.tracing.reporter import TraceReporter
from agent_framework.tracing.cost import CostTracker, PricingTable

__all__ = [
    "Agent",
    "AgentConfig",
    "CompactionConfig",
    "TransportConfig",
    "ProviderSpec",
    "ProviderResolutionError",
    "register_provider",
    "Message",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ToolCallRequest",
    "ToolResult",
    "ReasoningContent",
    "StopReason",
    "message_from_dict",
    "ToolChoice",
    "tool",
    "ToolRegistry",
    "ToolMetadata",
    "RetryableToolError",
    "PermissionCategory",
    "Verdict",
    "PermissionConfig",
    "AllowRule",
    "DenyRule",
    "TerminalRenderer",
    "CallbackRenderer",
    "HeadlessRenderer",
    "InterruptEvent",
    "InterruptResponse",
    "MemoryManager",
    "ShortTermMemory",
    "LongTermMemory",
    "SQLiteBackend",
    "MCPServerRegistry",
    "MCPConnectionConfig",
    "ExecutionTrace",
    "TraceEvent",
    "ErrorInfo",
    "TraceReporter",
    "CostTracker",
    "PricingTable",
]

__version__ = "0.1.0"
