"""Configuration types.

Two layers of config:
  - `ProviderConfig`: model name, API key, transport-level parameters.
  - `AgentConfig`: framework-level behavior (system prompt, max turns,
    budget, compaction, capabilities). Tools/MCP/memory/permissions are
    passed separately to `BaseAgent.create()`.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import Any


class ProviderKind(str, enum.Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Best-effort context-window table, matched by model-name prefix. Used only
# when `AgentConfig.context_window` is left unset. Conservative fallbacks.
_CONTEXT_WINDOWS: list[tuple[str, int]] = [
    ("claude-3", 200_000),
    ("claude-haiku", 200_000),
    ("claude", 1_000_000),  # current-generation Claude models (Opus/Sonnet 4.6+, 5.x)
    ("gpt-4o", 128_000),
    ("gpt-4.1", 1_000_000),
    ("gpt-5", 400_000),
    ("gpt-4", 128_000),
    ("o1", 200_000),
    ("o3", 200_000),
]

_DEFAULT_CONTEXT_WINDOW = 200_000


def model_context_window(model: str) -> int:
    """Best-effort context window for a model name (prefix match)."""
    for prefix, window in _CONTEXT_WINDOWS:
        if model.startswith(prefix):
            return window
    return _DEFAULT_CONTEXT_WINDOW


@dataclass
class ProviderConfig:
    provider: ProviderKind
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 60.0
    # Passed to the underlying SDK client; both SDKs retry 429/5xx with
    # exponential backoff up to this many times.
    max_retries: int = 2
    # Extra kwargs forwarded verbatim to the SDK client constructor
    # (e.g. custom http_client, default_headers).
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        env_name = {
            ProviderKind.OPENAI: "OPENAI_API_KEY",
            ProviderKind.ANTHROPIC: "ANTHROPIC_API_KEY",
        }[self.provider]
        key = os.environ.get(env_name)
        if not key:
            raise RuntimeError(
                f"No API key for {self.provider.value}: set ProviderConfig.api_key "
                f"or environment variable {env_name}"
            )
        return key


@dataclass
class CompactionConfig:
    """Knobs for context_manager compaction pipeline."""

    enabled: bool = True
    trigger_ratio: float = 0.70  # compact when >70% of budget used
    tool_result_truncate_threshold: int = 500  # tokens
    sliding_window_min_turns: int = 6  # always keep at least N most-recent turns
    full_compaction_min_recent_turns: int = 3
    summary_max_tokens: int = 800


@dataclass
class AgentConfig:
    """Framework-level agent configuration."""

    provider: ProviderConfig
    system_prompt: str = ""
    max_turns: int = 10
    max_budget_usd: float | None = None
    stall_threshold: int = 3
    max_tokens_per_response: int = 4096
    # None = omit the parameter and let the model use its default. Note:
    # current-generation Anthropic models (Opus 4.7+/Opus 5/Sonnet 5) reject
    # explicit sampling parameters with a 400 — only set this when you know
    # the target model accepts it.
    temperature: float | None = None
    # None = resolve from the model name (see `model_context_window`).
    context_window: int | None = None
    response_buffer_tokens: int = 8192  # leave headroom for the model's response
    compaction_buffer_tokens: int = 4096  # leave headroom to run compaction itself
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    environment: dict[str, str] = field(default_factory=dict)
    # Duplicate tool descriptions into the system prompt (<capabilities> block).
    # Off by default: the tool definitions already reach the model via the
    # tools parameter; duplicating them doubles their token cost.
    include_capabilities_block: bool = False
    # Anthropic: place an incremental cache breakpoint on the conversation
    # (see AnthropicAdapter). No-op for providers that cache automatically.
    enable_prompt_caching: bool = True
    # Cap on concurrently executing tool calls within one turn.
    max_parallel_tools: int = 8
    # Extra provider-specific request kwargs
    # (e.g. `thinking={"type": "adaptive"}` for Claude)
    provider_extra: dict[str, Any] = field(default_factory=dict)

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def provider_kind(self) -> ProviderKind:
        return self.provider.provider

    @property
    def resolved_context_window(self) -> int:
        if self.context_window is not None:
            return self.context_window
        return model_context_window(self.model)
