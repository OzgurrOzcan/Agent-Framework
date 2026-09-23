"""Configuration types.

`AgentConfig` is the one config object. It is flat, frozen, validated at
construction, and serializable:

    AgentConfig(model="claude-opus-5", system_prompt="...", max_turns=6)

The model id carries the provider (see `providers.catalog`), so nothing has
to name a provider in the common case. Two nested records hold the knobs that
are genuinely rare, so the flat surface stays short:

  - `TransportConfig`: how we reach the API (timeouts, retries, SDK kwargs).
  - `CompactionConfig`: the context-compaction pipeline's thresholds.

Every setting lives in exactly one place. Tools, MCP, memory and permissions
are wiring, not configuration, and are passed to `Agent(...)` directly.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from agent_framework.providers.catalog import (
    DEFAULT_CONTEXT_WINDOW,
    ProviderSpec,
    get_provider,
    resolve_model,
)

# Re-exported for callers who budget context themselves.
__all__ = [
    "DEFAULT_CONTEXT_WINDOW",
    "LOCAL_API_KEY_PLACEHOLDER",
    "AgentConfig",
    "CompactionConfig",
    "TransportConfig",
]

# Sent as the bearer token when a custom `base_url` is set and no key is
# configured. Local runtimes (vLLM, llama.cpp, Ollama) require the header to
# exist but ignore its value.
LOCAL_API_KEY_PLACEHOLDER = "not-needed"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class TransportConfig:
    """How the SDK client reaches the provider. Rarely needs changing."""

    timeout_seconds: float = 60.0
    # Both vendor SDKs retry 429/5xx with exponential backoff up to this many
    # times.
    max_retries: int = 2
    # Forwarded verbatim to the SDK client constructor (e.g. a custom
    # http_client or default_headers). Not to be confused with
    # `AgentConfig.request_extra`, which is per-request.
    client_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(
            self.timeout_seconds > 0,
            f"transport.timeout_seconds must be > 0, got {self.timeout_seconds}",
        )
        _require(
            self.max_retries >= 0,
            f"transport.max_retries must be >= 0, got {self.max_retries}",
        )


@dataclass(frozen=True)
class CompactionConfig:
    """Thresholds for the context_manager compaction pipeline."""

    enabled: bool = True
    trigger_ratio: float = 0.70  # compact when >70% of budget used
    tool_result_truncate_threshold: int = 500  # tokens
    sliding_window_min_turns: int = 6  # always keep at least N most-recent turns
    full_compaction_min_recent_turns: int = 3
    summary_max_tokens: int = 800

    def __post_init__(self) -> None:
        _require(
            0.0 < self.trigger_ratio <= 1.0,
            f"compaction.trigger_ratio must be in (0, 1], got {self.trigger_ratio}",
        )
        _require(
            self.tool_result_truncate_threshold >= 0,
            "compaction.tool_result_truncate_threshold must be >= 0, "
            f"got {self.tool_result_truncate_threshold}",
        )
        _require(
            self.sliding_window_min_turns >= 1,
            "compaction.sliding_window_min_turns must be >= 1, "
            f"got {self.sliding_window_min_turns}",
        )
        _require(
            self.full_compaction_min_recent_turns >= 1,
            "compaction.full_compaction_min_recent_turns must be >= 1, "
            f"got {self.full_compaction_min_recent_turns}",
        )
        _require(
            self.summary_max_tokens >= 1,
            f"compaction.summary_max_tokens must be >= 1, got {self.summary_max_tokens}",
        )


@dataclass(frozen=True)
class AgentConfig:
    """Everything that configures one agent. Frozen, validated, serializable.

    `model` may name its provider explicitly (``"openai:llama-3.3-70b"``).
    After construction it is always the bare name that reaches the wire, and
    `provider` is always the resolved provider name — so a constructed config
    is fully explicit and round-trips through `to_dict()` without re-inferring
    anything.

    Derive a variant with `dataclasses.replace(config, max_turns=2)`; the
    frozen dataclass guarantees nothing mutates a config mid-run.
    """

    model: str
    system_prompt: str = ""
    max_turns: int = 10
    max_budget_usd: float | None = None
    # None = read from the provider's environment variable at construction.
    api_key: str | None = None
    # Point at any OpenAI-compatible endpoint (vLLM, a proxy, a gateway).
    base_url: str | None = None
    # None = infer from the model id. Set it to override, or to name a
    # provider whose models carry no recognizable prefix.
    provider: str | None = None
    stall_threshold: int = 3
    max_tokens_per_response: int = 4096
    # None = omit the parameter and let the model use its default. Note:
    # current-generation Anthropic models (Opus 4.7+/Opus 5/Sonnet 5) reject
    # explicit sampling parameters with a 400 — only set this when you know
    # the target model accepts it.
    temperature: float | None = None
    # None = resolve from the model id via the provider catalog.
    context_window: int | None = None
    response_buffer_tokens: int = 8192  # leave headroom for the model's response
    compaction_buffer_tokens: int = 4096  # leave headroom to run compaction itself
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    environment: dict[str, str] = field(default_factory=dict)
    # Duplicate tool descriptions into the system prompt (<capabilities>
    # block). Off by default: the tool definitions already reach the model via
    # the tools parameter; duplicating them doubles their token cost.
    include_capabilities_block: bool = False
    # Anthropic: place an incremental cache breakpoint on the conversation
    # (see AnthropicAdapter). No-op for providers that cache automatically.
    enable_prompt_caching: bool = True
    # Cap on concurrently executing tool calls within one turn.
    max_parallel_tools: int = 8
    # Extra provider-specific kwargs merged into every request
    # (e.g. `thinking={"type": "adaptive"}` for Claude). Per-request, unlike
    # `transport.client_kwargs`.
    request_extra: dict[str, Any] = field(default_factory=dict)

    # ---------------- normalization + validation ----------------
    def __post_init__(self) -> None:
        spec, bare_model = resolve_model(self.model)
        if self.provider is not None and self.provider != spec.name:
            raise ValueError(
                f"Model id {self.model!r} resolves to provider {spec.name!r}, "
                f"which contradicts provider={self.provider!r}. Drop one of "
                "the two, or make them agree."
            )
        # Frozen dataclass: normalize through object.__setattr__ so that after
        # construction `model` is always the wire name and `provider` is
        # always resolved.
        object.__setattr__(self, "model", bare_model)
        object.__setattr__(self, "provider", spec.name)

        _require(self.max_turns >= 1, f"max_turns must be >= 1, got {self.max_turns}")
        _require(
            self.max_budget_usd is None or self.max_budget_usd > 0,
            f"max_budget_usd must be > 0 or None, got {self.max_budget_usd}",
        )
        _require(
            self.stall_threshold >= 1,
            f"stall_threshold must be >= 1, got {self.stall_threshold}",
        )
        _require(
            self.max_tokens_per_response >= 1,
            f"max_tokens_per_response must be >= 1, got {self.max_tokens_per_response}",
        )
        _require(
            self.temperature is None or 0.0 <= self.temperature <= 2.0,
            f"temperature must be within [0.0, 2.0] or None, got {self.temperature}",
        )
        _require(
            self.context_window is None or self.context_window >= 1024,
            f"context_window must be >= 1024 or None, got {self.context_window}",
        )
        _require(
            self.response_buffer_tokens >= 0,
            f"response_buffer_tokens must be >= 0, got {self.response_buffer_tokens}",
        )
        _require(
            self.compaction_buffer_tokens >= 0,
            f"compaction_buffer_tokens must be >= 0, got {self.compaction_buffer_tokens}",
        )
        _require(
            self.max_parallel_tools >= 1,
            f"max_parallel_tools must be >= 1, got {self.max_parallel_tools}",
        )
        # An unrecognized model silently inherits the provider's fallback
        # window. Budgeting a locally served 8k model as 200k means compaction
        # never fires and the endpoint rejects the overflow instead.
        if self.context_window is None and spec.known_context_window(bare_model) is None:
            warnings.warn(
                f"Unknown context window for model {bare_model!r}; assuming "
                f"{spec.default_context_window} tokens. If the real window is "
                "smaller, compaction will not fire in time and the provider "
                "will reject the request — pass context_window=... to set it.",
                stacklevel=3,
            )

        # Buffers larger than the window leave a 1-token conversation budget
        # (ContextManager.budget clamps), which compacts every single turn.
        window = self.resolved_context_window
        buffers = self.response_buffer_tokens + self.compaction_buffer_tokens
        _require(
            buffers < window,
            f"response_buffer_tokens + compaction_buffer_tokens ({buffers}) must "
            f"be less than the context window ({window}); otherwise no tokens "
            "remain for the conversation and every turn triggers compaction.",
        )

    # ---------------- derived ----------------
    @property
    def provider_spec(self) -> ProviderSpec:
        """The resolved provider's catalog record."""
        assert self.provider is not None  # guaranteed by __post_init__
        return get_provider(self.provider)

    @property
    def resolved_context_window(self) -> int:
        if self.context_window is not None:
            return self.context_window
        return self.provider_spec.context_window(self.model)

    def resolved_api_key(self) -> str:
        """The API key, from `api_key` or the provider's environment variable.

        A custom `base_url` makes the key optional: that endpoint is not the
        vendor's own API, and a local runtime usually accepts any token — so
        demanding one would mean inventing a fake value for every local model.
        An explicit key and the environment variable still win, so a gateway
        that does authenticate keeps working; one that wants a key and gets the
        placeholder answers 401, which is a clear error rather than a silent one.
        """
        if self.api_key:
            return self.api_key
        env_key = self.provider_spec.env_key
        key = os.environ.get(env_key)
        if key:
            return key
        if self.base_url:
            return LOCAL_API_KEY_PLACEHOLDER
        raise RuntimeError(
            f"No API key for provider {self.provider!r}: pass "
            f"AgentConfig(api_key=...) / Agent(..., api_key=...), or set "
            f"the {env_key} environment variable. (A self-hosted endpoint "
            "needs no key — set base_url= and the key becomes optional.)"
        )

    def replace(self, **overrides: Any) -> AgentConfig:
        """A copy with *overrides* applied (`dataclasses.replace`)."""
        return replace(self, **overrides)

    # ---------------- serialization ----------------
    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable snapshot, safe to log or put in a trace.

        The API key is never included — only whether one was set explicitly.
        Everything else round-trips through `from_dict()`.
        """
        data = asdict(self)
        data.pop("api_key", None)
        data["api_key_set"] = self.api_key is not None
        data["resolved_context_window"] = self.resolved_context_window
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        """Rebuild a config from `to_dict()` output or a parsed config file.

        Unknown keys are rejected rather than ignored, so a typo in a YAML
        file fails at load time instead of silently doing nothing.
        """
        known = {f.name for f in fields(cls)}
        # Derived keys that to_dict() adds; they are outputs, not inputs.
        payload = {
            k: v
            for k, v in data.items()
            if k not in ("api_key_set", "resolved_context_window")
        }
        unknown = set(payload) - known
        if unknown:
            raise ValueError(
                f"Unknown AgentConfig field(s): {', '.join(sorted(unknown))}. "
                f"Valid fields: {', '.join(sorted(known))}."
            )
        nested = {"compaction": CompactionConfig, "transport": TransportConfig}
        for key, factory in nested.items():
            value = payload.get(key)
            if isinstance(value, dict):
                payload[key] = factory(**value)
        return cls(**payload)

    @classmethod
    def from_env(cls, prefix: str = "AGENT_", **overrides: Any) -> AgentConfig:
        """Build a config from environment variables, for 12-factor deploys.

        Reads the flat scalar fields only — `{prefix}MODEL`,
        `{prefix}SYSTEM_PROMPT`, `{prefix}MAX_TURNS`, and so on. The nested
        `compaction` / `transport` records and the dict-valued fields are not
        env-configurable; pass them via *overrides*, which win over the
        environment. Empty variables are treated as unset, and a value that
        does not parse raises rather than falling back to the default.
        """
        parsers: dict[str, Any] = {
            "model": str,
            "system_prompt": str,
            "provider": str,
            "base_url": str,
            "api_key": str,
            "max_turns": int,
            "max_tokens_per_response": int,
            "stall_threshold": int,
            "context_window": int,
            "response_buffer_tokens": int,
            "compaction_buffer_tokens": int,
            "max_parallel_tools": int,
            "max_budget_usd": float,
            "temperature": float,
            "include_capabilities_block": _parse_bool,
            "enable_prompt_caching": _parse_bool,
        }
        kwargs: dict[str, Any] = {}
        for name, parse in parsers.items():
            raw = os.environ.get(f"{prefix}{name.upper()}")
            if raw is None or not raw.strip():
                continue
            try:
                kwargs[name] = parse(raw.strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{prefix}{name.upper()}={raw!r} is not a valid "
                    f"value for AgentConfig.{name}: {exc}"
                ) from exc
        kwargs.update(overrides)
        if "model" not in kwargs:
            raise ValueError(
                f"No model configured: set {prefix}MODEL or pass model=... "
                "to AgentConfig.from_env()."
            )
        return cls(**kwargs)


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().casefold()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(
        f"expected one of {sorted(_TRUE | _FALSE)}, got {raw!r}"
    )
