"""Configuration types.

`AgentConfig` is the one config object. It is flat, frozen, validated at
construction, and serializable:

    AgentConfig(model="claude-opus-5", system_prompt="...", max_turns=6)

The model id carries the provider (see `providers.catalog`), so nothing has
to name a provider in the common case. One nested record holds the knobs that
are genuinely rare, so the flat surface stays short:

  - `TransportConfig`: how we reach the API (timeouts, retries, SDK kwargs).

Context management has two flat limits and no record of its own: the prompt
is trimmed when it reaches `max_context_tokens`, and a tool result bigger than
`max_tool_result_tokens` is cut when it arrives. Both default to values derived
from the model's context window.

An agent is stateless unless `memory` is set: every run starts from an empty
history. `MemoryConfig()` keeps the conversation in RAM across runs;
`MemoryConfig(store="app.db")` also keeps long-term memory in SQLite.

Every setting lives in exactly one place. Tools, MCP, memory and permissions
are wiring, not configuration, and are passed to `Agent(...)` directly.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import PurePath
from typing import Any

from agent_framework.providers.catalog import (
    DEFAULT_CONTEXT_WINDOW,
    ProviderSpec,
    get_provider,
    resolve_model,
    split_model_id,
)

# Re-exported for callers who budget context themselves.
__all__ = [
    "DEFAULT_CONTEXT_WINDOW",
    "LOCAL_API_KEY_PLACEHOLDER",
    "AgentConfig",
    "MemoryConfig",
    "TransportConfig",
]

# Sent as the bearer token when a custom `base_url` is set and no key is
# configured. Local runtimes (vLLM, llama.cpp, Ollama) require the header to
# exist but ignore its value.
LOCAL_API_KEY_PLACEHOLDER = "not-needed"

# Automatic context limits. The prompt is trimmed at 70 % of the window (never
# above 150k, where long prompts get slow and expensive while rarely helping),
# leaving room for the response and for the turn that triggers the trim.
_AUTO_CONTEXT_SHARE = 0.70
_AUTO_CONTEXT_CEILING = 150_000
# A single tool result may use at most 10 % of the window, 16k tokens, or a
# fifth of the trim limit — so a few recent results always fit after a trim.
_AUTO_TOOL_RESULT_CEILING = 16_000

# Fields that existed before and what replaces them, so a stale config file
# fails with a migration hint instead of a bare "unknown field".
_REMOVED_FIELDS = {
    "compaction": "replaced by max_context_tokens and max_tool_result_tokens",
    "response_buffer_tokens": (
        "removed: the usable window is context_window - max_tokens_per_response"
    ),
    "compaction_buffer_tokens": "removed: trimming makes no LLM call, so needs no headroom",
}


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
class MemoryConfig:
    """Turns on conversation memory. Without one, every run starts fresh.

        MemoryConfig()                                   # history kept in RAM
        MemoryConfig(store="support.db", namespace="customer-42")   # + SQLite

    `namespace`, `summarize` and `summary_model` describe long-term memory, so
    they need a `store` (or a custom `Agent(memory_backend=...)`); `Agent`
    rejects them without one rather than silently ignoring them.
    """

    # A SQLite file path → long-term memory. None = the history lives in RAM.
    store: str | None = None
    # Whose long-term memory this is: a user, a customer, a project.
    namespace: str = "default"
    # Long-term: summarize the turns a trim removes (one LLM call per trim).
    summarize: bool = True
    # Long-term: a cheaper model of the same provider for those summaries.
    # None = the agent's own model.
    summary_model: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.store, PurePath):
            object.__setattr__(self, "store", str(self.store))
        _require(
            self.store is None or (isinstance(self.store, str) and self.store.strip() != ""),
            f"MemoryConfig.store must be a non-empty path or None, got {self.store!r}",
        )
        _require(
            isinstance(self.namespace, str) and self.namespace.strip() != "",
            f"MemoryConfig.namespace must be a non-empty string, got {self.namespace!r}",
        )
        _require(
            isinstance(self.summarize, bool),
            f"MemoryConfig.summarize must be a bool, got {self.summarize!r}",
        )
        _require(
            self.summary_model is None
            or (isinstance(self.summary_model, str) and self.summary_model.strip() != ""),
            f"MemoryConfig.summary_model must be a model id or None, got {self.summary_model!r}",
        )

    def long_term_settings(self) -> list[str]:
        """The long-term-only fields that were changed from their defaults."""
        changed = []
        if self.namespace != "default":
            changed.append("namespace")
        if self.summarize is not True:
            changed.append("summarize")
        if self.summary_model is not None:
            changed.append("summary_model")
        return changed


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
    # None = infer from the model id. Set it to override inference (Claude
    # through an OpenAI-compatible gateway), or to name the provider of a
    # model that carries no recognizable prefix. Equivalent to writing
    # "provider:model"; the two may appear together only if they agree.
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
    # The prompt (system + tools + messages) is trimmed when it reaches this
    # size. None = min(70 % of the window, 150k, window - max_tokens_per_response).
    max_context_tokens: int | None = None
    # A tool result bigger than this is cut (head + tail) when it arrives;
    # `@tool(max_result_tokens=...)` overrides it per tool. None = the smallest
    # of 16k, 10 % of the window and a fifth of max_context_tokens.
    max_tool_result_tokens: int | None = None
    # None = stateless: every run starts from an empty history. A
    # MemoryConfig keeps the conversation (see MemoryConfig).
    memory: MemoryConfig | None = None
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
        # An explicit provider — a "provider:" prefix or the `provider` field —
        # is authoritative, and inference runs only when neither is given.
        # That is what makes normalization idempotent: a constructed config
        # stores the bare model plus the resolved provider, and feeding those
        # two fields back in (replace, from_dict, from_env) must reproduce the
        # same config — including for a model no provider's prefixes claim,
        # which re-inferring from the bare name could never resolve.
        prefix, bare_model = split_model_id(self.model)
        if prefix is not None and self.provider is not None and prefix != self.provider:
            raise ValueError(
                f"Model id {self.model!r} names provider {prefix!r}, which "
                f"contradicts provider={self.provider!r}. Drop one of the two, "
                "or make them agree."
            )
        explicit = prefix if prefix is not None else self.provider
        if explicit is not None:
            spec = get_provider(explicit)
        else:
            spec, bare_model = resolve_model(self.model)
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
        for name in ("max_context_tokens", "max_tool_result_tokens"):
            value = getattr(self, name)
            _require(
                value is None
                or (isinstance(value, int) and not isinstance(value, bool) and value >= 1),
                f"{name} must be a positive int or None, got {value!r}",
            )
        _require(
            self.max_parallel_tools >= 1,
            f"max_parallel_tools must be >= 1, got {self.max_parallel_tools}",
        )
        if self.memory is not None and not isinstance(self.memory, MemoryConfig):
            raise TypeError(
                "memory must be a MemoryConfig or None, got "
                f"{type(self.memory).__name__}. A custom long-term backend is "
                "passed as Agent(..., memory_backend=...)."
            )
        # An unrecognized model silently inherits the provider's fallback
        # window. Budgeting a locally served 8k model as 200k means trimming
        # never fires and the endpoint rejects the overflow instead.
        if self.context_window is None and spec.known_context_window(bare_model) is None:
            warnings.warn(
                f"Unknown context window for model {bare_model!r}; assuming "
                f"{spec.default_context_window} tokens. If the real window is "
                "smaller, trimming will not fire in time and the provider "
                "will reject the request — pass context_window=... to set it.",
                stacklevel=3,
            )

        # The response shares the window with the prompt, so the prompt can
        # use at most what the response leaves.
        window = self.resolved_context_window
        _require(
            self.max_tokens_per_response < window,
            f"max_tokens_per_response ({self.max_tokens_per_response}) must be "
            f"less than the context window ({window}); otherwise no tokens remain "
            "for the prompt. Lower max_tokens_per_response.",
        )
        usable = self.usable_context_tokens
        _require(
            self.max_context_tokens is None or self.max_context_tokens <= usable,
            f"max_context_tokens ({self.max_context_tokens}) must not exceed the "
            f"usable window ({usable} = context_window {window} - "
            f"max_tokens_per_response {self.max_tokens_per_response}); a prompt "
            "that size leaves no room for the response.",
        )
        context_limit = self.resolved_max_context_tokens
        _require(
            self.max_tool_result_tokens is None
            or self.max_tool_result_tokens <= context_limit,
            f"max_tool_result_tokens ({self.max_tool_result_tokens}) must not "
            f"exceed max_context_tokens ({context_limit}); one result that size "
            "would fill the whole prompt.",
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

    @property
    def reasoning_binds_to_prefix(self) -> bool:
        """Whether the model rejects reasoning once earlier messages change.

        Such a model is never trimmed client-side in the middle of a run.
        """
        return self.provider_spec.binds_reasoning_to_prefix(self.model)

    @property
    def usable_context_tokens(self) -> int:
        """The most the prompt may use: the window minus the response's share."""
        return self.resolved_context_window - self.max_tokens_per_response

    @property
    def resolved_max_context_tokens(self) -> int:
        """The prompt size that triggers a trim (`max_context_tokens` or auto)."""
        if self.max_context_tokens is not None:
            return self.max_context_tokens
        window = self.resolved_context_window
        return min(
            int(window * _AUTO_CONTEXT_SHARE),
            _AUTO_CONTEXT_CEILING,
            self.usable_context_tokens,
        )

    @property
    def resolved_max_tool_result_tokens(self) -> int:
        """The default cap on one tool result (`max_tool_result_tokens` or auto)."""
        if self.max_tool_result_tokens is not None:
            return self.max_tool_result_tokens
        return max(
            1,
            min(
                _AUTO_TOOL_RESULT_CEILING,
                self.resolved_context_window // 10,
                self.resolved_max_context_tokens // 5,
            ),
        )

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
        data["resolved_max_context_tokens"] = self.resolved_max_context_tokens
        data["resolved_max_tool_result_tokens"] = self.resolved_max_tool_result_tokens
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        """Rebuild a config from `to_dict()` output or a parsed config file.

        Unknown keys are rejected rather than ignored, so a typo in a YAML
        file fails at load time instead of silently doing nothing.
        """
        known = {f.name for f in fields(cls)}
        # Derived keys that to_dict() adds; they are outputs, not inputs.
        payload = {k: v for k, v in data.items() if k not in _DERIVED_KEYS}
        removed = sorted(set(payload) & set(_REMOVED_FIELDS))
        if removed:
            hints = "; ".join(f"{name}: {_REMOVED_FIELDS[name]}" for name in removed)
            raise ValueError(f"Removed AgentConfig field(s) — {hints}.")
        unknown = set(payload) - known
        if unknown:
            raise ValueError(
                f"Unknown AgentConfig field(s): {', '.join(sorted(unknown))}. "
                f"Valid fields: {', '.join(sorted(known))}."
            )
        nested = {"transport": TransportConfig, "memory": MemoryConfig}
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
        `transport` / `memory` records and the dict-valued fields are not
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
            "max_context_tokens": int,
            "max_tool_result_tokens": int,
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


_DERIVED_KEYS = frozenset(
    {
        "api_key_set",
        "resolved_context_window",
        "resolved_max_context_tokens",
        "resolved_max_tool_result_tokens",
    }
)

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
