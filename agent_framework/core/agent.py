"""Agent — the single user-facing entry point.

Construction is one call, and the model id is the only thing you must give:

    agent = Agent("claude-opus-5", system_prompt="...", tools=[add])

The first argument is a model id or a full `AgentConfig`; the handful of
keyword arguments that shadow config fields override that config, so a
variant is a one-liner and nothing has to be rebuilt by hand:

    summarizer = Agent(base_config, system_prompt="Summarize.", max_turns=2)

An agent is stateless by default: every run starts from an empty history.
`memory=MemoryConfig()` keeps the conversation across runs (in RAM), and
`MemoryConfig(store="app.db")` adds long-term memory in SQLite:

    chat = Agent("claude-opus-5", memory=MemoryConfig())

The execution surface is deliberately small — two mental models, one method
each:

  - `complete()`   → ONE model call. No agent loop; requested tool calls
                     are returned as data (`result.tool_calls`), never run.
  - `run()`        → the agent loop. Shape is controlled by parameters
                     (`max_turns`, `max_tool_rounds`, `tool_choice`), not
                     by extra methods.
  - `run_stream()` → `run()` with a streaming transport (separate method
                     because the return type differs).
  - `to_json()`    → schema-constrained single call (structured output).

`LoopPolicy` is an internal detail; the public API never requires it.
There is no `set_mode`. The framework has no modes.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import uuid
import warnings
from collections.abc import AsyncGenerator, Iterable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from agent_framework.core.config import AgentConfig, MemoryConfig
from agent_framework.core.context_manager import ContextManager
from agent_framework.core.loop import AgentLoop, AgentResult
from agent_framework.core.message import AnyMessage, message_from_dict
from agent_framework.core.policy import LoopPolicy
from agent_framework.interrupts.manager import InterruptManager
from agent_framework.interrupts.renderers.base import UIRenderer
from agent_framework.mcp.registry import MCPServerRegistry
from agent_framework.memory.backends.base import LongTermBackend
from agent_framework.memory.backends.sqlite import SQLiteBackend
from agent_framework.memory.long_term import LongTermMemory
from agent_framework.memory.manager import MemoryManager
from agent_framework.memory.short_term import ContextState
from agent_framework.permissions.config import PermissionConfig
from agent_framework.permissions.resolver import PermissionResolver
from agent_framework.providers.base import BaseProviderAdapter, StreamChunk, ToolChoice
from agent_framework.tools.builtin.ask_user import ask_user_tool
from agent_framework.tools.executor import ToolExecutor
from agent_framework.tools.models import Tool
from agent_framework.tools.registry import ToolRegistry, to_registry
from agent_framework.tracing.cost import CostTracker, PricingTable
from agent_framework.tracing.trace import EventType, ExecutionTrace

CheckpointId = str


class _Unset:
    """Sentinel for "argument not supplied"."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


# Annotated as Any so the keyword arguments below can keep their real types
# while still defaulting to the sentinel. `None` cannot serve as the sentinel:
# for `temperature` and `max_budget_usd` it is a meaningful value.
_UNSET: Any = _Unset()

# Config fields that `Agent(...)` accepts directly. Every name here must be an
# `AgentConfig` field and a keyword argument below —
# `test_every_override_is_a_real_config_field` and
# `test_every_override_is_an_actual_keyword_argument` enforce both, so the
# flat surface can never drift from the dataclass.
_CONFIG_OVERRIDES = (
    "system_prompt",
    "max_turns",
    "max_budget_usd",
    "temperature",
    "api_key",
    "base_url",
    # Pairs with base_url: a self-hosted endpoint's window is not in the
    # catalog, and getting it wrong means trimming never fires.
    "context_window",
    # None (the default) = stateless; a MemoryConfig keeps the conversation.
    "memory",
)


class Agent:
    """One agent: a config, a provider adapter, and the loop that drives it."""

    def __init__(
        self,
        config: str | AgentConfig,
        *,
        # --- config overrides (see _CONFIG_OVERRIDES) ---
        system_prompt: str = _UNSET,
        max_turns: int = _UNSET,
        max_budget_usd: float | None = _UNSET,
        temperature: float | None = _UNSET,
        api_key: str | None = _UNSET,
        base_url: str | None = _UNSET,
        context_window: int | None = _UNSET,
        memory: MemoryConfig | None = _UNSET,
        # --- wiring ---
        tools: ToolRegistry | Iterable[Tool] | None = None,
        permissions: PermissionConfig | None = None,
        mcp: MCPServerRegistry | None = None,
        memory_backend: LongTermBackend | None = None,
        session_id: str | None = None,
        trace: ExecutionTrace | None = None,
        pricing: PricingTable | None = None,
        agent_id: str | None = None,
        parent_permissions: PermissionConfig | None = None,
        provider: BaseProviderAdapter | None = None,
        # Internal: set by `connect()`, which has already merged the MCP tools
        # into `tools`, so the session span records the true discovery mode.
        _mcp_discovered: bool = False,
    ) -> None:
        """Build an agent.

        Args:
            config: a model id (``"claude-opus-5"``, ``"openai:llama-3.3-70b"``)
                or an `AgentConfig`.
            system_prompt, max_turns, max_budget_usd, temperature, api_key,
                base_url, context_window, memory: override the corresponding
                `AgentConfig` field. Anything not listed here is set on an
                `AgentConfig` instead. `memory=None` (the default) makes the
                agent stateless; `memory=MemoryConfig(...)` keeps the
                conversation.
            tools: `Tool` objects (from `@tool`) or a `ToolRegistry`. Copied,
                so a registry shared between agents is never mutated.
            permissions: rules and the renderer that answers `ASK` prompts.
            mcp: MCP servers whose tools this agent may call. Discovery is
                deferred to the first run, and a failing server is recorded in
                the trace and skipped. Use `await Agent.connect(...)` to
                discover eagerly and raise on any failure instead.
            memory_backend: a custom long-term backend instead of SQLite
                (advanced). Requires `memory=MemoryConfig(...)` without a
                `store`.
            session_id: which conversation this agent saves and resumes.
                Defaults to a fresh id per agent; it is recorded in the trace.
            trace, pricing: injected subsystems; sensible defaults are
                built when omitted.
            agent_id: stable id for traces; generated when omitted.
            parent_permissions: when building a sub-agent, the parent's
                permissions. Construction fails if this agent's permissions
                are not a subset of them.
            provider: a pre-built adapter, bypassing the catalog. Mainly for
                tests and for adapters constructed by hand.
        """
        self.config = _resolve_config(
            config,
            system_prompt=system_prompt,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            context_window=context_window,
            memory=memory,
        )
        _validate_memory(self.config.memory, memory_backend)
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise ValueError(f"session_id must be a non-empty string, got {session_id!r}")

        # Copy the tools: registering `ask_user` (and later the MCP tools)
        # must not reach into a registry the caller may share between agents.
        registry = ToolRegistry(to_registry(tools))
        if ask_user_tool.metadata.name not in registry:
            registry.add(ask_user_tool)

        perms = permissions or PermissionConfig()
        if parent_permissions is not None and not perms.is_stricter_than(parent_permissions):
            raise PermissionError(
                "Sub-agent permissions must be a subset of parent permissions; "
                "widening is rejected at construction."
            )
        renderer = perms.on_ask
        if renderer is not None and not isinstance(renderer, UIRenderer):
            raise TypeError(
                f"PermissionConfig.on_ask must be a UIRenderer instance, "
                f"got {type(renderer).__name__}"
            )

        # Generate the agent id BEFORE the trace so spans always carry it.
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:12]}"
        self._trace = trace or ExecutionTrace(agent_id=self.agent_id)
        self._cost = CostTracker(pricing=pricing or PricingTable.default())
        self._tools = registry
        # Never the trace's id: two agents may share one trace, and must not
        # share — and interleave — one conversation.
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:16]}"
        self._memory_backend = memory_backend
        self._memory = _build_memory(self.config.memory, memory_backend)
        self._permissions_cfg = perms
        self._renderer = renderer
        self._provider = provider if provider is not None else _build_provider(self.config)

        self._mcp = mcp
        self._mcp_discovered = mcp is None or _mcp_discovered
        self._mcp_lock = asyncio.Lock()
        if not self._mcp_discovered and self.config.include_capabilities_block:
            warnings.warn(
                "include_capabilities_block=True with lazily discovered MCP "
                "tools: the <capabilities> block is built once at construction "
                "(the system prompt must stay byte-stable for prompt caching), "
                "so MCP tools will be missing from it. Use "
                "`await Agent.connect(...)` to discover before the prompt is built.",
                stacklevel=2,
            )

        # Wire the subsystems together.
        self._interrupts = (
            InterruptManager(self._renderer, trace=self._trace) if self._renderer else None
        )
        self._resolver = PermissionResolver(
            perms,
            interrupt_manager=self._interrupts,
            trace=self._trace,
            session_id=self.session_id,
        )
        self._executor = ToolExecutor(
            trace=self._trace,
            interrupt_manager=self._interrupts,
            session_id=self.session_id,
            max_parallel=self.config.max_parallel_tools,
            default_max_result_tokens=self.config.resolved_max_tool_result_tokens,
        )
        # Built once: the prompt must stay byte-stable for prompt caching, and
        # the session span fingerprints exactly this text.
        self._system_prompt = self._build_system_prompt()
        self._context_manager = ContextManager(
            system_prompt=self._system_prompt,
            max_context_tokens=self.config.resolved_max_context_tokens,
            max_tool_result_tokens=self.config.resolved_max_tool_result_tokens,
            tools=registry,
            memory=self._memory,
            trace=self._trace,
        )
        self._checkpoints: dict[CheckpointId, dict[str, Any]] = {}

        self._record_session_span()

        # Loop engine — one per agent.
        self._loop = AgentLoop(
            config=self.config,
            provider=self._provider,
            tools=registry,
            short_term=self._memory.short_term,
            context_manager=self._context_manager,
            permission_resolver=self._resolver,
            tool_executor=self._executor,
            cost_tracker=self._cost,
            trace=self._trace,
            agent_id=self.agent_id,
            retain_history=self.config.memory is not None,
        )

    # ===================== async construction =====================
    @classmethod
    async def connect(
        cls,
        config: str | AgentConfig,
        *,
        mcp: MCPServerRegistry,
        **kwargs: Any,
    ) -> Agent:
        """Build an agent and discover its MCP tools now, not on first run.

        Prefer this at service start-up. Discovery is strict here: if any
        server cannot be reached or listed, `MCPDiscoveryError` names every
        one that failed, so a misconfigured deployment crashes at boot rather
        than serving a silently reduced toolset. (The plain constructor defers
        discovery to the first run, where a failing server is recorded in the
        trace and skipped instead — errors inside a run are never raised.)
        Takes the same arguments as `Agent(...)`.
        """
        tools = ToolRegistry(to_registry(kwargs.pop("tools", None)))
        for discovered in await mcp.discover_all(strict=True):
            tools.add(discovered)
        return cls(config, tools=tools, mcp=mcp, _mcp_discovered=True, **kwargs)

    async def _ensure_mcp(self) -> None:
        """Discover MCP tools once, before the first provider call."""
        if self._mcp_discovered:
            return
        async with self._mcp_lock:
            if self._mcp_discovered or self._mcp is None:
                return
            # add(), not replace(): an MCP tool shadowing a native one of the
            # same name deserves the registry's overwrite warning.
            for discovered in await self._mcp.discover_all():
                self._tools.add(discovered)
            self._mcp_discovered = True

    # ===================== execution =====================
    def _run_policy(
        self,
        *,
        max_turns: int | None,
        max_tool_rounds: int | None,
        tool_choice: ToolChoice | None,
    ) -> LoopPolicy:
        return LoopPolicy(
            max_turns=max_turns if max_turns is not None else self.config.max_turns,
            execute_tools=True,
            max_tool_rounds=max_tool_rounds,
            tool_choice=tool_choice,
        )

    async def run(
        self,
        user_input: str,
        *,
        max_turns: int | None = None,
        max_tool_rounds: int | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> AgentResult:
        """Run the agent loop.

        Parameters shape the run — there are no separate methods per shape:
          - max_turns: hard ceiling on loop iterations (default: config).
          - max_tool_rounds: cap on tool-execution rounds; `1` gives the
            classic "one tool round + synthesis" pattern. None = unlimited
            (bounded by max_turns).
          - tool_choice: constrain the FIRST model call ("any"/"tool" force
            tool use once; later turns fall back to auto).
        """
        await self._ensure_mcp()
        return await self._loop.run(
            user_input,
            policy=self._run_policy(
                max_turns=max_turns,
                max_tool_rounds=max_tool_rounds,
                tool_choice=tool_choice,
            ),
        )

    async def run_stream(
        self,
        user_input: str,
        *,
        max_turns: int | None = None,
        max_tool_rounds: int | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """`run()` with a streaming transport (yields StreamChunk events).

        Closing this generator early (``await stream.aclose()``, or leaving an
        ``async with contextlib.aclosing(...)`` block) stops the run cleanly:
        its spans end as cancelled and the agent is immediately reusable.
        """
        await self._ensure_mcp()
        # aclosing: without it, closing this generator leaves the loop's own
        # generator suspended, so its "run in progress" flag is never cleared
        # and every later run() on this agent raises.
        async with contextlib.aclosing(
            self._loop.stream(
                user_input,
                policy=self._run_policy(
                    max_turns=max_turns,
                    max_tool_rounds=max_tool_rounds,
                    tool_choice=tool_choice,
                ),
            )
        ) as chunks:
            async for c in chunks:
                yield c

    async def complete(
        self,
        user_input: str,
        *,
        tool_choice: ToolChoice | None = None,
    ) -> AgentResult:
        """ONE model call — no agent loop.

        Tools are advertised to the model but never executed; any requested
        calls are returned as data on `result.tool_calls`. Use `run()` when
        tools should actually execute.
        """
        await self._ensure_mcp()
        return await self._loop.run(
            user_input,
            policy=LoopPolicy(
                max_turns=1,
                execute_tools=False,
                max_tool_rounds=0,
                tool_choice=tool_choice,
            ),
        )

    # ===================== structured output (optional) =====================
    async def to_json(self, prompt: str, *, schema: Any) -> Any:
        """Generate JSON validated against a Pydantic model (or jsonschema dict)."""
        from agent_framework.output.structured import generate_structured

        return await generate_structured(self, prompt, schema)

    # ===================== inspection =====================
    def get_history(self) -> list[AnyMessage]:
        """The stored conversation. Without memory: the last run's messages."""
        return self._memory.short_term.snapshot()

    def clear_history(self) -> None:
        """Forget the conversation (messages and context state)."""
        self._memory.short_term.clear()

    def get_trace(self) -> ExecutionTrace:
        return self._trace

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def permissions(self) -> PermissionConfig:
        return self._permissions_cfg

    @property
    def cost(self) -> CostTracker:
        return self._cost

    # ===================== checkpointing =====================
    def checkpoint(self) -> CheckpointId:
        cid = f"ckpt_{uuid.uuid4().hex[:12]}"
        short_term = self._memory.short_term
        self._checkpoints[cid] = {
            "messages": [m.to_dict() for m in short_term.snapshot()],
            # Without it a restored history would be rendered as if never
            # trimmed: cleared results back in full, cut reasoning back in.
            "context_state": short_term.state.to_dict(),
            "cost": self._cost.total_cost_usd,
        }
        return cid

    def restore(self, checkpoint_id: CheckpointId) -> None:
        """Full-fidelity restore: tool_calls, tool results, reasoning,
        metadata and the context state all survive the round-trip."""
        ckpt = self._checkpoints.get(checkpoint_id)
        if ckpt is None:
            raise KeyError(f"Unknown checkpoint: {checkpoint_id}")
        self._memory.short_term.replace_all(
            (message_from_dict(raw) for raw in ckpt["messages"]),
            state=ContextState.from_dict(ckpt.get("context_state")),
        )
        self._cost.total_cost_usd = ckpt.get("cost", self._cost.total_cost_usd)

    # ===================== internals =====================
    def _record_session_span(self) -> None:
        """Record how this agent was built, once, into the trace.

        Without it a trace cannot explain its own behaviour: nothing else
        records the resolved context window (so a compaction span cannot say
        whether it fired at the configured ratio or at a silently defaulted
        window), whether the model was priced at all (so a $0.00 run cannot be
        told apart from an inactive budget guard), or the sampling parameters.

        It records settings, never content — the convention the LLM spans
        already follow. Traces are exported (JSONL, SQLite, OTel), so:

          - the prompt is fingerprinted, not copied: a hash of the prompt
            actually sent answers "did these two runs use the same prompt?"
            without shipping its text to a trace store;
          - `base_url` loses userinfo, query and fragment, which can carry
            credentials (``user:pass@host``, ``?api-key=...``);
          - the API key is not part of `AgentConfig.to_dict()` at all, and the
            trace redacts sensitive keys on top of that.

        Emitted at construction, so a trace passed in with subscribers
        attached sees it live; one subscribed afterwards finds it in the
        exported trace.
        """
        payload = self.config.to_dict()
        payload.pop("system_prompt", None)
        payload.update(
            agent_id=self.agent_id,
            system_prompt_sha256=hashlib.sha256(
                self._system_prompt.encode("utf-8")
            ).hexdigest()[:16],
            system_prompt_chars=len(self._system_prompt),
            base_url=_endpoint_for_trace(self.config.base_url),
            tools=sorted(self._tools.names),
            mcp_servers=self._mcp.names() if self._mcp is not None else [],
            mcp_discovery=(
                "none"
                if self._mcp is None
                else ("eager" if self._mcp_discovered else "lazy")
            ),
            # A model missing from the PricingTable records 0.0 cost and
            # silently disables max_budget_usd — worth knowing per run.
            pricing_resolved=self._cost.pricing.get(self.config.model) is not None,
            framework_version=_framework_version(),
            # The conversation this agent saves and resumes. Not the span's
            # own `session_id`, which is the trace's.
            session_id=self.session_id,
            retain_history=self.config.memory is not None,
            memory_backend=(
                type(self._memory_backend).__name__
                if self._memory_backend is not None
                else None
            ),
        )
        self._trace.record_event(EventType.SESSION, name="session", payload=payload)

    def _build_system_prompt(self) -> str:
        """Compose the byte-stable system prompt at session start.

        Structure (stable order, designed for prefix caching):
          <identity> <capabilities> <rules> <environment>
        """
        sp = (self.config.system_prompt or "").rstrip()
        parts: list[str] = [sp] if sp else []

        # Capabilities block — opt-in: the tool definitions already reach the
        # model via the tools parameter; duplicating them doubles token cost.
        if self.config.include_capabilities_block and len(self._tools):
            tool_lines = [
                f"  - {t.metadata.name}: {t.metadata.description}" for t in self._tools
            ]
            parts.append("<capabilities>\n" + "\n".join(tool_lines) + "\n</capabilities>")

        # Environment block — config-provided.
        if self.config.environment:
            env_lines = [f"  {k}: {v}" for k, v in self.config.environment.items()]
            parts.append("<environment>\n" + "\n".join(env_lines) + "\n</environment>")

        return "\n\n".join(parts)


def _endpoint_for_trace(url: str | None) -> str | None:
    """*url* reduced to what identifies the endpoint: scheme, host, port, path.

    Userinfo, query and fragment are dropped because they are where
    credentials hide. Anything that is not a parseable http(s) URL — the only
    kind the SDKs accept — is replaced rather than echoed, since without a
    scheme and host a credential can hide in what parses as the path.
    """
    if not url:
        return url
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        return "[unparseable url]"
    if parts.scheme not in ("http", "https") or not host:
        return "[unparseable url]"
    if ":" in host:  # IPv6 literal: urlsplit strips the brackets
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _framework_version() -> str:
    from agent_framework import __version__

    return __version__


def _resolve_config(config: str | AgentConfig, **overrides: Any) -> AgentConfig:
    """Turn the first positional argument plus overrides into an AgentConfig.

    A supplied keyword always beats the config's own field; anything left as
    the sentinel is not touched.
    """
    supplied = {k: v for k, v in overrides.items() if v is not _UNSET}
    if isinstance(config, str):
        return AgentConfig(model=config, **supplied)
    if not isinstance(config, AgentConfig):
        raise TypeError(
            "Agent(config=...) takes a model id string or an AgentConfig, got "
            f"{type(config).__name__}."
        )
    return config.replace(**supplied) if supplied else config


def _validate_memory(memory: MemoryConfig | None, backend: LongTermBackend | None) -> None:
    """Reject memory settings that would silently do nothing.

    Lives here, not in `MemoryConfig`: whether long-term memory exists depends
    on `memory_backend`, which is wiring and never part of a config.
    """
    if backend is not None:
        if not isinstance(backend, LongTermBackend):
            raise TypeError(
                "memory_backend must be a LongTermBackend, got "
                f"{type(backend).__name__}."
            )
        if memory is None:
            raise ValueError(
                "memory_backend= needs memory=MemoryConfig(...): an agent is "
                "stateless unless memory is turned on."
            )
        if memory.store is not None:
            raise ValueError(
                "Pass either MemoryConfig(store=...) or memory_backend=, not "
                "both: each one is the long-term store."
            )
        return
    if memory is not None and memory.store is None:
        unused = memory.long_term_settings()
        if unused:
            raise ValueError(
                f"MemoryConfig.{', MemoryConfig.'.join(unused)} only apply to "
                "long-term memory: also set MemoryConfig(store='<file>.db') or "
                "pass memory_backend=."
            )


def _build_memory(
    memory: MemoryConfig | None, backend: LongTermBackend | None
) -> MemoryManager:
    """The agent's memory: short-term always, long-term when configured."""
    if memory is None:
        return MemoryManager()
    if backend is None and memory.store is not None:
        backend = SQLiteBackend(memory.store)
    long_term = (
        LongTermMemory(backend=backend, namespace=memory.namespace)
        if backend is not None
        else None
    )
    return MemoryManager(long_term=long_term)


def _build_provider(config: AgentConfig) -> BaseProviderAdapter:
    """Construct the adapter the config's provider spec asks for."""
    spec = config.provider_spec
    return spec.build(
        api_key=config.resolved_api_key(),
        base_url=config.base_url,
        timeout=config.transport.timeout_seconds,
        max_retries=config.transport.max_retries,
        client_kwargs=config.transport.client_kwargs or None,
    )
