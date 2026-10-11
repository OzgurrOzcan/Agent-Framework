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
from agent_framework.core.context_filter import ContextFilter
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
from agent_framework.memory.sessions import (
    ConversationSession,
    SessionStore,
    tool_set_fingerprint,
)
from agent_framework.memory.short_term import ContextState
from agent_framework.memory.summaries import Summarizer
from agent_framework.permissions.config import PermissionConfig
from agent_framework.permissions.resolver import PermissionResolver
from agent_framework.providers.base import BaseProviderAdapter, StreamChunk, ToolChoice
from agent_framework.providers.catalog import (
    ProviderResolutionError,
    resolve_model,
    split_model_id,
)
from agent_framework.tools.builtin.ask_user import ask_user_tool
from agent_framework.tools.executor import ToolExecutor
from agent_framework.tools.models import Tool
from agent_framework.tools.registry import ToolRegistry, to_registry
from agent_framework.tracing.cost import CostTracker, PricingTable
from agent_framework.tracing.trace import (
    ErrorInfo,
    EventType,
    ExecutionTrace,
    SpanStatus,
)

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
        context_filter: ContextFilter | None = None,
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
            context_filter: ``fn(messages, info) -> messages`` (sync or
                async), called before every LLM call to change what is sent —
                never what is stored (advanced; see
                `agent_framework.core.context_filter`).
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
        self._summary_model = _summary_model(self.config)
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise ValueError(f"session_id must be a non-empty string, got {session_id!r}")
        if context_filter is not None and not callable(context_filter):
            raise TypeError(
                "context_filter must be a function (messages, info) -> messages, got "
                f"{type(context_filter).__name__}"
            )
        self._context_filter = context_filter

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
        self._memory = _build_memory(self.config.memory, memory_backend, self._trace)
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
        # Saved conversation, when the long-term store keeps sessions. Loaded
        # once before the first run, saved after every run.
        self._session = _build_session(self.config.memory, self._memory, self.session_id)
        self._session_ready = self._session is None
        self._session_lock = asyncio.Lock()
        # With a log to write, messages leaving the history unlogged are held.
        self._memory.short_term.track_unlogged = self._session is not None
        # Long-term summaries live next to the sessions they come from.
        self._summarizer = (
            Summarizer(
                provider=self._provider,
                model=self._summary_model,
                session=self._session,
                cost=self._cost,
                trace=self._trace,
                auto=self.config.memory.summarize if self.config.memory else False,
                # The summary heads every new conversation; no trim removes it.
                budget_tokens=self.config.resolved_max_context_tokens // 5,
            )
            if self._session is not None
            else None
        )
        self._memory.set_summarize(self._summarize_now)
        self._context_manager = ContextManager(
            system_prompt=self._system_prompt,
            max_context_tokens=self.config.resolved_max_context_tokens,
            max_tool_result_tokens=self.config.resolved_max_tool_result_tokens,
            tools=registry,
            memory=self._memory,
            trace=self._trace,
            context_filter=context_filter,
            binds_reasoning=self.config.reasoning_binds_to_prefix,
            keeps_reasoning=self.config.reasoning_required_in_history,
            model=self.config.model,
            summarizer=self._summarizer,
            usable_tokens=self.config.usable_context_tokens,
        )
        self._checkpoints: dict[CheckpointId, dict[str, Any]] = {}
        self._native_clearing = _native_clearing_available(self.config, self._provider)

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
            native_clearing=self._native_clearing,
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
        Takes the same arguments as `Agent(...)`. With `MemoryConfig(store=...)`
        the saved conversation of `session_id` is loaded here as well.
        """
        tools = ToolRegistry(to_registry(kwargs.pop("tools", None)))
        for discovered in await mcp.discover_all(strict=True):
            tools.add(discovered)
        agent = cls(config, tools=tools, mcp=mcp, _mcp_discovered=True, **kwargs)
        # A saved conversation is loaded now too, so get_history() shows it
        # before the first run.
        await agent._ensure_session()
        return agent

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
        dry_run: bool = False,
    ) -> AgentResult:
        """Run the agent loop.

        Parameters shape the run — there are no separate methods per shape:
          - max_turns: hard ceiling on loop iterations (default: config).
          - max_tool_rounds: cap on tool-execution rounds; `1` gives the
            classic "one tool round + synthesis" pattern. None = unlimited
            (bounded by max_turns).
          - tool_choice: constrain the FIRST model call ("any"/"tool" force
            tool use once; later turns fall back to auto).
          - dry_run: build what the first call would send — trimmed if it
            would be — and return it as `result.messages` with
            `result.context`, without calling the model, summarizing or
            saving. The stored history is untouched.

        `result.context` reports what the last call was sent: message counts,
        the token estimate, cleared results and the provider's cache hits.
        """
        await self._ensure_mcp()
        policy = self._run_policy(
            max_turns=max_turns,
            max_tool_rounds=max_tool_rounds,
            tool_choice=tool_choice,
        )
        await self._ensure_session()
        if dry_run:
            return await self._loop.dry_run(user_input, policy=policy)
        try:
            return await self._loop.run(user_input, policy=policy)
        finally:
            await self._flush_session()

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
        await self._ensure_session()
        try:
            # aclosing: without it, closing this generator leaves the loop's
            # own generator suspended, so its "run in progress" flag is never
            # cleared and every later run() on this agent raises.
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
        finally:
            # Also when the consumer closes the stream early.
            await self._flush_session()

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
        await self._ensure_session()
        try:
            return await self._loop.run(
                user_input,
                policy=LoopPolicy(
                    max_turns=1,
                    execute_tools=False,
                    max_tool_rounds=0,
                    tool_choice=tool_choice,
                ),
            )
        finally:
            await self._flush_session()

    # ===================== structured output (optional) =====================
    async def to_json(self, prompt: str, *, schema: Any) -> Any:
        """Generate JSON validated against a Pydantic model (or jsonschema dict)."""
        from agent_framework.output.structured import generate_structured

        return await generate_structured(self, prompt, schema)

    # ===================== inspection =====================
    def get_history(self) -> list[AnyMessage]:
        """The stored conversation. Without memory: the last run's messages."""
        return self._memory.short_term.snapshot()

    async def clear_history(self) -> None:
        """Forget the conversation (messages and context state).

        With a store, the emptied conversation is saved too, so resuming the
        same session_id later starts from nothing rather than the old history.
        """
        await self._ensure_mcp()  # the saved fingerprint covers the MCP tools
        async with self._session_lock:
            if not self._session_ready and self._session is not None:
                # Never loaded: continue its log numbering, but not its history.
                try:
                    await self._session.load_counters()
                except Exception as exc:
                    self._session_unreadable(exc)
            # Never load the old conversation after this.
            self._session_ready = True
        self._memory.short_term.clear()
        await self._flush_session()

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

    # ===================== sessions =====================
    async def _summarize_now(self) -> bool:
        """`await agent.memory.summarize()`: save, then summarize what this
        conversation added since its last summary (one LLM call, or none
        when there is nothing new). False without long-term summaries."""
        if self._summarizer is None:
            return False
        await self._ensure_mcp()  # the session fingerprint covers the MCP tools
        await self._ensure_session()
        if self._session is None:  # it could not be read: nothing to summarize safely
            return False
        await self._flush_session()
        return await self._summarizer.summarize_session()

    def _fingerprint(self) -> str:
        return tool_set_fingerprint(self._system_prompt, self._tools)

    async def _ensure_session(self) -> None:
        """Load the saved conversation of `session_id` once, before the first run.

        A store that cannot be read is recorded and the run continues from an
        empty history — but nothing is saved for this agent afterwards, so
        the conversation in the store is never overwritten by a fresh one.
        """
        if self._session_ready:
            return
        async with self._session_lock:
            if self._session_ready or self._session is None:
                return
            session = self._session
            try:
                resumed, reasoning_dropped = await session.load(
                    self._memory.short_term, self._fingerprint()
                )
            except Exception as exc:
                self._session_unreadable(exc)
            else:
                self._trace.record_event(
                    EventType.MEMORY_READ,
                    name="session",
                    payload={
                        "session_id": self.session_id,
                        "resumed": resumed,
                        "messages": len(self._memory.short_term),
                        # The prompt or tool set changed since it was saved:
                        # reasoning bound to the old ones is left out.
                        "reasoning_dropped": reasoning_dropped,
                    },
                )
            self._session_ready = True

    def _session_unreadable(self, exc: BaseException) -> None:
        """The stored session could not be read: write nothing for it again.

        Its numbering and summary marker are unknown, so saving or summarizing
        would overwrite or double-count what the store holds.
        """
        if self._session is not None:
            self._session.broken = True  # the summarizer holds it too
        self._session = None
        self._trace.record_event(
            EventType.ERROR,
            name="session_load_failed",
            status=SpanStatus.ERROR,
            payload={"session_id": self.session_id},
            error=ErrorInfo.from_exception(exc),
        )

    async def _flush_session(self) -> None:
        """Save the conversation after a run. Failures are traced, not raised."""
        if self._session is None or not self._session_ready:
            return
        try:
            logged = await self._session.flush(self._memory.short_term, self._fingerprint())
        except Exception as exc:
            self._trace.record_event(
                EventType.ERROR,
                name="session_save_failed",
                status=SpanStatus.ERROR,
                payload={"session_id": self.session_id},
                error=ErrorInfo.from_exception(exc),
            )
            return
        self._trace.record_event(
            EventType.MEMORY_WRITE,
            name="session",
            payload={
                "session_id": self.session_id,
                "logged": logged,
                "messages": len(self._memory.short_term),
            },
        )

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
            native_tool_result_clearing=self._native_clearing,
            reasoning_binds_to_prefix=self.config.reasoning_binds_to_prefix,
            reasoning_required_in_history=self.config.reasoning_required_in_history,
            summary_model=self._summary_model if self._summarizer is not None else None,
            context_filter=(
                getattr(self._context_filter, "__qualname__", type(self._context_filter).__name__)
                if self._context_filter is not None
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


def _native_clearing_available(config: AgentConfig, provider: BaseProviderAdapter) -> bool:
    """Whether to ask the provider to clear old tool results server-side.

    On for adapters that support it, talking to the vendor's own API. Off
    for a custom `base_url` — a gateway or proxy may not forward the beta —
    and when `request_extra` already carries the caller's own
    `context_management`, which is theirs to control.
    """
    return (
        bool(getattr(provider, "supports_tool_result_clearing", False))
        and config.base_url is None
        and "context_management" not in (config.request_extra or {})
    )


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
    memory: MemoryConfig | None,
    backend: LongTermBackend | None,
    trace: ExecutionTrace,
) -> MemoryManager:
    """The agent's memory: short-term always, long-term when configured.

    A stateless agent (no MemoryConfig) never opens a store. A store path
    builds a SQLiteBackend that the manager owns — `await agent.memory.close()`
    closes it; a backend passed in stays the caller's to close.
    """
    if memory is None:
        return MemoryManager(trace=trace)
    owns = False
    if backend is None and memory.store is not None:
        backend = SQLiteBackend(memory.store)
        owns = True
    long_term = (
        LongTermMemory(backend=backend, namespace=memory.namespace)
        if backend is not None
        else None
    )
    return MemoryManager(long_term=long_term, owns_backend=owns, trace=trace)


def _summary_model(config: AgentConfig) -> str:
    """The model long-term summaries are written with (its wire name).

    It is called through the agent's own adapter, so it must belong to the
    agent's provider; a model of another provider is rejected here rather
    than failing on the first summary.
    """
    requested = config.memory.summary_model if config.memory is not None else None
    if requested is None:
        return config.model
    prefix, bare = split_model_id(requested)
    if prefix is None:
        try:
            spec, bare = resolve_model(requested)
        except ProviderResolutionError:
            if config.provider == "openai" and config.base_url:
                return bare  # an endpoint's own model name, as for the agent
            raise
        prefix = spec.name
    if prefix != config.provider:
        raise ValueError(
            f"MemoryConfig.summary_model={requested!r} belongs to provider "
            f"{prefix!r}, but this agent uses {config.provider!r}; summaries are "
            "written through the agent's own connection. Pick a model of the "
            "same provider (or leave summary_model unset)."
        )
    return bare


def _build_session(
    memory: MemoryConfig | None, manager: MemoryManager, session_id: str
) -> ConversationSession | None:
    """The saved conversation, if the long-term store keeps sessions."""
    if memory is None or manager.long_term is None:
        return None
    backend = manager.long_term.backend
    if not isinstance(backend, SessionStore):
        warnings.warn(
            f"{type(backend).__name__} does not implement SessionStore, so "
            "conversations are not saved or resumed — only its facts are used. "
            "Implement SessionStore to keep sessions.",
            stacklevel=3,
        )
        return None
    return ConversationSession(backend, namespace=manager.long_term.namespace, session_id=session_id)


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
