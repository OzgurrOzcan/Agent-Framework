"""BaseAgent — the single user-facing entry point.

User code only ever references `BaseAgent.create(...)`. The execution
surface is deliberately small — two mental models, one method each:

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

import abc
import uuid
from typing import Any, AsyncIterator, Iterable

from agent_framework.core.config import AgentConfig, ProviderKind
from agent_framework.core.context_manager import ContextManager
from agent_framework.core.loop import AgentLoop, AgentResult
from agent_framework.core.message import AnyMessage, message_from_dict
from agent_framework.core.policy import LoopPolicy
from agent_framework.interrupts.manager import InterruptManager
from agent_framework.interrupts.renderers.base import UIRenderer
from agent_framework.memory.manager import MemoryManager
from agent_framework.mcp.registry import MCPServerRegistry
from agent_framework.permissions.config import PermissionConfig
from agent_framework.permissions.resolver import PermissionResolver
from agent_framework.providers.base import BaseProviderAdapter, StreamChunk, ToolChoice
from agent_framework.tools.builtin.ask_user import ask_user_tool
from agent_framework.tools.executor import ToolExecutor
from agent_framework.tools.models import Tool
from agent_framework.tools.registry import ToolRegistry, to_registry
from agent_framework.tracing.cost import CostTracker, PricingTable
from agent_framework.tracing.trace import ExecutionTrace


CheckpointId = str


class BaseAgent(abc.ABC):
    """Abstract base. Use `BaseAgent.create(...)`."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        provider: BaseProviderAdapter,
        tools: ToolRegistry,
        memory: MemoryManager,
        permissions: PermissionConfig,
        renderer: UIRenderer | None,
        trace: ExecutionTrace,
        cost_tracker: CostTracker,
        mcp: MCPServerRegistry | None = None,
        agent_id: str | None = None,
    ) -> None:
        self.config = config
        self._provider = provider
        self._tools = tools
        self._memory = memory
        self._permissions_cfg = permissions
        self._renderer = renderer
        self._trace = trace
        self._cost = cost_tracker
        self._mcp = mcp
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:12]}"

        # Wire the subsystems together.
        self._interrupts = InterruptManager(renderer, trace=trace) if renderer else None
        self._resolver = PermissionResolver(
            permissions,
            interrupt_manager=self._interrupts,
            trace=trace,
            session_id=trace.session_id,
        )
        self._executor = ToolExecutor(
            trace=trace,
            interrupt_manager=self._interrupts,
            session_id=trace.session_id,
            max_parallel=config.max_parallel_tools,
        )
        self._context_manager = ContextManager(
            system_prompt=self._build_system_prompt(),
            context_window=config.resolved_context_window,
            response_buffer_tokens=config.response_buffer_tokens,
            compaction_buffer_tokens=config.compaction_buffer_tokens,
            compaction=config.compaction,
            memory=memory,
            trace=trace,
            summarizer=provider,
            summarizer_model=config.model,
        )
        self._checkpoints: dict[CheckpointId, dict[str, Any]] = {}

        # Loop engine — one per agent.
        self._loop = AgentLoop(
            config=config,
            provider=provider,
            tools=tools,
            short_term=memory.short_term,
            context_manager=self._context_manager,
            permission_resolver=self._resolver,
            tool_executor=self._executor,
            cost_tracker=cost_tracker,
            trace=trace,
            agent_id=self.agent_id,
        )

    # ===================== factory =====================
    @classmethod
    def create(
        cls,
        config: AgentConfig,
        *,
        tools: ToolRegistry | Iterable[Tool] | None = None,
        mcp: MCPServerRegistry | None = None,
        memory: MemoryManager | None = None,
        permissions: PermissionConfig | None = None,
        trace: ExecutionTrace | None = None,
        pricing: PricingTable | None = None,
        agent_id: str | None = None,
        parent_permissions: PermissionConfig | None = None,
    ) -> "BaseAgent":
        registry = to_registry(tools)
        # The built-in `ask_user` is always registered on every agent.
        if ask_user_tool.metadata.name not in registry:
            registry.add(ask_user_tool)

        # Auto-discover MCP tools and merge in.
        if mcp is not None:
            for t in mcp.discover_all_sync():
                registry.add(t)

        memory_mgr = memory or MemoryManager()

        perms = permissions or PermissionConfig()
        if parent_permissions is not None and not perms.is_stricter_than(parent_permissions):
            raise PermissionError(
                "Sub-agent permissions must be a subset of parent permissions; "
                "widening is rejected at construction."
            )
        # Validate renderer — must be a UIRenderer instance or None.
        renderer = perms.on_ask
        if renderer is not None and not isinstance(renderer, UIRenderer):
            raise TypeError(
                f"PermissionConfig.on_ask must be a UIRenderer instance, "
                f"got {type(renderer).__name__}"
            )

        # Generate the agent id BEFORE the trace so spans always carry it.
        agent_id = agent_id or f"agent_{uuid.uuid4().hex[:12]}"
        trace = trace or ExecutionTrace(agent_id=agent_id)
        cost = CostTracker(pricing=pricing or PricingTable.default())

        # Pick provider adapter.
        api_key = config.provider.resolved_api_key()
        if config.provider_kind == ProviderKind.ANTHROPIC:
            from agent_framework.providers.anthropic_agent import AnthropicAdapter
            provider = AnthropicAdapter(
                api_key=api_key,
                base_url=config.provider.base_url,
                timeout=config.provider.timeout_seconds,
                max_retries=config.provider.max_retries,
                client_kwargs=config.provider.extra or None,
            )
            klass = AnthropicAgent
        elif config.provider_kind == ProviderKind.OPENAI:
            from agent_framework.providers.openai_agent import OpenAIAdapter
            provider = OpenAIAdapter(
                api_key=api_key,
                base_url=config.provider.base_url,
                timeout=config.provider.timeout_seconds,
                max_retries=config.provider.max_retries,
                client_kwargs=config.provider.extra or None,
            )
            klass = OpenAIAgent
        else:
            raise ValueError(f"Unsupported provider: {config.provider_kind}")

        return klass(
            config=config,
            provider=provider,
            tools=registry,
            memory=memory_mgr,
            permissions=perms,
            renderer=renderer,
            trace=trace,
            cost_tracker=cost,
            mcp=mcp,
            agent_id=agent_id,
        )

    @classmethod
    async def acreate(
        cls,
        config: AgentConfig,
        *,
        mcp: MCPServerRegistry | None = None,
        **kwargs: Any,
    ) -> "BaseAgent":
        """Async factory — use this from async code when passing `mcp`.

        `create()` cannot run MCP discovery while an event loop is running;
        this variant awaits discovery and registers the discovered tools.
        """
        agent = cls.create(config, mcp=None, **kwargs)
        if mcp is not None:
            for t in await mcp.discover_all():
                agent._tools.add(t)
            agent._mcp = mcp
        return agent

    # ===================== execution =====================
    def _run_policy(
        self,
        *,
        max_turns: int | None,
        max_tool_rounds: int | None,
        tool_choice: "ToolChoice | None",
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
        tool_choice: "ToolChoice | None" = None,
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
        return await self._loop.run(
            user_input,
            policy=self._run_policy(
                max_turns=max_turns, max_tool_rounds=max_tool_rounds,
                tool_choice=tool_choice,
            ),
        )

    async def run_stream(
        self,
        user_input: str,
        *,
        max_turns: int | None = None,
        max_tool_rounds: int | None = None,
        tool_choice: "ToolChoice | None" = None,
    ) -> AsyncIterator[StreamChunk]:
        """`run()` with a streaming transport (yields StreamChunk events)."""
        async for c in self._loop.stream(
            user_input,
            policy=self._run_policy(
                max_turns=max_turns, max_tool_rounds=max_tool_rounds,
                tool_choice=tool_choice,
            ),
        ):
            yield c

    async def complete(
        self,
        user_input: str,
        *,
        tool_choice: "ToolChoice | None" = None,
    ) -> AgentResult:
        """ONE model call — no agent loop.

        Tools are advertised to the model but never executed; any requested
        calls are returned as data on `result.tool_calls`. Use `run()` when
        tools should actually execute.
        """
        return await self._loop.run(
            user_input,
            policy=LoopPolicy(
                max_turns=1, execute_tools=False, max_tool_rounds=0,
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
        return self._memory.short_term.snapshot()

    def clear_history(self) -> None:
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
        self._checkpoints[cid] = {
            "messages": [m.to_dict() for m in self._memory.short_term.snapshot()],
            "cost": self._cost.total_cost_usd,
        }
        return cid

    def restore(self, checkpoint_id: CheckpointId) -> None:
        """Full-fidelity restore: tool_calls, tool results, reasoning and
        metadata all survive the round-trip (see `message_from_dict`)."""
        ckpt = self._checkpoints.get(checkpoint_id)
        if ckpt is None:
            raise KeyError(f"Unknown checkpoint: {checkpoint_id}")
        self._memory.short_term.replace_all(
            message_from_dict(raw) for raw in ckpt["messages"]
        )
        self._cost.total_cost_usd = ckpt.get("cost", self._cost.total_cost_usd)

    # ===================== internals =====================
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
                f"  - {t.metadata.name}: {t.metadata.description}"
                for t in self._tools
            ]
            parts.append("<capabilities>\n" + "\n".join(tool_lines) + "\n</capabilities>")

        # Environment block — config-provided.
        if self.config.environment:
            env_lines = [f"  {k}: {v}" for k, v in self.config.environment.items()]
            parts.append("<environment>\n" + "\n".join(env_lines) + "\n</environment>")

        return "\n\n".join(parts)


class AnthropicAgent(BaseAgent):
    pass


class OpenAIAgent(BaseAgent):
    pass
