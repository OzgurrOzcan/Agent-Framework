"""The provider catalog — one record per provider, and the model-id resolver.

This table is the single source of truth for everything provider-shaped:

  - which adapter serves a model id
  - which environment variable holds that provider's API key
  - which model-name prefixes belong to the provider
  - that provider's context windows
  - which of its models bind reasoning to the conversation prefix

Adding a provider is therefore one `ProviderSpec` plus one
`BaseProviderAdapter` subclass — nothing else in the framework changes.

Resolution is a lookup table, never detection. It does **not** look at which
API keys happen to be set in the environment: which model you run must not
depend on ambient state. A model id may name its provider explicitly with a
``provider:model`` prefix, which always wins:

    "claude-opus-5"             -> anthropic  (prefix "claude-")
    "gpt-5"                     -> openai     (prefix "gpt-")
    "anthropic:claude-opus-5"   -> explicit
    "openai:llama-3.3-70b"      -> explicit; how OpenAI-compatible endpoints work
    "llama-3.3-70b"             -> ProviderResolutionError (never a silent default)

Prefixes are deliberately broad (``"claude-"``, not a model list) so a newly
released model works the day it ships, without a framework release.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agent_framework.providers.base import BaseProviderAdapter


# Conservative fallback for a model whose provider is known but whose context
# window is not in that provider's table.
DEFAULT_CONTEXT_WINDOW = 200_000


class ProviderResolutionError(ValueError):
    """Raised when a model id cannot be resolved to exactly one provider."""


class AdapterFactory(Protocol):
    """How the catalog constructs an adapter. Keyword-only by design."""

    def __call__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        timeout: float,
        max_retries: int,
        client_kwargs: dict[str, Any] | None,
    ) -> BaseProviderAdapter: ...


@dataclass(frozen=True)
class ProviderSpec:
    """Everything the framework needs to know about one provider."""

    name: str
    env_key: str
    # Model-name prefixes owned by this provider, used to infer the provider
    # from a bare model id. Longest match wins across the whole catalog.
    model_prefixes: tuple[str, ...]
    build: AdapterFactory
    # (model-name prefix, context window) pairs; longest prefix wins.
    context_windows: tuple[tuple[str, int], ...] = ()
    default_context_window: int = DEFAULT_CONTEXT_WINDOW
    # Whether a model's reasoning blocks are only valid while every message
    # before them is unchanged (the provider rejects them otherwise). On such
    # a model the framework never edits the history in the middle of a run.
    # Exact model names win over (prefix, binds) pairs; longest prefix wins;
    # `reasoning_binding_default` covers models in neither.
    reasoning_binding_exact: tuple[tuple[str, bool], ...] = ()
    reasoning_binding_prefixes: tuple[tuple[str, bool], ...] = ()
    reasoning_binding_default: bool = False
    # Models that reject a historical assistant tool-call message whose
    # reasoning was left out (Kimi's `reasoning_content`). Distinct from
    # binding: such reasoning is not invalidated by editing earlier messages,
    # it only has to stay — so a trim may still clear results and drop whole
    # turns, but must not cut reasoning from the turns it keeps.
    reasoning_required_prefixes: tuple[str, ...] = ()

    def prefix_score(self, model: str) -> int:
        """Length of the longest owned prefix matching *model*; 0 if none.

        Length is the score so that a more specific provider prefix beats a
        less specific one deterministically, independent of declaration order.
        """
        return max(
            (len(p) for p in self.model_prefixes if model.startswith(p)),
            default=0,
        )

    def known_context_window(self, model: str) -> int | None:
        """The declared context window for *model*, or None if undeclared.

        The distinction matters: a model this provider does not recognize gets
        a fallback that can be wildly wrong (a locally served 8k model budgeted
        as if it had 200k never compacts, and overflows at the endpoint), so
        callers that can warn about it need to tell the two cases apart.
        """
        best_len = -1
        window: int | None = None
        for prefix, size in self.context_windows:
            if model.startswith(prefix) and len(prefix) > best_len:
                best_len, window = len(prefix), size
        return window

    def context_window(self, model: str) -> int:
        """Best-effort context window for *model* (longest prefix wins)."""
        known = self.known_context_window(model)
        return known if known is not None else self.default_context_window

    def binds_reasoning_to_prefix(self, model: str) -> bool:
        """True if *model*'s reasoning is invalidated by editing earlier messages."""
        for name, binds in self.reasoning_binding_exact:
            if model == name:
                return binds
        best_len = -1
        result = self.reasoning_binding_default
        for prefix, binds in self.reasoning_binding_prefixes:
            if model.startswith(prefix) and len(prefix) > best_len:
                best_len, result = len(prefix), binds
        return result

    def requires_reasoning_in_history(self, model: str) -> bool:
        """True if *model* rejects a kept assistant message whose reasoning was cut."""
        return any(model.startswith(p) for p in self.reasoning_required_prefixes)


# --------------------------------------------------------------------------
# Built-in adapter factories. Imports are deferred so that importing the
# catalog does not pull in either vendor SDK.
# --------------------------------------------------------------------------


def _build_anthropic(
    *,
    api_key: str,
    base_url: str | None,
    timeout: float,
    max_retries: int,
    client_kwargs: dict[str, Any] | None,
) -> BaseProviderAdapter:
    from agent_framework.providers.anthropic_agent import AnthropicAdapter

    return AnthropicAdapter(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        client_kwargs=client_kwargs,
    )


def _build_openai(
    *,
    api_key: str,
    base_url: str | None,
    timeout: float,
    max_retries: int,
    client_kwargs: dict[str, Any] | None,
) -> BaseProviderAdapter:
    from agent_framework.providers.openai_agent import OpenAIAdapter

    return OpenAIAdapter(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        client_kwargs=client_kwargs,
    )


ANTHROPIC = ProviderSpec(
    name="anthropic",
    env_key="ANTHROPIC_API_KEY",
    model_prefixes=("claude-",),
    build=_build_anthropic,
    context_windows=(
        ("claude-3", 200_000),
        ("claude-haiku", 200_000),
        # Current-generation Claude models (Opus/Sonnet 4.6+, 5.x).
        ("claude-", 1_000_000),
    ),
    # "Preserved thinking": on these models a thinking block's signature
    # covers the system prompt, the tool set and every message before it, so
    # editing the history mid-run turns the next request into a 400. Claude
    # Mythos 5.1 does not run that check. A model this table does not know is
    # assumed to bind — the safe side: it only gives up client-side trims in
    # the middle of a run, while assuming the opposite would end runs in 400s.
    reasoning_binding_exact=(
        ("claude-fable-5-1", True),
        ("claude-opus-5-5", True),
        ("claude-sonnet-5-5", True),
        ("claude-mythos-5-1", False),
        ("claude-fable-5", False),
        ("claude-mythos-5", False),
        ("claude-opus-5", False),
        ("claude-sonnet-5", False),
    ),
    reasoning_binding_prefixes=(
        ("claude-2", False),
        ("claude-3", False),
        ("claude-instant", False),
        ("claude-haiku-4", False),
        ("claude-sonnet-4", False),
        ("claude-opus-4", False),
    ),
    reasoning_binding_default=True,
)

OPENAI = ProviderSpec(
    name="openai",
    env_key="OPENAI_API_KEY",
    model_prefixes=("gpt-", "chatgpt-", "o1", "o3", "o4-"),
    build=_build_openai,
    context_windows=(
        ("gpt-4", 128_000),
        ("gpt-4o", 128_000),
        ("gpt-4.1", 1_000_000),
        ("gpt-5", 400_000),
        ("chatgpt-4o", 128_000),
        ("o1", 200_000),
        ("o3", 200_000),
        ("o4-", 200_000),
    ),
    # Chat Completions never sends reasoning back, so there is none to bind.
    reasoning_binding_default=False,
    # Kimi behind this adapter (`openai:kimi-k3` + base_url): its thinking
    # models return `reasoning_content` and reject a historical tool-call
    # message without it. Other such endpoints (DeepSeek, a vLLM thinking
    # model) are not inferable from the id — `keep_reasoning_in_history=True`
    # on the config names them explicitly.
    reasoning_required_prefixes=("kimi-",),
)


_REGISTRY: dict[str, ProviderSpec] = {}


def register_provider(spec: ProviderSpec, *, replace: bool = False) -> None:
    """Add *spec* to the catalog.

    Pass ``replace=True`` to deliberately override a registered provider (for
    example to swap an adapter in a test); without it, re-registering a name
    raises so a typo cannot silently shadow a built-in.
    """
    if not spec.name:
        raise ValueError("ProviderSpec.name must be a non-empty string")
    if ":" in spec.name:
        raise ValueError(
            f"ProviderSpec.name must not contain ':' (got {spec.name!r}); "
            "the colon separates provider from model in a model id"
        )
    if spec.name in _REGISTRY and not replace:
        raise ValueError(
            f"Provider {spec.name!r} is already registered; "
            "pass replace=True to override it deliberately"
        )
    _REGISTRY[spec.name] = spec


def unregister_provider(name: str) -> None:
    """Remove a provider from the catalog. No-op if it is not registered."""
    _REGISTRY.pop(name, None)


def provider_names() -> list[str]:
    """Registered provider names, sorted."""
    return sorted(_REGISTRY)


def get_provider(name: str) -> ProviderSpec:
    """Look up a provider by name, or raise listing the registered ones."""
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ProviderResolutionError(
            f"Unknown provider {name!r}. Registered providers: "
            f"{', '.join(provider_names()) or '(none)'}. "
            "Register your own with providers.catalog.register_provider()."
        )
    return spec


def split_model_id(model_id: str) -> tuple[str | None, str]:
    """Split ``"provider:model"`` into its parts without resolving anything.

    Only a head that names a *registered* provider is treated as a provider,
    so model ids that legitimately contain a colon (``"llama3:8b"``) are left
    intact. Splits on the first colon only, so ``"ollama:llama3:8b"`` yields
    ``("ollama", "llama3:8b")``.
    """
    if not isinstance(model_id, str) or not model_id.strip():
        raise ProviderResolutionError(
            f"model must be a non-empty string, got {model_id!r}"
        )
    model_id = model_id.strip()
    head, sep, tail = model_id.partition(":")
    if sep and head in _REGISTRY:
        if not tail:
            raise ProviderResolutionError(
                f"Model id {model_id!r} names provider {head!r} but no model. "
                f'Write e.g. "{head}:<model-name>".'
            )
        return head, tail
    return None, model_id


def infer_provider(model: str) -> ProviderSpec:
    """Infer the provider that owns a bare model name.

    Raises `ProviderResolutionError` when no provider claims the name, or when
    more than one claims it equally strongly — never guesses, never defaults.
    """
    scored = [(spec.prefix_score(model), spec) for spec in _REGISTRY.values()]
    best = max((score for score, _ in scored), default=0)
    if best == 0:
        raise ProviderResolutionError(
            f"Cannot infer a provider from model {model!r}. Name it explicitly "
            f'as "provider:model" — e.g. "openai:{model}" for an '
            f"OpenAI-compatible endpoint. Registered providers: "
            f"{', '.join(provider_names()) or '(none)'}."
        )
    winners = [spec for score, spec in scored if score == best]
    if len(winners) > 1:
        names = ", ".join(sorted(s.name for s in winners))
        raise ProviderResolutionError(
            f"Model {model!r} is claimed by more than one provider ({names}). "
            f'Name one explicitly as "provider:model".'
        )
    return winners[0]


def resolve_model(model_id: str) -> tuple[ProviderSpec, str]:
    """Resolve a model id to ``(spec, bare_model_name)``.

    The bare name is what reaches the wire and the pricing table; the spec
    only ever selects the adapter, so a mis-resolution surfaces immediately
    as a provider-side error rather than as subtly different behaviour.
    """
    provider, model = split_model_id(model_id)
    if provider is not None:
        return get_provider(provider), model
    try:
        return infer_provider(model), model
    except ProviderResolutionError:
        # A colon whose head is not a registered provider is far more likely a
        # misspelled provider than a model name that happens to contain ':',
        # so say so instead of suggesting "openai:nope:gpt-5".
        head, sep, _ = model.partition(":")
        if sep and head:
            raise ProviderResolutionError(
                f"Model id {model_id!r} looks like \"provider:model\", but "
                f"{head!r} is not a registered provider. If the colon belongs "
                "to the model name (an Ollama tag such as \"llama3.2:3b\"), "
                f'name the adapter explicitly: "openai:{model}". Registered '
                f"providers: {', '.join(provider_names()) or '(none)'}."
            ) from None
        raise


def model_context_window(model_id: str) -> int:
    """Best-effort context window for a model id.

    Falls back to `DEFAULT_CONTEXT_WINDOW` when the provider cannot be
    resolved at all, so context budgeting never hard-fails on an unknown name.
    """
    try:
        spec, model = resolve_model(model_id)
    except ProviderResolutionError:
        return DEFAULT_CONTEXT_WINDOW
    return spec.context_window(model)


register_provider(ANTHROPIC)
register_provider(OPENAI)
