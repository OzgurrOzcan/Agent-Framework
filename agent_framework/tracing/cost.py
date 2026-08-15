"""Cost tracking — accumulates token usage across LLM calls.

Pricing is **injected**, never hardcoded. Callers either pass a
`PricingTable` they own or rely on `PricingTable.default()` which
ships a small starter set of public prices you can override.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field

from agent_framework.core.message import TokenUsage


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0


class PricingTable:
    """Lookup of model → ModelPricing. Mutable; injected at agent creation.

    With ``strict=True``, `estimate` raises `KeyError` for unknown models
    instead of silently returning 0.0 — for deployments that require the
    budget guard to be a hard guarantee.
    """

    def __init__(
        self,
        prices: dict[str, ModelPricing] | None = None,
        *,
        strict: bool = False,
    ) -> None:
        self._prices: dict[str, ModelPricing] = dict(prices or {})
        self.strict = strict

    def set(self, model: str, pricing: ModelPricing) -> None:
        self._prices[model] = pricing

    def get(self, model: str) -> ModelPricing | None:
        if model in self._prices:
            return self._prices[model]
        # Best-effort prefix match (e.g. "claude-opus-4-7-20260101" → "claude-opus-4-7").
        for key, value in self._prices.items():
            if model.startswith(key):
                return value
        return None

    def estimate(self, model: str, usage: TokenUsage) -> float:
        p = self.get(model)
        if p is None:
            if self.strict:
                raise KeyError(
                    f"No pricing entry for model {model!r} (strict PricingTable)"
                )
            return 0.0
        cost = (usage.input_tokens / 1_000_000.0) * p.input_per_mtok
        cost += (usage.output_tokens / 1_000_000.0) * p.output_per_mtok
        cost += (usage.cache_read_input_tokens / 1_000_000.0) * p.cache_read_per_mtok
        cost += (usage.cache_creation_input_tokens / 1_000_000.0) * p.cache_write_per_mtok
        return cost

    @classmethod
    def default(cls) -> "PricingTable":
        """Starter pricing — override for your own deployment."""
        return cls(
            {
                "claude-fable-5": ModelPricing(10.0, 50.0, 1.0, 12.5),
                "claude-opus-5": ModelPricing(5.0, 25.0, 0.5, 6.25),
                "claude-opus-4-8": ModelPricing(5.0, 25.0, 0.5, 6.25),
                "claude-opus-4-7": ModelPricing(5.0, 25.0, 0.5, 6.25),
                "claude-opus-4-6": ModelPricing(5.0, 25.0, 0.5, 6.25),
                "claude-sonnet-5": ModelPricing(3.0, 15.0, 0.3, 3.75),
                "claude-sonnet-4-6": ModelPricing(3.0, 15.0, 0.3, 3.75),
                "claude-haiku-4-5": ModelPricing(1.0, 5.0, 0.1, 1.25),
                "gpt-5": ModelPricing(1.25, 10.0, 0.125, 0.0),
                "gpt-4.1": ModelPricing(2.0, 8.0, 0.5, 0.0),
                "gpt-4o": ModelPricing(2.5, 10.0, 1.25, 0.0),
                "gpt-4o-mini": ModelPricing(0.15, 0.6, 0.075, 0.0),
                "o1": ModelPricing(15.0, 60.0, 7.5, 0.0),
            }
        )


@dataclass
class CostTracker:
    """Thread-safe cumulative cost & token tracker. Wired into the budget guard."""

    pricing: PricingTable = field(default_factory=PricingTable.default)
    total_cost_usd: float = 0.0
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _warned_models: set = field(default_factory=set, repr=False)

    def record(self, model: str, usage: TokenUsage) -> float:
        if not self.pricing.strict and self.pricing.get(model) is None:
            with self._lock:
                should_warn = model not in self._warned_models
                if should_warn:
                    self._warned_models.add(model)
            if should_warn:
                warnings.warn(
                    f"No pricing for model {model!r}; cost is recorded as 0.0 "
                    "(budget guard inactive for this model). Register the model "
                    "in the PricingTable to enable cost tracking.",
                    UserWarning,
                    stacklevel=2,
                )
        cost = self.pricing.estimate(model, usage)
        with self._lock:
            self.total_cost_usd += cost
            self.total_usage = self.total_usage + usage
        return cost

    def snapshot(self) -> tuple[float, TokenUsage]:
        with self._lock:
            return self.total_cost_usd, TokenUsage(
                input_tokens=self.total_usage.input_tokens,
                output_tokens=self.total_usage.output_tokens,
                cache_read_input_tokens=self.total_usage.cache_read_input_tokens,
                cache_creation_input_tokens=self.total_usage.cache_creation_input_tokens,
            )
