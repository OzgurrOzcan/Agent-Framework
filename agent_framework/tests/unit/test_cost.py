"""Tests for cost tracking: pricing table, unknown-model warning, strict mode."""
import warnings

import pytest

from agent_framework.core.message import TokenUsage
from agent_framework.tracing.cost import CostTracker, ModelPricing, PricingTable


def _user_warnings(caught):
    return [w for w in caught if issubclass(w.category, UserWarning)]


def test_unknown_model_warns_once_and_costs_zero():
    tracker = CostTracker(pricing=PricingTable())
    usage = TokenUsage(input_tokens=1_000, output_tokens=1_000)

    with pytest.warns(UserWarning, match="budget guard inactive") as caught:
        cost = tracker.record("mystery-model", usage)
    assert cost == 0.0
    assert "mystery-model" in str(caught[0].message)

    # Second record for the same model: no warning.
    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        cost2 = tracker.record("mystery-model", usage)
    assert cost2 == 0.0
    assert _user_warnings(second) == []

    # A different unknown model warns again (dedup is per model).
    with pytest.warns(UserWarning, match="budget guard inactive"):
        tracker.record("other-mystery-model", usage)


def test_strict_pricing_table_raises_keyerror():
    table = PricingTable({"gpt-4o": ModelPricing(2.5, 10.0)}, strict=True)
    with pytest.raises(KeyError):
        table.estimate("unknown-model", TokenUsage(input_tokens=10))
    # Known models still estimate normally.
    assert table.estimate("gpt-4o", TokenUsage(input_tokens=1_000_000)) == pytest.approx(2.5)


def test_prefix_match_no_warning():
    tracker = CostTracker(pricing=PricingTable.default())
    usage = TokenUsage(input_tokens=1_000_000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cost = tracker.record("claude-opus-5-20260101", usage)
    assert cost == pytest.approx(5.0)  # matches the "claude-opus-5" entry
    assert _user_warnings(caught) == []


def test_known_model_cost_with_cache_tokens():
    tracker = CostTracker(pricing=PricingTable.default())
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_input_tokens=500_000,
        cache_creation_input_tokens=200_000,
    )
    cost = tracker.record("claude-sonnet-5", usage)
    # 3.0 (input) + 1.5 (output) + 0.15 (cache read) + 0.75 (cache write)
    assert cost == pytest.approx(5.4)
    total, snap_usage = tracker.snapshot()
    assert total == pytest.approx(5.4)
    assert snap_usage.cache_read_input_tokens == 500_000


def test_default_table_updated_prices():
    table = PricingTable.default()
    assert table.get("claude-fable-5") == ModelPricing(10.0, 50.0, 1.0, 12.5)
    assert table.get("claude-opus-5") == ModelPricing(5.0, 25.0, 0.5, 6.25)
    assert table.get("claude-haiku-4-5") == ModelPricing(1.0, 5.0, 0.1, 1.25)
    assert table.get("gpt-5") == ModelPricing(1.25, 10.0, 0.125, 0.0)
