"""AgentConfig — normalization, validation, and serialization.

The config is flat, frozen and validated at construction: a bad value must
fail here rather than several turns into a run.
"""

from __future__ import annotations

import dataclasses

import pytest

from agent_framework.core.config import AgentConfig, CompactionConfig, TransportConfig
from agent_framework.providers.catalog import (
    DEFAULT_CONTEXT_WINDOW,
    ProviderResolutionError,
)


def cfg(**kw) -> AgentConfig:
    kw.setdefault("model", "claude-opus-5")
    kw.setdefault("api_key", "test-key")
    return AgentConfig(**kw)


# ---------------- normalization ----------------

def test_model_is_the_only_required_field():
    config = AgentConfig(model="claude-opus-5")
    assert config.model == "claude-opus-5"
    assert config.provider == "anthropic"


def test_model_may_be_positional():
    assert AgentConfig("gpt-5").provider == "openai"


def test_an_explicit_prefix_is_stripped_from_the_wire_name():
    """`model` must be the bare name: it goes to the API and the price table."""
    config = AgentConfig(model="anthropic:claude-opus-5")
    assert config.model == "claude-opus-5"
    assert config.provider == "anthropic"


def test_provider_field_and_prefix_may_agree():
    config = AgentConfig(model="anthropic:claude-opus-5", provider="anthropic")
    assert (config.provider, config.model) == ("anthropic", "claude-opus-5")


def test_a_prefix_contradicting_the_provider_field_is_rejected():
    """Two explicit statements that disagree: refuse rather than pick one."""
    with pytest.raises(ValueError, match="contradicts"):
        AgentConfig(model="anthropic:claude-opus-5", provider="openai")


def test_the_provider_field_names_a_provider_for_an_unprefixed_model():
    """The documented use: a model whose name no provider's prefixes claim."""
    config = AgentConfig(model="llama-3.3-70b", provider="openai", context_window=32_768)
    assert (config.provider, config.model) == ("openai", "llama-3.3-70b")


def test_the_provider_field_overrides_inference():
    """Same as "openai:claude-opus-5": Claude through an OpenAI-compatible gateway."""
    config = AgentConfig(model="claude-opus-5", provider="openai")
    assert (config.provider, config.model) == ("openai", "claude-opus-5")


def test_an_unknown_provider_field_is_rejected():
    with pytest.raises(ProviderResolutionError, match="bedrock"):
        AgentConfig(model="claude-opus-5", provider="bedrock")


# Every shape a model id can take. Normalization must reach a fixed point on
# all of them: a constructed config stores the bare model plus the resolved
# provider, and feeding those back in must reproduce the same config — that is
# what replace(), from_dict() and every Agent(config, override=...) rely on.
_MODEL_ID_SHAPES = [
    "claude-opus-5",              # inferred
    "gpt-5",                      # inferred
    "anthropic:claude-opus-5",    # explicit, agrees with inference
    "openai:claude-opus-5",       # explicit, overrides inference (gateway)
    "openai:llama-3.3-70b",       # explicit, nothing to infer from
    "openai:llama3.2:3b",         # explicit, colon inside the model name (Ollama tag)
]


@pytest.fixture
def _quiet_unknown_windows():
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unknown context window")
        yield


@pytest.mark.usefixtures("_quiet_unknown_windows")
@pytest.mark.parametrize("model_id", _MODEL_ID_SHAPES)
def test_normalization_is_idempotent(model_id):
    config = cfg(model=model_id)
    rebuilt = AgentConfig(
        **{f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    )
    assert rebuilt == config


@pytest.mark.usefixtures("_quiet_unknown_windows")
@pytest.mark.parametrize("model_id", _MODEL_ID_SHAPES)
def test_replace_works_for_every_model_id_shape(model_id):
    derived = cfg(model=model_id).replace(max_turns=2)
    assert derived.max_turns == 2
    assert derived.provider == cfg(model=model_id).provider


@pytest.mark.usefixtures("_quiet_unknown_windows")
@pytest.mark.parametrize("model_id", _MODEL_ID_SHAPES)
def test_to_dict_round_trips_every_model_id_shape(model_id):
    config = cfg(model=model_id)
    assert AgentConfig.from_dict(config.to_dict()) == config.replace(api_key=None)


def test_from_env_accepts_a_provider_for_an_unprefixed_model(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "llama-3.3-70b")
    monkeypatch.setenv("AGENT_PROVIDER", "openai")
    monkeypatch.setenv("AGENT_CONTEXT_WINDOW", "32768")
    config = AgentConfig.from_env()
    assert (config.provider, config.model) == ("openai", "llama-3.3-70b")


def test_unresolvable_model_is_rejected_at_construction():
    with pytest.raises(ProviderResolutionError):
        AgentConfig(model="llama-3.3-70b")


def test_openai_compatible_endpoint_needs_only_a_prefix_and_base_url():
    config = AgentConfig(
        model="openai:llama-3.3-70b", base_url="http://localhost:8000/v1"
    )
    assert (config.provider, config.model) == ("openai", "llama-3.3-70b")
    assert config.resolved_context_window == DEFAULT_CONTEXT_WINDOW


# ---------------- immutability ----------------

def test_config_is_frozen():
    config = cfg()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_turns = 99  # type: ignore[misc]


def test_replace_returns_a_copy_and_leaves_the_original_alone():
    base = cfg(max_turns=10, system_prompt="base")
    derived = base.replace(max_turns=2, system_prompt="derived")
    assert (derived.max_turns, derived.system_prompt) == (2, "derived")
    assert (base.max_turns, base.system_prompt) == (10, "base")


def test_replace_revalidates():
    with pytest.raises(ValueError, match="max_turns"):
        cfg().replace(max_turns=0)


# ---------------- context window ----------------

def test_context_window_resolves_from_the_model():
    assert cfg(model="claude-opus-5").resolved_context_window == 1_000_000
    assert cfg(model="gpt-4o").resolved_context_window == 128_000


def test_explicit_context_window_wins():
    assert cfg(context_window=32_000).resolved_context_window == 32_000


# ---------------- validation ----------------

@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"max_turns": 0}, "max_turns"),
        ({"max_turns": -1}, "max_turns"),
        ({"max_budget_usd": 0}, "max_budget_usd"),
        ({"max_budget_usd": -0.5}, "max_budget_usd"),
        ({"stall_threshold": 0}, "stall_threshold"),
        ({"max_tokens_per_response": 0}, "max_tokens_per_response"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": 2.5}, "temperature"),
        ({"context_window": 512}, "context_window"),
        ({"response_buffer_tokens": -1}, "response_buffer_tokens"),
        ({"compaction_buffer_tokens": -1}, "compaction_buffer_tokens"),
        ({"max_parallel_tools": 0}, "max_parallel_tools"),
    ],
)
def test_out_of_range_values_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        cfg(**kwargs)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"trigger_ratio": 0.0}, "trigger_ratio"),
        ({"trigger_ratio": 1.5}, "trigger_ratio"),
        ({"sliding_window_min_turns": 0}, "sliding_window_min_turns"),
        ({"full_compaction_min_recent_turns": 0}, "full_compaction_min_recent_turns"),
        ({"summary_max_tokens": 0}, "summary_max_tokens"),
        ({"tool_result_truncate_threshold": -1}, "tool_result_truncate_threshold"),
    ],
)
def test_compaction_values_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CompactionConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"max_retries": -1}, "max_retries"),
    ],
)
def test_transport_values_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TransportConfig(**kwargs)


def test_buffers_larger_than_the_window_are_rejected():
    """ContextManager.budget() clamps to 1 token, compacting every turn."""
    with pytest.raises(ValueError, match="less than the context window"):
        cfg(
            context_window=8192,
            response_buffer_tokens=8192,
            compaction_buffer_tokens=4096,
        )


def test_valid_edge_values_are_accepted():
    config = cfg(
        max_turns=1,
        temperature=0.0,
        max_budget_usd=0.0001,
        context_window=1024,
        response_buffer_tokens=0,
        compaction_buffer_tokens=0,
        compaction=CompactionConfig(trigger_ratio=1.0),
    )
    assert config.max_turns == 1


# ---------------- api key resolution ----------------

def test_explicit_api_key_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert cfg(api_key="explicit").resolved_api_key() == "explicit"


def test_api_key_falls_back_to_the_providers_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert AgentConfig(model="claude-opus-5").resolved_api_key() == "from-env"


def test_missing_api_key_names_both_ways_to_supply_it(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        AgentConfig(model="claude-opus-5").resolved_api_key()
    assert "ANTHROPIC_API_KEY" in str(exc.value)
    assert "api_key" in str(exc.value)


def test_each_provider_reads_its_own_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert AgentConfig(model="gpt-5").resolved_api_key() == "sk-openai"
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AgentConfig(model="claude-opus-5").resolved_api_key()


# ---------------- serialization ----------------

def test_to_dict_round_trips():
    base = cfg(
        system_prompt="hello",
        max_turns=4,
        max_budget_usd=0.25,
        temperature=0.3,
        environment={"env": "prod"},
        request_extra={"thinking": {"type": "adaptive"}},
        compaction=CompactionConfig(trigger_ratio=0.85),
        transport=TransportConfig(timeout_seconds=12.0, max_retries=5),
    )
    restored = AgentConfig.from_dict(base.to_dict())
    # The API key is deliberately not serialized; everything else survives.
    assert restored == base.replace(api_key=None)
    assert restored.compaction.trigger_ratio == 0.85
    assert restored.transport.max_retries == 5


def test_to_dict_never_leaks_the_api_key():
    data = cfg(api_key="sk-secret").to_dict()
    assert "api_key" not in data
    assert data["api_key_set"] is True
    assert "sk-secret" not in repr(data)


def test_to_dict_reports_whether_a_key_was_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert AgentConfig(model="claude-opus-5").to_dict()["api_key_set"] is False


def test_to_dict_is_json_serializable_and_self_describing():
    import json

    data = json.loads(json.dumps(cfg().to_dict()))
    # The values a reader needs at 3am to explain a run's behaviour.
    for key in (
        "model",
        "provider",
        "resolved_context_window",
        "max_turns",
        "max_budget_usd",
        "temperature",
        "enable_prompt_caching",
        "compaction",
    ):
        assert key in data, key
    assert data["resolved_context_window"] == 1_000_000


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="max_turn"):
        AgentConfig.from_dict({"model": "claude-opus-5", "max_turn": 3})


def test_from_dict_accepts_already_built_nested_records():
    config = AgentConfig.from_dict(
        {"model": "claude-opus-5", "transport": TransportConfig(max_retries=7)}
    )
    assert config.transport.max_retries == 7


# ---------------- from_env ----------------

def test_from_env_reads_the_flat_fields(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "claude-opus-5")
    monkeypatch.setenv("AGENT_MAX_TURNS", "3")
    monkeypatch.setenv("AGENT_MAX_BUDGET_USD", "1.5")
    monkeypatch.setenv("AGENT_TEMPERATURE", "0.2")
    monkeypatch.setenv("AGENT_ENABLE_PROMPT_CACHING", "false")
    monkeypatch.setenv("AGENT_SYSTEM_PROMPT", "from env")
    config = AgentConfig.from_env()
    assert config.model == "claude-opus-5"
    assert config.max_turns == 3
    assert config.max_budget_usd == 1.5
    assert config.temperature == 0.2
    assert config.enable_prompt_caching is False
    assert config.system_prompt == "from env"


def test_from_env_honours_a_custom_prefix(monkeypatch):
    monkeypatch.setenv("MYAPP_MODEL", "gpt-5")
    assert AgentConfig.from_env(prefix="MYAPP_").provider == "openai"


def test_from_env_overrides_win_over_the_environment(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "claude-opus-5")
    monkeypatch.setenv("AGENT_MAX_TURNS", "3")
    config = AgentConfig.from_env(max_turns=9)
    assert config.max_turns == 9


def test_from_env_treats_empty_variables_as_unset(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "claude-opus-5")
    monkeypatch.setenv("AGENT_MAX_TURNS", "   ")
    assert AgentConfig.from_env().max_turns == 10


def test_from_env_without_a_model_says_what_to_set(monkeypatch):
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    with pytest.raises(ValueError, match="AGENT_MODEL"):
        AgentConfig.from_env()


def test_from_env_rejects_an_unparseable_value(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "claude-opus-5")
    monkeypatch.setenv("AGENT_MAX_TURNS", "many")
    with pytest.raises(ValueError, match="AGENT_MAX_TURNS"):
        AgentConfig.from_env()


def test_from_env_rejects_an_unparseable_bool(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "claude-opus-5")
    monkeypatch.setenv("AGENT_ENABLE_PROMPT_CACHING", "maybe")
    with pytest.raises(ValueError, match="AGENT_ENABLE_PROMPT_CACHING"):
        AgentConfig.from_env()


def test_from_env_still_validates(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "claude-opus-5")
    monkeypatch.setenv("AGENT_MAX_TURNS", "0")
    with pytest.raises(ValueError, match="max_turns"):
        AgentConfig.from_env()
