"""Provider catalog — model-id resolution, registration, context windows.

Resolution must be a deterministic table lookup: never ambient (it must not
read API keys from the environment), never a silent default, and never
order-dependent.
"""

from __future__ import annotations

import pytest

from agent_framework.providers.catalog import (
    DEFAULT_CONTEXT_WINDOW,
    ProviderResolutionError,
    ProviderSpec,
    get_provider,
    infer_provider,
    model_context_window,
    provider_names,
    register_provider,
    resolve_model,
    split_model_id,
    unregister_provider,
)


def _spec(name: str, prefixes: tuple[str, ...], **kw) -> ProviderSpec:
    return ProviderSpec(
        name=name,
        env_key=f"{name.upper()}_API_KEY",
        model_prefixes=prefixes,
        build=lambda **_: None,  # type: ignore[arg-type,return-value]
        **kw,
    )


@pytest.fixture
def custom_provider():
    """Register a throwaway provider and always remove it again."""
    registered: list[str] = []

    def _register(spec: ProviderSpec, **kw) -> ProviderSpec:
        register_provider(spec, **kw)
        registered.append(spec.name)
        return spec

    yield _register
    for name in registered:
        unregister_provider(name)


# ---------------- built-ins ----------------

def test_builtin_providers_are_registered():
    assert provider_names() == ["anthropic", "openai"]


@pytest.mark.parametrize(
    "model_id,provider,bare",
    [
        ("claude-opus-5", "anthropic", "claude-opus-5"),
        ("claude-sonnet-5", "anthropic", "claude-sonnet-5"),
        ("claude-haiku-4-5", "anthropic", "claude-haiku-4-5"),
        ("gpt-5", "openai", "gpt-5"),
        ("gpt-4o-mini", "openai", "gpt-4o-mini"),
        ("chatgpt-4o-latest", "openai", "chatgpt-4o-latest"),
        ("o1-preview", "openai", "o1-preview"),
        ("o3-mini", "openai", "o3-mini"),
    ],
)
def test_bare_model_names_infer_their_provider(model_id, provider, bare):
    spec, model = resolve_model(model_id)
    assert (spec.name, model) == (provider, bare)


@pytest.mark.parametrize(
    "model_id,provider,bare",
    [
        ("anthropic:claude-opus-5", "anthropic", "claude-opus-5"),
        # The whole point of the explicit form: a model whose name carries no
        # usable prefix, served by an OpenAI-compatible endpoint.
        ("openai:llama-3.3-70b", "openai", "llama-3.3-70b"),
        # An explicit prefix may also contradict inference; explicit wins.
        ("openai:claude-opus-5", "openai", "claude-opus-5"),
    ],
)
def test_explicit_prefix_wins_over_inference(model_id, provider, bare):
    spec, model = resolve_model(model_id)
    assert (spec.name, model) == (provider, bare)


def test_bare_name_reaches_the_wire_unchanged():
    """The spec selects the adapter; it must never rewrite the model name."""
    _, model = resolve_model("anthropic:claude-opus-5")
    assert model == "claude-opus-5"
    assert ":" not in model


# ---------------- failure modes ----------------

def test_unknown_model_raises_with_the_fix_in_the_message():
    with pytest.raises(ProviderResolutionError) as exc:
        resolve_model("llama-3.3-70b")
    msg = str(exc.value)
    assert "openai:llama-3.3-70b" in msg
    assert "anthropic, openai" in msg


def test_misspelled_provider_prefix_says_so():
    with pytest.raises(ProviderResolutionError, match="not a registered provider"):
        resolve_model("nope:gpt-5")


def test_an_ollama_tag_is_steered_to_the_explicit_form():
    """Ollama names always carry a colon; the fix is a prefix, not a provider."""
    with pytest.raises(ProviderResolutionError) as exc:
        resolve_model("llama3.2:3b")
    assert '"openai:llama3.2:3b"' in str(exc.value)


def test_provider_without_a_model_is_rejected():
    with pytest.raises(ProviderResolutionError, match="no model"):
        resolve_model("anthropic:")


@pytest.mark.parametrize("bad", ["", "   ", None, 42])
def test_empty_or_non_string_model_is_rejected(bad):
    with pytest.raises(ProviderResolutionError):
        split_model_id(bad)


def test_unknown_provider_name_lists_the_registered_ones():
    with pytest.raises(ProviderResolutionError) as exc:
        get_provider("bedrock")
    assert "anthropic, openai" in str(exc.value)


def test_resolution_never_reads_api_keys_from_the_environment(monkeypatch):
    """Ambient keys must not influence which provider is chosen."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec, _ = resolve_model("claude-opus-5")
    assert spec.name == "anthropic"


def test_ambiguous_model_raises_and_names_the_candidates(custom_provider):
    custom_provider(_spec("clonedropic", ("claude-",)))
    with pytest.raises(ProviderResolutionError) as exc:
        infer_provider("claude-opus-5")
    assert "anthropic" in str(exc.value)
    assert "clonedropic" in str(exc.value)


def test_more_specific_prefix_wins_over_a_shorter_one(custom_provider):
    """Scoring is by prefix length, so declaration order cannot matter."""
    custom_provider(_spec("specific", ("claude-opus-",)))
    assert infer_provider("claude-opus-5").name == "specific"
    assert infer_provider("claude-sonnet-5").name == "anthropic"


# ---------------- registration ----------------

def test_registering_an_existing_name_requires_replace(custom_provider):
    custom_provider(_spec("dupe", ("dupe-",)))
    with pytest.raises(ValueError, match="already registered"):
        register_provider(_spec("dupe", ("dupe-",)))
    register_provider(_spec("dupe", ("other-",)), replace=True)
    assert infer_provider("other-1").name == "dupe"


@pytest.mark.parametrize("name", ["", "has:colon"])
def test_invalid_provider_names_are_rejected(name):
    with pytest.raises(ValueError):
        register_provider(_spec(name, ("x-",)))


def test_a_registered_provider_is_fully_usable(custom_provider):
    """One ProviderSpec is all it takes: inference, env key, and window."""
    custom_provider(
        _spec(
            "bedrock",
            ("bedrock.",),
            context_windows=(("bedrock.big", 500_000),),
            default_context_window=100_000,
        )
    )
    spec, model = resolve_model("bedrock.big-1")
    assert (spec.name, model) == ("bedrock", "bedrock.big-1")
    assert spec.env_key == "BEDROCK_API_KEY"
    assert model_context_window("bedrock.big-1") == 500_000
    assert model_context_window("bedrock.small-1") == 100_000


def test_model_ids_may_contain_a_colon_after_the_provider(custom_provider):
    """Split on the first colon only: "ollama:llama3:8b" -> "llama3:8b"."""
    custom_provider(_spec("ollama", ("ollama-",)))
    spec, model = resolve_model("ollama:llama3:8b")
    assert (spec.name, model) == ("ollama", "llama3:8b")


# ---------------- context windows ----------------

@pytest.mark.parametrize(
    "model_id,window",
    [
        # Reproduces the pre-refactor table exactly.
        ("claude-3-opus-20240229", 200_000),
        ("claude-haiku-4-5", 200_000),
        ("claude-opus-5", 1_000_000),
        ("claude-sonnet-5", 1_000_000),
        ("gpt-4o", 128_000),
        ("gpt-4o-mini", 128_000),
        ("gpt-4.1", 1_000_000),
        ("gpt-5", 400_000),
        ("gpt-4-turbo", 128_000),
        ("o1", 200_000),
        ("o3-mini", 200_000),
        # Prefixes the catalog claims must also be budgetable.
        ("o4-mini", 200_000),
        ("chatgpt-4o-latest", 128_000),
    ],
)
def test_context_windows_match_the_known_models(model_id, window):
    assert model_context_window(model_id) == window


def test_context_window_falls_back_instead_of_raising():
    """Context budgeting must never hard-fail on an unresolvable model."""
    assert model_context_window("totally-unknown-model") == DEFAULT_CONTEXT_WINDOW


def test_context_window_honours_an_explicit_prefix():
    assert model_context_window("openai:llama-3.3-70b") == DEFAULT_CONTEXT_WINDOW
