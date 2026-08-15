"""Structured output tests — native mechanisms, schema prep, history isolation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Optional

import pytest
from pydantic import BaseModel

from agent_framework.core.config import AgentConfig, ProviderConfig, ProviderKind
from agent_framework.output.schema_prep import prepare_for_strict, strip_optional_nulls
from agent_framework.output.structured import StructuredOutputError, generate_structured
from agent_framework.tests.mocks.mock_llm import MockLLMProvider, ScriptedTurn


class Contact(BaseModel):
    name: str
    email: str
    nickname: Optional[str] = None
    priority: int = 1


def _fake_agent(mock: MockLLMProvider):
    config = AgentConfig(
        provider=ProviderConfig(
            provider=ProviderKind.ANTHROPIC, model="claude-opus-5", api_key="test-key"
        )
    )
    return SimpleNamespace(_provider=mock, config=config)


# ---------------- schema preparation ----------------

def test_prepare_for_strict_objects():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 3},
            "age": {"type": "integer", "minimum": 0},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        },
        "required": ["name"],
    }
    prepared = prepare_for_strict(schema)
    # unsupported constraints removed
    assert "minLength" not in prepared["properties"]["name"]
    assert "minimum" not in prepared["properties"]["age"]
    assert "maxItems" not in prepared["properties"]["tags"]
    # strict shape: additionalProperties false + everything required
    assert prepared["additionalProperties"] is False
    assert set(prepared["required"]) == {"name", "age", "tags"}
    # originally-optional props became nullable
    assert prepared["properties"]["age"]["type"] == ["integer", "null"]
    # original schema untouched (deep copy)
    assert schema["required"] == ["name"]
    assert "minLength" in schema["properties"]["name"]


def test_prepare_for_strict_recurses_defs_and_nested():
    schema = {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/Item"}},
        "required": ["item"],
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"n": {"type": "integer", "maximum": 10}},
            }
        },
    }
    prepared = prepare_for_strict(schema)
    item = prepared["$defs"]["Item"]
    assert item["additionalProperties"] is False
    assert "maximum" not in item["properties"]["n"]
    assert item["required"] == ["n"]


def test_strip_optional_nulls():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "nickname": {"type": "string"},
        },
        "required": ["name"],
    }
    data = {"name": "Jane", "nickname": None}
    assert strip_optional_nulls(data, schema) == {"name": "Jane"}
    # required nulls are kept (validation will catch them)
    data2 = {"name": None, "nickname": "J"}
    assert strip_optional_nulls(data2, schema) == {"name": None, "nickname": "J"}


# ---------------- generation ----------------

async def test_pydantic_roundtrip_with_nullable_optionals():
    mock = MockLLMProvider()
    mock.structured_script = [
        json.dumps({"name": "Jane", "email": "j@x.io", "nickname": None, "priority": None})
    ]
    agent = _fake_agent(mock)
    result = await generate_structured(agent, "extract", Contact)
    assert isinstance(result, Contact)
    assert result.name == "Jane"
    assert result.nickname is None
    assert result.priority == 1  # default restored after null-strip
    # wire schema was strict-prepared
    sent = mock.structured_calls[0]["schema"]
    assert sent["additionalProperties"] is False
    assert set(sent["required"]) == {"name", "email", "nickname", "priority"}


async def test_invalid_json_retries_with_feedback():
    mock = MockLLMProvider()
    mock.structured_script = ["not json at all", json.dumps({"a": 1})]
    agent = _fake_agent(mock)
    result = await generate_structured(
        agent, "go",
        {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
    )
    assert result == {"a": 1}
    assert len(mock.structured_calls) == 2
    retry_msg = mock.structured_calls[1]["messages"][0].content
    assert "previous attempt was invalid" in retry_msg
    assert "not json at all" in retry_msg


async def test_client_side_constraint_enforced():
    # minLength stripped from the wire schema must still be enforced locally.
    mock = MockLLMProvider()
    mock.structured_script = [json.dumps({"code": "ab"}), json.dumps({"code": "ab"})]
    agent = _fake_agent(mock)
    schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "minLength": 5}},
        "required": ["code"],
    }
    with pytest.raises(StructuredOutputError, match="validation"):
        await generate_structured(agent, "go", schema)
    # the wire schema did not contain minLength
    assert "minLength" not in mock.structured_calls[0]["schema"]["properties"]["code"]


async def test_history_is_never_touched():
    """to_json must not pollute the agent's conversation history."""
    from agent_framework.core.base_agent import BaseAgent

    config = AgentConfig(
        provider=ProviderConfig(
            provider=ProviderKind.ANTHROPIC, model="claude-opus-5", api_key="test-key"
        ),
        system_prompt="test",
    )
    agent = BaseAgent.create(config)
    mock = MockLLMProvider()
    mock.structured_script = [json.dumps({"name": "J", "email": "j@x.io"})]
    agent._provider = mock
    agent._loop.provider = mock

    before = agent.get_history()
    result = await agent.to_json("extract contact", schema=Contact)
    assert result.name == "J"
    assert agent.get_history() == before == []


async def test_prompt_based_fallback_on_base_adapter():
    """Adapters without a structured() override use the prompt-based default."""
    from agent_framework.providers.base import BaseProviderAdapter

    class MinimalProvider(MockLLMProvider):
        # Remove the override → fall back to BaseProviderAdapter.structured.
        structured = BaseProviderAdapter.structured

    mock = MinimalProvider([ScriptedTurn(text='```json\n{"a": 2}\n```')])
    agent = _fake_agent(mock)
    result = await generate_structured(
        agent, "go",
        {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
    )
    assert result == {"a": 2}
    # The fallback embeds the schema in the system prompt of a normal call.
    assert "json" in mock.call_log[0]["system"].lower()


async def test_rejects_unknown_schema_type():
    mock = MockLLMProvider()
    agent = _fake_agent(mock)
    with pytest.raises(TypeError):
        await generate_structured(agent, "go", schema=42)
