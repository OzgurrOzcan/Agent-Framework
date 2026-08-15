"""Structured / JSON output — schema-constrained generation.

Uses each provider's NATIVE structured-output mechanism (Anthropic
`output_config.format`, OpenAI `response_format json_schema strict`) via
`BaseProviderAdapter.structured()`; providers without an override fall back
to the adapter base class's prompt-based implementation.

Two hard guarantees:
  - The agent's conversation history is NEVER touched — structured calls go
    straight to the provider with an isolated message list.
  - The parsed result is validated against the ORIGINAL schema client-side
    (including constraints the provider-side decoder can't enforce), plus
    Pydantic validation when a model class was given.

Usage:
    result = await agent.to_json(prompt, schema=MyModel)
    # or via the standalone function:
    result = await generate_structured(agent, prompt, schema=MyModel)
"""

from __future__ import annotations

import json
import re
from typing import Any, TYPE_CHECKING

from agent_framework.core.message import UserMessage
from agent_framework.output.schema_prep import prepare_for_strict, strip_optional_nulls

if TYPE_CHECKING:
    from agent_framework.core.base_agent import BaseAgent


class StructuredOutputError(Exception):
    """Raised when the model's output cannot be parsed/validated."""


async def generate_structured(
    agent: "BaseAgent",
    prompt: str,
    schema: Any,
    *,
    max_retries: int = 1,
) -> Any:
    """Generate JSON output validated against `schema`.

    `schema` may be:
      - A Pydantic BaseModel class → uses `.model_json_schema()` + `.model_validate()`
      - A plain `dict` (jsonschema) → validated with `jsonschema.validate()`

    Does not read or write the agent's conversation history.
    """
    schema_dict, validator = _resolve_schema(schema)
    wire_schema = prepare_for_strict(schema_dict)
    provider = agent._provider
    config = agent.config

    current_prompt = prompt
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        raw_text = await provider.structured(
            system="",
            messages=[UserMessage(content=current_prompt)],
            schema=wire_schema,
            model=config.model,
            max_tokens=config.max_tokens_per_response,
        )
        text = _strip_code_fences(raw_text.strip())

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            last_exc = StructuredOutputError(
                f"Model output is not valid JSON (attempt {attempt + 1}): {e}\n"
                f"Raw output: {text[:500]}"
            )
            current_prompt = _retry_prompt(prompt, text, str(e))
            continue

        parsed = strip_optional_nulls(parsed, schema_dict)

        try:
            _validate_original(parsed, schema_dict)
            return validator(parsed) if validator is not None else parsed
        except Exception as e:
            last_exc = StructuredOutputError(
                f"JSON failed schema validation (attempt {attempt + 1}): {e}"
            )
            current_prompt = _retry_prompt(prompt, text, str(e))
            continue

    raise last_exc or StructuredOutputError("Structured output generation failed.")


def _retry_prompt(original_prompt: str, previous_output: str, error: str) -> str:
    return (
        f"{original_prompt}\n\n"
        f"Your previous attempt was invalid.\n"
        f"Previous output:\n{previous_output[:2000]}\n"
        f"Error: {error}\n"
        f"Return the corrected JSON object only."
    )


def _validate_original(parsed: Any, schema_dict: dict[str, Any]) -> None:
    """Enforce the ORIGINAL schema — including constraints stripped from the
    wire schema (minLength, minimum, ...) — client-side."""
    import jsonschema

    jsonschema.validate(instance=parsed, schema=schema_dict)


def _resolve_schema(schema: Any) -> tuple[dict[str, Any], Any]:
    """Returns (schema_dict, validator_fn_or_None).

    validator_fn takes a parsed dict and returns a validated instance (or
    raises on failure).
    """
    # Pydantic BaseModel class.
    try:
        import pydantic
        if isinstance(schema, type) and issubclass(schema, pydantic.BaseModel):
            schema_dict = schema.model_json_schema()
            def _pydantic_validator(data: dict) -> Any:
                return schema.model_validate(data)
            return schema_dict, _pydantic_validator
    except ImportError:
        pass

    # Raw jsonschema dict.
    if isinstance(schema, dict):
        return schema, None

    raise TypeError(
        f"schema must be a Pydantic BaseModel class or a jsonschema dict, "
        f"got {type(schema).__name__}"
    )


def _strip_code_fences(text: str) -> str:
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()
