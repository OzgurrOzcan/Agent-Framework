"""Optional output layer — structured/JSON output (schema-constrained)."""

from agent_framework.output.structured import generate_structured, StructuredOutputError

__all__ = ["generate_structured", "StructuredOutputError"]
