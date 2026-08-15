"""Provider adapters — OpenAI + Anthropic. Normalize to framework Message types."""

from agent_framework.providers.base import (
    BaseProviderAdapter,
    NormalizedResponse,
    StreamChunk,
)

__all__ = ["BaseProviderAdapter", "NormalizedResponse", "StreamChunk"]
