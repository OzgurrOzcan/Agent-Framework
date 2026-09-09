"""Provider adapters — OpenAI + Anthropic. Normalize to framework Message types.

`catalog` holds one `ProviderSpec` per provider and resolves a model id to the
adapter that serves it; see `agent_framework.providers.catalog`.
"""

from agent_framework.providers.base import (
    BaseProviderAdapter,
    NormalizedResponse,
    StreamChunk,
)
from agent_framework.providers.catalog import (
    ProviderResolutionError,
    ProviderSpec,
    register_provider,
    resolve_model,
)

__all__ = [
    "BaseProviderAdapter",
    "NormalizedResponse",
    "ProviderResolutionError",
    "ProviderSpec",
    "StreamChunk",
    "register_provider",
    "resolve_model",
]
