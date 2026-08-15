"""Built-in `ask_user` tool — always registered on every agent.

This tool is special: its body is never executed directly. Instead, the
`ToolExecutor` intercepts calls to it and routes them through the
`InterruptManager` as a `CLARIFICATION` interrupt. The renderer (terminal
or callback) handles delivery; the answer comes back as the tool result.

Permission category is `AUTO` — asking the human a question is always safe.
"""

from __future__ import annotations

from agent_framework.permissions.models import PermissionCategory
from agent_framework.tools.models import Tool, ToolMetadata


ASK_USER_TOOL_NAME = "ask_user"


_ASK_USER_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "The clarifying question to ask the user.",
        },
        "options": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional suggested choices to present.",
        },
        "allow_free_text": {
            "type": "boolean",
            "description": "Whether the user may type a custom answer beyond the options.",
            "default": True,
        },
        "urgency": {
            "enum": ["info", "warning", "critical"],
            "default": "info",
        },
    },
    "required": ["question"],
    "additionalProperties": False,
}


async def _ask_user_placeholder(**kwargs: object) -> object:
    """Never called — the executor intercepts this tool."""
    raise RuntimeError(
        "ask_user must be routed through the InterruptManager, not invoked directly."
    )


ask_user_tool = Tool(
    metadata=ToolMetadata(
        name=ASK_USER_TOOL_NAME,
        description=(
            "Ask the user a clarifying question when you cannot proceed without their "
            "input. Use sparingly — only when genuinely blocked."
        ),
        parameters_schema=_ASK_USER_SCHEMA,
        permission=PermissionCategory.AUTO,
        timeout_seconds=600.0,  # humans can be slow
        max_retries=0,
        idempotent=False,
        cache_ttl_seconds=None,
        tags=("builtin", "interrupt"),
        is_async=True,
        source="builtin",
    ),
    fn=_ask_user_placeholder,
)
