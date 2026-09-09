"""@tool decorator — introspects type hints to generate a JSON schema.

Usage:

    @tool(
        description="Search the web",
        permission=PermissionCategory.AUTO,
        timeout_seconds=30,
        max_retries=2,
        idempotent=True,
        cache_ttl_seconds=None,
        tags=["research", "web"],
    )
    async def web_search(
        query: Annotated[str, "The search phrase"],
        max_results: int = 5,
    ) -> list[dict]:
        ...

Per-parameter descriptions are picked up from two sources (in priority
order): `Annotated[T, "description"]` metadata, then a Google-style
docstring ``Args:`` section (`param: description` lines).

The decorator returns a `Tool` (callable wrapper) so it can be used
directly in a `list[Tool]` passed to `Agent(...)`.
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from typing import Any, Callable, Sequence, get_args, get_origin, get_type_hints

from agent_framework.permissions.models import PermissionCategory
from agent_framework.tools.models import Tool, ToolMetadata


def tool(
    *,
    description: str | None = None,
    permission: PermissionCategory = PermissionCategory.ASK,
    timeout_seconds: float = 60.0,
    max_retries: int = 0,
    idempotent: bool = False,
    cache_ttl_seconds: float | None = None,
    tags: Sequence[str] = (),
    name: str | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    def decorator(fn: Callable[..., Any]) -> Tool:
        tool_name = name or fn.__name__
        desc = description or (inspect.getdoc(fn) or "").strip() or tool_name
        schema = _build_schema(fn)
        meta = ToolMetadata(
            name=tool_name,
            description=desc,
            parameters_schema=schema,
            permission=permission,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            idempotent=idempotent,
            cache_ttl_seconds=cache_ttl_seconds,
            tags=tuple(tags),
            is_async=inspect.iscoroutinefunction(fn),
            source="native",
        )
        return Tool(metadata=meta, fn=fn)

    return decorator


def _build_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Generate JSON Schema from a function's signature and type hints.

    Falls back to `string` for unannotated or non-trivial parameters.
    Handles Optional/Union (as `anyOf` with `{"type": "null"}`), list[T],
    dict[str, V], int/float/bool/str, and Pydantic models (via
    .model_json_schema()).

    Parameter descriptions come from `Annotated[T, "description"]`
    metadata first, then from a Google-style docstring ``Args:`` section.
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}
    doc_args = _parse_docstring_args(inspect.getdoc(fn))

    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        hint = hints.get(pname, str)
        description: str | None = None
        # `Annotated[T, "description"]` — checked structurally because
        # `get_origin(Annotated[...])` is version-dependent.
        if hasattr(hint, "__metadata__"):
            for item in hint.__metadata__:
                if isinstance(item, str):
                    description = item
                    break
            hint = hint.__origin__
        if description is None:
            description = doc_args.get(pname)
        prop = _type_to_schema(hint)
        if description:
            prop["description"] = description
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            try:
                prop["default"] = _jsonable(param.default)
            except Exception:
                pass
        properties[pname] = prop

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


_ARGS_SECTION_HEADERS = ("args:", "arguments:", "parameters:")
_ARG_LINE_RE = re.compile(r"^(\*{0,2}\w+)\s*(?:\(([^)]*)\))?:\s*(.+)$")


def _parse_docstring_args(doc: str | None) -> dict[str, str]:
    """Extract `{param: description}` from a Google-style ``Args:`` section.

    Only single-line `param: description` (or `param (type): description`)
    entries are captured; continuation lines are ignored. The section ends
    at the next top-level section header (``Returns:``, ``Raises:``, ...).
    """
    if not doc:
        return {}
    out: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        stripped = line.strip()
        if not in_args:
            if stripped.lower() in _ARGS_SECTION_HEADERS:
                in_args = True
            continue
        if not stripped:
            continue
        if line[:1] not in (" ", "\t"):
            break  # a new top-level section ends the Args block
        match = _ARG_LINE_RE.match(stripped)
        if match:
            out[match.group(1).lstrip("*")] = match.group(3).strip()
    return out


def _type_to_schema(t: Any) -> dict[str, Any]:
    # Unwrap stray Annotated (e.g. nested inside list[...]).
    if hasattr(t, "__metadata__"):
        t = t.__origin__
    if t is None or t is type(None):
        return {"type": "null"}
    if t is str:
        return {"type": "string"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is bool:
        return {"type": "boolean"}
    if t is list:
        return {"type": "array", "items": {}}
    if t is dict:
        return {"type": "object"}
    if t is Any:
        return {}

    origin = get_origin(t)
    args = get_args(t)

    # Optional / Union — both typing.Union and PEP 604 `X | Y`.
    # Optional[T] becomes {"anyOf": [<T schema>, {"type": "null"}]}
    # (proper JSON Schema; the OpenAPI-style `nullable` key is never emitted).
    if origin is typing.Union or origin is types.UnionType:
        return {"anyOf": [_type_to_schema(a) for a in args]}

    if origin in (list, typing.List):
        inner = args[0] if args else Any
        return {"type": "array", "items": _type_to_schema(inner)}

    if origin in (dict, typing.Dict):
        value_t = args[1] if len(args) >= 2 else Any
        return {"type": "object", "additionalProperties": _type_to_schema(value_t)}

    if origin in (tuple, typing.Tuple):
        return {"type": "array", "items": [_type_to_schema(a) for a in args]}

    if origin is typing.Literal:
        return {"enum": list(args)}

    # Pydantic v2
    try:
        import pydantic
        if isinstance(t, type) and issubclass(t, pydantic.BaseModel):
            return t.model_json_schema()
    except Exception:
        pass

    # Enum
    try:
        import enum as _enum
        if isinstance(t, type) and issubclass(t, _enum.Enum):
            return {"enum": [m.value for m in t]}
    except Exception:
        pass

    return {"type": "string"}
