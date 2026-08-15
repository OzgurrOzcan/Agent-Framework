"""JSON-Schema preparation for native structured-output APIs.

Both providers' strict modes share the same core requirements:
  - every object node must set `additionalProperties: false`
  - (OpenAI strict) every property must be listed in `required`; originally
    optional properties are made nullable so the model can express "absent"
  - value-level constraints the constrained decoders don't support
    (minLength, minimum, ...) must be removed from the wire schema and
    enforced client-side against the ORIGINAL schema instead.

`prepare_for_strict()` returns a deep-copied wire schema; callers validate
the parsed result against the original schema (jsonschema/Pydantic) and use
`strip_optional_nulls()` first so nulls injected for originally-optional
fields don't break validation.
"""

from __future__ import annotations

import copy
from typing import Any

# Constraints not supported by provider-side constrained decoding.
UNSUPPORTED_CONSTRAINTS = frozenset(
    {
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "pattern",
        "minProperties",
        "maxProperties",
    }
)

_NESTED_LIST_KEYS = ("anyOf", "allOf", "oneOf", "prefixItems")
_NESTED_DICT_KEYS = ("$defs", "definitions", "properties", "patternProperties")


def prepare_for_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy `schema` and rewrite it for provider strict modes."""
    return _walk(copy.deepcopy(schema))


def _walk(node: Any) -> Any:
    if isinstance(node, list):
        return [_walk(n) for n in node]
    if not isinstance(node, dict):
        return node

    for key in UNSUPPORTED_CONSTRAINTS:
        node.pop(key, None)

    for key in _NESTED_LIST_KEYS:
        if isinstance(node.get(key), list):
            node[key] = [_walk(n) for n in node[key]]
    for key in _NESTED_DICT_KEYS:
        if isinstance(node.get(key), dict):
            node[key] = {k: _walk(v) for k, v in node[key].items()}
    if isinstance(node.get("items"), (dict, list)):
        node["items"] = _walk(node["items"])
    if isinstance(node.get("additionalProperties"), dict):
        node["additionalProperties"] = _walk(node["additionalProperties"])

    props = node.get("properties")
    if isinstance(props, dict):
        node["additionalProperties"] = False
        originally_required = set(node.get("required") or [])
        for name, prop_schema in props.items():
            if name not in originally_required:
                props[name] = _nullable(prop_schema)
        node["required"] = list(props.keys())
    elif node.get("type") == "object" and "additionalProperties" not in node:
        node["additionalProperties"] = False

    return node


def _nullable(prop_schema: Any) -> Any:
    """Wrap a property schema so `null` is an accepted value."""
    if not isinstance(prop_schema, dict):
        return prop_schema
    if prop_schema.get("type") == "null":
        return prop_schema
    any_of = prop_schema.get("anyOf")
    if isinstance(any_of, list):
        if any(isinstance(a, dict) and a.get("type") == "null" for a in any_of):
            return prop_schema
        prop_schema["anyOf"] = any_of + [{"type": "null"}]
        return prop_schema
    t = prop_schema.get("type")
    if isinstance(t, str):
        prop_schema["type"] = [t, "null"]
        return prop_schema
    if isinstance(t, list):
        if "null" not in t:
            prop_schema["type"] = t + ["null"]
        return prop_schema
    # enum/const/$ref etc. → wrap in anyOf.
    return {"anyOf": [prop_schema, {"type": "null"}]}


def strip_optional_nulls(data: Any, original_schema: dict[str, Any]) -> Any:
    """Remove `null` values for properties the ORIGINAL schema did not require.

    Strict modes force every property to be present; the model emits `null`
    for originally-optional fields. Dropping those keys restores the shape
    the original schema (and Pydantic defaults) expect.
    """
    if not isinstance(data, dict) or not isinstance(original_schema, dict):
        return data
    resolved = _resolve_ref(original_schema, original_schema)
    props = resolved.get("properties")
    if not isinstance(props, dict):
        return data
    required = set(resolved.get("required") or [])
    out: dict[str, Any] = {}
    for key, value in data.items():
        if value is None and key in props and key not in required:
            continue
        child_schema = props.get(key)
        if isinstance(child_schema, dict) and isinstance(value, dict):
            child = _resolve_ref(child_schema, original_schema)
            out[key] = strip_optional_nulls(value, {**child, "$defs": original_schema.get("$defs", {})})
        elif isinstance(child_schema, dict) and isinstance(value, list):
            items = child_schema.get("items")
            if isinstance(items, dict):
                item_schema = _resolve_ref(items, original_schema)
                out[key] = [
                    strip_optional_nulls(v, {**item_schema, "$defs": original_schema.get("$defs", {})})
                    if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                out[key] = value
        else:
            out[key] = value
    return out


def _resolve_ref(node: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return node
    target: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(target, dict) or part not in target:
            return node
        target = target[part]
    return target if isinstance(target, dict) else node
