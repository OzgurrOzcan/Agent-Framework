"""Tests for @tool decorator schema generation (T2.7).

Covers `Annotated[T, "description"]` metadata, Google-style docstring
``Args:`` parsing, their precedence, the Optional[T] -> anyOf fix, and
regressions for the pre-existing schema behaviours.
"""
import enum
import json
from typing import Annotated, Literal, Optional

import pydantic

from agent_framework.permissions.models import PermissionCategory
from agent_framework.tools.decorator import tool


def _props(t) -> dict:
    return t.metadata.parameters_schema["properties"]


# ---- Annotated[T, "description"] ----

def test_annotated_description():
    @tool(description="Search the web", permission=PermissionCategory.AUTO)
    def web_search(query: Annotated[str, "aranacak ifade"], max_results: int = 5) -> list:
        return []

    props = _props(web_search)
    assert props["query"]["description"] == "aranacak ifade"
    assert props["query"]["type"] == "string"
    assert props["max_results"]["type"] == "integer"
    assert props["max_results"]["default"] == 5
    assert web_search.metadata.parameters_schema["required"] == ["query"]


def test_annotated_first_string_metadata_wins():
    @tool(description="x", permission=PermissionCategory.AUTO)
    def f(n: Annotated[int, 42, "count of things", "second string"]) -> int:
        return n

    props = _props(f)
    assert props["n"]["type"] == "integer"
    assert props["n"]["description"] == "count of things"


def test_annotated_optional_combo():
    @tool(description="x", permission=PermissionCategory.AUTO)
    def f(x: Annotated[Optional[int], "maybe a number"] = None) -> None:
        return None

    prop = _props(f)["x"]
    assert prop["description"] == "maybe a number"
    assert prop["anyOf"] == [{"type": "integer"}, {"type": "null"}]
    assert "nullable" not in prop


# ---- docstring Args fallback ----

def test_docstring_args_parsed():
    @tool(permission=PermissionCategory.AUTO)
    def lookup(city: str, country: str = "TR") -> str:
        """Look up the weather.

        Args:
            city: Name of the city.
            country: ISO country code.

        Returns:
            A forecast string.
        """
        return ""

    props = _props(lookup)
    assert props["city"]["description"] == "Name of the city."
    assert props["country"]["description"] == "ISO country code."


def test_docstring_args_with_type_in_parens():
    @tool(permission=PermissionCategory.AUTO)
    def f(a: int) -> int:
        """Do something.

        Args:
            a (int): The first operand.
        """
        return a

    assert _props(f)["a"]["description"] == "The first operand."


def test_annotated_wins_over_docstring():
    @tool(permission=PermissionCategory.AUTO)
    def f(query: Annotated[str, "from annotated"]) -> str:
        """Search.

        Args:
            query: from docstring
        """
        return query

    assert _props(f)["query"]["description"] == "from annotated"


def test_no_description_sources_yields_no_description():
    @tool(description="d", permission=PermissionCategory.AUTO)
    def f(x: int) -> int:
        return x

    assert "description" not in _props(f)["x"]


# ---- Optional / Union -> anyOf (no `nullable`) ----

def test_optional_int_anyof():
    @tool(description="d", permission=PermissionCategory.AUTO)
    def f(count: Optional[int] = None) -> int:
        return count or 0

    prop = _props(f)["count"]
    assert prop["anyOf"] == [{"type": "integer"}, {"type": "null"}]
    assert "nullable" not in prop
    assert prop["default"] is None


def test_pipe_none_union_anyof():
    @tool(description="d", permission=PermissionCategory.AUTO)
    def f(name: str | None = None) -> str:
        return name or ""

    prop = _props(f)["name"]
    assert prop["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert "nullable" not in prop


def test_multi_union_with_none_keeps_null_member():
    @tool(description="d", permission=PermissionCategory.AUTO)
    def f(v: int | str | None = None) -> None:
        return None

    prop = _props(f)["v"]
    assert {"type": "null"} in prop["anyOf"]
    assert {"type": "integer"} in prop["anyOf"]
    assert {"type": "string"} in prop["anyOf"]


def test_nullable_key_never_emitted():
    @tool(description="d", permission=PermissionCategory.AUTO)
    def f(
        a: Optional[int] = None,
        b: Optional[list[str]] = None,
        c: int | str | None = None,
    ) -> None:
        return None

    assert "nullable" not in json.dumps(f.metadata.parameters_schema)


# ---- regressions: existing schema behaviours ----

def test_list_and_dict_schema_regression():
    @tool(description="d", permission=PermissionCategory.AUTO)
    def f(items: list[str], mapping: dict[str, int]) -> None:
        return None

    props = _props(f)
    assert props["items"] == {"type": "array", "items": {"type": "string"}}
    assert props["mapping"] == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }


def test_literal_and_enum_regression():
    class Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

    @tool(description="d", permission=PermissionCategory.AUTO)
    def f(mode: Literal["fast", "slow"], color: Color = Color.RED) -> None:
        return None

    props = _props(f)
    assert props["mode"] == {"enum": ["fast", "slow"]}
    assert props["color"]["enum"] == ["red", "blue"]


def test_pydantic_model_regression():
    class Query(pydantic.BaseModel):
        text: str
        limit: int = 10

    @tool(description="d", permission=PermissionCategory.AUTO)
    def f(q: Query) -> None:
        return None

    prop = _props(f)["q"]
    assert prop["type"] == "object"
    assert "text" in prop["properties"]
    assert "limit" in prop["properties"]


def test_required_and_defaults_regression():
    @tool(description="d", permission=PermissionCategory.AUTO)
    def f(a: int, b: str = "x") -> None:
        return None

    schema = f.metadata.parameters_schema
    assert schema["type"] == "object"
    assert schema["required"] == ["a"]
    assert schema["properties"]["b"]["default"] == "x"
    assert schema["additionalProperties"] is False
