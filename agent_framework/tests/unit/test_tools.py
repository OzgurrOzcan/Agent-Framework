"""Tests for tool decorator, registry, and executor."""
import asyncio
import pytest
from agent_framework.permissions.models import PermissionCategory
from agent_framework.tools.decorator import tool
from agent_framework.tools.executor import ToolExecutor
from agent_framework.tools.models import RetryableToolError, Tool
from agent_framework.tools.registry import ToolRegistry, to_registry
from agent_framework.core.message import ToolCallRequest


# ---- decorator ----

def test_tool_decorator_sync():
    @tool(description="Add two numbers", permission=PermissionCategory.AUTO)
    def add(a: int, b: int) -> int:
        return a + b

    assert isinstance(add, Tool)
    assert add.metadata.name == "add"
    assert add.metadata.permission == PermissionCategory.AUTO
    assert "a" in add.metadata.parameters_schema["properties"]
    assert "b" in add.metadata.parameters_schema["properties"]
    assert "a" in add.metadata.parameters_schema["required"]


def test_tool_decorator_async():
    @tool(description="Fetch data", permission=PermissionCategory.AUTO, timeout_seconds=5)
    async def fetch(url: str) -> dict:
        return {}

    assert fetch.metadata.is_async is True
    assert fetch.metadata.timeout_seconds == 5


def test_tool_schema_optional_param():
    from typing import Optional

    @tool(description="Optional test", permission=PermissionCategory.AUTO)
    def greet(name: str, greeting: Optional[str] = "Hello") -> str:
        return f"{greeting}, {name}"

    props = greet.metadata.parameters_schema["properties"]
    assert "name" in props
    assert "greeting" in props
    assert "name" in greet.metadata.parameters_schema["required"]
    assert "greeting" not in greet.metadata.parameters_schema.get("required", [])


# ---- registry ----

def test_registry_add_and_get():
    @tool(description="t1", permission=PermissionCategory.AUTO)
    def t1() -> str:
        return "t1"

    reg = ToolRegistry()
    reg.add(t1)
    assert reg.get("t1") is t1
    assert "t1" in reg
    assert len(reg) == 1


def test_registry_filter_by_permission():
    @tool(description="safe", permission=PermissionCategory.AUTO)
    def safe_tool() -> str:
        return ""

    @tool(description="risky", permission=PermissionCategory.ASK)
    def risky_tool() -> str:
        return ""

    reg = ToolRegistry([safe_tool, risky_tool])
    auto_only = reg.filter(permission=PermissionCategory.AUTO)
    assert "safe_tool" in auto_only
    assert "risky_tool" not in auto_only


def test_registry_add_operator():
    @tool(description="a", permission=PermissionCategory.AUTO)
    def tool_a() -> str:
        return ""

    @tool(description="b", permission=PermissionCategory.AUTO)
    def tool_b() -> str:
        return ""

    r1 = ToolRegistry([tool_a])
    r2 = ToolRegistry([tool_b])
    merged = r1 + r2
    assert "tool_a" in merged
    assert "tool_b" in merged


def test_to_registry_from_list():
    @tool(description="x", permission=PermissionCategory.AUTO)
    def tool_x() -> str:
        return ""

    reg = to_registry([tool_x])
    assert isinstance(reg, ToolRegistry)
    assert "tool_x" in reg


# ---- executor ----

@pytest.mark.asyncio
async def test_executor_runs_async_tool():
    @tool(description="echo", permission=PermissionCategory.AUTO)
    async def echo(msg: str) -> str:
        return f"echo:{msg}"

    executor = ToolExecutor()
    call = ToolCallRequest(id="c1", name="echo", arguments={"msg": "hello"})
    result = await executor.run(echo, call)
    assert result.is_error is False
    assert result.content == "echo:hello"
    assert result.tool_use_id == "c1"


@pytest.mark.asyncio
async def test_executor_runs_sync_tool():
    @tool(description="double", permission=PermissionCategory.AUTO)
    def double(n: int) -> int:
        return n * 2

    executor = ToolExecutor()
    call = ToolCallRequest(id="c2", name="double", arguments={"n": 7})
    result = await executor.run(double, call)
    assert result.is_error is False
    assert result.content == 14


@pytest.mark.asyncio
async def test_executor_handles_exception():
    @tool(description="boom", permission=PermissionCategory.AUTO)
    async def boom() -> str:
        raise ValueError("intentional error")

    executor = ToolExecutor()
    call = ToolCallRequest(id="c3", name="boom", arguments={})
    result = await executor.run(boom, call)
    assert result.is_error is True
    assert "intentional error" in result.content


@pytest.mark.asyncio
async def test_executor_retries_idempotent_tool():
    call_count = 0

    @tool(description="flaky", permission=PermissionCategory.AUTO, max_retries=2, idempotent=True)
    async def flaky() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RetryableToolError("not ready")
        return "ok"

    executor = ToolExecutor()
    call = ToolCallRequest(id="c4", name="flaky", arguments={})
    result = await executor.run(flaky, call)
    assert result.is_error is False
    assert result.content == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_executor_no_retry_non_idempotent():
    call_count = 0

    @tool(description="nonidempotent", permission=PermissionCategory.AUTO, max_retries=2, idempotent=False)
    async def non_idem() -> str:
        nonlocal call_count
        call_count += 1
        raise RetryableToolError("fail")

    executor = ToolExecutor()
    call = ToolCallRequest(id="c5", name="non_idem", arguments={})
    result = await executor.run(non_idem, call)
    assert result.is_error is True
    assert call_count == 1  # no retries because not idempotent


@pytest.mark.asyncio
async def test_executor_timeout():
    @tool(description="slow", permission=PermissionCategory.AUTO, timeout_seconds=0.05)
    async def slow_tool() -> str:
        await asyncio.sleep(10)
        return "never"

    executor = ToolExecutor()
    call = ToolCallRequest(id="c6", name="slow_tool", arguments={})
    result = await executor.run(slow_tool, call)
    assert result.is_error is True
    assert "timed out" in result.content.lower()


@pytest.mark.asyncio
async def test_executor_run_many_parallel():
    results_order = []

    @tool(description="ordered", permission=PermissionCategory.AUTO)
    async def ordered(n: int) -> int:
        await asyncio.sleep(0.01)
        results_order.append(n)
        return n * 10

    executor = ToolExecutor()
    calls = [
        (ordered, ToolCallRequest(id=f"c{i}", name="ordered", arguments={"n": i}))
        for i in range(3)
    ]
    results = await executor.run_many(calls)
    assert len(results) == 3
    assert [r.content for r in results] == [0, 10, 20]
