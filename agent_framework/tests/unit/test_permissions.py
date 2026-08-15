"""Tests for the permission engine."""
from agent_framework.permissions.config import AllowRule, DenyRule, PermissionConfig
from agent_framework.permissions.models import PermissionCategory, Verdict
from agent_framework.permissions.rule_engine import RuleEngine
from agent_framework.tools.models import ToolMetadata


def _make_tool(name: str, cat: PermissionCategory) -> ToolMetadata:
    return ToolMetadata(
        name=name, description="", parameters_schema={}, permission=cat
    )


def test_auto_tool_allows():
    engine = RuleEngine(PermissionConfig())
    tool = _make_tool("read_file", PermissionCategory.AUTO)
    d = engine.evaluate(tool, {})
    assert d.verdict == Verdict.ALLOW


def test_ask_tool_asks():
    engine = RuleEngine(PermissionConfig())
    tool = _make_tool("write_file", PermissionCategory.ASK)
    d = engine.evaluate(tool, {})
    assert d.verdict == Verdict.ASK


def test_deny_tool_denies():
    engine = RuleEngine(PermissionConfig())
    tool = _make_tool("drop_db", PermissionCategory.DENY)
    d = engine.evaluate(tool, {})
    assert d.verdict == Verdict.DENY


def test_deny_rule_overrides_auto():
    config = PermissionConfig(deny=[DenyRule(tool="rm_rf")])
    engine = RuleEngine(config)
    tool = _make_tool("rm_rf", PermissionCategory.AUTO)
    d = engine.evaluate(tool, {})
    assert d.verdict == Verdict.DENY


def test_deny_rule_glob_match():
    config = PermissionConfig(deny=[DenyRule(tool="delete_*")])
    engine = RuleEngine(config)
    tool = _make_tool("delete_user", PermissionCategory.AUTO)
    d = engine.evaluate(tool, {})
    assert d.verdict == Verdict.DENY


def test_allow_rule_overrides_ask():
    config = PermissionConfig(allow=[AllowRule(tool="write_file", arg_pattern=r"^path=/tmp/")])
    engine = RuleEngine(config)
    tool = _make_tool("write_file", PermissionCategory.ASK)
    d = engine.evaluate(tool, {"path": "/tmp/myfile.txt"})
    assert d.verdict == Verdict.ALLOW


def test_danger_pattern_escalates_auto_to_ask():
    config = PermissionConfig(danger_patterns=[r"--force"])
    engine = RuleEngine(config)
    tool = _make_tool("deploy", PermissionCategory.AUTO)
    d = engine.evaluate(tool, {"flags": "--force"})
    assert d.verdict == Verdict.ASK


def test_danger_pattern_escalates_ask_to_deny():
    config = PermissionConfig(danger_patterns=[r"DROP\s+TABLE"])
    engine = RuleEngine(config)
    tool = _make_tool("run_sql", PermissionCategory.ASK)
    d = engine.evaluate(tool, {"sql": "DROP TABLE users"})
    assert d.verdict == Verdict.DENY


def test_deny_rule_with_arg_pattern():
    config = PermissionConfig(deny=[DenyRule(tool="run_shell", arg_pattern=r"rm\s+-rf")])
    engine = RuleEngine(config)
    tool = _make_tool("run_shell", PermissionCategory.AUTO)
    # Dangerous args → deny
    d = engine.evaluate(tool, {"cmd": "rm -rf /"})
    assert d.verdict == Verdict.DENY
    # Safe args → pass through to AUTO → ALLOW
    d2 = engine.evaluate(tool, {"cmd": "ls -la"})
    assert d2.verdict == Verdict.ALLOW


def test_category_override_in_config():
    config = PermissionConfig(
        category_overrides={"write_file": PermissionCategory.AUTO}
    )
    engine = RuleEngine(config)
    tool = _make_tool("write_file", PermissionCategory.ASK)
    d = engine.evaluate(tool, {})
    assert d.verdict == Verdict.ALLOW
