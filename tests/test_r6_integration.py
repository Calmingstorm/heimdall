"""Tests for Round 6 cross-feature integration.

Covers:
- set_permission tool: tier name consistency (child→guest fix)
- Tool memory + permissions: hints filtered by user tier
- PermissionManager.allowed_tool_names()
- Skill operations: system prompt rebuild preserves user_id and channel
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.permissions.manager import PermissionManager, USER_TIER_TOOLS, VALID_TIERS  # noqa: E402
from src.tools.tool_memory import ToolMemory, extract_keywords  # noqa: E402
from src.tools.registry import TOOLS, get_tool_definitions  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pm(tmp_path: Path) -> PermissionManager:
    return PermissionManager(
        config_tiers={
            "100000000000000001": "admin",
            "100000000000000002": "admin",
        },
        default_tier="user",
        overrides_path=str(tmp_path / "permissions.json"),
    )


def _make_entry(query, tools, success=True):
    return {
        "query": query,
        "keywords": extract_keywords(query),
        "tools_used": tools,
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# set_permission tool description consistency
# ---------------------------------------------------------------------------


class TestSetPermissionDescription:
    """The set_permission tool description must match actual tier names."""

    def _get_tool(self):
        for t in get_tool_definitions():
            if t["name"] == "set_permission":
                return t
        pytest.fail("set_permission tool not found")

    async def test_description_says_guest_not_child(self):
        tool = self._get_tool()
        assert "guest" in tool["description"]
        assert "child" not in tool["description"]

    async def test_enum_matches_valid_tiers(self):
        tool = self._get_tool()
        enum = tool["input_schema"]["properties"]["tier"]["enum"]
        assert set(enum) == set(VALID_TIERS)

    async def test_description_lists_all_tiers(self):
        tool = self._get_tool()
        desc = tool["description"]
        for tier in VALID_TIERS:
            assert tier in desc, f"tier '{tier}' missing from description"


# ---------------------------------------------------------------------------
# PermissionManager.allowed_tool_names()
# ---------------------------------------------------------------------------


class TestAllowedToolNames:
    async def test_admin_returns_none(self, pm):
        """Admin gets None (no restriction)."""
        result = pm.allowed_tool_names("100000000000000001")
        assert result is None

    async def test_user_returns_user_tier_tools(self, pm):
        """User tier gets the USER_TIER_TOOLS set."""
        result = pm.allowed_tool_names("999")
        assert result == set(USER_TIER_TOOLS)

    async def test_guest_returns_empty_set(self, pm):
        pm.set_tier("123", "guest")
        result = pm.allowed_tool_names("123")
        assert result == set()

    async def test_user_set_is_frozen(self, pm):
        """Returned set for user tier can't accidentally mutate USER_TIER_TOOLS."""
        result = pm.allowed_tool_names("999")
        result.add("dangerous_tool")
        assert "dangerous_tool" not in USER_TIER_TOOLS


# ---------------------------------------------------------------------------
# Tool memory: find_patterns with allowed_tools filter
# ---------------------------------------------------------------------------


class TestFindPatternsAllowedTools:
    async def test_no_filter_returns_all(self, tmp_path):
        """Without allowed_tools, all matching patterns are returned."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            _make_entry("check disk space server", ["check_disk", "run_command"]),
        ]
        results = await tm.find_patterns("check disk space server")
        assert len(results) == 1

    async def test_allowed_tools_passes_matching(self, tmp_path):
        """Patterns where all tools are allowed are returned."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            _make_entry("check disk space server", ["check_disk", "query_prometheus"]),
        ]
        allowed = {"check_disk", "query_prometheus", "check_memory"}
        results = await tm.find_patterns("check disk space server", allowed_tools=allowed)
        assert len(results) == 1

    async def test_allowed_tools_filters_out_admin_tools(self, tmp_path):
        """Patterns with admin-only tools are excluded for non-admin users."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            _make_entry("check disk space server", ["check_disk", "run_command"]),
        ]
        # run_command is admin-only, not in USER_TIER_TOOLS
        allowed = set(USER_TIER_TOOLS)
        results = await tm.find_patterns("check disk space server", allowed_tools=allowed)
        assert len(results) == 0

    async def test_allowed_tools_partial_match(self, tmp_path):
        """If any tool in the sequence is disallowed, the whole pattern is excluded."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            _make_entry("check disk server logs", ["check_disk", "check_logs", "write_file"]),
        ]
        allowed = {"check_disk", "check_logs"}  # write_file not allowed
        results = await tm.find_patterns("check disk server logs", allowed_tools=allowed)
        assert len(results) == 0

    async def test_empty_allowed_tools_returns_nothing(self, tmp_path):
        """Guest tier (empty set) should get no patterns."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            _make_entry("check disk space server", ["check_disk", "query_prometheus"]),
        ]
        results = await tm.find_patterns("check disk space server", allowed_tools=set())
        assert len(results) == 0

    async def test_mixed_allowed_and_disallowed(self, tmp_path):
        """Only patterns with fully-allowed tool sets are returned."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            _make_entry("check disk space server", ["check_disk", "query_prometheus"]),
            _make_entry("check disk space restart", ["check_disk", "run_command"]),
        ]
        allowed = {"check_disk", "query_prometheus", "check_memory"}
        results = await tm.find_patterns("check disk space server", allowed_tools=allowed)
        assert len(results) == 1
        assert results[0]["tools_used"] == ["check_disk", "query_prometheus"]

    async def test_none_allowed_tools_means_no_filter(self, tmp_path):
        """None means admin — no filtering applied."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            _make_entry("check disk space server", ["check_disk", "run_command"]),
        ]
        results = await tm.find_patterns("check disk space server", allowed_tools=None)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Tool memory: format_hints with allowed_tools filter
# ---------------------------------------------------------------------------


class TestFormatHintsAllowedTools:
    async def test_format_hints_with_allowed_tools(self, tmp_path):
        """format_hints passes allowed_tools through to find_patterns."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            _make_entry("check disk space server", ["check_disk", "query_prometheus"]),
        ]
        allowed = {"check_disk", "query_prometheus"}
        hints = await tm.format_hints("check disk space server", allowed_tools=allowed)
        assert "Tool Use Patterns" in hints

    async def test_format_hints_filters_admin_tools(self, tmp_path):
        """format_hints excludes patterns with admin-only tools for user tier."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            _make_entry("check disk space server", ["check_disk", "run_command"]),
        ]
        allowed = set(USER_TIER_TOOLS)
        hints = await tm.format_hints("check disk space server", allowed_tools=allowed)
        assert hints == ""

    async def test_format_hints_no_filter_for_admin(self, tmp_path):
        """Admin (None) sees all hints including admin-only tools."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            _make_entry("check disk space server", ["check_disk", "run_command"]),
        ]
        hints = await tm.format_hints("check disk space server", allowed_tools=None)
        assert "run_command" in hints

    async def test_format_hints_empty_for_guest(self, tmp_path):
        """Guest (empty set) gets no hints."""
        tm = ToolMemory(str(tmp_path / "tm.json"))
        tm._entries = [
            _make_entry("check disk space server", ["check_disk", "query_prometheus"]),
        ]
        hints = await tm.format_hints("check disk space server", allowed_tools=set())
        assert hints == ""


# ---------------------------------------------------------------------------
# Skill operations: system prompt rebuild preserves user_id and channel
# ---------------------------------------------------------------------------


class TestSkillPromptRebuildUserContext:
    """After create/edit/delete skill, _build_system_prompt must receive
    user_id and channel so per-user memory, learned context, recent actions,
    and channel personality are preserved in the rebuilt prompt."""

    def _make_bot_stub(self, memory=None):
        """Create a minimal bot stub with _build_system_prompt bound."""
        stub = MagicMock()
        host_mock = MagicMock()
        host_mock.ssh_user = "root"
        host_mock.address = "10.0.0.2"
        stub.config.tools.hosts = {"desktop": host_mock}
        stub.config.tools.allowed_services = ["nginx"]
        stub.config.tools.allowed_playbooks = ["update.yml"]
        stub.context_loader.context = "Context."
        stub.voice_manager = None
        stub.tool_executor._load_memory_for_user = MagicMock(
            return_value=memory or {}
        )
        stub.reflector = MagicMock()
        stub.reflector.get_prompt_section = MagicMock(return_value="")
        stub.skill_manager = MagicMock()
        stub.skill_manager.list_skills = MagicMock(return_value=[])
        stub.config.timezone = "UTC"
        stub._recent_actions = {}
        stub._recent_actions_max = 10
        stub._recent_actions_expiry = 3600
        stub.permissions = MagicMock()
        stub.permissions.allowed_tool_names = MagicMock(return_value=None)
        stub.tool_memory = MagicMock()
        stub.tool_memory.format_hints = MagicMock(return_value="")

        # Cache attributes for prompt caching helpers
        stub._cached_hosts = None
        stub._cached_skills_text = None
        stub._memory_cache = {}
        stub._memory_cache_ttl = 60.0
        stub._reflector_cache = {}
        stub._reflector_cache_ttl = 60.0

        from src.discord.client import HeimdallBot
        stub._build_system_prompt = HeimdallBot._build_system_prompt.__get__(stub)
        stub._get_cached_hosts = HeimdallBot._get_cached_hosts.__get__(stub)
        stub._get_cached_skills_text = HeimdallBot._get_cached_skills_text.__get__(stub)
        stub._get_cached_memory = HeimdallBot._get_cached_memory.__get__(stub)
        stub._get_cached_reflector = HeimdallBot._get_cached_reflector.__get__(stub)
        return stub

    async def test_create_skill_passes_user_id(self):
        """create_skill prompt rebuild should pass user_id."""
        stub = self._make_bot_stub(memory={"birthday": "Feb 7"})
        # Simulate what _process_with_tools does after skill creation
        channel = MagicMock()
        channel.id = 12345
        channel.topic = None
        user_id = "100000000000000001"

        prompt = stub._build_system_prompt(channel=channel, user_id=user_id)
        stub.tool_executor._load_memory_for_user.assert_called_with(user_id)
        assert "birthday" in prompt

    async def test_create_skill_passes_channel(self):
        """create_skill prompt rebuild should pass channel for recent actions."""
        stub = self._make_bot_stub()
        channel = MagicMock()
        channel.id = 12345
        channel.topic = "Be helpful and friendly"
        user_id = "100000000000000001"

        prompt = stub._build_system_prompt(channel=channel, user_id=user_id)
        assert "Be helpful and friendly" in prompt

    async def test_no_user_id_loses_personal_memory(self):
        """Without user_id, personal memory is not injected (the old bug)."""
        stub = self._make_bot_stub(memory={"birthday": "Feb 7"})
        # Call without user_id — the old code path
        stub._build_system_prompt()
        stub.tool_executor._load_memory_for_user.assert_called_with(None)

    async def test_no_channel_loses_personality(self):
        """Without channel, channel personality is not injected (the old bug)."""
        stub = self._make_bot_stub()
        channel = MagicMock()
        channel.id = 12345
        channel.topic = "Be a pirate"

        # With channel — personality injected
        prompt_with = stub._build_system_prompt(channel=channel, user_id="42")
        assert "Be a pirate" in prompt_with

        # Without channel — personality missing
        prompt_without = stub._build_system_prompt()
        assert "Be a pirate" not in prompt_without

    async def test_skill_rebuild_includes_reflector_user_context(self):
        """Prompt rebuild after skill op should pass user_id to reflector."""
        stub = self._make_bot_stub()
        stub.reflector.get_prompt_section = MagicMock(
            return_value="## Learned\n- User prefers verbose output"
        )
        user_id = "100000000000000001"
        prompt = stub._build_system_prompt(user_id=user_id)
        stub.reflector.get_prompt_section.assert_called_with(user_id=user_id)
        assert "verbose output" in prompt
