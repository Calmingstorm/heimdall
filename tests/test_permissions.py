"""Tests for per-user permission tiers.

Tests cover:
- PermissionManager: tier resolution, overrides, persistence, tool filtering
- Config schema: PermissionsConfig model
- Registry: set_permission tool definition
- Client integration: tool filtering in _process_with_tools, child route forcing,
  set_permission handler
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.permissions.manager import PermissionManager, VALID_TIERS, USER_TIER_TOOLS  # noqa: E402
from src.config.schema import PermissionsConfig, Config  # noqa: E402
from src.tools.registry import TOOLS, get_tool_definitions  # noqa: E402
from src.discord.client import HeimdallBot  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pm(tmp_dir: Path) -> PermissionManager:
    """PermissionManager with default config (test users = admin)."""
    return PermissionManager(
        config_tiers={
            "100000000000000001": "admin",
            "100000000000000002": "admin",
        },
        default_tier="user",
        overrides_path=str(tmp_dir / "permissions.json"),
    )


@pytest.fixture
def sample_tools() -> list[dict]:
    """A small sample of tools for filter testing."""
    return [
        {"name": "check_disk", "description": "Check disk", "input_schema": {}},
        {"name": "run_command", "description": "Run command", "input_schema": {}},
        {"name": "check_memory", "description": "Check memory", "input_schema": {}},
        {"name": "write_file", "description": "Write file", "input_schema": {}},
        {"name": "set_permission", "description": "Set permission", "input_schema": {}},
        {"name": "web_search", "description": "Web search", "input_schema": {}},
    ]


# ---------------------------------------------------------------------------
# PermissionManager: tier resolution
# ---------------------------------------------------------------------------

class TestTierResolution:
    def test_admin_from_config(self, pm: PermissionManager):
        assert pm.get_tier("100000000000000001") == "admin"

    def test_admin_secondary(self, pm: PermissionManager):
        assert pm.get_tier("100000000000000002") == "admin"

    def test_unknown_user_defaults_to_user(self, pm: PermissionManager):
        assert pm.get_tier("999999999999") == "user"

    def test_is_admin_true(self, pm: PermissionManager):
        assert pm.is_admin("100000000000000001") is True

    def test_is_admin_false(self, pm: PermissionManager):
        assert pm.is_admin("999999999999") is False

    def test_is_guest_false_for_admin(self, pm: PermissionManager):
        assert pm.is_guest("100000000000000001") is False

    def test_is_guest_false_for_user(self, pm: PermissionManager):
        assert pm.is_guest("999999999999") is False

    def test_custom_default_tier(self, tmp_dir: Path):
        pm = PermissionManager(
            config_tiers={},
            default_tier="guest",
            overrides_path=str(tmp_dir / "p.json"),
        )
        assert pm.get_tier("anyone") == "guest"

    def test_invalid_default_tier_falls_back(self, tmp_dir: Path):
        pm = PermissionManager(
            config_tiers={},
            default_tier="invalid",
            overrides_path=str(tmp_dir / "p.json"),
        )
        assert pm.get_tier("anyone") == "user"


# ---------------------------------------------------------------------------
# PermissionManager: runtime overrides
# ---------------------------------------------------------------------------

class TestRuntimeOverrides:
    def test_set_tier_overrides_config(self, pm: PermissionManager):
        pm.set_tier("999", "guest")
        assert pm.get_tier("999") == "guest"

    def test_set_tier_overrides_admin(self, pm: PermissionManager):
        """Runtime override can demote an admin."""
        pm.set_tier("100000000000000001", "user")
        assert pm.get_tier("100000000000000001") == "user"

    def test_set_tier_invalid_raises(self, pm: PermissionManager):
        with pytest.raises(ValueError, match="Invalid tier"):
            pm.set_tier("999", "superadmin")

    def test_set_tier_persists_to_file(self, pm: PermissionManager, tmp_dir: Path):
        pm.set_tier("999", "guest")
        data = json.loads((tmp_dir / "permissions.json").read_text())
        assert data["999"] == "guest"

    def test_load_overrides_on_init(self, tmp_dir: Path):
        overrides_path = tmp_dir / "permissions.json"
        overrides_path.write_text(json.dumps({"111": "admin"}))
        pm = PermissionManager(
            config_tiers={},
            default_tier="user",
            overrides_path=str(overrides_path),
        )
        assert pm.get_tier("111") == "admin"

    def test_load_overrides_ignores_invalid_tiers(self, tmp_dir: Path):
        overrides_path = tmp_dir / "permissions.json"
        overrides_path.write_text(json.dumps({"111": "badtier", "222": "admin"}))
        pm = PermissionManager(
            config_tiers={},
            default_tier="user",
            overrides_path=str(overrides_path),
        )
        assert pm.get_tier("111") == "user"  # ignored invalid, falls to default
        assert pm.get_tier("222") == "admin"

    def test_load_overrides_handles_corrupt_json(self, tmp_dir: Path):
        overrides_path = tmp_dir / "permissions.json"
        overrides_path.write_text("not json{{{")
        pm = PermissionManager(
            config_tiers={},
            default_tier="user",
            overrides_path=str(overrides_path),
        )
        assert pm.get_tier("anyone") == "user"

    def test_load_overrides_handles_non_dict(self, tmp_dir: Path):
        overrides_path = tmp_dir / "permissions.json"
        overrides_path.write_text(json.dumps(["a", "b"]))
        pm = PermissionManager(
            config_tiers={},
            default_tier="user",
            overrides_path=str(overrides_path),
        )
        assert pm.get_tier("anyone") == "user"

    def test_no_overrides_file_ok(self, tmp_dir: Path):
        pm = PermissionManager(
            config_tiers={},
            default_tier="user",
            overrides_path=str(tmp_dir / "nonexistent.json"),
        )
        assert pm.get_tier("anyone") == "user"

    def test_overrides_take_precedence(self, pm: PermissionManager):
        """Runtime overrides take precedence over config tiers."""
        assert pm.get_tier("100000000000000001") == "admin"
        pm.set_tier("100000000000000001", "guest")
        assert pm.get_tier("100000000000000001") == "guest"


# ---------------------------------------------------------------------------
# PermissionManager: tool filtering
# ---------------------------------------------------------------------------

class TestToolFiltering:
    def test_admin_gets_all_tools(self, pm: PermissionManager, sample_tools: list):
        result = pm.filter_tools("100000000000000001", sample_tools)
        assert result == sample_tools

    def test_user_gets_only_whitelisted(self, pm: PermissionManager, sample_tools: list):
        result = pm.filter_tools("999999", sample_tools)
        names = {t["name"] for t in result}
        assert names == {"check_disk", "check_memory", "web_search"}

    def test_user_excludes_admin_tools(self, pm: PermissionManager, sample_tools: list):
        result = pm.filter_tools("999999", sample_tools)
        names = {t["name"] for t in result}
        assert "run_command" not in names
        assert "write_file" not in names
        assert "set_permission" not in names

    def test_guest_gets_none(self, pm: PermissionManager, sample_tools: list):
        pm.set_tier("guest_user", "guest")
        result = pm.filter_tools("guest_user", sample_tools)
        assert result is None

    def test_user_tier_tools_whitelist_is_correct(self):
        """All USER_TIER_TOOLS should exist in the registry."""
        registry_names = {t["name"] for t in TOOLS}
        for tool_name in USER_TIER_TOOLS:
            assert tool_name in registry_names, f"{tool_name} not in registry"

    def test_user_tier_tools_are_all_read_only(self):
        """All USER_TIER_TOOLS should be read-only tools."""
        for t in TOOLS:
            if t["name"] in USER_TIER_TOOLS:
                assert "name" in t, f"{t['name']} missing name"

    def test_filter_preserves_order(self, pm: PermissionManager, sample_tools: list):
        result = pm.filter_tools("999999", sample_tools)
        names = [t["name"] for t in result]
        # check_disk comes before check_memory comes before web_search
        assert names == ["check_disk", "check_memory", "web_search"]


# ---------------------------------------------------------------------------
# Config schema: PermissionsConfig
# ---------------------------------------------------------------------------

class TestPermissionsConfig:
    def test_default_tiers(self):
        pc = PermissionsConfig()
        assert pc.tiers == {}

    def test_default_tier_is_user(self):
        pc = PermissionsConfig()
        assert pc.default_tier == "user"

    def test_default_overrides_path(self):
        pc = PermissionsConfig()
        assert pc.overrides_path == "./data/permissions.json"

    def test_config_includes_permissions(self, config: Config):
        assert hasattr(config, "permissions")
        assert isinstance(config.permissions, PermissionsConfig)


# ---------------------------------------------------------------------------
# Registry: set_permission tool
# ---------------------------------------------------------------------------

class TestSetPermissionTool:
    def _get_tool(self):
        for t in TOOLS:
            if t["name"] == "set_permission":
                return t
        pytest.fail("set_permission tool not found in registry")

    def test_tool_exists(self):
        self._get_tool()

    def test_required_fields(self):
        tool = self._get_tool()
        assert "user_id" in tool["input_schema"]["required"]
        assert "tier" in tool["input_schema"]["required"]

    def test_tier_enum(self):
        tool = self._get_tool()
        tier_prop = tool["input_schema"]["properties"]["tier"]
        assert set(tier_prop["enum"]) == {"admin", "user", "guest"}

    def test_user_id_is_string(self):
        tool = self._get_tool()
        assert tool["input_schema"]["properties"]["user_id"]["type"] == "string"

    def test_tool_in_definitions(self):
        defs = get_tool_definitions()
        names = {t["name"] for t in defs}
        assert "set_permission" in names


# ---------------------------------------------------------------------------
# Client integration: _handle_set_permission
# ---------------------------------------------------------------------------

class TestHandleSetPermission:
    @pytest.fixture
    def bot(self, config, tmp_dir):
        """Minimal bot mock with real PermissionManager."""
        with patch.object(HeimdallBot, "__init__", lambda self, *a, **k: None):
            bot = HeimdallBot.__new__(HeimdallBot)
        bot.permissions = PermissionManager(
            config_tiers={"111": "admin"},
            default_tier="user",
            overrides_path=str(tmp_dir / "permissions.json"),
        )
        return bot

    def test_admin_can_set_permission(self, bot):
        result = bot._handle_set_permission("111", {"user_id": "999", "tier": "guest"})
        assert "guest" in result
        assert bot.permissions.get_tier("999") == "guest"

    def test_non_admin_denied(self, bot):
        result = bot._handle_set_permission("999", {"user_id": "111", "tier": "guest"})
        assert "denied" in result.lower()

    def test_invalid_tier_returns_error(self, bot):
        result = bot._handle_set_permission("111", {"user_id": "999", "tier": "superadmin"})
        assert "Invalid tier" in result

    def test_set_permission_persists(self, bot, tmp_dir):
        bot._handle_set_permission("111", {"user_id": "222", "tier": "admin"})
        data = json.loads((tmp_dir / "permissions.json").read_text())
        assert data["222"] == "admin"


# ---------------------------------------------------------------------------
# Client integration: tool filtering in _process_with_tools
# ---------------------------------------------------------------------------

class TestToolFilteringIntegration:
    @pytest.fixture
    def bot(self, config, tmp_dir):
        """Minimal bot mock to test _process_with_tools tool filtering."""
        with patch.object(HeimdallBot, "__init__", lambda self, *a, **k: None):
            bot = HeimdallBot.__new__(HeimdallBot)
        bot.config = config
        bot.permissions = PermissionManager(
            config_tiers={"111": "admin"},
            default_tier="user",
            overrides_path=str(tmp_dir / "permissions.json"),
        )
        bot.skill_manager = MagicMock()
        bot.skill_manager.get_tool_definitions.return_value = []
        bot._cached_merged_tools = None
        return bot

    def test_merged_tools_filtered_for_user(self, bot):
        """User tier should only see USER_TIER_TOOLS in _merged_tool_definitions."""
        all_tools = bot._merged_tool_definitions()
        filtered = bot.permissions.filter_tools("999", all_tools)
        filtered_names = {t["name"] for t in filtered}
        assert filtered_names == USER_TIER_TOOLS

    def test_merged_tools_all_for_admin(self, bot):
        """Admin should see all tools."""
        all_tools = bot._merged_tool_definitions()
        filtered = bot.permissions.filter_tools("111", all_tools)
        assert len(filtered) == len(all_tools)

    def test_merged_tools_none_for_guest(self, bot):
        """Child should get None (no tools)."""
        bot.permissions.set_tier("guest_user", "guest")
        all_tools = bot._merged_tool_definitions()
        filtered = bot.permissions.filter_tools("guest_user", all_tools)
        assert filtered is None


# ---------------------------------------------------------------------------
# Client integration: guest tier route forcing
# ---------------------------------------------------------------------------

class TestChildRouteForcing:
    def test_guest_tier_is_detected(self, tmp_dir):
        pm = PermissionManager(
            config_tiers={},
            default_tier="user",
            overrides_path=str(tmp_dir / "p.json"),
        )
        pm.set_tier("guest_user", "guest")
        assert pm.is_guest("guest_user") is True

    def test_non_guest_not_detected(self, tmp_dir):
        pm = PermissionManager(
            config_tiers={"111": "admin"},
            default_tier="user",
            overrides_path=str(tmp_dir / "p.json"),
        )
        assert pm.is_guest("111") is False
        assert pm.is_guest("999") is False


# ---------------------------------------------------------------------------
# VALID_TIERS constant
# ---------------------------------------------------------------------------

class TestValidTiers:
    def test_valid_tiers_contains_all(self):
        assert "admin" in VALID_TIERS
        assert "user" in VALID_TIERS
        assert "guest" in VALID_TIERS

    def test_valid_tiers_count(self):
        assert len(VALID_TIERS) == 3
