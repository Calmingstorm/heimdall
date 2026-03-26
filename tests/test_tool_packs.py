"""Tests for the Tool Pack system (registry.py).

Covers: pack filtering, backward compat, unknown packs, skill bypass,
pack completeness, and combined packs.
"""
from __future__ import annotations

import pytest

from src.tools.registry import (
    TOOL_PACKS,
    TOOLS,
    _ALL_PACK_TOOLS,
    get_pack_tool_names,
    get_tool_definitions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_TOOL_NAMES = {t["name"] for t in TOOLS}
CORE_TOOL_NAMES = ALL_TOOL_NAMES - _ALL_PACK_TOOLS


# ---------------------------------------------------------------------------
# Backward compatibility — no packs = all tools
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Empty or absent tool_packs returns all tools."""

    def test_empty_packs_returns_all_tools(self):
        """Passing enabled_packs=[] returns every tool."""
        result = get_tool_definitions(enabled_packs=[])
        names = {t["name"] for t in result}
        assert names == ALL_TOOL_NAMES

    def test_none_packs_returns_all_tools(self):
        """Passing enabled_packs=None returns every tool."""
        result = get_tool_definitions(enabled_packs=None)
        names = {t["name"] for t in result}
        assert names == ALL_TOOL_NAMES

    def test_no_arg_returns_all_tools(self):
        """Default call (no argument) returns every tool."""
        result = get_tool_definitions()
        names = {t["name"] for t in result}
        assert names == ALL_TOOL_NAMES

    def test_backward_compat_no_config_field(self):
        """Config with no tool_packs field defaults to empty list → all tools."""
        from src.config.schema import ToolsConfig

        cfg = ToolsConfig()
        assert cfg.tool_packs == []
        # Empty list → all tools
        result = get_tool_definitions(enabled_packs=cfg.tool_packs)
        assert len(result) == len(TOOLS)


# ---------------------------------------------------------------------------
# Single pack filtering
# ---------------------------------------------------------------------------


class TestSinglePack:
    """Enabling a single pack returns core + that pack's tools."""

    def test_single_pack_returns_core_plus_pack(self):
        """Systemd pack returns core tools + systemd tools only."""
        result = get_tool_definitions(enabled_packs=["systemd"])
        names = {t["name"] for t in result}

        systemd_tools = set(TOOL_PACKS["systemd"])
        expected = CORE_TOOL_NAMES | systemd_tools
        assert names == expected

    def test_single_pack_excludes_other_packs(self):
        """Enabling only 'systemd' excludes incus, prometheus, etc."""
        result = get_tool_definitions(enabled_packs=["systemd"])
        names = {t["name"] for t in result}

        for pack_name, pack_tools in TOOL_PACKS.items():
            if pack_name == "systemd":
                continue
            for tool_name in pack_tools:
                if tool_name not in CORE_TOOL_NAMES:
                    assert tool_name not in names, (
                        f"{tool_name} from pack '{pack_name}' should not be in systemd-only results"
                    )

    def test_each_pack_works_alone(self):
        """Every defined pack can be enabled individually."""
        for pack_name in TOOL_PACKS:
            result = get_tool_definitions(enabled_packs=[pack_name])
            names = {t["name"] for t in result}
            pack_tools = set(TOOL_PACKS[pack_name])
            # Core tools always present
            assert CORE_TOOL_NAMES <= names
            # Pack tools present
            assert pack_tools <= names


# ---------------------------------------------------------------------------
# Multiple packs
# ---------------------------------------------------------------------------


class TestMultiplePacks:
    """Enabling multiple packs combines them."""

    def test_multiple_packs_combine(self):
        """Systemd + prometheus returns core + systemd tools + prometheus tools."""
        result = get_tool_definitions(enabled_packs=["systemd", "prometheus"])
        names = {t["name"] for t in result}

        expected = CORE_TOOL_NAMES | set(TOOL_PACKS["systemd"]) | set(TOOL_PACKS["prometheus"])
        assert names == expected

    def test_all_packs_equals_all_tools(self):
        """Enabling every pack returns the same as no packs."""
        all_packs = list(TOOL_PACKS.keys())
        result = get_tool_definitions(enabled_packs=all_packs)
        names = {t["name"] for t in result}
        assert names == ALL_TOOL_NAMES

    def test_duplicate_packs_no_duplicates(self):
        """Passing the same pack twice doesn't duplicate tools."""
        result = get_tool_definitions(enabled_packs=["systemd", "systemd"])
        names = [t["name"] for t in result]
        assert len(names) == len(set(names)), "Duplicate tool names found"


# ---------------------------------------------------------------------------
# Unknown packs
# ---------------------------------------------------------------------------


class TestUnknownPack:
    """Unknown pack names are silently ignored."""

    def test_unknown_pack_ignored(self):
        """An unrecognized pack name does not crash, returns core tools."""
        result = get_tool_definitions(enabled_packs=["nonexistent_pack"])
        names = {t["name"] for t in result}
        assert names == CORE_TOOL_NAMES

    def test_mixed_known_unknown_packs(self):
        """Known packs work even when mixed with unknown ones."""
        result = get_tool_definitions(enabled_packs=["systemd", "fake_pack"])
        names = {t["name"] for t in result}
        expected = CORE_TOOL_NAMES | set(TOOL_PACKS["systemd"])
        assert names == expected


# ---------------------------------------------------------------------------
# Skill tools bypass packs
# ---------------------------------------------------------------------------


class TestSkillToolsBypassPacks:
    """Skill-defined tools are not affected by pack filtering.

    Skills are merged in _merged_tool_definitions (client.py), not in
    get_tool_definitions. So get_tool_definitions only returns built-in tools.
    Skills are always added regardless of packs.
    """

    def test_skill_tools_bypass_packs(self):
        """get_tool_definitions never removes a core tool even with packs enabled.

        Skills are added separately by the client, so they are always present.
        This test verifies that core tools (like run_command, read_file, etc.)
        are never filtered out by pack filtering.
        """
        for pack_name in TOOL_PACKS:
            result = get_tool_definitions(enabled_packs=[pack_name])
            names = {t["name"] for t in result}
            # Core tools must always be present
            assert CORE_TOOL_NAMES <= names


# ---------------------------------------------------------------------------
# Pack completeness — no drift
# ---------------------------------------------------------------------------


class TestPackCompleteness:
    """Verify pack tool names actually exist in TOOLS."""

    def test_pack_tool_names_complete(self):
        """Every tool name in TOOL_PACKS exists in TOOLS. No drift."""
        for pack_name, pack_tools in TOOL_PACKS.items():
            for tool_name in pack_tools:
                assert tool_name in ALL_TOOL_NAMES, (
                    f"Tool '{tool_name}' in pack '{pack_name}' does not exist in TOOLS"
                )

    def test_all_pack_tools_set_matches(self):
        """_ALL_PACK_TOOLS matches the union of all pack tool lists."""
        expected = set()
        for tools in TOOL_PACKS.values():
            expected.update(tools)
        assert _ALL_PACK_TOOLS == expected

    def test_no_tool_in_multiple_packs(self):
        """No tool appears in more than one pack."""
        seen: dict[str, str] = {}
        for pack_name, pack_tools in TOOL_PACKS.items():
            for tool_name in pack_tools:
                assert tool_name not in seen, (
                    f"Tool '{tool_name}' in both '{seen[tool_name]}' and '{pack_name}'"
                )
                seen[tool_name] = pack_name


# ---------------------------------------------------------------------------
# get_pack_tool_names helper
# ---------------------------------------------------------------------------


class TestGetPackToolNames:
    """Test the get_pack_tool_names helper function."""

    def test_single_pack(self):
        names = get_pack_tool_names(["systemd"])
        assert names == set(TOOL_PACKS["systemd"])

    def test_multiple_packs(self):
        names = get_pack_tool_names(["systemd", "prometheus"])
        assert names == set(TOOL_PACKS["systemd"]) | set(TOOL_PACKS["prometheus"])

    def test_empty_list(self):
        names = get_pack_tool_names([])
        assert names == set()

    def test_unknown_pack_returns_empty(self):
        names = get_pack_tool_names(["nonexistent"])
        assert names == set()

    def test_mixed_known_unknown(self):
        names = get_pack_tool_names(["systemd", "nonexistent"])
        assert names == set(TOOL_PACKS["systemd"])


# ---------------------------------------------------------------------------
# Tool definition format
# ---------------------------------------------------------------------------


class TestToolDefinitionFormat:
    """Verify tool definitions have the required fields."""

    def test_all_tools_have_required_fields(self):
        """Every tool has name, description, and input_schema."""
        result = get_tool_definitions()
        for tool in result:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert isinstance(tool["name"], str)
            assert isinstance(tool["description"], str)
            assert isinstance(tool["input_schema"], dict)

    def test_filtered_tools_have_required_fields(self):
        """Filtered tool definitions retain all required fields."""
        result = get_tool_definitions(enabled_packs=["systemd"])
        for tool in result:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
