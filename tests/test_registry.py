"""Tests for tools/registry.py."""
from __future__ import annotations

from src.tools.registry import TOOLS, get_tool_definitions


class TestGetToolDefinitions:
    def test_returns_list(self):
        defs = get_tool_definitions()
        assert isinstance(defs, list)
        assert len(defs) > 0

    def test_strips_internal_fields(self):
        defs = get_tool_definitions()
        for d in defs:
            assert "requires_approval" not in d
            assert "name" in d
            assert "description" in d
            assert "input_schema" in d

    def test_all_tools_have_definitions(self):
        defs = get_tool_definitions()
        assert len(defs) == len(TOOLS)


class TestToolDescriptions:
    """Tests for tool description content and quality."""

    def test_no_requires_approval_text_in_descriptions(self):
        """No tool description should contain 'REQUIRES APPROVAL' text."""
        for tool in TOOLS:
            assert "REQUIRES APPROVAL" not in tool["description"], (
                f"Tool '{tool['name']}' description still contains "
                f"'REQUIRES APPROVAL' text"
            )

    def test_generate_file_mentions_code_and_scripts(self):
        """generate_file description should guide Codex to use it for code output."""
        gen_file = next(t for t in TOOLS if t["name"] == "generate_file")
        desc = gen_file["description"].lower()
        assert "script" in desc, "generate_file should mention scripts"
        assert "code" in desc, "generate_file should mention code"

    def test_no_approval_fields_in_tools(self):
        """No tool should have a requires_approval field anymore."""
        for tool in TOOLS:
            assert "requires_approval" not in tool, (
                f"Tool '{tool['name']}' still has requires_approval field"
            )
