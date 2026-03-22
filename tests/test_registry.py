"""Tests for tools/registry.py."""
from __future__ import annotations

from src.tools.registry import TOOLS, get_tool_definitions, requires_approval


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


class TestRequiresApproval:
    def test_read_only_tools(self):
        assert requires_approval("check_service") is False
        assert requires_approval("check_disk") is False
        assert requires_approval("check_memory") is False
        assert requires_approval("check_docker") is False
        assert requires_approval("check_logs") is False
        assert requires_approval("query_prometheus") is False
        assert requires_approval("read_file") is False
        assert requires_approval("post_file") is False
        assert requires_approval("search_history") is False
        assert requires_approval("memory_manage") is False
        assert requires_approval("search_audit") is False
        assert requires_approval("list_schedules") is False
        assert requires_approval("list_skills") is False

    def test_dangerous_tools(self):
        assert requires_approval("restart_service") is True
        assert requires_approval("run_ansible_playbook") is True
        assert requires_approval("run_command") is True
        assert requires_approval("write_file") is True
        assert requires_approval("purge_messages") is True
        assert requires_approval("schedule_task") is True
        assert requires_approval("delete_schedule") is True
        assert requires_approval("create_digest") is True
        assert requires_approval("create_skill") is True
        assert requires_approval("delete_skill") is True

    def test_unknown_tool_defaults_to_approval(self):
        assert requires_approval("unknown_tool_xyz") is True

    def test_edit_skill_no_approval(self):
        assert requires_approval("edit_skill") is False


class TestToolDescriptions:
    """Tests for tool description content and quality."""

    def test_no_requires_approval_text_in_descriptions(self):
        """No tool description should contain 'REQUIRES APPROVAL' text.

        The approval mechanism is enforced in code (_run_tool checks
        requires_approval boolean flag), not by the LLM. Including this
        text in descriptions confuses Codex into asking text-based
        permission instead of just calling the tool.
        """
        for tool in TOOLS:
            assert "REQUIRES APPROVAL" not in tool["description"], (
                f"Tool '{tool['name']}' description still contains "
                f"'REQUIRES APPROVAL' text — remove it (the requires_approval "
                f"boolean flag handles approval in code)"
            )

    def test_generate_file_mentions_code_and_scripts(self):
        """generate_file description should guide Codex to use it for code output."""
        gen_file = next(t for t in TOOLS if t["name"] == "generate_file")
        desc = gen_file["description"].lower()
        assert "script" in desc, "generate_file should mention scripts"
        assert "code" in desc, "generate_file should mention code"

    def test_approval_tools_still_have_requires_approval_flag(self):
        """Removing text from descriptions must NOT remove the boolean flag."""
        approval_tools = [
            "restart_service", "run_ansible_playbook", "run_command",
            "write_file", "purge_messages", "schedule_task", "delete_schedule",
            "create_digest", "create_skill", "delete_skill",
            "docker_compose_action", "git_pull", "git_commit", "git_push",
            "git_branch", "run_command_multi", "delegate_task",
            "delete_knowledge", "browser_click", "browser_fill",
            "browser_evaluate", "incus_exec", "incus_start", "incus_stop",
            "incus_restart", "incus_snapshot", "incus_launch", "incus_delete",
        ]
        for name in approval_tools:
            assert requires_approval(name) is True, (
                f"Tool '{name}' should still require approval (boolean flag)"
            )
