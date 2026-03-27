"""Round 19: Tool description optimization tests.

Verifies descriptions are concise, cross-references valid, and no regressions.
"""

import json
import re
import pytest
from src.tools.registry import TOOLS, get_tool_definitions, TOOL_PACKS


class TestDescriptionOptimization:
    """Verify description token savings and quality."""

    def test_total_description_chars_reduced(self):
        """Total description chars should be under 11000 (base 10000 + agent tools)."""
        total = sum(len(t["description"]) for t in TOOLS)
        assert total < 11000, f"Description chars {total} should be < 11000"

    def test_total_json_chars_reduced(self):
        """Total JSON payload should be under 38000 (base 36000 + agent tools)."""
        total = sum(
            len(json.dumps({
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }))
            for t in TOOLS
        )
        assert total < 38000, f"JSON chars {total} should be < 38000"

    def test_no_empty_descriptions(self):
        """Every tool must have a non-empty description."""
        for t in TOOLS:
            assert t["description"].strip(), f"{t['name']} has empty description"

    def test_no_description_over_600_chars(self):
        """No single description should exceed 600 chars after optimization."""
        for t in TOOLS:
            length = len(t["description"])
            assert length <= 600, (
                f"{t['name']} description is {length} chars (max 600)"
            )

    def test_tool_count_unchanged(self):
        """Still 73 tools (67 base + 6 agent tools)."""
        assert len(TOOLS) == 73


class TestCrossReferences:
    """Verify all tool cross-references point to existing tools."""

    TOOL_NAMES = {t["name"] for t in TOOLS}

    def test_use_references_valid(self):
        """All 'use X' references in descriptions point to real tools."""
        for t in TOOLS:
            for m in re.finditer(r"\buse (\w+)", t["description"]):
                ref = m.group(1)
                if "_" in ref:  # Only check tool-like names
                    assert ref in self.TOOL_NAMES, (
                        f"{t['name']} references non-existent tool: {ref}"
                    )

    def test_see_references_valid(self):
        """All 'See X' references in descriptions point to real tools or paths."""
        for t in TOOLS:
            for m in re.finditer(r"\bSee (\w+)", t["description"]):
                ref = m.group(1)
                if ref.startswith("data"):
                    continue  # file path
                if "_" in ref:
                    assert ref in self.TOOL_NAMES, (
                        f"{t['name']} See-references non-existent tool: {ref}"
                    )

    def test_track_stop_check_references_valid(self):
        """All 'Track/stop/Check with X' references valid."""
        for t in TOOLS:
            for m in re.finditer(
                r"(?:Track|stop|Check) with (\w+)", t["description"]
            ):
                ref = m.group(1)
                assert ref in self.TOOL_NAMES, (
                    f"{t['name']} references non-existent: {ref}"
                )

    def test_no_stale_docker_git_references(self):
        """No descriptions reference removed docker/git pack tools."""
        removed = {
            "check_docker", "docker_logs", "docker_compose_action",
            "docker_compose_status", "docker_compose_logs", "docker_stats",
            "git_status", "git_log", "git_diff", "git_show",
            "git_pull", "git_commit", "git_push", "git_branch",
        }
        for t in TOOLS:
            desc_lower = t["description"].lower()
            for r in removed:
                assert r not in desc_lower, (
                    f"{t['name']} references removed tool {r}"
                )


class TestDescriptionQuality:
    """Verify description quality after optimization."""

    def test_critical_tools_still_informative(self):
        """Key tools still have informative descriptions."""
        tools = {t["name"]: t["description"] for t in TOOLS}

        # run_command — failure format preserved
        assert "Command failed" in tools["run_command"]

        # run_script — interpreter list preserved
        assert "python3" in tools["run_script"]
        assert "perl" in tools["run_script"]

        # create_skill — SkillContext API preserved
        assert "run_on_host" in tools["create_skill"]
        assert "execute_tool" in tools["create_skill"]
        assert "http_get" in tools["create_skill"]
        assert "search_knowledge" in tools["create_skill"]

        # claude_code — key guidance preserved
        assert "3+" in tools["claude_code"]
        assert "read_file" in tools["claude_code"]
        assert "run_command" in tools["claude_code"]

        # schedule_task — action types preserved
        assert "reminder" in tools["schedule_task"]
        assert "workflow" in tools["schedule_task"]
        assert "parse_time" in tools["schedule_task"]

        # delegate_task — features preserved
        assert "prev_output" in tools["delegate_task"]
        assert "store_as" in tools["delegate_task"]

    def test_search_knowledge_instructs_search_first(self):
        """search_knowledge still says to search here FIRST."""
        desc = next(
            t["description"] for t in TOOLS if t["name"] == "search_knowledge"
        )
        assert "FIRST" in desc

    def test_browser_tools_mention_javascript(self):
        """Browser tools still explain JS rendering advantage."""
        browser_tools = {
            t["name"]: t["description"]
            for t in TOOLS
            if t["name"].startswith("browser_")
        }
        assert "JavaScript" in browser_tools["browser_screenshot"]
        assert "JavaScript" in browser_tools["browser_read_page"]
        assert "JavaScript" in browser_tools["browser_read_table"]

    def test_memory_manage_mentions_scopes(self):
        """memory_manage still explains personal vs global scope."""
        desc = next(
            t["description"] for t in TOOLS if t["name"] == "memory_manage"
        )
        assert "personal" in desc
        assert "global" in desc

    def test_start_loop_mentions_all_tools(self):
        """start_loop description says all tools available."""
        desc = next(
            t["description"] for t in TOOLS if t["name"] == "start_loop"
        )
        assert "all tools" in desc

    def test_manage_process_limits_preserved(self):
        """manage_process still mentions limits."""
        desc = next(
            t["description"] for t in TOOLS if t["name"] == "manage_process"
        )
        assert "20" in desc
        assert "1hr" in desc or "1 hour" in desc


class TestParameterDescriptions:
    """Verify parameter descriptions are clear after optimization."""

    def test_run_command_host_has_example(self):
        """run_command still has host alias example."""
        tool = next(t for t in TOOLS if t["name"] == "run_command")
        host_desc = tool["input_schema"]["properties"]["host"]["description"]
        assert "myserver" in host_desc or "webhost" in host_desc

    def test_run_script_interpreter_concise(self):
        """run_script interpreter param is concise (list in main desc)."""
        tool = next(t for t in TOOLS if t["name"] == "run_script")
        interp_desc = tool["input_schema"]["properties"]["interpreter"]["description"]
        assert len(interp_desc) < 50, f"Interpreter desc too long: {interp_desc}"

    def test_schedule_task_workflow_steps_concise(self):
        """schedule_task workflow steps description is concise."""
        tool = next(t for t in TOOLS if t["name"] == "schedule_task")
        steps_desc = tool["input_schema"]["properties"]["steps"]["description"]
        assert len(steps_desc) < 40, f"Steps desc too long: {steps_desc}"

    def test_start_loop_goal_param_concise(self):
        """start_loop goal parameter is shorter than before."""
        tool = next(t for t in TOOLS if t["name"] == "start_loop")
        goal_desc = tool["input_schema"]["properties"]["goal"]["description"]
        assert len(goal_desc) < 200, f"Goal desc too long ({len(goal_desc)}): {goal_desc}"

    def test_all_required_params_have_descriptions(self):
        """Every required parameter has a description."""
        for t in TOOLS:
            schema = t["input_schema"]
            required = schema.get("required", [])
            props = schema.get("properties", {})
            for param in required:
                assert param in props, (
                    f"{t['name']} requires {param} but no property defined"
                )
                assert "description" in props[param], (
                    f"{t['name']}.{param} has no description"
                )
