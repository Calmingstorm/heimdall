"""Session 4/10: Connect and document the 3-layer instruction architecture.

Covers:
- Defense mechanisms documented in architecture.md (fabrication, hedging, premature failure)
- Tool lifecycle documentation (background tasks, scheduling, knowledge base)
- Cross-reference bidirectionality in tool descriptions
- Redundancy de-duplication between system prompt and architecture.md
- Architecture.md content integrity
"""
from __future__ import annotations

from pathlib import Path

from src.llm.system_prompt import (
    build_system_prompt,
    SYSTEM_PROMPT_TEMPLATE,
)
from src.tools.registry import TOOLS

_ARCH_PATH = Path(__file__).parent.parent / "data" / "context" / "architecture.md"
_ARCH_CONTENT = _ARCH_PATH.read_text()


def _build_prompt() -> str:
    return build_system_prompt(
        context=_ARCH_CONTENT, hosts={}, services=[], playbooks=[],
    )


def _tool_desc(name: str) -> str:
    for t in TOOLS:
        if t["name"] == name:
            return t["description"]
    raise ValueError(f"Tool {name!r} not found")


# ---------------------------------------------------------------------------
# Defense Mechanisms in architecture.md
# ---------------------------------------------------------------------------

class TestDefenseMechanismsDocumented:
    """architecture.md documents the three runtime detection patterns."""

    def test_fabrication_detection_documented(self):
        assert "fabrication detection" in _ARCH_CONTENT.lower()

    def test_hedging_detection_documented(self):
        assert "hedging detection" in _ARCH_CONTENT.lower()

    def test_premature_failure_detection_documented(self):
        assert "premature failure detection" in _ARCH_CONTENT.lower()

    def test_defense_section_exists(self):
        assert "## Defense Mechanisms" in _ARCH_CONTENT

    def test_fabrication_teaches_always_call_tool(self):
        assert "call the tool" in _ARCH_CONTENT.lower()

    def test_hedging_teaches_execute_immediately(self):
        assert "execute immediately" in _ARCH_CONTENT.lower()

    def test_premature_failure_references_rule_9(self):
        assert "rule 9" in _ARCH_CONTENT.lower()

    def test_first_iteration_documented(self):
        assert "first" in _ARCH_CONTENT.lower() and "iteration" in _ARCH_CONTENT.lower()

    def test_defense_in_built_prompt(self):
        prompt = _build_prompt()
        assert "defense mechanisms" in prompt.lower()


# ---------------------------------------------------------------------------
# Tool Lifecycle Documentation
# ---------------------------------------------------------------------------

class TestToolLifecycleDocs:
    """architecture.md documents lifecycle for complex tool chains."""

    def test_delegate_task_lifecycle(self):
        assert "delegate_task" in _ARCH_CONTENT
        assert "list_tasks" in _ARCH_CONTENT
        assert "cancel_task" in _ARCH_CONTENT

    def test_schedule_lifecycle(self):
        assert "schedule_task" in _ARCH_CONTENT
        assert "list_schedules" in _ARCH_CONTENT
        assert "delete_schedule" in _ARCH_CONTENT

    def test_knowledge_lifecycle(self):
        assert "search_knowledge" in _ARCH_CONTENT
        assert "ingest_document" in _ARCH_CONTENT
        assert "list_knowledge" in _ARCH_CONTENT
        assert "delete_knowledge" in _ARCH_CONTENT

    def test_lifecycle_keyword_present(self):
        assert "lifecycle" in _ARCH_CONTENT.lower()

    def test_delegate_returns_task_id(self):
        assert "task ID" in _ARCH_CONTENT or "task_id" in _ARCH_CONTENT


# ---------------------------------------------------------------------------
# Cross-Reference Bidirectionality
# ---------------------------------------------------------------------------

class TestCrossReferenceBidirectionality:
    """Tool descriptions cross-reference in both directions."""

    def test_search_knowledge_to_list_knowledge(self):
        assert "list_knowledge" in _tool_desc("search_knowledge")

    def test_delete_knowledge_to_list_knowledge(self):
        assert "list_knowledge" in _tool_desc("delete_knowledge")

    def test_delete_schedule_to_list_schedules(self):
        assert "list_schedules" in _tool_desc("delete_schedule")

    def test_parse_time_to_schedule_task(self):
        assert "schedule_task" in _tool_desc("parse_time")

    def test_read_file_to_claude_code(self):
        assert "claude_code" in _tool_desc("read_file")

    def test_incus_info_to_incus_list(self):
        assert "incus_list" in _tool_desc("incus_info")

    def test_incus_exec_to_incus_logs(self):
        assert "incus_logs" in _tool_desc("incus_exec")

    def test_incus_delete_to_incus_launch(self):
        assert "incus_launch" in _tool_desc("incus_delete")


# ---------------------------------------------------------------------------
# Redundancy De-duplication
# ---------------------------------------------------------------------------

class TestRedundancyDedup:
    """Common Patterns section no longer duplicates Rule 9's failure guidance."""

    def test_rule_9_in_system_prompt(self):
        assert "exhaust ALL alternatives" in SYSTEM_PROMPT_TEMPLATE

    def test_common_patterns_no_duplicate_failure_guidance(self):
        # architecture.md should NOT contain the same "exhaust ALL alternatives"
        # phrasing as Rule 9 — defense section references Rule 9 instead
        common_section = _ARCH_CONTENT.split("## Common Patterns")[1].split("##")[0]
        assert "exhaust ALL alternatives" not in common_section

    def test_defense_section_references_rule_9(self):
        defense_section = _ARCH_CONTENT.split("## Defense Mechanisms")[1]
        assert "Rule 9" in defense_section


# ---------------------------------------------------------------------------
# Architecture.md Content Integrity
# ---------------------------------------------------------------------------

class TestArchitectureMdIntegrity:
    """architecture.md has all expected sections and no stale content."""

    def test_has_claude_code_delegation(self):
        assert "## Claude Code Delegation" in _ARCH_CONTENT

    def test_has_knowledge_base(self):
        assert "## Knowledge Base" in _ARCH_CONTENT

    def test_has_background_tasks(self):
        assert "## Background Tasks" in _ARCH_CONTENT

    def test_has_reminders_and_scheduling(self):
        assert "## Reminders and Scheduling" in _ARCH_CONTENT

    def test_has_common_patterns(self):
        assert "## Common Patterns" in _ARCH_CONTENT

    def test_has_defense_mechanisms(self):
        assert "## Defense Mechanisms" in _ARCH_CONTENT

    def test_no_approval_references(self):
        assert "approval" not in _ARCH_CONTENT.lower()

    def test_no_classifier_references(self):
        assert "classifier" not in _ARCH_CONTENT.lower()

    def test_no_permission_language(self):
        # Defense section is allowed to mention permission in context of "ask permission"
        # but should not contain "would you like" or "may I" as standalone phrases
        assert "may i" not in _ARCH_CONTENT.lower()


# ---------------------------------------------------------------------------
# Cross-Layer Consistency
# ---------------------------------------------------------------------------

class TestCrossLayerConsistency:
    """System prompt, architecture.md, and tool descriptions are consistent."""

    def test_claude_code_deep_reasoning_in_all_layers(self):
        assert "deep reasoning" in _ARCH_CONTENT
        assert "deep reasoning" in _tool_desc("claude_code").lower()

    def test_run_script_in_architecture(self):
        assert "run_script" in _ARCH_CONTENT

    def test_post_file_in_architecture(self):
        assert "post_file" in _ARCH_CONTENT

    def test_prompt_under_5000_chars(self):
        assert len(SYSTEM_PROMPT_TEMPLATE) < 5000

    def test_architecture_loaded_in_prompt(self):
        prompt = _build_prompt()
        # architecture.md sections should appear in built prompt via {context}
        assert "Claude Code Delegation" in prompt
        assert "Defense Mechanisms" in prompt

    def test_tool_descriptions_action_first(self):
        """All tool descriptions start with action verbs (OpenClaw pattern)."""
        action_starts = (
            "Returns", "Runs", "Lists", "Fetches", "Deletes", "Removes",
            "Creates", "Writes", "Searches", "Navigates", "Executes",
            "Pushes", "Pulls", "Stages", "Starts", "Stops", "Restarts",
            "Manages", "Sets", "Persistent", "Converts", "Deep",
            "Saves", "Schedules", "Replaces", "Cancels", "Ingests",
            "Launches", "Extracts", "Adds", "Generates", "Analyzes",
            "Sends", "Takes", "Reads", "Evaluates",
        )
        for tool in TOOLS:
            desc = tool["description"]
            # Handle multi-line descriptions (tuples joined by Python)
            if isinstance(desc, tuple):
                desc = "".join(desc)
            assert any(desc.startswith(s) for s in action_starts), (
                f"Tool {tool['name']!r} description doesn't start with action verb: {desc[:60]}"
            )
