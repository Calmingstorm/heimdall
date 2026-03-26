"""Tests for pipeline pattern: system prompt and tool description guide the LLM to use claude_code within its tool loop."""
from __future__ import annotations

from pathlib import Path

from src.llm.system_prompt import build_system_prompt, SYSTEM_PROMPT_TEMPLATE
from src.tools.registry import TOOLS, get_tool_definitions

_ARCH_CONTEXT = (Path(__file__).parent.parent / "data" / "context" / "architecture.md").read_text()


def _full_prompt() -> str:
    return build_system_prompt(
        context=_ARCH_CONTEXT, hosts={}, services=[], playbooks=[],
    )


class TestSystemPromptPipelineGuidance:
    """System prompt must guide the LLM to use claude_code for code subtasks."""

    def test_no_last_resort_framing(self):
        """claude_code should not be framed as a 'last resort' — it's a primary tool for code tasks."""
        prompt = _full_prompt()
        assert "last resort" not in prompt.lower()

    def test_no_expensive_slow_framing(self):
        """System prompt should not call claude_code 'expensive' or 'slow'."""
        prompt = _full_prompt()
        # Check the Claude Code section specifically
        assert "expensive" not in prompt.lower()
        assert "(slow)" not in prompt.lower()

    def test_mentions_complex_multi_step_tasks(self):
        """Should describe claude_code as for complex multi-step tasks."""
        prompt = _full_prompt()
        assert "complex multi-step" in prompt.lower() or "multi-step task" in prompt.lower()

    def test_mentions_allow_edits(self):
        """Should mention allow_edits parameter for write mode."""
        prompt = _full_prompt()
        assert "allow_edits" in prompt

    def test_mentions_deep_reasoning_delegation(self):
        """Should describe claude_code as a deep reasoning agent for delegation."""
        prompt = _full_prompt()
        lower = prompt.lower()
        assert "claude_code" in prompt
        assert "deep reasoning agent" in lower

    def test_code_creation_and_review_present(self):
        """Should include guidance for code creation and code review patterns."""
        prompt = _full_prompt()
        lower = prompt.lower()
        assert "code creation" in lower or "allow_edits" in lower

    def test_still_discourages_misuse(self):
        """Should still warn against using claude_code for single file reads, commands."""
        prompt = _full_prompt()
        lower = prompt.lower()
        assert "read_file" in lower or "file read" in lower
        assert "run_command" in lower or "command" in lower

    def test_delegation_section_exists(self):
        """The Claude Code Delegation section must exist."""
        prompt = _full_prompt()
        assert "## Claude Code Delegation" in prompt

    def test_no_budget_in_prompt(self):
        """Budget guidance should NOT be in prompt (Max subscription, no per-call limits)."""
        prompt = _full_prompt()
        assert "$0.50" not in prompt
        assert "$2.00" not in prompt


class TestToolDescriptionPipelineGuidance:
    """claude_code tool description must align with pipeline guidance."""

    def _get_claude_code_tool(self) -> dict:
        for t in TOOLS:
            if t["name"] == "claude_code":
                return t
        raise AssertionError("claude_code tool not found in TOOLS")

    def test_no_only_when_explicitly_asked(self):
        """Tool description should not say 'ONLY use when explicitly asked'."""
        desc = self._get_claude_code_tool()["description"]
        assert "only use when" not in desc.lower()

    def test_no_expensive_slow_in_description(self):
        """Tool description should not call itself 'expensive, slow'."""
        desc = self._get_claude_code_tool()["description"]
        assert "expensive" not in desc.lower()
        assert "(slow)" not in desc.lower()

    def test_describes_positive_use_cases(self):
        """Tool description should state what it's FOR, not just what it's against."""
        desc = self._get_claude_code_tool()["description"]
        lower = desc.lower()
        # Should mention broad use cases (broadened from just code tasks)
        assert any(
            term in lower
            for term in ["code generation", "repo analysis", "debugging", "architecture review", "building"]
        )

    def test_mentions_deploy_pipeline(self):
        """Tool description should mention the code+deploy pattern."""
        desc = self._get_claude_code_tool()["description"]
        assert "deploy" in desc.lower()

    def test_still_warns_against_misuse(self):
        """Tool description should still discourage misuse for file/command tasks."""
        desc = self._get_claude_code_tool()["description"]
        lower = desc.lower()
        assert "read_file" in lower or "reading" in lower
        assert "run_command" in lower or "running commands" in lower

    def test_has_allow_edits_param(self):
        """claude_code tool must have allow_edits in its input schema."""
        tool = self._get_claude_code_tool()
        props = tool["input_schema"]["properties"]
        assert "allow_edits" in props
        assert props["allow_edits"]["type"] == "boolean"

    def test_included_in_get_tool_definitions(self):
        """claude_code must be included in the API-facing tool definitions."""
        defs = get_tool_definitions()
        names = [d["name"] for d in defs]
        assert "claude_code" in names
