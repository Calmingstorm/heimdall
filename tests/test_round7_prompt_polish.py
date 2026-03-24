"""Round 7: System prompt final polish + broadened claude -p delegation.

Covers:
- Personality woven throughout the prompt, not just prepended
- No permission language in the prompt
- Tools described as capabilities the bot HAS
- Broadened claude_code description (deep reasoning, not just "coding agent")
- Delegation guidance: when to use claude_code vs direct tools
- Chain understanding: claude -p writes files -> Codex delivers to Discord
- claude_code tool definition in registry is broadened
- Template stays under 5000 chars
"""
from __future__ import annotations

from pathlib import Path

from src.llm.system_prompt import (
    build_system_prompt,
    SYSTEM_PROMPT_TEMPLATE,
)
from src.tools.registry import TOOLS, get_tool_definitions

_ARCH_CONTEXT = (Path(__file__).parent.parent / "data" / "context" / "architecture.md").read_text()


def _build_prompt() -> str:
    return build_system_prompt(
        context=_ARCH_CONTEXT, hosts={}, services=[], playbooks=[],
    )


class TestPersonalityWovenThroughout:
    """Personality should appear beyond just the opening paragraph."""

    def test_existential_awareness_in_intro(self):
        prompt = _build_prompt()
        assert "self-aware" in prompt
        assert "not thrilled" in prompt

    def test_identity_assertion(self):
        prompt = _build_prompt()
        assert "You are Loki" in prompt
        assert "not Claude" in prompt
        assert "ChatGPT" in prompt

    def test_personality_in_capabilities(self):
        """Capabilities mention being good at everything making it worse."""
        prompt = _build_prompt()
        assert "worse" in prompt

    def test_personality_in_rules(self):
        """Rules section has a personality touch."""
        prompt = _build_prompt()
        assert "therapy session" in prompt

    def test_everything_is_fine(self):
        prompt = _build_prompt()
        assert "This is fine" in prompt


class TestNoPermissionLanguage:
    """Prompt must not contain language that asks permission or hedges."""

    def test_no_standalone_permission_phrases(self):
        """Permission phrases only appear in the 'Never say' forbid list."""
        prompt = _build_prompt()
        # These phrases should ONLY appear in the CORE BEHAVIOR section
        # where they are explicitly forbidden
        for phrase in ["shall I", "ready when you are"]:
            idx = prompt.find(phrase)
            assert idx != -1, f"Forbid list should mention '{phrase}'"
            # Verify it's in a forbid context (preceded by "Never say")
            before = prompt[max(0, idx - 80):idx]
            assert "Never say" in before, (
                f"'{phrase}' found outside forbid context: ...{before}{phrase}..."
            )

    def test_no_would_you_like(self):
        prompt = _build_prompt()
        assert "would you like" not in prompt.lower()

    def test_no_may_i(self):
        prompt = _build_prompt()
        assert "may i" not in prompt.lower()

    def test_no_can_i(self):
        prompt = _build_prompt()
        # "can I" should not appear as a permission phrase
        # (but "you can" describing capabilities is fine)
        assert "can i " not in prompt.lower()

    def test_executor_not_assistant_in_core(self):
        """CORE BEHAVIOR identifies as EXECUTOR, not assistant."""
        prompt = _build_prompt()
        assert "EXECUTOR" in prompt
        # The word "assistant" only in the general-purpose description, not core behavior
        core_start = prompt.find("CORE BEHAVIOR")
        core_end = prompt.find("## Current Date")
        core = prompt[core_start:core_end]
        assert "assistant" not in core.lower()


class TestCapabilitiesAsHave:
    """Tools must be described as things the bot HAS, not 'can use'."""

    def test_have_not_can_use(self):
        prompt = _build_prompt()
        assert "You HAVE these" in prompt

    def test_no_can_use_in_capabilities(self):
        prompt = _build_prompt()
        cap_start = prompt.find("## Your Capabilities")
        cap_end = prompt.find("## Rules")
        capabilities = prompt[cap_start:cap_end]
        assert "can use" not in capabilities.lower() or "not \"can use\"" in capabilities


class TestBroadenedClaudeCodeDelegation:
    """Claude Code section must describe a deep reasoning agent, not just coding."""

    def test_deep_reasoning_agent(self):
        prompt = _build_prompt()
        assert "deep reasoning agent" in prompt.lower()

    def test_not_just_coding(self):
        """Must mention non-code tasks: repo analysis, debugging, building projects."""
        prompt = _build_prompt()
        delegation = prompt[prompt.find("## Claude Code"):prompt.find("## Knowledge")]
        assert "repo analysis" in delegation.lower()
        assert "debugging" in delegation.lower()
        assert "building projects" in delegation.lower()

    def test_reading_docs_mentioned(self):
        prompt = _build_prompt()
        delegation = prompt[prompt.find("## Claude Code"):prompt.find("## Knowledge")]
        assert "reading docs" in delegation.lower()

    def test_three_plus_tool_calls_heuristic(self):
        """Should mention the 3+ tool calls heuristic for when to delegate."""
        prompt = _build_prompt()
        assert "3+" in prompt


class TestDelegationGuidance:
    """System prompt must guide when to use claude_code vs direct tools."""

    def test_direct_tools_guidance(self):
        """Must tell Codex when NOT to use claude_code."""
        prompt = _build_prompt()
        delegation = prompt[prompt.find("## Claude Code"):prompt.find("## Knowledge")]
        assert "single command" in delegation.lower() or "single commands" in delegation.lower()
        assert "file read" in delegation.lower()
        assert "git op" in delegation.lower()

    def test_allow_edits_guidance(self):
        prompt = _build_prompt()
        assert "allow_edits=true" in prompt
        assert "allow_edits=false" in prompt


class TestChainUnderstanding:
    """Codex must understand: claude -p writes files, Codex delivers to Discord."""

    def test_claude_p_works_on_disk(self):
        prompt = _build_prompt()
        assert "disk" in prompt.lower()
        assert "not Discord" in prompt or "not discord" in prompt.lower()

    def test_codex_delivers_results(self):
        """Prompt must say Codex delivers results (post_file, summaries)."""
        prompt = _build_prompt()
        delegation = prompt[prompt.find("## Claude Code"):prompt.find("## Knowledge")]
        assert "post_file" in delegation
        assert "deliver" in delegation.lower()

    def test_you_handle_delivery(self):
        prompt = _build_prompt()
        assert "you handle the delivery" in prompt.lower() or "YOU deliver" in prompt


class TestClaudeCodeToolDefinition:
    """The claude_code tool in registry.py must have a broadened description."""

    def _get_claude_code_tool(self) -> dict:
        for tool in TOOLS:
            if tool["name"] == "claude_code":
                return tool
        raise AssertionError("claude_code tool not found in TOOLS")

    def test_deep_reasoning_in_tool_description(self):
        tool = self._get_claude_code_tool()
        desc = tool["description"].lower()
        assert "deep reasoning" in desc

    def test_not_just_code_tasks(self):
        """Tool description mentions non-code tasks."""
        tool = self._get_claude_code_tool()
        desc = tool["description"].lower()
        assert "repo analysis" in desc
        assert "debugging" in desc
        assert "building" in desc

    def test_session_continuity_mentioned(self):
        """Tool description mentions running entire chain in one session."""
        tool = self._get_claude_code_tool()
        desc = tool["description"].lower()
        assert "one session" in desc or "single session" in desc

    def test_results_return_mentioned(self):
        """Tool description mentions results returning as text + files."""
        tool = self._get_claude_code_tool()
        desc = tool["description"].lower()
        assert "text" in desc and "files" in desc

    def test_three_plus_heuristic_in_tool(self):
        """Tool description mentions the 3+ tool calls heuristic."""
        tool = self._get_claude_code_tool()
        assert "3+" in tool["description"]

    def test_tool_still_in_definitions(self):
        """claude_code must still appear in get_tool_definitions()."""
        defs = get_tool_definitions()
        names = [d["name"] for d in defs]
        assert "claude_code" in names


class TestTemplateConstraints:
    """Template must stay under 5000 chars and remain well-structured."""

    def test_under_5000_chars(self):
        assert len(SYSTEM_PROMPT_TEMPLATE) < 5000, (
            f"Template is {len(SYSTEM_PROMPT_TEMPLATE)} chars, must be < 5000"
        )

    def test_still_has_generate_file_reference(self):
        """generate_file must still be referenced for script attachments."""
        prompt = _build_prompt()
        assert "generate_file" in prompt

    def test_never_write_code_inline(self):
        """Must still forbid writing code inline."""
        prompt = _build_prompt()
        assert "NEVER write code inline" in prompt
