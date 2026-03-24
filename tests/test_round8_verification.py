"""Round 8: Verify Round 7 — system prompt polish + broadened claude -p delegation.

Cross-cutting verification that Round 7 changes are consistent with Rounds 1-6
and that the full architecture is coherent after 7 rounds of changes.

Note: Removal checks (approval, classifier, routing, schedule guard, AnthropicConfig,
_last_tool_use) are consolidated into test_round10_verification.py.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.llm.system_prompt import (
    build_system_prompt,
    build_chat_system_prompt,
    SYSTEM_PROMPT_TEMPLATE,
    CHAT_SYSTEM_PROMPT_TEMPLATE,
)
from src.tools.registry import TOOLS

_ARCH_CONTEXT = (Path(__file__).parent.parent / "data" / "context" / "architecture.md").read_text()


def _build_prompt() -> str:
    return build_system_prompt(
        context=_ARCH_CONTEXT, hosts={"web": "10.0.0.1"}, services=["nginx"], playbooks=["deploy"],
    )


# ---------------------------------------------------------------------------
# Cross-round consistency: prompt + registry agree
# ---------------------------------------------------------------------------

class TestPromptRegistryConsistency:
    """System prompt claims and registry definitions must agree."""

    def test_claude_code_in_both(self):
        """claude_code referenced in prompt AND present in registry."""
        prompt = _build_prompt()
        assert "claude_code" in prompt
        names = [t["name"] for t in TOOLS]
        assert "claude_code" in names

    def test_run_script_in_both(self):
        """run_script referenced in prompt AND present in registry."""
        prompt = _build_prompt()
        assert "run_script" in prompt
        names = [t["name"] for t in TOOLS]
        assert "run_script" in names

    def test_search_knowledge_in_both(self):
        prompt = _build_prompt()
        assert "search_knowledge" in prompt
        names = [t["name"] for t in TOOLS]
        assert "search_knowledge" in names

    def test_delegate_task_in_both(self):
        prompt = _build_prompt()
        assert "delegate_task" in prompt
        names = [t["name"] for t in TOOLS]
        assert "delegate_task" in names

    def test_deep_reasoning_consistent(self):
        """Both prompt and tool description call it 'deep reasoning'."""
        prompt = _build_prompt()
        cc = next(t for t in TOOLS if t["name"] == "claude_code")
        assert "deep reasoning" in prompt.lower()
        assert "deep reasoning" in cc["description"].lower()


# ---------------------------------------------------------------------------
# Prompt architecture coherence
# ---------------------------------------------------------------------------

class TestPromptArchitectureCoherence:
    """The two-tier execution model is properly expressed in the prompt."""

    def test_two_tier_in_prompt(self):
        """Prompt mentions direct tools AND claude_code as two tiers."""
        prompt = _build_prompt()
        delegation = prompt[prompt.find("## Claude Code"):prompt.find("## Knowledge")]
        assert "direct tools" in delegation.lower()
        assert "complex" in delegation.lower()

    def test_executor_identity(self):
        """CORE BEHAVIOR declares EXECUTOR identity."""
        prompt = _build_prompt()
        assert "EXECUTOR" in prompt

    def test_no_hesitation_patterns(self):
        """Prompt forbids hedging phrases."""
        prompt = _build_prompt()
        core = prompt[prompt.find("CORE BEHAVIOR"):prompt.find("## Current Date")]
        assert "if you want" in core.lower()  # it's in the forbid list
        assert "Never say" in core  # the forbid instruction itself

    def test_host_injection_works(self):
        """Hosts passed to build_system_prompt appear in output."""
        prompt = _build_prompt()
        assert "web" in prompt
        assert "10.0.0.1" in prompt

    def test_services_injection_works(self):
        prompt = _build_prompt()
        assert "nginx" in prompt

    def test_playbooks_injection_works(self):
        prompt = _build_prompt()
        assert "deploy" in prompt


# ---------------------------------------------------------------------------
# No removed systems in prompt
# ---------------------------------------------------------------------------

class TestNoRemovedSystemsInPrompt:
    """Prompt must not reference removed systems."""

    def test_no_approval_in_prompt(self):
        prompt = _build_prompt()
        assert "approval" not in prompt.lower()

    def test_no_classifier_in_prompt(self):
        prompt = _build_prompt()
        assert "classifier" not in prompt.lower()
        assert "haiku" not in prompt.lower()


# ---------------------------------------------------------------------------
# Chat prompt still works (guest tier)
# ---------------------------------------------------------------------------

class TestChatPromptIntact:
    """Chat prompt (guest tier) is still functional and separate."""

    def test_chat_prompt_builds(self):
        prompt = build_chat_system_prompt()
        assert "Loki" in prompt

    def test_chat_prompt_no_tools(self):
        """Chat prompt should not mention tool execution capabilities."""
        prompt = build_chat_system_prompt()
        assert "run_command" not in prompt
        assert "claude_code" not in prompt

    def test_chat_prompt_under_limit(self):
        assert len(CHAT_SYSTEM_PROMPT_TEMPLATE) < 2000


# ---------------------------------------------------------------------------
# Template safety
# ---------------------------------------------------------------------------

class TestTemplateSafety:
    """Template must be well-formed and safe."""

    def test_all_format_fields_filled(self):
        """build_system_prompt fills all format fields without KeyError."""
        prompt = build_system_prompt(
            context="ctx", hosts={"h": "1.2.3.4"}, services=["s"], playbooks=["p"],
        )
        unfilled = re.findall(r"\{[a-z_]+\}", prompt)
        unfilled = [m for m in unfilled if m not in ('{alertstate}',)]
        assert not unfilled, f"Unfilled format fields: {unfilled}"

    def test_template_has_expected_sections(self):
        """Template has identity sections; operational sections are in context files."""
        template_sections = [
            "## Current Date",
            "## Your Capabilities",
            "## Rules",
            "## Available Hosts",
        ]
        for section in template_sections:
            assert section in SYSTEM_PROMPT_TEMPLATE, f"Missing section: {section}"
        # Operational sections moved to architecture.md context file
        context_sections = [
            "## Claude Code Delegation",
            "## Knowledge Base",
            "## Background Tasks",
        ]
        for section in context_sections:
            assert section in _ARCH_CONTEXT, f"Missing in context: {section}"

    def test_char_budget_remaining(self):
        """At least 100 chars of budget remaining."""
        remaining = 5000 - len(SYSTEM_PROMPT_TEMPLATE)
        assert remaining >= 100, f"Only {remaining} chars remaining in budget"
