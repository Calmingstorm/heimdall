"""Round 8: Verify Round 7 — system prompt polish + broadened claude -p delegation.

Cross-cutting verification that Round 7 changes are consistent with Rounds 1-6
and that the full architecture is coherent after 7 rounds of changes.
"""
from __future__ import annotations

import ast
import re

import pytest

from src.llm.system_prompt import (
    build_system_prompt,
    build_chat_system_prompt,
    SYSTEM_PROMPT_TEMPLATE,
    CHAT_SYSTEM_PROMPT_TEMPLATE,
)
from src.tools.registry import TOOLS, get_tool_definitions


def _build_prompt() -> str:
    return build_system_prompt(
        context="", hosts={"web": "10.0.0.1"}, services=["nginx"], playbooks=["deploy"],
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
# No removed systems leak into Round 7 changes
# ---------------------------------------------------------------------------

class TestNoRemovedSystemLeaks:
    """Round 7 prompt changes must not re-introduce any removed system."""

    def test_no_approval_in_prompt(self):
        prompt = _build_prompt()
        # "approval" should not appear in the main prompt
        assert "approval" not in prompt.lower()

    def test_no_classifier_in_prompt(self):
        prompt = _build_prompt()
        assert "classifier" not in prompt.lower()
        assert "haiku" not in prompt.lower()

    def test_no_requires_approval_in_tools(self):
        for tool in TOOLS:
            assert "requires_approval" not in tool, f"{tool['name']} has requires_approval"

    def test_no_anthropic_config_import(self):
        """schema.py should not define AnthropicConfig."""
        import src.config.schema as schema
        assert not hasattr(schema, "AnthropicConfig")

    def test_routing_module_removed(self):
        """routing.py should be completely removed (dead code cleanup)."""
        import importlib
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("src.discord.routing")


# ---------------------------------------------------------------------------
# Prompt architecture coherence
# ---------------------------------------------------------------------------

class TestPromptArchitectureCoherence:
    """The two-tier execution model is properly expressed in the prompt."""

    def test_two_tier_in_prompt(self):
        """Prompt mentions direct tools AND claude_code as two tiers."""
        prompt = _build_prompt()
        delegation = prompt[prompt.find("## Claude Code"):prompt.find("## Knowledge")]
        # Direct tools guidance
        assert "direct tools" in delegation.lower()
        # Complex task delegation
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
        # No remaining {field_name} format placeholders (PromQL braces like
        # ALERTS{alertstate="firing"} are fine — they're literal content)
        import re
        unfilled = re.findall(r"\{[a-z_]+\}", prompt)
        # Filter out known PromQL/content braces
        unfilled = [m for m in unfilled if m not in ('{alertstate}',)]
        assert not unfilled, f"Unfilled format fields: {unfilled}"

    def test_template_has_expected_sections(self):
        sections = [
            "## Current Date",
            "## Your Capabilities",
            "## Claude Code Delegation",
            "## Knowledge Base",
            "## Background Tasks",
            "## Rules",
            "## Available Hosts",
        ]
        for section in sections:
            assert section in SYSTEM_PROMPT_TEMPLATE, f"Missing section: {section}"

    def test_char_budget_remaining(self):
        """At least 100 chars of budget remaining."""
        remaining = 5000 - len(SYSTEM_PROMPT_TEMPLATE)
        assert remaining >= 100, f"Only {remaining} chars remaining in budget"


# ---------------------------------------------------------------------------
# Source code: no dangling old references
# ---------------------------------------------------------------------------

class TestNoDanglingReferences:
    """Verify removed systems have no executable references in src/."""

    def test_no_approval_py(self):
        """approval.py must not exist."""
        import importlib
        try:
            importlib.import_module("src.discord.approval")
            raise AssertionError("src.discord.approval should not be importable")
        except (ImportError, ModuleNotFoundError):
            pass

    def test_no_haiku_classifier_py(self):
        """haiku_classifier.py must not exist."""
        import importlib
        try:
            importlib.import_module("src.llm.haiku_classifier")
            raise AssertionError("src.llm.haiku_classifier should not be importable")
        except (ImportError, ModuleNotFoundError):
            pass

    def test_no_schedule_intent_re(self):
        """_SCHEDULE_INTENT_RE should be gone from client."""
        from src.discord.client import LokiBot
        assert not hasattr(LokiBot, "_SCHEDULE_INTENT_RE")

    def test_no_last_tool_use(self):
        """_last_tool_use tracking should be gone from client."""
        from src.discord.client import LokiBot
        # Check the __init__ method doesn't reference _last_tool_use
        import inspect
        source = inspect.getsource(LokiBot.__init__)
        assert "_last_tool_use" not in source
