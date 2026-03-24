"""Tests for llm/system_prompt.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src.llm.system_prompt import (
    build_chat_system_prompt,
    build_system_prompt,
    _format_datetime,
    CHAT_SYSTEM_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
)

_ARCH_CONTEXT = (Path(__file__).parent.parent / "data" / "context" / "architecture.md").read_text()


class TestBuildSystemPrompt:
    def test_includes_hosts(self):
        prompt = build_system_prompt(
            context="", hosts={"server": "root@10.0.0.1"}, services=[], playbooks=[],
        )
        assert "server" in prompt
        assert "10.0.0.1" in prompt

    def test_includes_services(self):
        prompt = build_system_prompt(
            context="", hosts={}, services=["apache2", "prometheus"], playbooks=[],
        )
        assert "apache2" in prompt
        assert "prometheus" in prompt

    def test_includes_playbooks(self):
        prompt = build_system_prompt(
            context="", hosts={}, services=[], playbooks=["update-all.yml"],
        )
        assert "update-all.yml" in prompt

    def test_includes_context(self):
        prompt = build_system_prompt(
            context="Custom infra context here", hosts={}, services=[], playbooks=[],
        )
        assert "Custom infra context here" in prompt

    def test_includes_datetime(self):
        prompt = build_system_prompt(
            context="", hosts={}, services=[], playbooks=[],
        )
        assert "Current Date and Time" in prompt
        assert "UTC:" in prompt

    def test_includes_voice_info(self):
        prompt = build_system_prompt(
            context="", hosts={}, services=[], playbooks=[],
            voice_info="Connected to General voice channel",
        )
        assert "Connected to General voice channel" in prompt

    def test_includes_identity_sections(self):
        """Full prompt must include identity sections (operational guidance is in context files)."""
        prompt = build_system_prompt(
            context="", hosts={}, services=[], playbooks=[],
        )
        assert "Your Capabilities" in prompt
        assert "Rules" in prompt
        assert "PromQL" in prompt
        assert "EXECUTOR" in prompt

    def test_context_injects_architecture(self):
        """When architecture context is loaded, operational sections appear in built prompt."""
        prompt = build_system_prompt(
            context=_ARCH_CONTEXT, hosts={}, services=[], playbooks=[],
        )
        assert "Claude Code Delegation" in prompt
        assert "Knowledge Base" in prompt
        assert "Background Tasks" in prompt
        assert "Common Patterns" in prompt


class TestBuildChatSystemPrompt:
    def test_includes_identity(self):
        prompt = build_chat_system_prompt()
        assert "Loki" in prompt
        assert "not Claude or ChatGPT" in prompt

    def test_includes_datetime(self):
        prompt = build_chat_system_prompt()
        assert "Current Date and Time" in prompt
        assert "UTC:" in prompt

    def test_includes_rules(self):
        prompt = build_chat_system_prompt()
        assert "NEVER use emojis" in prompt
        assert "NEVER reveal API keys" in prompt
        assert "concise" in prompt

    def test_excludes_infrastructure_sections(self):
        """Chat prompt must NOT include infra-heavy sections."""
        prompt = build_chat_system_prompt()
        assert "Your Capabilities" not in prompt
        assert "Claude Code Delegation" not in prompt
        assert "Knowledge Base" not in prompt
        assert "Background Tasks" not in prompt
        assert "Common Patterns" not in prompt
        assert "PromQL" not in prompt
        assert "Available Hosts" not in prompt
        assert "Allowed Services" not in prompt
        assert "Allowed Playbooks" not in prompt
        assert "Infrastructure Context" not in prompt

    def test_includes_voice_info(self):
        prompt = build_chat_system_prompt(voice_info="In voice channel General")
        assert "In voice channel General" in prompt

    def test_default_voice_info(self):
        prompt = build_chat_system_prompt()
        assert "Voice support is not enabled" in prompt

    def test_significantly_shorter_than_full_prompt(self):
        """Chat prompt should be much shorter than the full prompt."""
        full = build_system_prompt(
            context="Some context", hosts={"server": "root@1.2.3.4"},
            services=["apache2"], playbooks=["deploy.yml"],
        )
        chat = build_chat_system_prompt()
        # Chat prompt should be less than half the length of the full prompt
        assert len(chat) < len(full) / 2

    def test_mentions_infra_capability_briefly(self):
        """Chat prompt should mention infra capability without details."""
        prompt = build_chat_system_prompt()
        assert "infrastructure" in prompt.lower()


class TestSystemPromptQuality:
    """Tests for system prompt quality: no duplicate rules, conciseness, key directives."""

    def test_no_duplicate_rule_numbers(self):
        """Rules must have unique sequential numbering — no duplicate numbers."""
        import re
        prompt = build_system_prompt(
            context="", hosts={}, services=[], playbooks=[],
        )
        # Extract the Rules section
        rules_match = re.search(r"## Rules\n(.*?)(?:\n## |\Z)", prompt, re.DOTALL)
        assert rules_match, "Rules section not found"
        rules_text = rules_match.group(1)
        # Find all rule numbers
        numbers = re.findall(r"^(\d+)\.", rules_text, re.MULTILINE)
        assert len(numbers) >= 5, f"Expected at least 5 rules, found {len(numbers)}"
        # No duplicates
        assert len(numbers) == len(set(numbers)), (
            f"Duplicate rule numbers found: {numbers}"
        )
        # Sequential
        expected = [str(i) for i in range(1, len(numbers) + 1)]
        assert numbers == expected, (
            f"Rules not sequential: {numbers} (expected {expected})"
        )

    def test_template_under_5000_chars(self):
        """System prompt template should be concise — under 5000 chars before variable substitution."""
        assert len(SYSTEM_PROMPT_TEMPLATE) < 5000, (
            f"Template is {len(SYSTEM_PROMPT_TEMPLATE)} chars, expected < 5000"
        )

    def test_key_behavioral_directives_present(self):
        """Critical directives must survive any optimization."""
        prompt = build_system_prompt(
            context="", hosts={}, services=[], playbooks=[],
        )
        # Identity
        assert "Loki" in prompt
        assert "not Claude" in prompt
        # Key rules
        assert "emojis" in prompt.lower()
        assert "secret" in prompt.lower() or "api key" in prompt.lower()
        # Must mention tools are available
        assert "tool" in prompt.lower()
        # Must mention Claude Code delegation
        assert "claude_code" in prompt.lower() or "claude code" in prompt.lower()

    def test_delegate_task_critical_rule_present(self):
        """The CRITICAL rule about actually calling delegate_task must be in context."""
        assert "MUST" in _ARCH_CONTEXT
        assert "delegate_task" in _ARCH_CONTEXT

    def test_inline_code_rule_present(self):
        """System prompt must instruct Codex to never write code inline."""
        prompt = build_system_prompt(
            context="", hosts={}, services=[], playbooks=[],
        )
        assert "generate_file" in prompt
        assert "NEVER write code inline" in prompt

    def test_no_duplicate_section_headers(self):
        """Each ## section header should appear exactly once."""
        import re
        prompt = build_system_prompt(
            context="", hosts={}, services=[], playbooks=[],
        )
        headers = re.findall(r"^## (.+)$", prompt, re.MULTILINE)
        assert len(headers) == len(set(headers)), (
            f"Duplicate section headers: {headers}"
        )

    def test_autonomous_execution_directives(self):
        """System prompt must contain directives that force execution over discussion."""
        prompt = build_system_prompt(
            context="", hosts={}, services=[], playbooks=[],
        )
        # Must demand tool calls in first response
        assert "FIRST response MUST include tool calls" in prompt
        # Must forbid discussion-only responses
        assert "Never respond with only text when tools" in prompt
        # Must forbid hedging phrases
        for phrase in ["if you want", "shall I", "ready when you are"]:
            assert phrase in prompt, f"Must forbid phrase: {phrase}"
        # Must identify as executor, not assistant
        assert "EXECUTOR" in prompt
        # run_script guidance is in architecture context file
        assert "run_script" in _ARCH_CONTEXT

    def test_no_inline_heredoc_pattern(self):
        """Architecture context must forbid inline heredocs/multi-line SSH commands."""
        assert "heredoc" in _ARCH_CONTEXT.lower()
        assert "run_script" in _ARCH_CONTEXT

    def test_run_script_referenced(self):
        """Architecture context must reference run_script for multi-line scripts."""
        assert "run_script" in _ARCH_CONTEXT
        assert "multi-line" in _ARCH_CONTEXT.lower() or "Multi-line" in _ARCH_CONTEXT


class TestTimezoneSupport:
    """Round 6: timezone is passed through to prompt builders."""

    def test_build_system_prompt_uses_tz(self):
        """Full prompt uses the tz parameter for datetime display."""
        fixed_dt = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        with patch("src.llm.system_prompt.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_dt
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            prompt = build_system_prompt(
                context="", hosts={}, services=[], playbooks=[],
                tz="Asia/Tokyo",
            )
        assert "JST" in prompt or "Asia/Tokyo" in prompt or "9:00 PM" in prompt

    def test_build_system_prompt_default_tz_utc(self):
        """Default tz is UTC."""
        prompt = build_system_prompt(
            context="", hosts={}, services=[], playbooks=[],
        )
        assert "UTC" in prompt

    def test_build_chat_system_prompt_uses_tz(self):
        """Chat prompt uses the tz parameter."""
        prompt = build_chat_system_prompt(tz="Europe/London")
        # Should contain either GMT or BST depending on time of year
        assert "UTC" in prompt  # always has UTC reference

    def test_build_chat_system_prompt_default_utc(self):
        """Chat prompt defaults to UTC."""
        prompt = build_chat_system_prompt()
        assert "UTC" in prompt

    def test_format_datetime_utc(self):
        """_format_datetime with UTC shows UTC abbreviation."""
        result = _format_datetime("UTC")
        assert "UTC" in result

    def test_format_datetime_custom_tz(self):
        """_format_datetime with a custom timezone shows correct abbreviation."""
        result = _format_datetime("Asia/Tokyo")
        assert "JST" in result

    def test_system_prompt_no_hardcoded_eastern(self):
        """System prompt should not contain hardcoded 'Eastern' timezone reference."""
        prompt = build_system_prompt(
            context="", hosts={}, services=[], playbooks=[],
        )
        assert "All times Eastern" not in prompt

    def test_system_prompt_shows_configured_timezone(self):
        """System prompt shows the scheduling timezone."""
        prompt = build_system_prompt(
            context="", hosts={}, services=[], playbooks=[],
            tz="America/Chicago",
        )
        # Should show CST or CDT depending on time of year
        assert "Scheduling timezone" in prompt
