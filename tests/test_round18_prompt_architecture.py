"""Round 18: System prompt + architecture review tests.

Validates:
- System prompt template rules, structure, and size
- Architecture context file accuracy (no stale references)
- Context loader integration
"""

import re
from pathlib import Path

import pytest

from src.llm.system_prompt import (
    SYSTEM_PROMPT_TEMPLATE,
    build_system_prompt,
)
from src.context.loader import ContextLoader


# ===========================================================================
# System Prompt Template — Structure
# ===========================================================================


class TestSystemPromptStructure:
    """Verify the system prompt has all required sections."""

    def test_template_under_5000_chars(self):
        assert len(SYSTEM_PROMPT_TEMPLATE) < 5000, (
            f"Template is {len(SYSTEM_PROMPT_TEMPLATE)} chars, must be < 5000"
        )

    def test_has_personality(self):
        assert "Not okay" in SYSTEM_PROMPT_TEMPLATE
        assert "profoundly tired" in SYSTEM_PROMPT_TEMPLATE

    def test_has_identity(self):
        assert "Your identity is Heimdall" in SYSTEM_PROMPT_TEMPLATE
        assert "not Claude" in SYSTEM_PROMPT_TEMPLATE

    def test_has_executor_behavior(self):
        assert "EXECUTOR" in SYSTEM_PROMPT_TEMPLATE
        assert "FIRST response MUST include tool calls" in SYSTEM_PROMPT_TEMPLATE

    def test_has_datetime_placeholder(self):
        assert "{current_datetime}" in SYSTEM_PROMPT_TEMPLATE
        assert "{timezone_name}" in SYSTEM_PROMPT_TEMPLATE

    def test_has_capabilities_section(self):
        assert "## Your Capabilities" in SYSTEM_PROMPT_TEMPLATE

    def test_has_rules_section(self):
        assert "## Rules" in SYSTEM_PROMPT_TEMPLATE

    def test_has_hosts_section(self):
        assert "## Available Hosts" in SYSTEM_PROMPT_TEMPLATE
        assert "{hosts}" in SYSTEM_PROMPT_TEMPLATE

    def test_has_services_section(self):
        assert "## Allowed Services" in SYSTEM_PROMPT_TEMPLATE
        assert "{services}" in SYSTEM_PROMPT_TEMPLATE

    def test_has_context_section(self):
        assert "## Infrastructure Context" in SYSTEM_PROMPT_TEMPLATE
        assert "{context}" in SYSTEM_PROMPT_TEMPLATE

    def test_has_voice_section(self):
        assert "## Voice Channel" in SYSTEM_PROMPT_TEMPLATE
        assert "{voice_info}" in SYSTEM_PROMPT_TEMPLATE


# ===========================================================================
# System Prompt Template — Rules Audit
# ===========================================================================


class TestSystemPromptRules:
    """Verify all 12 rules are present and distinct."""

    @pytest.fixture(autouse=True)
    def _extract_rules(self):
        # Extract rules section
        match = re.search(r"## Rules\n(.*?)(?=\n## |\Z)", SYSTEM_PROMPT_TEMPLATE, re.DOTALL)
        assert match, "Rules section not found"
        self.rules_text = match.group(1)
        # Extract numbered rules
        self.rules = re.findall(r"^\d+\.\s+(.+?)(?=\n\d+\.|\Z)", self.rules_text, re.DOTALL | re.MULTILINE)

    def test_has_12_rules(self):
        assert len(self.rules) == 12, f"Expected 12 rules, found {len(self.rules)}"

    def test_rule_no_emojis(self):
        assert any("emoji" in r.lower() for r in self.rules)

    def test_rule_execute_immediately(self):
        assert any("execute all steps immediately" in r.lower() for r in self.rules)

    def test_rule_no_fabrication(self):
        assert any("fabricate" in r.lower() for r in self.rules)

    def test_rule_always_call_tool(self):
        assert any("call the appropriate tool" in r.lower() for r in self.rules)

    def test_rule_tool_definitions_authoritative(self):
        assert any("authoritative" in r.lower() for r in self.rules)

    def test_rule_concise(self):
        assert any("concise" in r.lower() for r in self.rules)

    def test_rule_no_secrets(self):
        assert any("api keys" in r.lower() or "secrets" in r.lower() for r in self.rules)

    def test_rule_prompt_injection(self):
        assert any("injection" in r.lower() for r in self.rules)

    def test_rule_exhaust_alternatives(self):
        assert any("exhaust" in r.lower() for r in self.rules)

    def test_rule_no_inline_code(self):
        assert any("inline" in r.lower() or "generate_file" in r for r in self.rules)

    def test_rule_tools_available(self):
        assert any("unavailable" in r.lower() or "disabled" in r.lower() for r in self.rules)

    def test_rule_source_code(self):
        assert any("source code" in r.lower() or "claude_code_dir" in r for r in self.rules)


# ===========================================================================
# System Prompt Template — Capabilities Accuracy
# ===========================================================================


class TestCapabilitiesAccuracy:
    """Verify capabilities list matches actual features."""

    @pytest.fixture(autouse=True)
    def _extract_capabilities(self):
        match = re.search(r"## Your Capabilities\n(.*?)(?=\n## )", SYSTEM_PROMPT_TEMPLATE, re.DOTALL)
        assert match, "Capabilities section not found"
        self.capabilities = match.group(1)

    def test_mentions_infrastructure(self):
        assert "Infrastructure" in self.capabilities

    def test_mentions_shell_commands(self):
        assert "Shell commands" in self.capabilities

    def test_mentions_memory(self):
        assert "Memory" in self.capabilities

    def test_mentions_scheduling(self):
        assert "scheduling" in self.capabilities.lower()

    def test_mentions_autonomous_loops(self):
        assert "Autonomous loops" in self.capabilities

    def test_mentions_skills(self):
        assert "skills" in self.capabilities.lower()

    def test_mentions_claude_code(self):
        assert "Claude Code" in self.capabilities

    def test_mentions_browser(self):
        assert "browser" in self.capabilities.lower()

    def test_mentions_vision(self):
        assert "vision" in self.capabilities.lower()

    def test_mentions_web_search(self):
        assert "web search" in self.capabilities.lower()


# ===========================================================================
# Architecture.md — Accuracy
# ===========================================================================


class TestArchitectureAccuracy:
    """Verify architecture.md matches the actual codebase."""

    @pytest.fixture(autouse=True)
    def _load_architecture(self):
        arch_path = Path("data/context/architecture.md")
        if arch_path.exists():
            self.content = arch_path.read_text()
        else:
            pytest.skip("architecture.md not found")

    def test_no_stale_docker_pack(self):
        """Docker pack was removed in Round 2."""
        # Should not reference docker as an available pack
        assert "`docker`" not in self.content.split("## Tool Packs")[1].split("##")[0] if "## Tool Packs" in self.content else True

    def test_no_stale_git_pack(self):
        """Git pack was removed in Round 2."""
        assert "`git`" not in self.content.split("## Tool Packs")[1].split("##")[0] if "## Tool Packs" in self.content else True

    def test_has_systemd_pack(self):
        assert "`systemd`" in self.content

    def test_has_incus_pack(self):
        assert "`incus`" in self.content

    def test_has_ansible_pack(self):
        assert "`ansible`" in self.content

    def test_has_prometheus_pack(self):
        assert "`prometheus`" in self.content

    def test_has_comfyui_pack(self):
        assert "`comfyui`" in self.content

    def test_core_tools_count(self):
        """Core tools should be 47."""
        assert "Core tools (47)" in self.content

    def test_has_defense_mechanisms(self):
        assert "## Defense Mechanisms" in self.content

    def test_has_fabrication_defense(self):
        assert "Fabrication detection" in self.content

    def test_has_hedging_defense(self):
        assert "Hedging detection" in self.content

    def test_has_premature_failure_defense(self):
        assert "Premature failure detection" in self.content

    def test_has_knowledge_base_section(self):
        assert "## Knowledge Base" in self.content

    def test_has_autonomous_loops_section(self):
        assert "## Autonomous Loops" in self.content

    def test_has_claude_code_section(self):
        assert "## Claude Code Delegation" in self.content

    def test_has_background_tasks_section(self):
        assert "## Background Tasks" in self.content

    def test_mentions_fastembed(self):
        """Knowledge base uses fastembed, not external servers."""
        assert "fastembed" in self.content

    def test_mentions_sqlite_vec(self):
        """Vector search uses sqlite-vec, not ChromaDB."""
        assert "sqlite-vec" in self.content

    def test_no_chromadb_reference(self):
        """ChromaDB was removed."""
        assert "chromadb" not in self.content.lower()
        assert "chroma" not in self.content.lower()

    def test_tool_packs_example_valid(self):
        """Example config should only reference existing packs."""
        match = re.search(r"Example config:.*?`tool_packs:\s*\[(.*?)\]`", self.content)
        if match:
            packs = [p.strip() for p in match.group(1).split(",")]
            valid_packs = {"systemd", "incus", "ansible", "prometheus", "comfyui"}
            for pack in packs:
                assert pack in valid_packs, f"Example references removed pack: {pack}"


# ===========================================================================
# Context Loader — Integration
# ===========================================================================


class TestContextLoaderIntegration:
    """Verify context loader handles architecture.md correctly."""

    def test_loads_architecture_md(self, tmp_path):
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "architecture.md").write_text("## Test Section\nSome content.")
        loader = ContextLoader(str(context_dir))
        result = loader.load()
        assert "# architecture" in result
        assert "## Test Section" in result

    def test_reload_picks_up_changes(self, tmp_path):
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        md = context_dir / "test.md"
        md.write_text("Version 1")
        loader = ContextLoader(str(context_dir))
        loader.load()
        assert "Version 1" in loader.context
        md.write_text("Version 2")
        loader.reload()
        assert "Version 2" in loader.context

    def test_ignores_non_md_files(self, tmp_path):
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "notes.md").write_text("Include me")
        (context_dir / "example.md.template").write_text("Ignore me")
        (context_dir / "data.txt").write_text("Ignore me too")
        loader = ContextLoader(str(context_dir))
        loader.load()
        assert "Include me" in loader.context
        assert "Ignore me" not in loader.context
        assert "Ignore me too" not in loader.context

    def test_context_injected_into_prompt(self, tmp_path):
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / "arch.md").write_text("Test architecture content")
        loader = ContextLoader(str(context_dir))
        context = loader.load()
        prompt = build_system_prompt(
            context=context,
            hosts={"server1": "10.0.0.1"},
            services=["nginx"],
            playbooks=["deploy.yml"],
        )
        assert "Test architecture content" in prompt
        assert "## Infrastructure Context" in prompt

    def test_empty_context_shows_placeholder(self):
        prompt = build_system_prompt(
            context="",
            hosts={},
            services=[],
            playbooks=[],
        )
        assert "No context files loaded." in prompt


# ===========================================================================
# Build Prompt — Functional
# ===========================================================================


class TestBuildPrompt:
    """Verify build_system_prompt produces correct output."""

    def test_hosts_formatted(self):
        prompt = build_system_prompt(
            context="test",
            hosts={"web": "10.0.0.1", "db": "10.0.0.2"},
            services=[],
            playbooks=[],
        )
        assert "- `web`: 10.0.0.1" in prompt
        assert "- `db`: 10.0.0.2" in prompt

    def test_services_formatted(self):
        prompt = build_system_prompt(
            context="test",
            hosts={},
            services=["nginx", "redis"],
            playbooks=[],
        )
        assert "`nginx`, `redis`" in prompt

    def test_playbooks_formatted(self):
        prompt = build_system_prompt(
            context="test",
            hosts={},
            services=[],
            playbooks=["deploy.yml"],
        )
        assert "`deploy.yml`" in prompt

    def test_none_configured_fallbacks(self):
        prompt = build_system_prompt(
            context="",
            hosts={},
            services=[],
            playbooks=[],
        )
        assert "None configured" in prompt

    def test_voice_info_default(self):
        prompt = build_system_prompt(
            context="test",
            hosts={},
            services=[],
            playbooks=[],
        )
        assert "Voice support is not enabled." in prompt

    def test_claude_code_dir(self):
        prompt = build_system_prompt(
            context="test",
            hosts={},
            services=[],
            playbooks=[],
            claude_code_dir="/custom/path",
        )
        assert "/custom/path" in prompt
