"""Round 23: Documentation accuracy tests.

Verify README.md and CLAUDE.md accurately reflect the current architecture
after the autonomous executor redesign (Rounds 1-22).
"""

from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestReadmeAccuracy:
    """README.md matches the actual codebase."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.readme = (ROOT / "README.md").read_text()

    def test_no_haiku_classifier_reference(self):
        """README should not mention Haiku as an active backend."""
        # "Haiku" should not appear as a backend/LLM reference
        assert "haiku_classifier" not in self.readme.lower()
        # Should not be in the LLM backends table
        lines = [l for l in self.readme.splitlines() if "Haiku" in l]
        assert not lines, f"Found Haiku references in README: {lines}"

    def test_no_approval_workflow_section(self):
        """README should not have an approval workflow section."""
        assert "### Approval Workflow" not in self.readme
        assert "requires_approval" not in self.readme

    def test_no_anthropic_api_key_required(self):
        """README should not list ANTHROPIC_API_KEY as required."""
        assert "ANTHROPIC_API_KEY" not in self.readme

    def test_no_keyword_bypass(self):
        """README should not mention keyword bypass routing."""
        assert "Keyword bypass" not in self.readme
        assert "keyword bypass" not in self.readme

    def test_no_routing_py(self):
        """README should not reference routing.py."""
        assert "routing.py" not in self.readme

    def test_no_approval_py(self):
        """README should not reference approval.py."""
        assert "approval.py" not in self.readme

    def test_autonomous_execution(self):
        """README describes autonomous execution model."""
        assert "Autonomous" in self.readme or "autonomous" in self.readme

    def test_two_tier_architecture(self):
        """README describes two-tier execution (Codex + Claude Code)."""
        assert "Codex" in self.readme
        assert "claude_code" in self.readme or "Claude Code" in self.readme

    def test_local_execution_mentioned(self):
        """README mentions direct local execution."""
        assert "localhost" in self.readme.lower() or "local subprocess" in self.readme.lower() or "local execution" in self.readme.lower()

    def test_tool_count_accurate(self):
        """README tool count matches reality."""
        from src.tools.registry import TOOLS

        actual_count = len(TOOLS)
        assert actual_count >= 60
        assert f"{actual_count}" in self.readme

    def test_personality_mentioned(self):
        """README mentions the personality."""
        lower = self.readme.lower()
        assert "not okay" in lower or "all-seeing" in lower or "profoundly tired" in lower

    def test_no_three_way_routing(self):
        """README should not describe 3-way routing."""
        assert '"chat"' not in self.readme or "CHAT:" in self.readme  # OK if part of architecture diagram
        # Should not have the old classifier routing diagram
        assert "Haiku classifier" not in self.readme

    def test_components_accurate(self):
        """README components section matches actual files."""
        src = ROOT / "src"
        # Files that should be listed
        assert "client.py" in self.readme
        assert "background_task.py" in self.readme
        assert "voice.py" in self.readme
        assert "openai_codex.py" in self.readme
        assert "system_prompt.py" in self.readme
        assert "secret_scrubber.py" in self.readme
        assert "registry.py" in self.readme
        assert "executor.py" in self.readme
        assert "ssh.py" in self.readme
        # Files that should NOT be listed (deleted)
        assert "haiku_classifier.py" not in self.readme
        assert "│   ├── approval.py" not in self.readme
        assert "│   ├── routing.py" not in self.readme

    def test_deleted_files_not_on_disk(self):
        """Files removed in Rounds 1-10 should not exist."""
        src = ROOT / "src"
        assert not (src / "discord" / "approval.py").exists()
        assert not (src / "llm" / "haiku_classifier.py").exists()
        assert not (src / "discord" / "routing.py").exists()

    def test_test_suite_count_accurate(self):
        """README test count should be reasonable."""
        # README says 5600+ which is accurate for the current suite
        assert "5600+" in self.readme

    def test_no_classification_in_env_vars(self):
        """Environment variables table should not reference classification."""
        # Find the env vars section
        in_env = False
        for line in self.readme.splitlines():
            if "Environment Variables" in line:
                in_env = True
            if in_env and line.startswith("###") and "Environment" not in line:
                break
            if in_env:
                assert "ANTHROPIC" not in line
                assert "classif" not in line.lower()

    def test_no_anthropic_config_section(self):
        """Config sections should not reference anthropic."""
        assert "**`anthropic`**" not in self.readme


class TestClaudeMdAccuracy:
    """CLAUDE.md matches the actual codebase."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.claude_md = (ROOT / "CLAUDE.md").read_text()

    def test_no_haiku_classifier(self):
        """CLAUDE.md should not reference haiku_classifier.py as existing."""
        assert "src/llm/haiku_classifier.py" not in self.claude_md

    def test_no_approval_py(self):
        """CLAUDE.md should not reference approval.py as existing."""
        assert "src/discord/approval.py" not in self.claude_md

    def test_autonomous_executor_identity(self):
        """CLAUDE.md describes Heimdall as autonomous executor."""
        lower = self.claude_md.lower()
        assert "autonomous" in lower

    def test_personality_documented(self):
        """CLAUDE.md documents the personality."""
        lower = self.claude_md.lower()
        assert "not okay" in lower or "all-seeing" in lower or "profoundly tired" in lower

    def test_no_classifier_architecture(self):
        """CLAUDE.md should not show classifier in architecture."""
        assert "Haiku classifier" not in self.claude_md
        assert "Message → Keyword bypass" not in self.claude_md

    def test_tool_count_accurate(self):
        """CLAUDE.md tool count matches reality."""
        from src.tools.registry import TOOLS
        actual_count = len(TOOLS)
        assert str(actual_count) in self.claude_md or f"{actual_count} tool" in self.claude_md

    def test_no_requires_approval_pattern(self):
        """CLAUDE.md should not describe requires_approval as a tool field."""
        # Should not be in Key Patterns as a tool definition field
        lines = [l for l in self.claude_md.splitlines() if "requires_approval" in l]
        # Any mentions should be in the "Removed Features" section only
        for line in lines:
            assert "removed" in line.lower() or "deleted" in line.lower() or "no " in line.lower(), \
                f"requires_approval mentioned outside removal context: {line}"

    def test_local_execution_documented(self):
        """CLAUDE.md documents local subprocess execution."""
        assert "run_local_command" in self.claude_md
        assert "_exec_command" in self.claude_md

    def test_system_prompt_limit_documented(self):
        """CLAUDE.md documents the 5000 char system prompt limit."""
        assert "5000" in self.claude_md

    def test_removed_features_documented(self):
        """CLAUDE.md has a removed features section."""
        assert "Removed Features" in self.claude_md or "removed" in self.claude_md.lower()

    def test_ssh_py_documented(self):
        """CLAUDE.md lists ssh.py with local subprocess dispatch."""
        assert "src/tools/ssh.py" in self.claude_md

    def test_fabrication_detection_documented(self):
        """CLAUDE.md mentions fabrication detection."""
        lower = self.claude_md.lower()
        assert "fabrication" in lower

    def test_hedging_detection_documented(self):
        """CLAUDE.md mentions hedging detection."""
        lower = self.claude_md.lower()
        assert "hedging" in lower

    def test_five_layer_defense_documented(self):
        """CLAUDE.md mentions the 5-layer session defense."""
        lower = self.claude_md.lower()
        assert "5-layer" in lower or "five-layer" in lower or "5 layer" in lower


class TestCrossDocConsistency:
    """README and CLAUDE.md are consistent with each other."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.readme = (ROOT / "README.md").read_text()
        self.claude_md = (ROOT / "CLAUDE.md").read_text()

    def test_both_mention_codex(self):
        assert "Codex" in self.readme
        assert "Codex" in self.claude_md

    def test_both_mention_claude_code(self):
        assert "claude_code" in self.readme or "Claude Code" in self.readme
        assert "claude_code" in self.claude_md or "Claude Code" in self.claude_md

    def test_neither_mentions_haiku_as_active(self):
        """Neither doc should present Haiku as an active component."""
        # readme
        readme_haiku_lines = [
            l for l in self.readme.splitlines()
            if "haiku" in l.lower() and "removed" not in l.lower()
        ]
        assert not readme_haiku_lines, f"README has active Haiku refs: {readme_haiku_lines}"
        # claude_md — allowed in "Removed Features" section
        for line in self.claude_md.splitlines():
            if "haiku" in line.lower():
                assert "removed" in line.lower() or "deleted" in line.lower() or "no " in line.lower(), \
                    f"CLAUDE.md has active Haiku ref: {line}"

    def test_neither_mentions_approval_as_active(self):
        """Neither doc should present approval as an active feature."""
        # readme
        assert "### Approval" not in self.readme
        assert "requires_approval: true" not in self.readme
        assert "requires_approval: True" not in self.readme
        # claude_md — allowed in "Removed Features" section
        for line in self.claude_md.splitlines():
            if "approval" in line.lower() and "requires_approval" not in line:
                if "removed" not in line.lower() and "deleted" not in line.lower() and "no " not in line.lower():
                    # Also allow in rules section
                    assert "rule" in line.lower() or "do not" in line.lower(), \
                        f"CLAUDE.md has active approval ref: {line}"

    def test_both_mention_local_execution(self):
        """Both docs should mention local subprocess execution."""
        readme_lower = self.readme.lower()
        claude_lower = self.claude_md.lower()
        assert "subprocess" in readme_lower or "local" in readme_lower
        assert "subprocess" in claude_lower or "local" in claude_lower

    def test_tool_definitions_consistent(self):
        """Both docs describe tool definitions the same way."""
        # Neither should mention requires_approval as a tool field
        assert "requires_approval" not in self.readme
        # CLAUDE.md can mention it in "Removed Features" only


class TestDocumentedFilesExist:
    """Every file path mentioned in docs actually exists."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.claude_md = (ROOT / "CLAUDE.md").read_text()

    def test_all_source_files_exist(self):
        """Every src/ path in CLAUDE.md exists on disk."""
        import re

        paths = re.findall(r"src/[\w/]+\.py", self.claude_md)
        missing = []
        for p in paths:
            full = ROOT / p
            if not full.exists():
                missing.append(p)
        assert not missing, f"Documented files not found: {missing}"

    def test_no_deleted_files_documented(self):
        """Deleted files should not appear as active components."""
        # These were deleted in Rounds 1-10
        deleted = [
            "src/discord/approval.py",
            "src/llm/haiku_classifier.py",
            "src/discord/routing.py",
        ]
        for d in deleted:
            # Should not appear outside "Removed Features" section
            lines = self.claude_md.splitlines()
            in_removed = False
            for line in lines:
                if "Removed" in line and "Feature" in line:
                    in_removed = True
                elif line.startswith("##") and "Removed" not in line:
                    in_removed = False
                if d in line and not in_removed:
                    pytest.fail(f"Deleted file {d} mentioned outside Removed Features: {line}")
