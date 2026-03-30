"""Round 10: Verify Round 9 dead code cleanup.

Comprehensive verification that all removed systems have zero stale references
in source code. Test files may contain intentional verification assertions
(testing absence) and comments (documenting history) — those are fine.
"""

from pathlib import Path

import pytest

from src.config.schema import Config
from src.discord.client import HeimdallBot
from src.tools.registry import TOOLS


# ---------------------------------------------------------------------------
# Source code cleanliness checks
# ---------------------------------------------------------------------------

class TestNoStaleClassifierInSource:
    """No executable classifier/haiku references in source code."""

    def test_no_haiku_in_source_strings(self):
        """No user-facing strings mentioning 'Haiku' in source."""
        for p in Path("src").rglob("*.py"):
            text = p.read_text()
            # Skip comments — only check string literals
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert "haiku" not in stripped.lower() or "detect" in stripped.lower(), (
                    f"Stale 'haiku' reference in {p}:{i}: {stripped}"
                )

    def test_no_classifier_import_in_source(self):
        """No file imports haiku_classifier or HaikuClassifier."""
        for p in Path("src").rglob("*.py"):
            text = p.read_text()
            assert "haiku_classifier" not in text, f"Stale import in {p}"
            assert "HaikuClassifier" not in text, f"Stale import in {p}"

    def test_classifier_module_not_importable(self):
        with pytest.raises(ImportError):
            import src.llm.haiku_classifier  # noqa: F401


class TestNoStaleApprovalInSource:
    """No executable approval references in source code."""

    def test_no_requires_approval_in_tools(self):
        for tool in TOOLS:
            assert "requires_approval" not in tool, tool.get("name", "?")

    def test_no_approval_module(self):
        with pytest.raises(ImportError):
            import src.discord.approval  # noqa: F401

    def test_no_auto_approve_in_config(self):
        cfg = Config(discord={"token": "t"})
        assert not hasattr(cfg.tools, "auto_approve")
        assert not hasattr(cfg.tools, "approval_timeout_seconds")

    def test_no_approval_import_in_client(self):
        source = Path("src/discord/client.py").read_text()
        assert "from .approval" not in source
        assert "request_approval" not in source
        assert "ApprovalView" not in source


class TestNoStaleRoutingInSource:
    """No references to removed routing module."""

    def test_routing_module_deleted(self):
        assert not Path("src/discord/routing.py").exists()

    def test_routing_not_importable(self):
        with pytest.raises(ImportError):
            import src.discord.routing  # noqa: F401

    def test_no_routing_import_in_client(self):
        source = Path("src/discord/client.py").read_text()
        assert "from .routing" not in source
        assert "resolve_claude_code_target" not in source
        assert "CLAUDE_CODE_DEFAULTS" not in source

    def test_no_init_routing_defaults(self):
        assert not hasattr(HeimdallBot, "_init_routing_defaults")
        source = Path("src/discord/client.py").read_text()
        assert "_init_routing_defaults" not in source


class TestNoStaleKeywordBypassInSource:
    """No keyword bypass or schedule intent guard references."""

    def test_no_is_task_by_keyword(self):
        for p in Path("src").rglob("*.py"):
            text = p.read_text()
            assert "is_task_by_keyword" not in text, f"Found in {p}"

    def test_no_schedule_intent_re(self):
        assert not hasattr(HeimdallBot, "_SCHEDULE_INTENT_RE")
        source = Path("src/discord/client.py").read_text()
        assert "_SCHEDULE_INTENT_RE" not in source

    def test_no_last_tool_use(self):
        source = Path("src/discord/client.py").read_text()
        assert "_last_tool_use" not in source


class TestNoStaleAnthropicConfig:
    """AnthropicConfig fully removed."""

    def test_no_anthropic_config_in_schema(self):
        source = Path("src/config/schema.py").read_text()
        assert "AnthropicConfig" not in source

    def test_no_anthropic_field_on_config(self):
        cfg = Config(discord={"token": "t"})
        assert not hasattr(cfg, "anthropic")


# ---------------------------------------------------------------------------
# /usage command accuracy
# ---------------------------------------------------------------------------

class TestUsageCommandAccuracy:
    """The /usage command should reflect current architecture."""

    def test_no_classifier_in_usage_response(self):
        """The /usage slash command should not mention Haiku or classifier."""
        source = Path("src/discord/client.py").read_text()
        # Find the cmd_usage function and check its response text
        in_usage = False
        for line in source.splitlines():
            if 'name="usage"' in line:
                in_usage = True
            if in_usage:
                assert "Haiku" not in line, "Stale 'Haiku' in /usage command"
                assert "Classifier" not in line, "Stale 'Classifier' in /usage command"
                if ")" in line and in_usage and "send_message" not in line:
                    # End of the send_message call
                    if line.strip() == ")":
                        break

    def test_usage_mentions_codex(self):
        source = Path("src/discord/client.py").read_text()
        assert "Codex" in source

    def test_usage_mentions_claude_code(self):
        source = Path("src/discord/client.py").read_text()
        assert "Claude Code" in source


# ---------------------------------------------------------------------------
# Test docstring cleanliness
# ---------------------------------------------------------------------------

class TestNoMisleadingDocstrings:
    """Test docstrings should not reference 'requires approval' for allowlist blocking."""

    def test_skill_context_gaps_no_approval_language(self):
        """test_skill_context_gaps.py should use 'allowlist' not 'approval' for blocking."""
        source = Path("tests/test_skill_context_gaps.py").read_text()
        assert "requires approval" not in source, (
            "Stale 'requires approval' docstrings in test_skill_context_gaps.py"
        )


# ---------------------------------------------------------------------------
# Cross-round consistency
# ---------------------------------------------------------------------------

class TestCrossRoundConsistency:
    """Verify all rounds 1-9 remain clean together."""

    def test_single_routing_path(self):
        """All messages go to Codex with tools (no 3-way routing)."""
        source = Path("src/discord/client.py").read_text()
        assert 'msg_type = "chat"' not in source
        assert 'msg_type = "claude_code"' not in source
        assert 'msg_type = "task"' not in source

    def test_no_dead_source_modules(self):
        """No deleted modules are importable."""
        dead_modules = [
            "src.discord.approval",
            "src.discord.routing",
            "src.llm.haiku_classifier",
        ]
        for mod in dead_modules:
            with pytest.raises(ImportError):
                __import__(mod)

    def test_system_prompt_under_5000(self):
        from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE
        assert len(SYSTEM_PROMPT_TEMPLATE) < 5000

    def test_local_execution_intact(self):
        """Round 5 local execution dispatch still works."""
        from src.tools.ssh import is_local_address, run_local_command
        assert is_local_address("127.0.0.1")
        assert is_local_address("localhost")
        assert not is_local_address("10.0.0.1")

    def test_personality_in_prompt(self):
        """Heimdall personality present in system prompt."""
        from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE
        assert "Heimdall" in SYSTEM_PROMPT_TEMPLATE
        assert "Exhausted omniscience" in SYSTEM_PROMPT_TEMPLATE

    def test_claude_code_in_architecture(self):
        """claude_code delegation guidance present in architecture.md."""
        from pathlib import Path
        arch = Path("data/context/architecture.md").read_text()
        assert "deep reasoning" in arch.lower()
        assert "claude_code" in arch

    def test_tool_count_stable(self):
        """Tool count should be stable (no accidental removals)."""
        assert len(TOOLS) >= 55, f"Expected 55+ tools, got {len(TOOLS)}"
