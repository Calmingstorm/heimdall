"""Round 24: Documentation accuracy tests (continued).

Verify README.md and CLAUDE.md accurately document features from Rounds 10-22:
session defense details, security model, tool execution flow, bot interop,
caching strategy, and performance optimizations.
"""

from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestClaudeMdSessionDefense:
    """CLAUDE.md documents all 5 session defense layers."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.claude_md = (ROOT / "CLAUDE.md").read_text()

    def test_context_separator_documented(self):
        assert "CONTEXT ABOVE IS HISTORY" in self.claude_md

    def test_selective_saving_documented(self):
        assert "Selective saving" in self.claude_md or "selective saving" in self.claude_md
        assert "tool-less" in self.claude_md.lower() or "Tool-less" in self.claude_md

    def test_abbreviated_task_history_documented(self):
        assert "get_task_history" in self.claude_md

    def test_compaction_error_omission_documented(self):
        assert "compaction" in self.claude_md.lower()
        assert "OMIT" in self.claude_md or "omit errors" in self.claude_md.lower()

    def test_fabrication_detection_documented(self):
        assert "detect_fabrication" in self.claude_md

    def test_hedging_detection_documented(self):
        assert "detect_hedging" in self.claude_md

    def test_compaction_triggers_reflection(self):
        assert "reflection" in self.claude_md.lower()
        assert "reflector" in self.claude_md.lower()


class TestClaudeMdSecurityModel:
    """CLAUDE.md documents security model from Rounds 19-20."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.claude_md = (ROOT / "CLAUDE.md").read_text()

    def test_secret_pattern_count(self):
        assert "10 patterns" in self.claude_md

    def test_github_tokens_documented(self):
        assert "ghp_" in self.claude_md

    def test_aws_tokens_documented(self):
        assert "AKIA" in self.claude_md

    def test_stripe_tokens_documented(self):
        assert "sk_live_" in self.claude_md

    def test_slack_tokens_documented(self):
        assert "xox" in self.claude_md

    def test_scrubbing_locations_documented(self):
        assert "9+" in self.claude_md or "9 " in self.claude_md
        assert "webhook" in self.claude_md.lower()
        assert "knowledge search" in self.claude_md.lower()

    def test_read_file_validation_documented(self):
        assert "read_file" in self.claude_md
        assert "lines" in self.claude_md
        assert "min(1000)" in self.claude_md or "line limit" in self.claude_md.lower()

    def test_prompt_injection_resistance_documented(self):
        assert "prompt injection" in self.claude_md.lower()


class TestClaudeMdToolExecution:
    """CLAUDE.md documents tool execution flow from Rounds 15-16."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.claude_md = (ROOT / "CLAUDE.md").read_text()

    def test_max_iterations_documented(self):
        assert "MAX_TOOL_ITERATIONS" in self.claude_md or "20" in self.claude_md

    def test_concurrent_tool_execution_documented(self):
        assert "asyncio.gather" in self.claude_md or "concurrently" in self.claude_md

    def test_tool_output_scrubbing_documented(self):
        assert "scrub" in self.claude_md.lower()
        assert "TOOL_OUTPUT_MAX_CHARS" in self.claude_md or "12000" in self.claude_md

    def test_tool_timeout_documented(self):
        assert "timeout" in self.claude_md.lower()
        assert "300" in self.claude_md

    def test_discord_native_tools_documented(self):
        assert "client.py" in self.claude_md
        assert "executor.py" in self.claude_md

    def test_skill_handoff_documented(self):
        assert "handoff" in self.claude_md


class TestClaudeMdBotInterop:
    """CLAUDE.md documents bot interop from Rounds 11-12."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.claude_md = (ROOT / "CLAUDE.md").read_text()

    def test_bot_buffer_documented(self):
        assert "combine_bot_messages" in self.claude_md

    def test_bot_preamble_documented(self):
        assert "EXECUTE immediately" in self.claude_md or "preamble" in self.claude_md

    def test_bot_mention_stripping_documented(self):
        assert "mention" in self.claude_md.lower()
        assert "strip" in self.claude_md.lower()

    def test_webhook_bypass_documented(self):
        assert "webhook" in self.claude_md.lower()
        assert "ALLOWED_WEBHOOK_IDS" in self.claude_md or "bypass" in self.claude_md.lower()

    def test_bot_dedup_documented(self):
        assert "dedup" in self.claude_md.lower() or "_processed_messages" in self.claude_md

    def test_tool_less_bot_anti_poisoning(self):
        """Tool-less bot responses not saved to history."""
        assert "anti-poisoning" in self.claude_md.lower() or "not saved" in self.claude_md.lower()


class TestClaudeMdCaching:
    """CLAUDE.md documents caching strategy from Rounds 21-22."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.claude_md = (ROOT / "CLAUDE.md").read_text()

    def test_tool_definition_caching_documented(self):
        assert "tool definition" in self.claude_md.lower() or "Tool definitions" in self.claude_md

    def test_system_prompt_caching_documented(self):
        assert "prompt" in self.claude_md.lower()
        assert "cached" in self.claude_md.lower() or "caching" in self.claude_md.lower()

    def test_connection_pooling_documented(self):
        assert "TCPConnector" in self.claude_md or "connection pool" in self.claude_md.lower()

    def test_ttl_caching_documented(self):
        assert "TTL" in self.claude_md or "ttl" in self.claude_md

    def test_cache_invalidation_documented(self):
        assert "/reload" in self.claude_md
        assert "invalidat" in self.claude_md.lower()

    def test_stale_cache_cleanup_documented(self):
        assert "_recent_actions" in self.claude_md or "stale" in self.claude_md.lower()


class TestReadmeSessionDefense:
    """README.md documents session defense details."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.readme = (ROOT / "README.md").read_text()

    def test_five_layers_listed(self):
        assert "Context separator" in self.readme
        assert "Selective saving" in self.readme
        assert "Abbreviated task history" in self.readme
        assert "Compaction error omission" in self.readme
        assert "Fabrication" in self.readme

    def test_context_separator_content(self):
        assert "HISTORY" in self.readme


class TestReadmeSecurity:
    """README.md documents security features."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.readme = (ROOT / "README.md").read_text()

    def test_secret_pattern_count(self):
        assert "10 secret patterns" in self.readme

    def test_github_tokens_listed(self):
        assert "ghp_" in self.readme

    def test_aws_tokens_listed(self):
        assert "AKIA" in self.readme

    def test_stripe_tokens_listed(self):
        assert "sk_live_" in self.readme

    def test_slack_tokens_listed(self):
        assert "xox" in self.readme

    def test_scrubbing_coverage(self):
        assert "9+" in self.readme

    def test_input_validation_mentioned(self):
        assert "read_file" in self.readme

    def test_prompt_injection_mentioned(self):
        assert "prompt injection" in self.readme.lower()


class TestReadmeToolLoop:
    """README.md documents tool loop mechanics."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.readme = (ROOT / "README.md").read_text()

    def test_iteration_limit(self):
        assert "20 iterations" in self.readme

    def test_concurrent_execution(self):
        assert "concurrently" in self.readme

    def test_output_truncation(self):
        assert "12000" in self.readme

    def test_timeout(self):
        assert "300s" in self.readme


class TestReadmeBotInteraction:
    """README.md documents bot interaction beyond config."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.readme = (ROOT / "README.md").read_text()

    def test_bot_buffer_mentioned(self):
        assert "combine_bot_messages" in self.readme or "buffered" in self.readme

    def test_bot_preamble_mentioned(self):
        assert "EXECUTE immediately" in self.readme or "preamble" in self.readme

    def test_webhook_bypass_mentioned(self):
        assert "webhook" in self.readme.lower()

    def test_anti_poisoning_mentioned(self):
        assert "anti-poisoning" in self.readme.lower() or "not saved" in self.readme.lower()


class TestReadmePerformance:
    """README.md mentions performance optimizations."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.readme = (ROOT / "README.md").read_text()

    def test_caching_mentioned(self):
        assert "caching" in self.readme.lower()

    def test_connection_pooling_mentioned(self):
        assert "connection pooling" in self.readme.lower()


class TestCrossDocConsistencyRound24:
    """README.md and CLAUDE.md are consistent on new sections."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.readme = (ROOT / "README.md").read_text()
        self.claude_md = (ROOT / "CLAUDE.md").read_text()

    def test_both_mention_5_layers(self):
        assert "5 Layer" in self.readme or "5 layer" in self.readme.lower()
        assert "5 Layer" in self.claude_md or "5 layer" in self.claude_md.lower()

    def test_both_mention_secret_patterns(self):
        assert "ghp_" in self.readme
        assert "ghp_" in self.claude_md

    def test_both_mention_tool_loop(self):
        assert "20" in self.readme
        assert "20" in self.claude_md

    def test_both_mention_scrubbing_locations(self):
        assert "9+" in self.readme
        assert "9+" in self.claude_md

    def test_both_mention_context_separator(self):
        assert "context separator" in self.readme.lower()
        assert "context separator" in self.claude_md.lower()

    def test_security_section_in_both(self):
        assert "### Security" in self.readme
        assert "### Security Model" in self.claude_md
