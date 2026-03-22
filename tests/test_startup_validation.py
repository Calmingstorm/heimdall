"""Tests for startup validation — config loading error handling and startup warnings (Round 10)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config.schema import load_config, _substitute_env_vars


# ---------------------------------------------------------------------------
# Config loading: error handling
# ---------------------------------------------------------------------------

class TestConfigLoadingErrors:
    """load_config should give clear, actionable error messages."""

    def test_empty_config_file_exits(self, tmp_path: Path):
        """Empty config.yml should raise SystemExit with helpful message."""
        config_path = tmp_path / "config.yml"
        config_path.write_text("")
        with pytest.raises(SystemExit, match="empty or invalid"):
            load_config(config_path)

    def test_non_mapping_config_exits(self, tmp_path: Path):
        """Config that parses to a non-dict (e.g. a list) should raise SystemExit."""
        config_path = tmp_path / "config.yml"
        config_path.write_text("- item1\n- item2\n")
        with pytest.raises(SystemExit, match="empty or invalid"):
            load_config(config_path)

    def test_scalar_config_exits(self, tmp_path: Path):
        """Config that parses to a scalar value should raise SystemExit."""
        config_path = tmp_path / "config.yml"
        config_path.write_text("just a string\n")
        with pytest.raises(SystemExit, match="empty or invalid"):
            load_config(config_path)

    def test_invalid_yaml_exits(self, tmp_path: Path):
        """Malformed YAML should raise SystemExit with syntax guidance."""
        config_path = tmp_path / "config.yml"
        config_path.write_text("discord:\n  token: {bad yaml\n    broken: [")
        with pytest.raises(SystemExit, match="Failed to parse"):
            load_config(config_path)

    def test_missing_env_var_exits(self, tmp_path: Path):
        """Missing required env var should raise SystemExit referencing .env.example."""
        os.environ.pop("_TEST_MISSING_VAR_XYZ", None)
        config_path = tmp_path / "config.yml"
        config_path.write_text("discord:\n  token: ${_TEST_MISSING_VAR_XYZ}\n")
        with pytest.raises(SystemExit, match="_TEST_MISSING_VAR_XYZ"):
            load_config(config_path)

    def test_missing_env_var_message_mentions_env_example(self, tmp_path: Path):
        """Error message should point users to .env.example."""
        os.environ.pop("_TEST_MISSING_VAR_XYZ", None)
        config_path = tmp_path / "config.yml"
        config_path.write_text("discord:\n  token: ${_TEST_MISSING_VAR_XYZ}\n")
        with pytest.raises(SystemExit, match=r"\.env"):
            load_config(config_path)

    def test_valid_config_still_loads(self, tmp_path: Path, monkeypatch):
        """Ensure the error handling doesn't break normal loading."""
        monkeypatch.setenv("DISCORD_TOKEN", "test-token")
        config_path = tmp_path / "config.yml"
        config_path.write_text("discord:\n  token: ${DISCORD_TOKEN}\n")
        cfg = load_config(config_path)
        assert cfg.discord.token == "test-token"


# ---------------------------------------------------------------------------
# Startup config warnings
# ---------------------------------------------------------------------------

class TestStartupConfigWarnings:
    """LokiBot._log_startup_config should log useful warnings for new users."""

    def _make_stub(self, **config_overrides):
        """Create a minimal LokiBot stub for _log_startup_config tests."""
        from src.discord.client import LokiBot

        stub = MagicMock(spec=LokiBot)
        stub.config = MagicMock()
        stub.config.tools.hosts = {}
        stub.config.tools.claude_code_host = ""
        stub.config.anthropic.api_key = "sk-test"
        stub.config.openai_codex.enabled = False
        stub.config.discord.respond_to_bots = False
        stub.config.discord.require_mention = False
        stub.codex_client = None

        for key, value in config_overrides.items():
            parts = key.split(".")
            obj = stub
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

        return stub

    def test_warns_on_empty_hosts(self, caplog):
        stub = self._make_stub()
        with caplog.at_level(logging.WARNING, logger="loki.discord"):
            from src.discord.client import LokiBot
            LokiBot._log_startup_config(stub)
        assert any("No hosts configured" in r.message for r in caplog.records)

    def test_logs_configured_hosts(self, caplog):
        stub = self._make_stub()
        stub.config.tools.hosts = {"webserver": MagicMock(), "dbserver": MagicMock()}
        with caplog.at_level(logging.INFO, logger="loki.discord"):
            from src.discord.client import LokiBot
            LokiBot._log_startup_config(stub)
        assert any("webserver" in r.message and "dbserver" in r.message for r in caplog.records)

    def test_warns_on_missing_anthropic_key(self, caplog):
        stub = self._make_stub(**{"config.anthropic.api_key": ""})
        with caplog.at_level(logging.WARNING, logger="loki.discord"):
            from src.discord.client import LokiBot
            LokiBot._log_startup_config(stub)
        assert any("No Anthropic API key" in r.message for r in caplog.records)

    def test_warns_on_codex_enabled_but_unconfigured(self, caplog):
        stub = self._make_stub(**{"config.openai_codex.enabled": True})
        stub.codex_client = None
        with caplog.at_level(logging.WARNING, logger="loki.discord"):
            from src.discord.client import LokiBot
            LokiBot._log_startup_config(stub)
        assert any("Codex enabled but not configured" in r.message for r in caplog.records)

    def test_logs_respond_to_bots_enabled(self, caplog):
        stub = self._make_stub(**{"config.discord.respond_to_bots": True})
        with caplog.at_level(logging.INFO, logger="loki.discord"):
            from src.discord.client import LokiBot
            LokiBot._log_startup_config(stub)
        assert any("Bot interaction enabled" in r.message for r in caplog.records)

    def test_logs_require_mention_enabled(self, caplog):
        stub = self._make_stub(**{"config.discord.require_mention": True})
        with caplog.at_level(logging.INFO, logger="loki.discord"):
            from src.discord.client import LokiBot
            LokiBot._log_startup_config(stub)
        assert any("Mention-only mode" in r.message for r in caplog.records)

    def test_no_codex_warning_when_disabled(self, caplog):
        stub = self._make_stub(**{"config.openai_codex.enabled": False})
        stub.codex_client = None
        with caplog.at_level(logging.WARNING, logger="loki.discord"):
            from src.discord.client import LokiBot
            LokiBot._log_startup_config(stub)
        assert not any("Codex" in r.message for r in caplog.records)

    def test_info_on_missing_claude_code_host(self, caplog):
        stub = self._make_stub()
        with caplog.at_level(logging.INFO, logger="loki.discord"):
            from src.discord.client import LokiBot
            LokiBot._log_startup_config(stub)
        assert any("claude_code_host not set" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Unused import verification
# ---------------------------------------------------------------------------

class TestCodeCleanup:
    """Verify cleanup items from Round 10."""

    def test_no_unused_llmresponse_import_in_client(self):
        """LLMResponse should not be imported in client.py (was unused)."""
        client_code = Path("src/discord/client.py").read_text()
        # Should not have a direct import of LLMResponse
        assert "from ..llm.types import LLMResponse" not in client_code
        # The type name can still appear in comments — that's fine
