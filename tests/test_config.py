"""Tests for config/schema.py."""
from __future__ import annotations

import os
import pytest
from pathlib import Path

from src.config.schema import (
    Config,
    DiscordConfig,
    ToolsConfig,
    ToolHost,
    _substitute_env_vars,
    load_config,
)


class TestSubstituteEnvVars:
    def test_replaces_known_var(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "abc123")
        assert _substitute_env_vars("key: ${MY_TOKEN}") == "key: abc123"

    def test_raises_on_missing_var(self):
        # Ensure the var doesn't exist
        os.environ.pop("NONEXISTENT_VAR_12345", None)
        with pytest.raises(ValueError, match="NONEXISTENT_VAR_12345"):
            _substitute_env_vars("${NONEXISTENT_VAR_12345}")

    def test_no_substitution_needed(self):
        assert _substitute_env_vars("plain text") == "plain text"

    def test_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("A", "1")
        monkeypatch.setenv("B", "2")
        result = _substitute_env_vars("${A} and ${B}")
        assert result == "1 and 2"


class TestLoadConfig:
    def test_loads_minimal_config(self, tmp_dir: Path, monkeypatch):
        monkeypatch.setenv("DISCORD_TOKEN", "test-token")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        config_path = tmp_dir / "config.yml"
        config_path.write_text(
            "discord:\n"
            "  token: ${DISCORD_TOKEN}\n"
            "anthropic:\n"
            "  api_key: ${ANTHROPIC_API_KEY}\n"
        )
        cfg = load_config(config_path)
        assert cfg.discord.token == "test-token"
        assert cfg.anthropic.api_key == "sk-test"
        # Defaults
        assert cfg.sessions.max_history == 50
        assert cfg.tools.command_timeout_seconds == 30


class TestToolsConfig:
    def test_host_resolution(self):
        tc = ToolsConfig(
            hosts={"server": ToolHost(address="1.2.3.4")},
        )
        assert tc.hosts["server"].address == "1.2.3.4"
        assert tc.hosts["server"].ssh_user == "root"
        assert tc.hosts["server"].os == "linux"

    def test_defaults(self):
        tc = ToolsConfig()
        assert tc.enabled is True
        assert tc.command_timeout_seconds == 30
        assert tc.allowed_services == []
