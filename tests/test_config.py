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

    def test_loads_actual_config_yml(self, monkeypatch):
        """Verify the real config.yml parses correctly after Round 1-4 changes."""
        monkeypatch.setenv("DISCORD_TOKEN", "test-token")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
        cfg = load_config("config.yml")
        assert cfg.discord.token == "test-token"
        assert cfg.anthropic.api_key == "sk-test"
        # Round 2: hosts should be empty (no personal data)
        assert cfg.tools.hosts == {}
        assert cfg.tools.allowed_services == []
        assert cfg.tools.allowed_playbooks == []
        # Round 2: new config fields have sensible defaults
        assert cfg.tools.prometheus_host == ""
        assert cfg.tools.ansible_host == ""
        assert cfg.tools.claude_code_host == ""
        assert cfg.tools.claude_code_user == ""
        assert cfg.tools.claude_code_dir == "/opt/project"
        # Round 3: bot interaction config
        assert cfg.discord.respond_to_bots is False
        assert cfg.discord.require_mention is False
        # Webhook disabled by default
        assert cfg.webhook.enabled is False
        assert cfg.webhook.secret == "test-secret"

    def test_actual_config_no_personal_data(self, monkeypatch):
        """Verify config.yml contains no personal IPs, user IDs, or hostnames."""
        monkeypatch.setenv("DISCORD_TOKEN", "t")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("WEBHOOK_SECRET", "s")
        raw = Path("config.yml").read_text()
        # No personal IPs
        assert "192.168.1" not in raw
        # No personal user IDs
        assert "441602773310767105" not in raw
        assert "757383353141035140" not in raw
        # No personal bot ID
        assert "1469121766910726210" not in raw
        # No personal hostnames as config values (not in comments)
        # Check that "ansiblex" is fully renamed
        assert "ansiblex" not in raw.lower()


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

    def test_new_config_fields_defaults(self):
        """Round 2 config fields have correct empty defaults."""
        tc = ToolsConfig()
        assert tc.prometheus_host == ""
        assert tc.ansible_host == ""
        assert tc.claude_code_host == ""
        assert tc.claude_code_user == ""
        assert tc.claude_code_dir == "/opt/project"
        assert tc.incus_host == ""

    def test_new_config_fields_set(self):
        """Round 2 config fields accept custom values."""
        tc = ToolsConfig(
            prometheus_host="monitor",
            ansible_host="controller",
            claude_code_host="devbox",
            claude_code_user="deploy",
            claude_code_dir="/home/deploy/project",
            incus_host="vmhost",
        )
        assert tc.prometheus_host == "monitor"
        assert tc.ansible_host == "controller"
        assert tc.claude_code_host == "devbox"
        assert tc.claude_code_user == "deploy"
        assert tc.claude_code_dir == "/home/deploy/project"
        assert tc.incus_host == "vmhost"
