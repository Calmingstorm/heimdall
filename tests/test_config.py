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
        config_path = tmp_dir / "config.yml"
        config_path.write_text(
            "discord:\n"
            "  token: ${DISCORD_TOKEN}\n"
        )
        cfg = load_config(config_path)
        assert cfg.discord.token == "test-token"
        # Defaults
        assert cfg.sessions.max_history == 50
        assert cfg.tools.command_timeout_seconds == 300

    def test_loads_actual_config_yml(self, monkeypatch):
        """Verify the real config.yml parses correctly after Round 1-4 changes."""
        monkeypatch.setenv("DISCORD_TOKEN", "test-token")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
        cfg = load_config("config.yml")
        assert cfg.discord.token == "test-token"
        # Round 2: hosts should be empty (no personal data)
        assert cfg.tools.hosts == {}
        # Round 2: new config fields have sensible defaults
        assert cfg.tools.claude_code_host == ""
        assert cfg.tools.claude_code_user == ""
        assert cfg.tools.claude_code_dir == "/opt/project"
        # Round 3: bot interaction config
        assert cfg.discord.respond_to_bots is False
        assert cfg.discord.require_mention is False
        # Webhook disabled by default
        assert cfg.webhook.enabled is False
        assert cfg.webhook.secret == "test-secret"
        # Round 6: timezone defaults to UTC
        assert cfg.timezone == "UTC"

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


class TestEnvVarDefaults:
    """Round 7: ${VAR:-default} env var substitution syntax."""

    def test_default_value_when_unset(self):
        os.environ.pop("_TEST_UNSET_VAR_99", None)
        result = _substitute_env_vars("val: ${_TEST_UNSET_VAR_99:-fallback}")
        assert result == "val: fallback"

    def test_default_value_empty_string(self):
        os.environ.pop("_TEST_UNSET_VAR_99", None)
        result = _substitute_env_vars("val: ${_TEST_UNSET_VAR_99:-}")
        assert result == "val: "

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("_TEST_SET_VAR_99", "real")
        result = _substitute_env_vars("val: ${_TEST_SET_VAR_99:-fallback}")
        assert result == "val: real"

    def test_required_var_still_raises(self):
        os.environ.pop("_TEST_UNSET_VAR_99", None)
        with pytest.raises(ValueError, match="_TEST_UNSET_VAR_99"):
            _substitute_env_vars("${_TEST_UNSET_VAR_99}")

    def test_mixed_required_and_optional(self, monkeypatch):
        monkeypatch.setenv("_TEST_REQ_VAR", "hello")
        os.environ.pop("_TEST_OPT_VAR", None)
        result = _substitute_env_vars("${_TEST_REQ_VAR} ${_TEST_OPT_VAR:-world}")
        assert result == "hello world"


class TestConfigWithoutWebhookSecret:
    """Round 7: config.yml loads without WEBHOOK_SECRET env var."""

    def test_loads_without_webhook_secret(self, monkeypatch):
        monkeypatch.setenv("DISCORD_TOKEN", "test-token")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
        cfg = load_config("config.yml")
        assert cfg.webhook.secret == ""
        assert cfg.webhook.enabled is False

    def test_loads_with_webhook_secret(self, monkeypatch):
        monkeypatch.setenv("DISCORD_TOKEN", "test-token")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("WEBHOOK_SECRET", "my-secret")
        cfg = load_config("config.yml")
        assert cfg.webhook.secret == "my-secret"


class TestDeploymentFiles:
    """Round 7: Docker and deployment files are generic."""

    def test_docker_compose_no_hardcoded_timezone(self):
        content = Path("docker-compose.yml").read_text()
        assert "America/New_York" not in content
        assert "${TZ:-UTC}" in content

    def test_docker_compose_service_names(self):
        content = Path("docker-compose.yml").read_text()
        assert "ansiblex" not in content.lower()
        assert "heimdall-bot" in content
        assert "heimdall-browser" in content
        assert "heimdall-voice" in content

    def test_dockerfile_uses_heimdall_user(self):
        content = Path("Dockerfile").read_text()
        assert "useradd" in content and "heimdall" in content
        assert "USER heimdall" in content
        assert "ansiblex" not in content.lower()

    def test_dockerfile_creates_ssh_dir(self):
        content = Path("Dockerfile").read_text()
        assert ".ssh" in content

    def test_dockerignore_excludes_ssh(self):
        content = Path(".dockerignore").read_text()
        assert "ssh/" in content

    def test_env_example_has_required_vars(self):
        content = Path(".env.example").read_text()
        assert "DISCORD_TOKEN" in content
        # Anthropic removed — no classifier

    def test_env_example_no_personal_data(self):
        content = Path(".env.example").read_text()
        assert "192.168.1" not in content
        assert "ansiblex" not in content.lower()


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
        assert tc.command_timeout_seconds == 300

    def test_new_config_fields_defaults(self):
        """Round 2 config fields have correct empty defaults."""
        tc = ToolsConfig()
        assert tc.claude_code_host == ""
        assert tc.claude_code_user == ""
        assert tc.claude_code_dir == "/opt/project"

    def test_new_config_fields_set(self):
        """Round 2 config fields accept custom values."""
        tc = ToolsConfig(
            claude_code_host="devbox",
            claude_code_user="deploy",
            claude_code_dir="/home/deploy/project",
        )
        assert tc.claude_code_host == "devbox"
        assert tc.claude_code_user == "deploy"
        assert tc.claude_code_dir == "/home/deploy/project"


class TestTimezoneConfig:
    """Round 6: timezone is configurable at the top level."""

    def test_default_timezone_is_utc(self):
        cfg = Config(discord=DiscordConfig(token="t"))
        assert cfg.timezone == "UTC"

    def test_custom_timezone(self):
        cfg = Config(discord=DiscordConfig(token="t"), timezone="America/New_York")
        assert cfg.timezone == "America/New_York"

    def test_timezone_in_actual_config(self, monkeypatch):
        """config.yml has timezone field set to UTC."""
        monkeypatch.setenv("DISCORD_TOKEN", "t")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("WEBHOOK_SECRET", "s")
        cfg = load_config("config.yml")
        assert cfg.timezone == "UTC"

    def test_no_hardcoded_america_new_york_in_source(self):
        """No source files should hardcode America/New_York anymore."""
        from pathlib import Path
        src_dir = Path("src")
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text()
            assert "America/New_York" not in content, (
                f"{py_file} still contains hardcoded America/New_York"
            )
