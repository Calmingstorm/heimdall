"""Tests for Heimdall CLI setup wizard."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.setup_wizard import (
    PLACEHOLDER_TOKEN,
    SetupWizard,
    build_config,
    build_env,
    detect_systemd,
    generate_api_token,
    is_setup_needed,
    validate_discord_token,
    validate_host_address,
    validate_ssh_user,
    validate_token_format,
)


# ---------------------------------------------------------------------------
# validate_token_format
# ---------------------------------------------------------------------------

class TestValidateTokenFormat:
    """Tests for Discord token format validation."""

    def test_valid_token_format(self):
        """Three-part dot-separated token is valid."""
        assert validate_token_format("MTIz.NDU2.Nzg5") is True

    def test_empty_string_invalid(self):
        """Empty string is not valid."""
        assert validate_token_format("") is False

    def test_none_invalid(self):
        """None input is not valid."""
        assert validate_token_format(None) is False

    def test_whitespace_only_invalid(self):
        """Whitespace-only string is not valid."""
        assert validate_token_format("   ") is False

    def test_two_parts_invalid(self):
        """Two-part token is not valid."""
        assert validate_token_format("abc.def") is False

    def test_four_parts_invalid(self):
        """Four-part token is not valid."""
        assert validate_token_format("a.b.c.d") is False

    def test_empty_part_invalid(self):
        """Token with empty part is not valid."""
        assert validate_token_format("abc..def") is False

    def test_real_looking_token(self):
        """Realistic token format passes."""
        token = "ODk5NjY3.OTk5OTk5.AbCdEfGhIjKlMnOpQrStUvWxYz"
        assert validate_token_format(token) is True

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is handled."""
        assert validate_token_format("  abc.def.ghi  ") is True


# ---------------------------------------------------------------------------
# validate_discord_token (async, mocked HTTP)
# ---------------------------------------------------------------------------

class TestValidateDiscordToken:
    """Tests for Discord token API validation."""

    async def test_valid_token_returns_bot_info(self):
        """Successful API call returns (True, info)."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "username": "Heimdall",
            "id": "123456789",
            "discriminator": "0",
        })

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("src.setup_wizard.aiohttp.ClientSession", return_value=mock_session):
            valid, info = await validate_discord_token("valid.token.here")

        assert valid is True
        assert info["username"] == "Heimdall"
        assert info["id"] == "123456789"

    async def test_invalid_token_returns_error(self):
        """401 response returns (False, error info)."""
        mock_resp = AsyncMock()
        mock_resp.status = 401

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("src.setup_wizard.aiohttp.ClientSession", return_value=mock_session):
            valid, info = await validate_discord_token("bad.token.here")

        assert valid is False
        assert "error" in info

    async def test_network_error_returns_false(self):
        """Network exception returns (False, error info)."""
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(side_effect=Exception("Connection refused"))

        with patch("src.setup_wizard.aiohttp.ClientSession", return_value=mock_session):
            valid, info = await validate_discord_token("any.token.here")

        assert valid is False
        assert "error" in info


# ---------------------------------------------------------------------------
# validate_host_address
# ---------------------------------------------------------------------------

class TestValidateHostAddress:
    """Tests for host address validation."""

    def test_valid_ipv4(self):
        assert validate_host_address("192.168.1.1") is True

    def test_valid_ipv4_zeros(self):
        assert validate_host_address("0.0.0.0") is True

    def test_invalid_ipv4_octet(self):
        """IP with octet > 255 is invalid."""
        assert validate_host_address("256.0.0.1") is False

    def test_valid_hostname(self):
        assert validate_host_address("myserver") is True

    def test_valid_fqdn(self):
        assert validate_host_address("server.example.com") is True

    def test_hostname_with_dashes(self):
        assert validate_host_address("my-server-01") is True

    def test_empty_string_invalid(self):
        assert validate_host_address("") is False

    def test_none_invalid(self):
        assert validate_host_address(None) is False

    def test_whitespace_only_invalid(self):
        assert validate_host_address("   ") is False

    def test_special_chars_invalid(self):
        assert validate_host_address("server!@#") is False


# ---------------------------------------------------------------------------
# validate_ssh_user
# ---------------------------------------------------------------------------

class TestValidateSshUser:
    """Tests for SSH username validation."""

    def test_valid_root(self):
        assert validate_ssh_user("root") is True

    def test_valid_deploy(self):
        assert validate_ssh_user("deploy") is True

    def test_valid_underscored(self):
        assert validate_ssh_user("_apt") is True

    def test_valid_with_numbers(self):
        assert validate_ssh_user("user01") is True

    def test_invalid_starts_with_number(self):
        assert validate_ssh_user("1user") is False

    def test_invalid_uppercase(self):
        assert validate_ssh_user("Root") is False

    def test_empty_string_invalid(self):
        assert validate_ssh_user("") is False

    def test_none_invalid(self):
        assert validate_ssh_user(None) is False


# ---------------------------------------------------------------------------
# generate_api_token
# ---------------------------------------------------------------------------

class TestGenerateApiToken:
    """Tests for API token generation."""

    def test_default_length(self):
        token = generate_api_token()
        assert len(token) == 32

    def test_custom_length(self):
        token = generate_api_token(48)
        assert len(token) == 48

    def test_alphanumeric_only(self):
        token = generate_api_token(100)
        assert token.isalnum()

    def test_tokens_are_unique(self):
        """Two generated tokens should differ."""
        t1 = generate_api_token()
        t2 = generate_api_token()
        assert t1 != t2


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------

class TestBuildConfig:
    """Tests for config dict generation from wizard answers."""

    def test_minimal_config(self):
        """Config with no optional answers has correct structure."""
        cfg = build_config()
        assert cfg["discord"]["token"] == "${DISCORD_TOKEN}"
        assert cfg["timezone"] == "UTC"
        assert cfg["tools"]["hosts"] == {}
        assert cfg["web"]["api_token"] == ""
        assert cfg["browser"]["enabled"] is False
        assert cfg["voice"]["enabled"] is False
        assert cfg["comfyui"]["enabled"] is False

    def test_custom_timezone(self):
        cfg = build_config(timezone="America/New_York")
        assert cfg["timezone"] == "America/New_York"

    def test_hosts_added(self):
        hosts = {
            "myserver": {"address": "10.0.0.1", "ssh_user": "deploy"},
            "other": {"address": "10.0.0.2"},
        }
        cfg = build_config(hosts=hosts)
        assert "myserver" in cfg["tools"]["hosts"]
        assert cfg["tools"]["hosts"]["myserver"]["address"] == "10.0.0.1"
        assert cfg["tools"]["hosts"]["myserver"]["ssh_user"] == "deploy"
        # Default ssh_user
        assert cfg["tools"]["hosts"]["other"]["ssh_user"] == "root"

    def test_features_enabled(self):
        features = {"browser": True, "voice": True, "comfyui": False}
        cfg = build_config(features=features)
        assert cfg["browser"]["enabled"] is True
        assert cfg["voice"]["enabled"] is True
        assert cfg["comfyui"]["enabled"] is False

    def test_web_api_token(self):
        cfg = build_config(web_api_token="my-secret-token")
        assert cfg["web"]["api_token"] == "my-secret-token"

    def test_claude_code_host(self):
        hosts = {"dev": {"address": "10.0.0.5", "ssh_user": "root"}}
        cfg = build_config(hosts=hosts, claude_code_host="dev")
        assert cfg["tools"]["claude_code_host"] == "dev"

    def test_claude_code_host_ignored_if_not_in_hosts(self):
        """claude_code_host is only set if the host exists."""
        cfg = build_config(claude_code_host="nonexistent")
        assert "claude_code_host" not in cfg["tools"]

    def test_config_has_all_required_sections(self):
        """Generated config has all sections needed by Config model."""
        cfg = build_config()
        required = [
            "discord", "openai_codex", "context", "sessions", "tools",
            "webhook", "learning", "search", "logging", "usage",
            "voice", "browser", "monitoring", "permissions", "comfyui", "web",
        ]
        for section in required:
            assert section in cfg, f"Missing section: {section}"

    def test_config_does_not_mutate_default(self):
        """build_config should not mutate the _DEFAULT_CONFIG template."""
        cfg1 = build_config(hosts={"a": {"address": "1.2.3.4"}})
        cfg2 = build_config()
        assert cfg2["tools"]["hosts"] == {}


# ---------------------------------------------------------------------------
# build_env
# ---------------------------------------------------------------------------

class TestBuildEnv:
    """Tests for .env file generation."""

    def test_contains_discord_token(self):
        content = build_env("my-test-token")
        assert "DISCORD_TOKEN=my-test-token" in content

    def test_includes_header(self):
        content = build_env("tok")
        assert "Generated by setup wizard" in content

    def test_extra_vars_added(self):
        content = build_env("tok", extra={"WEBHOOK_SECRET": "abc123"})
        assert "WEBHOOK_SECRET=abc123" in content

    def test_no_trailing_whitespace(self):
        content = build_env("tok")
        for line in content.splitlines():
            assert line == line.rstrip(), f"Trailing whitespace: {line!r}"

    def test_ends_with_newline(self):
        content = build_env("tok")
        assert content.endswith("\n")


# ---------------------------------------------------------------------------
# is_setup_needed
# ---------------------------------------------------------------------------

class TestIsSetupNeeded:
    """Tests for first-boot detection."""

    def test_no_config_file(self, tmp_path):
        """Setup needed when config.yml doesn't exist."""
        assert is_setup_needed(
            config_path=tmp_path / "config.yml",
            env_path=tmp_path / ".env",
        ) is True

    def test_no_env_file(self, tmp_path):
        """Setup needed when .env doesn't exist."""
        (tmp_path / "config.yml").write_text("discord:\n  token: test\n")
        assert is_setup_needed(
            config_path=tmp_path / "config.yml",
            env_path=tmp_path / ".env",
        ) is True

    def test_placeholder_token(self, tmp_path):
        """Setup needed when DISCORD_TOKEN is the placeholder."""
        (tmp_path / "config.yml").write_text("discord:\n  token: test\n")
        (tmp_path / ".env").write_text(f"DISCORD_TOKEN={PLACEHOLDER_TOKEN}\n")
        assert is_setup_needed(
            config_path=tmp_path / "config.yml",
            env_path=tmp_path / ".env",
        ) is True

    def test_empty_token(self, tmp_path):
        """Setup needed when DISCORD_TOKEN is empty."""
        (tmp_path / "config.yml").write_text("discord:\n  token: test\n")
        (tmp_path / ".env").write_text("DISCORD_TOKEN=\n")
        assert is_setup_needed(
            config_path=tmp_path / "config.yml",
            env_path=tmp_path / ".env",
        ) is True

    def test_valid_setup(self, tmp_path):
        """Setup NOT needed when both files exist and token is set."""
        (tmp_path / "config.yml").write_text("discord:\n  token: test\n")
        (tmp_path / ".env").write_text("DISCORD_TOKEN=MTIz.NDU2.Nzg5\n")
        assert is_setup_needed(
            config_path=tmp_path / "config.yml",
            env_path=tmp_path / ".env",
        ) is False

    def test_corrupt_env_file(self, tmp_path):
        """Setup needed when .env file is unreadable."""
        (tmp_path / "config.yml").write_text("discord:\n  token: test\n")
        env = tmp_path / ".env"
        env.write_text("DISCORD_TOKEN=valid.token.here\n")
        # Make unreadable
        env.chmod(0o000)
        result = is_setup_needed(
            config_path=tmp_path / "config.yml",
            env_path=env,
        )
        env.chmod(0o644)  # Restore for cleanup
        assert result is True


# ---------------------------------------------------------------------------
# detect_systemd
# ---------------------------------------------------------------------------

class TestDetectSystemd:
    """Tests for systemd detection."""

    def test_returns_bool(self):
        """detect_systemd returns a boolean."""
        result = detect_systemd()
        assert isinstance(result, bool)

    def test_with_systemctl_available(self):
        """Returns True when systemctl is on PATH."""
        with patch("src.setup_wizard.shutil.which", return_value="/usr/bin/systemctl"):
            assert detect_systemd() is True

    def test_without_systemctl(self):
        """Returns False when systemctl is not found."""
        with patch("src.setup_wizard.shutil.which", return_value=None):
            assert detect_systemd() is False


# ---------------------------------------------------------------------------
# SetupWizard — individual steps
# ---------------------------------------------------------------------------

class TestWizardStepDiscordToken:
    """Tests for the Discord token step of the wizard."""

    def _make_wizard(self, inputs):
        """Create wizard with mocked I/O."""
        input_iter = iter(inputs)
        return SetupWizard(
            config_path=Path("/tmp/test_config.yml"),
            env_path=Path("/tmp/test.env"),
            input_fn=lambda _prompt: next(input_iter),
            print_fn=lambda *a, **k: None,  # Suppress output
        )

    def test_valid_token_accepted(self):
        """Valid token is accepted on first try."""
        wizard = self._make_wizard(["MTIz.NDU2.Nzg5"])

        async def _mock_validate(token):
            return (True, {"username": "TestBot", "id": "123"})

        with patch("src.setup_wizard.validate_discord_token", _mock_validate):
            result = wizard.step_discord_token()

        assert result is True
        assert wizard.discord_token == "MTIz.NDU2.Nzg5"
        assert wizard.bot_info["username"] == "TestBot"

    def test_invalid_format_retried(self):
        """Invalid format prompts retry, then accepts valid token."""
        wizard = self._make_wizard(["bad", "still-bad", "MTIz.NDU2.Nzg5"])

        async def _mock_validate(token):
            return (True, {"username": "Bot", "id": "1"})

        with patch("src.setup_wizard.validate_discord_token", _mock_validate):
            result = wizard.step_discord_token()

        assert result is True

    def test_three_failures_abort(self):
        """Three failed attempts returns False."""
        wizard = self._make_wizard(["bad", "bad2", "bad3"])

        result = wizard.step_discord_token()
        assert result is False

    def test_empty_token_retried(self):
        """Empty input counts as an attempt."""
        wizard = self._make_wizard(["", "", ""])

        result = wizard.step_discord_token()
        assert result is False


class TestWizardStepHosts:
    """Tests for the hosts configuration step."""

    def _make_wizard(self, inputs):
        input_iter = iter(inputs)
        return SetupWizard(
            config_path=Path("/tmp/test_config.yml"),
            env_path=Path("/tmp/test.env"),
            input_fn=lambda _prompt: next(input_iter),
            print_fn=lambda *a, **k: None,
        )

    def test_add_one_host(self):
        """Adding a single host works correctly."""
        wizard = self._make_wizard([
            "y",            # Add a remote host?
            "myserver",     # Host name
            "10.0.0.1",    # Address
            "deploy",       # SSH user
            "n",            # Add another?
        ])
        wizard.step_hosts()
        assert "myserver" in wizard.hosts
        assert wizard.hosts["myserver"]["address"] == "10.0.0.1"
        assert wizard.hosts["myserver"]["ssh_user"] == "deploy"

    def test_add_multiple_hosts(self):
        """Adding two hosts works."""
        wizard = self._make_wizard([
            "y", "server1", "10.0.0.1", "root",
            "y", "server2", "10.0.0.2", "deploy",
            "n",
        ])
        wizard.step_hosts()
        assert len(wizard.hosts) == 2

    def test_skip_hosts(self):
        """Declining adds no hosts."""
        wizard = self._make_wizard(["n"])
        wizard.step_hosts()
        assert wizard.hosts == {}

    def test_invalid_address_skipped(self):
        """Invalid address doesn't add the host."""
        wizard = self._make_wizard([
            "y", "bad", "not!valid", # Invalid address
            "n",
        ])
        wizard.step_hosts()
        assert wizard.hosts == {}

    def test_default_ssh_user(self):
        """Empty SSH user defaults to root."""
        wizard = self._make_wizard([
            "y", "myhost", "10.0.0.1", "",  # Empty = default
            "n",
        ])
        wizard.step_hosts()
        assert wizard.hosts["myhost"]["ssh_user"] == "root"


class TestWizardStepFeatures:
    """Tests for the feature toggle step."""

    def _make_wizard(self, inputs):
        input_iter = iter(inputs)
        return SetupWizard(
            config_path=Path("/tmp/test_config.yml"),
            env_path=Path("/tmp/test.env"),
            input_fn=lambda _prompt: next(input_iter),
            print_fn=lambda *a, **k: None,
        )

    def test_all_features_enabled(self):
        """Answering yes to all enables all features."""
        wizard = self._make_wizard(["y", "y", "y"])
        wizard.step_features()
        assert wizard.features["browser"] is True
        assert wizard.features["voice"] is True
        assert wizard.features["comfyui"] is True

    def test_all_features_disabled(self):
        """Answering no to all disables all features."""
        wizard = self._make_wizard(["n", "n", "n"])
        wizard.step_features()
        assert wizard.features["browser"] is False
        assert wizard.features["voice"] is False
        assert wizard.features["comfyui"] is False

    def test_claude_code_offered_when_hosts_exist(self):
        """Claude Code option appears when hosts are configured."""
        wizard = self._make_wizard([
            "n", "n", "n",  # browser, voice, comfyui
            "y",             # claude code
        ])
        wizard.hosts = {"dev": {"address": "10.0.0.1", "ssh_user": "root"}}
        wizard.step_features()
        assert wizard.features["claude_code"] is True
        assert wizard.claude_code_host == "dev"

    def test_claude_code_not_offered_without_hosts(self):
        """Claude Code option is skipped when no hosts configured."""
        wizard = self._make_wizard(["n", "n", "n"])
        wizard.step_features()
        # Claude Code question never asked, so not in features
        assert "claude_code" not in wizard.features


class TestWizardStepWebToken:
    """Tests for the web UI token step."""

    def _make_wizard(self, inputs):
        input_iter = iter(inputs)
        return SetupWizard(
            config_path=Path("/tmp/test_config.yml"),
            env_path=Path("/tmp/test.env"),
            input_fn=lambda _prompt: next(input_iter),
            print_fn=lambda *a, **k: None,
        )

    def test_generate_random_token(self):
        """Choosing yes generates a random token."""
        wizard = self._make_wizard(["y"])
        wizard.step_web_token()
        assert len(wizard.web_api_token) == 32
        assert wizard.web_api_token.isalnum()

    def test_custom_token(self):
        """Choosing no and entering a custom token works."""
        wizard = self._make_wizard(["n", "my-custom-token"])
        wizard.step_web_token()
        assert wizard.web_api_token == "my-custom-token"

    def test_no_auth(self):
        """Choosing no and empty custom means no auth."""
        wizard = self._make_wizard(["n", ""])
        wizard.step_web_token()
        assert wizard.web_api_token == ""


# ---------------------------------------------------------------------------
# SetupWizard — write config step
# ---------------------------------------------------------------------------

class TestWizardStepWriteConfig:
    """Tests for the config file writing step."""

    def test_writes_config_and_env(self, tmp_path):
        """Wizard writes both config.yml and .env."""
        config_path = tmp_path / "config.yml"
        env_path = tmp_path / ".env"

        inputs = iter(["", ""])  # No overwrite prompts (files don't exist)
        wizard = SetupWizard(
            config_path=config_path,
            env_path=env_path,
            input_fn=lambda _: next(inputs),
            print_fn=lambda *a, **k: None,
        )
        wizard.discord_token = "MTIz.NDU2.Nzg5"
        wizard.timezone = "UTC"

        config_written, env_written = wizard.step_write_config()

        assert config_written is True
        assert env_written is True
        assert config_path.exists()
        assert env_path.exists()

    def test_config_is_valid_yaml(self, tmp_path):
        """Generated config.yml is valid YAML."""
        import yaml
        config_path = tmp_path / "config.yml"

        inputs = iter(["", ""])
        wizard = SetupWizard(
            config_path=config_path,
            env_path=tmp_path / ".env",
            input_fn=lambda _: next(inputs),
            print_fn=lambda *a, **k: None,
        )
        wizard.discord_token = "tok"
        wizard.step_write_config()

        data = yaml.safe_load(config_path.read_text())
        assert isinstance(data, dict)
        assert "discord" in data
        assert data["discord"]["token"] == "${DISCORD_TOKEN}"

    def test_env_contains_token(self, tmp_path):
        """Generated .env contains the Discord token."""
        env_path = tmp_path / ".env"

        inputs = iter(["", ""])
        wizard = SetupWizard(
            config_path=tmp_path / "config.yml",
            env_path=env_path,
            input_fn=lambda _: next(inputs),
            print_fn=lambda *a, **k: None,
        )
        wizard.discord_token = "my-secret-token"
        wizard.step_write_config()

        content = env_path.read_text()
        assert "DISCORD_TOKEN=my-secret-token" in content

    def test_env_permissions_restricted(self, tmp_path):
        """Generated .env has 0600 permissions."""
        env_path = tmp_path / ".env"

        inputs = iter(["", ""])
        wizard = SetupWizard(
            config_path=tmp_path / "config.yml",
            env_path=env_path,
            input_fn=lambda _: next(inputs),
            print_fn=lambda *a, **k: None,
        )
        wizard.discord_token = "tok"
        wizard.step_write_config()

        mode = env_path.stat().st_mode & 0o777
        assert mode == 0o600

    def test_existing_config_overwrite_declined(self, tmp_path):
        """Existing config.yml not overwritten when user declines."""
        config_path = tmp_path / "config.yml"
        config_path.write_text("original: content\n")

        inputs = iter(["n", ""])  # Decline config overwrite
        wizard = SetupWizard(
            config_path=config_path,
            env_path=tmp_path / ".env",
            input_fn=lambda _: next(inputs),
            print_fn=lambda *a, **k: None,
        )
        wizard.discord_token = "tok"
        config_written, _ = wizard.step_write_config()

        assert config_written is False
        assert config_path.read_text() == "original: content\n"

    def test_hosts_in_generated_config(self, tmp_path):
        """Host configuration appears in generated YAML."""
        import yaml
        config_path = tmp_path / "config.yml"

        inputs = iter(["", ""])
        wizard = SetupWizard(
            config_path=config_path,
            env_path=tmp_path / ".env",
            input_fn=lambda _: next(inputs),
            print_fn=lambda *a, **k: None,
        )
        wizard.discord_token = "tok"
        wizard.hosts = {"myhost": {"address": "10.0.0.1", "ssh_user": "deploy"}}
        wizard.step_write_config()

        data = yaml.safe_load(config_path.read_text())
        assert "myhost" in data["tools"]["hosts"]
        assert data["tools"]["hosts"]["myhost"]["address"] == "10.0.0.1"

    def test_features_in_generated_config(self, tmp_path):
        """Feature toggles appear in generated YAML."""
        import yaml
        config_path = tmp_path / "config.yml"

        inputs = iter(["", ""])
        wizard = SetupWizard(
            config_path=config_path,
            env_path=tmp_path / ".env",
            input_fn=lambda _: next(inputs),
            print_fn=lambda *a, **k: None,
        )
        wizard.discord_token = "tok"
        wizard.features = {"browser": True, "voice": False, "comfyui": True}
        wizard.step_write_config()

        data = yaml.safe_load(config_path.read_text())
        assert data["browser"]["enabled"] is True
        assert data["voice"]["enabled"] is False
        assert data["comfyui"]["enabled"] is True


# ---------------------------------------------------------------------------
# SetupWizard — systemd start step
# ---------------------------------------------------------------------------

class TestWizardStepStartService:
    """Tests for the systemd start step."""

    def _make_wizard(self, inputs):
        input_iter = iter(inputs)
        return SetupWizard(
            config_path=Path("/tmp/test_config.yml"),
            env_path=Path("/tmp/test.env"),
            input_fn=lambda _prompt: next(input_iter),
            print_fn=lambda *a, **k: None,
        )

    def test_skipped_without_systemd(self):
        """Step is silently skipped when systemd not detected."""
        wizard = self._make_wizard([])
        with patch("src.setup_wizard.detect_systemd", return_value=False):
            wizard.step_start_service()  # Should not prompt

    def test_start_declined(self):
        """Declining to start does nothing."""
        wizard = self._make_wizard(["n"])
        with patch("src.setup_wizard.detect_systemd", return_value=True):
            with patch("src.setup_wizard.subprocess.run") as mock_run:
                wizard.step_start_service()
                mock_run.assert_not_called()

    def test_start_accepted(self):
        """Accepting starts the service via systemctl."""
        wizard = self._make_wizard(["y"])
        with patch("src.setup_wizard.detect_systemd", return_value=True):
            with patch("src.setup_wizard.subprocess.run") as mock_run:
                wizard.step_start_service()
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert "systemctl" in args
                assert "start" in args
                assert "heimdall" in args


# ---------------------------------------------------------------------------
# SetupWizard — full run
# ---------------------------------------------------------------------------

class TestWizardFullRun:
    """Integration tests for the full wizard flow."""

    def test_full_run_minimal(self, tmp_path):
        """Full wizard run with minimal answers (just token, skip everything)."""
        config_path = tmp_path / "config.yml"
        env_path = tmp_path / ".env"

        inputs = iter([
            "MTIz.NDU2.Nzg5",  # Discord token
            "n",                # Skip Codex auth
            "n",                # No hosts
            "n", "n", "n",     # No features
            "n", "",            # No web token
            # No overwrite prompts (files don't exist)
            # Systemd skipped (mocked)
        ])

        wizard = SetupWizard(
            config_path=config_path,
            env_path=env_path,
            input_fn=lambda _: next(inputs),
            print_fn=lambda *a, **k: None,
        )

        async def _mock_validate(token):
            return (True, {"username": "Bot", "id": "1"})

        with patch("src.setup_wizard.validate_discord_token", _mock_validate):
            with patch("src.setup_wizard.detect_systemd", return_value=False):
                success = wizard.run()

        assert success is True
        assert config_path.exists()
        assert env_path.exists()

    def test_full_run_with_hosts_and_features(self, tmp_path):
        """Full wizard run with hosts and features configured."""
        config_path = tmp_path / "config.yml"
        env_path = tmp_path / ".env"

        inputs = iter([
            "MTIz.NDU2.Nzg5",  # Discord token
            "n",                # Skip Codex auth
            "y", "myhost", "10.0.0.1", "deploy",  # Add host
            "n",                # No more hosts
            "y", "n", "n",     # browser=yes, voice=no, comfyui=no
            "y",                # claude code
            "y",                # Generate web token
            # No overwrite prompts (files don't exist)
        ])

        wizard = SetupWizard(
            config_path=config_path,
            env_path=env_path,
            input_fn=lambda _: next(inputs),
            print_fn=lambda *a, **k: None,
        )

        async def _mock_validate(token):
            return (True, {"username": "Bot", "id": "1"})

        with patch("src.setup_wizard.validate_discord_token", _mock_validate):
            with patch("src.setup_wizard.detect_systemd", return_value=False):
                success = wizard.run()

        assert success is True

        import yaml
        data = yaml.safe_load(config_path.read_text())
        assert "myhost" in data["tools"]["hosts"]
        assert data["browser"]["enabled"] is True
        assert data["voice"]["enabled"] is False
        assert data["tools"]["claude_code_host"] == "myhost"
        assert len(data["web"]["api_token"]) == 32

    def test_failed_token_aborts(self, tmp_path):
        """Wizard returns False when token validation fails."""
        inputs = iter(["bad", "bad", "bad"])

        wizard = SetupWizard(
            config_path=tmp_path / "config.yml",
            env_path=tmp_path / ".env",
            input_fn=lambda _: next(inputs),
            print_fn=lambda *a, **k: None,
        )

        success = wizard.run()
        assert success is False
        assert not (tmp_path / "config.yml").exists()

    def test_codex_auth_failure_doesnt_abort(self, tmp_path):
        """Codex auth failure doesn't abort the whole wizard."""
        config_path = tmp_path / "config.yml"
        env_path = tmp_path / ".env"

        inputs = iter([
            "MTIz.NDU2.Nzg5",  # Discord token
            "y",                # Try Codex auth (will fail via mocked import)
            "n",                # No hosts
            "n", "n", "n",     # No features
            "n", "",            # No web token
        ])

        wizard = SetupWizard(
            config_path=config_path,
            env_path=env_path,
            input_fn=lambda _: next(inputs),
            print_fn=lambda *a, **k: None,
        )

        async def _mock_validate(token):
            return (True, {"username": "Bot", "id": "1"})

        # Mock the browser auth to raise an error (simulating auth failure)
        with patch("src.setup_wizard.validate_discord_token", _mock_validate):
            with patch("src.setup_wizard.detect_systemd", return_value=False):
                with patch("src.setup.webbrowser"):
                    with patch("src.setup._get_auth_code_browser", side_effect=RuntimeError("mocked auth failure")):
                        success = wizard.run()

        assert success is True

    def test_wizard_output_captures(self, tmp_path):
        """Wizard output goes through print_fn."""
        captured = []

        inputs = iter(["MTIz.NDU2.Nzg5", "n", "n", "n", "n", "n", "n", ""])

        wizard = SetupWizard(
            config_path=tmp_path / "config.yml",
            env_path=tmp_path / ".env",
            input_fn=lambda _: next(inputs),
            print_fn=lambda *a, **k: captured.append(" ".join(str(x) for x in a)),
        )

        async def _mock_validate(token):
            return (True, {"username": "Bot", "id": "1"})

        with patch("src.setup_wizard.validate_discord_token", _mock_validate):
            with patch("src.setup_wizard.detect_systemd", return_value=False):
                wizard.run()

        output = "\n".join(captured)
        assert "Setup Wizard" in output
        assert "Setup complete" in output


# ---------------------------------------------------------------------------
# SetupWizard prompt helpers
# ---------------------------------------------------------------------------

class TestWizardPromptHelpers:
    """Tests for wizard prompt utility methods."""

    def _make_wizard(self, inputs):
        input_iter = iter(inputs)
        return SetupWizard(
            config_path=Path("/tmp/config.yml"),
            env_path=Path("/tmp/.env"),
            input_fn=lambda _: next(input_iter),
            print_fn=lambda *a, **k: None,
        )

    def test_prompt_with_default(self):
        """Empty input returns the default value."""
        wizard = self._make_wizard([""])
        assert wizard._prompt("Enter value", default="fallback") == "fallback"

    def test_prompt_override_default(self):
        """Non-empty input overrides the default."""
        wizard = self._make_wizard(["custom"])
        assert wizard._prompt("Enter value", default="fallback") == "custom"

    def test_prompt_yes_no_default_yes(self):
        """Empty input with default=True returns True."""
        wizard = self._make_wizard([""])
        assert wizard._prompt_yes_no("Continue?", default=True) is True

    def test_prompt_yes_no_default_no(self):
        """Empty input with default=False returns False."""
        wizard = self._make_wizard([""])
        assert wizard._prompt_yes_no("Continue?", default=False) is False

    def test_prompt_yes_no_y(self):
        wizard = self._make_wizard(["y"])
        assert wizard._prompt_yes_no("Continue?") is True

    def test_prompt_yes_no_yes(self):
        wizard = self._make_wizard(["yes"])
        assert wizard._prompt_yes_no("Continue?") is True

    def test_prompt_yes_no_n(self):
        wizard = self._make_wizard(["n"])
        assert wizard._prompt_yes_no("Continue?", default=True) is False

    def test_prompt_yes_no_no(self):
        wizard = self._make_wizard(["no"])
        assert wizard._prompt_yes_no("Continue?", default=True) is False

    def test_prompt_yes_no_case_insensitive(self):
        wizard = self._make_wizard(["Y"])
        assert wizard._prompt_yes_no("Continue?") is True
