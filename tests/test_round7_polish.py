"""
Round 7 — Polish and edge case handling tests.

Tests for:
  - ANSI colour helpers (color_header, color_success, etc.)
  - Per-step Ctrl+C / EOFError handling in prompt helpers
  - Shared write_env_file function consolidation
  - Reconfigure mode (--reconfigure): loads existing config as defaults
  - load_existing_config helper
  - Postinstall upgrade detection messaging
  - Systemd service Restart=always
  - Web wizard: removed unused api import, reload countdown, spinner
"""
from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.setup_wizard import (
    SetupWizard,
    _DEFAULT_CONFIG,
    build_config,
    build_env,
    color_dim,
    color_error,
    color_header,
    color_success,
    color_warn,
    load_existing_config,
    write_env_file,
    _supports_color,
)
from src.packaging.validate import (
    REQUIRED_SERVICE_DIRECTIVES,
    parse_systemd_unit,
    validate_postinstall,
    validate_service_file,
)

# Paths
PACKAGING_DIR = Path(__file__).resolve().parent.parent / "packaging"
UI_DIR = Path(__file__).resolve().parent.parent / "ui"
SETUP_WIZARD_JS = UI_DIR / "js" / "pages" / "setup-wizard.js"
APP_JS = UI_DIR / "js" / "app.js"


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

class TestColorHelpers:
    """Tests for CLI colour output functions."""

    def test_color_header_contains_text(self):
        """Header wrapper preserves the original text."""
        result = color_header("hello")
        assert "hello" in result

    def test_color_success_contains_text(self):
        result = color_success("ok")
        assert "ok" in result

    def test_color_error_contains_text(self):
        result = color_error("fail")
        assert "fail" in result

    def test_color_warn_contains_text(self):
        result = color_warn("caution")
        assert "caution" in result

    def test_color_dim_contains_text(self):
        result = color_dim("subtle")
        assert "subtle" in result

    def test_no_color_env_disables_ansi(self):
        """NO_COLOR environment variable disables colour codes."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            assert _supports_color() is False

    def test_force_color_env_enables_ansi(self):
        """FORCE_COLOR environment variable forces colour codes on."""
        with patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=False):
            assert _supports_color() is True


# ---------------------------------------------------------------------------
# Per-step Ctrl+C handling in prompt helpers
# ---------------------------------------------------------------------------

class TestPromptCtrlCHandling:
    """Ctrl+C and EOFError in prompts return defaults gracefully."""

    def _make_wizard(self, input_fn):
        """Create a wizard with custom input_fn."""
        output = []
        return SetupWizard(
            config_path=Path("/tmp/test_cfg.yml"),
            env_path=Path("/tmp/test_env"),
            input_fn=input_fn,
            print_fn=lambda *a: output.append(str(a)),
        ), output

    def test_prompt_returns_default_on_keyboard_interrupt(self):
        """_prompt returns default when user presses Ctrl+C."""
        def raise_interrupt(msg):
            raise KeyboardInterrupt()
        wiz, _ = self._make_wizard(raise_interrupt)
        result = wiz._prompt("Name", default="fallback")
        assert result == "fallback"

    def test_prompt_returns_empty_on_eof_no_default(self):
        """_prompt returns empty string on EOFError with no default."""
        def raise_eof(msg):
            raise EOFError()
        wiz, _ = self._make_wizard(raise_eof)
        result = wiz._prompt("Name")
        assert result == ""

    def test_prompt_yes_no_returns_default_on_ctrl_c(self):
        """_prompt_yes_no returns default (True) on Ctrl+C."""
        def raise_interrupt(msg):
            raise KeyboardInterrupt()
        wiz, _ = self._make_wizard(raise_interrupt)
        result = wiz._prompt_yes_no("Continue?", default=True)
        assert result is True

    def test_prompt_yes_no_returns_false_default_on_ctrl_c(self):
        """_prompt_yes_no returns default (False) on Ctrl+C."""
        def raise_interrupt(msg):
            raise KeyboardInterrupt()
        wiz, _ = self._make_wizard(raise_interrupt)
        result = wiz._prompt_yes_no("Continue?", default=False)
        assert result is False

    def test_prompt_secret_returns_empty_on_ctrl_c(self):
        """_prompt_secret returns empty string on Ctrl+C."""
        def raise_interrupt(msg):
            raise KeyboardInterrupt()
        wiz, _ = self._make_wizard(raise_interrupt)
        result = wiz._prompt_secret("Token")
        assert result == ""

    def test_prompt_secret_returns_empty_on_eof(self):
        """_prompt_secret returns empty string on EOFError."""
        def raise_eof(msg):
            raise EOFError()
        wiz, _ = self._make_wizard(raise_eof)
        result = wiz._prompt_secret("Token")
        assert result == ""


# ---------------------------------------------------------------------------
# Shared write_env_file
# ---------------------------------------------------------------------------

class TestSharedWriteEnvFile:
    """Tests for the shared write_env_file function."""

    def test_writes_content(self, tmp_path):
        """write_env_file writes content to the specified path."""
        env_path = tmp_path / "sub" / ".env"
        write_env_file(env_path, "DISCORD_TOKEN=abc\n")
        assert env_path.read_text() == "DISCORD_TOKEN=abc\n"

    def test_creates_parent_dirs(self, tmp_path):
        """write_env_file creates parent directories."""
        env_path = tmp_path / "a" / "b" / ".env"
        write_env_file(env_path, "X=1\n")
        assert env_path.exists()

    def test_sets_restricted_permissions(self, tmp_path):
        """write_env_file sets 0600 permissions."""
        env_path = tmp_path / ".env"
        write_env_file(env_path, "SECRET=x\n")
        mode = env_path.stat().st_mode & 0o777
        assert mode == 0o600

    def test_api_uses_shared_function(self):
        """src/web/api.py imports write_env_file from setup_wizard."""
        import src.web.api as api_mod
        assert hasattr(api_mod, 'write_env_file')


class TestApiWriteEnvFileDelegation:
    """Verify the web API _write_env_file delegates to the shared function."""

    def test_delegation(self, tmp_path):
        """api._write_env_file calls shared write_env_file."""
        from src.web.api import _write_env_file
        env_path = tmp_path / ".env"
        _write_env_file(env_path, "TOKEN=abc\n")
        assert env_path.read_text() == "TOKEN=abc\n"
        mode = env_path.stat().st_mode & 0o777
        assert mode == 0o600


# ---------------------------------------------------------------------------
# load_existing_config
# ---------------------------------------------------------------------------

class TestLoadExistingConfig:
    """Tests for loading existing config in reconfigure mode."""

    def test_returns_none_when_missing(self, tmp_path):
        """Returns None if config file doesn't exist."""
        result = load_existing_config(tmp_path / "missing.yml")
        assert result is None

    def test_returns_dict_for_valid_yaml(self, tmp_path):
        """Returns parsed dict for valid YAML config."""
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text(yaml.dump({"timezone": "US/Eastern", "tools": {"hosts": {}}}))
        result = load_existing_config(cfg_path)
        assert result["timezone"] == "US/Eastern"

    def test_returns_none_for_invalid_yaml(self, tmp_path):
        """Returns None for unparseable YAML."""
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text("{{{invalid yaml")
        result = load_existing_config(cfg_path)
        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path):
        """Returns None for empty YAML file."""
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text("")
        result = load_existing_config(cfg_path)
        assert result is None


# ---------------------------------------------------------------------------
# Reconfigure mode
# ---------------------------------------------------------------------------

class TestReconfigureMode:
    """Tests for --reconfigure flag on SetupWizard."""

    def test_loads_existing_hosts(self, tmp_path):
        """Reconfigure mode pre-populates hosts from existing config."""
        cfg_path = tmp_path / "config.yml"
        cfg = build_config(hosts={"srv1": {"address": "10.0.0.5", "ssh_user": "admin"}})
        cfg_path.write_text(yaml.dump(cfg))

        wiz = SetupWizard(
            config_path=cfg_path,
            env_path=tmp_path / ".env",
            reconfigure=True,
            input_fn=lambda m: "",
            print_fn=lambda *a: None,
        )
        assert "srv1" in wiz.hosts
        assert wiz.hosts["srv1"]["address"] == "10.0.0.5"

    def test_loads_existing_features(self, tmp_path):
        """Reconfigure mode pre-populates feature flags."""
        cfg_path = tmp_path / "config.yml"
        cfg = build_config(features={"browser": True, "voice": False, "comfyui": True})
        cfg_path.write_text(yaml.dump(cfg))

        wiz = SetupWizard(
            config_path=cfg_path,
            env_path=tmp_path / ".env",
            reconfigure=True,
            input_fn=lambda m: "",
            print_fn=lambda *a: None,
        )
        assert wiz.features["browser"] is True
        assert wiz.features["voice"] is False
        assert wiz.features["comfyui"] is True

    def test_loads_timezone(self, tmp_path):
        """Reconfigure mode pre-populates timezone."""
        cfg_path = tmp_path / "config.yml"
        cfg = build_config(timezone="US/Pacific")
        cfg_path.write_text(yaml.dump(cfg))

        wiz = SetupWizard(
            config_path=cfg_path,
            env_path=tmp_path / ".env",
            reconfigure=True,
            input_fn=lambda m: "",
            print_fn=lambda *a: None,
        )
        assert wiz.timezone == "US/Pacific"

    def test_loads_web_api_token(self, tmp_path):
        """Reconfigure mode pre-populates web API token."""
        cfg_path = tmp_path / "config.yml"
        cfg = build_config(web_api_token="my-secret-token")
        cfg_path.write_text(yaml.dump(cfg))

        wiz = SetupWizard(
            config_path=cfg_path,
            env_path=tmp_path / ".env",
            reconfigure=True,
            input_fn=lambda m: "",
            print_fn=lambda *a: None,
        )
        assert wiz.web_api_token == "my-secret-token"

    def test_no_existing_config_is_safe(self, tmp_path):
        """Reconfigure mode with missing config still works (no pre-populated values)."""
        wiz = SetupWizard(
            config_path=tmp_path / "nonexistent.yml",
            env_path=tmp_path / ".env",
            reconfigure=True,
            input_fn=lambda m: "",
            print_fn=lambda *a: None,
        )
        assert wiz.hosts == {}
        assert wiz.timezone == "UTC"

    def test_reconfigure_banner(self, tmp_path):
        """Reconfigure mode shows 'Reconfigure' in the banner."""
        output = []
        # The wizard will abort at step_discord_token since input returns empty
        wiz = SetupWizard(
            config_path=tmp_path / "config.yml",
            env_path=tmp_path / ".env",
            reconfigure=True,
            input_fn=lambda m: "",
            print_fn=lambda *a: output.append(" ".join(str(x) for x in a)),
        )
        wiz.run()
        banner_text = " ".join(output)
        assert "Reconfigure" in banner_text

    def test_non_reconfigure_banner(self, tmp_path):
        """Non-reconfigure mode shows 'Setup' in the banner."""
        output = []
        wiz = SetupWizard(
            config_path=tmp_path / "config.yml",
            env_path=tmp_path / ".env",
            reconfigure=False,
            input_fn=lambda m: "",
            print_fn=lambda *a: output.append(" ".join(str(x) for x in a)),
        )
        wiz.run()
        banner_text = " ".join(output)
        assert "Setup Wizard" in banner_text


# ---------------------------------------------------------------------------
# Postinstall upgrade detection
# ---------------------------------------------------------------------------

class TestPostinstallUpgradeDetection:
    """Tests for upgrade vs fresh install messaging in postinstall.sh."""

    @pytest.fixture
    def postinstall_content(self):
        return (PACKAGING_DIR / "postinstall.sh").read_text()

    def test_has_is_upgrade_variable(self, postinstall_content):
        """Script sets IS_UPGRADE based on existing config files."""
        assert 'IS_UPGRADE=' in postinstall_content

    def test_checks_config_and_env_exist(self, postinstall_content):
        """Upgrade detection checks both config.yml and .env."""
        assert '"$CONFIG_DIR/config.yml"' in postinstall_content
        assert '"$CONFIG_DIR/.env"' in postinstall_content

    def test_shows_upgrade_message(self, postinstall_content):
        """Script prints 'upgraded' message for upgrade installs."""
        assert "upgraded successfully" in postinstall_content

    def test_shows_fresh_install_message(self, postinstall_content):
        """Script prints 'installed' message for fresh installs."""
        assert "installed successfully" in postinstall_content

    def test_upgrade_preserves_existing_config(self, postinstall_content):
        """Upgrade message mentions config preservation."""
        assert "Existing configuration preserved" in postinstall_content

    def test_upgrade_suggests_restart(self, postinstall_content):
        """Upgrade message suggests restarting the service."""
        assert "systemctl restart heimdall" in postinstall_content

    def test_upgrade_mentions_reconfigure(self, postinstall_content):
        """Upgrade message mentions --reconfigure flag."""
        assert "--reconfigure" in postinstall_content

    def test_fresh_install_mentions_wizard(self, postinstall_content):
        """Fresh install message mentions setup wizard."""
        assert "src.setup wizard" in postinstall_content


# ---------------------------------------------------------------------------
# Service file Restart=always
# ---------------------------------------------------------------------------

class TestServiceRestartAlways:
    """Tests for systemd service Restart=always change."""

    @pytest.fixture
    def service_content(self):
        return (PACKAGING_DIR / "heimdall.service").read_text()

    def test_restart_directive_is_always(self, service_content):
        """Service file uses Restart=always for web wizard restart support."""
        parsed = parse_systemd_unit(service_content)
        assert parsed["Service"]["Restart"] == "always"

    def test_required_directives_constant_updated(self):
        """REQUIRED_SERVICE_DIRECTIVES reflects Restart=always."""
        assert REQUIRED_SERVICE_DIRECTIVES["Restart"] == "always"

    def test_still_has_restart_delay(self, service_content):
        """Service file still has RestartSec to prevent crash loops."""
        parsed = parse_systemd_unit(service_content)
        assert int(parsed["Service"]["RestartSec"]) >= 5

    def test_service_validates_cleanly(self, service_content):
        """Full service file passes validation."""
        errors = validate_service_file(service_content)
        assert errors == []


# ---------------------------------------------------------------------------
# Web wizard polish
# ---------------------------------------------------------------------------

class TestWebWizardPolish:
    """Tests for web wizard UI improvements."""

    @pytest.fixture
    def setup_wizard_js(self):
        return SETUP_WIZARD_JS.read_text()

    def test_no_unused_api_import(self, setup_wizard_js):
        """setup-wizard.js no longer imports unused api module."""
        # Should NOT have "import { api }" or "import { api } from"
        assert "import { api }" not in setup_wizard_js

    def test_has_loading_spinner(self, setup_wizard_js):
        """Submit button shows a spinner SVG while submitting."""
        assert "animate-spin" in setup_wizard_js

    def test_has_reload_countdown(self, setup_wizard_js):
        """Success state shows a reload countdown."""
        assert "reloadCountdown" in setup_wizard_js

    def test_has_window_reload(self, setup_wizard_js):
        """After countdown, the page auto-reloads."""
        assert "window.location.reload()" in setup_wizard_js

    def test_has_select_all_on_token(self, setup_wizard_js):
        """Generated token display uses select-all for easy copying."""
        assert "select-all" in setup_wizard_js

    def test_still_uses_raw_fetch(self, setup_wizard_js):
        """Setup wizard still uses raw fetch() for the POST."""
        assert "fetch('/api/setup/complete'" in setup_wizard_js

    def test_has_page_fade_in(self, setup_wizard_js):
        """Setup wizard has page-fade-in class."""
        assert "page-fade-in" in setup_wizard_js

    def test_has_error_banner(self, setup_wizard_js):
        """Setup wizard has error alert banner."""
        assert 'role="alert"' in setup_wizard_js


# ---------------------------------------------------------------------------
# Step output uses colour functions
# ---------------------------------------------------------------------------

class TestWizardColoredOutput:
    """Verify wizard output goes through colour helpers."""

    def test_step_headers_use_color_header(self):
        """Step methods output coloured headers (when TTY)."""
        output = []
        wiz = SetupWizard(
            config_path=Path("/tmp/test_cfg.yml"),
            env_path=Path("/tmp/test_env"),
            input_fn=lambda m: "",
            print_fn=lambda *a: output.append(" ".join(str(x) for x in a)),
        )
        # Run step that doesn't need token validation
        wiz.step_hosts()
        header_output = " ".join(output)
        assert "Step 3" in header_output
        assert "Remote Hosts" in header_output

    def test_step_discord_token_shows_attempt_count(self):
        """Token step shows remaining attempts on failure."""
        output = []
        wiz = SetupWizard(
            config_path=Path("/tmp/test_cfg.yml"),
            env_path=Path("/tmp/test_env"),
            input_fn=lambda m: "",
            print_fn=lambda *a: output.append(" ".join(str(x) for x in a)),
        )
        wiz.step_discord_token()
        all_output = " ".join(output)
        assert "attempt" in all_output.lower() or "left" in all_output.lower()


# ---------------------------------------------------------------------------
# Wizard _write_env_file delegates to shared function
# ---------------------------------------------------------------------------

class TestWizardWriteEnvFileDelegation:
    """Verify SetupWizard._write_env_file uses the shared write_env_file."""

    def test_sets_permissions_via_shared(self, tmp_path):
        """Wizard's _write_env_file produces same 0600 permissions."""
        output = []
        wiz = SetupWizard(
            config_path=tmp_path / "config.yml",
            env_path=tmp_path / ".env",
            input_fn=lambda m: "",
            print_fn=lambda *a: output.append(str(a)),
        )
        wiz._write_env_file("TOKEN=abc\n")
        mode = (tmp_path / ".env").stat().st_mode & 0o777
        assert mode == 0o600

    def test_creates_file_content(self, tmp_path):
        """Wizard's _write_env_file writes correct content."""
        wiz = SetupWizard(
            config_path=tmp_path / "config.yml",
            env_path=tmp_path / ".env",
            input_fn=lambda m: "",
            print_fn=lambda *a: None,
        )
        wiz._write_env_file("DISCORD_TOKEN=xyz\n")
        assert (tmp_path / ".env").read_text() == "DISCORD_TOKEN=xyz\n"
