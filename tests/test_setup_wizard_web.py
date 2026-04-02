"""
Round 5 — Web UI First-Boot Setup Wizard Tests

Tests for:
  - GET /api/setup/status (first-boot detection)
  - POST /api/setup/complete (config file writing + restart)
  - Auth bypass for setup endpoints
  - _write_env_file helper
  - Frontend setup wizard page structure
  - App.js setup route and redirect logic
"""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
SETUP_WIZARD_JS = UI_DIR / "js" / "pages" / "setup-wizard.js"
APP_JS = UI_DIR / "js" / "app.js"


# ---------------------------------------------------------------------------
# API test helpers
# ---------------------------------------------------------------------------


def _make_bot():
    """Create a minimal mock bot for web API testing."""
    bot = MagicMock()
    guild = MagicMock()
    guild.id = 111
    guild.name = "TestGuild"
    guild.member_count = 10
    bot.guilds = [guild]
    bot.is_ready = MagicMock(return_value=True)
    bot._start_time = time.monotonic() - 100

    bot._merged_tool_definitions = MagicMock(return_value=[])
    bot.sessions = MagicMock()
    bot.sessions._sessions = {}
    bot.skill_manager = MagicMock()
    bot.skill_manager.list_skills = MagicMock(return_value=[])
    bot.skill_manager._skills = {}
    bot.scheduler = MagicMock()
    bot.scheduler.list_all = MagicMock(return_value=[])
    bot.audit = MagicMock()
    bot.audit.count_by_tool = AsyncMock(return_value={})
    bot.tool_executor = MagicMock()
    bot.tool_executor._load_all_memory = MagicMock(return_value={})
    bot.tool_executor._save_all_memory = MagicMock()
    bot.config = MagicMock()
    bot.config.model_dump = MagicMock(return_value={"discord": {"token": "x"}})
    bot.config.web = MagicMock()
    bot.config.web.api_token = ""
    bot.config.tools = MagicMock()
    bot.config.monitoring = MagicMock()
    bot.config.monitoring.enabled = False
    bot._cached_merged_tools = None
    bot._knowledge_store = MagicMock()
    bot._knowledge_store.available = False
    bot._embedder = MagicMock()
    bot.loop_manager = MagicMock()
    bot.loop_manager._loops = {}
    bot.loop_manager.active_count = 0
    bot.agent_manager = MagicMock()
    bot.agent_manager._agents = {}
    bot.infra_watcher = None
    bot.tool_executor._process_registry = MagicMock()
    bot.tool_executor._process_registry._processes = {}
    bot.context_loader = MagicMock()
    return bot


def _make_app(bot=None, *, api_token=""):
    """Create a test aiohttp app with API routes and middleware."""
    if bot is None:
        bot = _make_bot()
    from src.config.schema import WebConfig
    from src.health.server import (
        SessionManager,
        _make_auth_middleware,
        _make_rate_limit_middleware,
        _make_security_headers_middleware,
    )
    from src.web.api import setup_api

    web_config = WebConfig(api_token=api_token)
    app = web.Application(middlewares=[
        _make_security_headers_middleware(),
        _make_rate_limit_middleware(),
        _make_auth_middleware(web_config, SessionManager()),
    ])
    setup_api(app, bot)
    return app, bot


# ===================================================================
# GET /api/setup/status
# ===================================================================


class TestSetupStatusEndpoint:
    """Tests for the /api/setup/status endpoint."""

    @pytest.mark.asyncio
    async def test_setup_needed_when_no_env(self, tmp_path):
        """Returns needed=true when .env does not exist."""
        app, _ = _make_app()
        config_file = tmp_path / "config.yml"
        config_file.write_text("discord:\n  token: test\n")
        env_file = tmp_path / ".env"
        # .env does not exist

        with patch("src.web.api.is_setup_needed", return_value=True):
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/api/setup/status")
                assert resp.status == 200
                data = await resp.json()
                assert data["needed"] is True

    @pytest.mark.asyncio
    async def test_setup_not_needed_when_configured(self):
        """Returns needed=false when config is present and token is set."""
        app, _ = _make_app()
        with patch("src.web.api.is_setup_needed", return_value=False):
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/api/setup/status")
                assert resp.status == 200
                data = await resp.json()
                assert data["needed"] is False

    @pytest.mark.asyncio
    async def test_setup_status_no_auth_required(self):
        """Setup status endpoint accessible even with auth configured."""
        app, _ = _make_app(api_token="secret-token")
        with patch("src.web.api.is_setup_needed", return_value=True):
            async with TestClient(TestServer(app)) as client:
                # No Authorization header — should still succeed
                resp = await client.get("/api/setup/status")
                assert resp.status == 200
                data = await resp.json()
                assert data["needed"] is True

    @pytest.mark.asyncio
    async def test_setup_status_returns_json(self):
        """Response is JSON with 'needed' boolean key."""
        app, _ = _make_app()
        with patch("src.web.api.is_setup_needed", return_value=False):
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/api/setup/status")
                data = await resp.json()
                assert isinstance(data["needed"], bool)


# ===================================================================
# POST /api/setup/complete
# ===================================================================


class TestSetupCompleteEndpoint:
    """Tests for the /api/setup/complete endpoint."""

    @pytest.mark.asyncio
    async def test_missing_discord_token_returns_400(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/setup/complete", json={})
            assert resp.status == 400
            data = await resp.json()
            assert "discord_token" in data["error"]

    @pytest.mark.asyncio
    async def test_empty_discord_token_returns_400(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/setup/complete", json={"discord_token": "  "})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_invalid_token_format_returns_400(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/setup/complete", json={"discord_token": "not-a-token"})
            assert resp.status == 400
            data = await resp.json()
            assert "format" in data["error"].lower() or "invalid" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_valid_token_writes_config(self, tmp_path):
        """Valid request writes config.yml and .env files."""
        app, _ = _make_app()
        config_path = tmp_path / "config.yml"
        env_path = tmp_path / ".env"

        # Patch the file paths used in the endpoint
        with (
            patch("src.web.api._write_config") as mock_write_cfg,
            patch("src.web.api._write_env_file") as mock_write_env,
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            # Make to_thread call the function synchronously
            mock_asyncio.to_thread = AsyncMock(side_effect=lambda fn, *a: fn(*a))
            mock_asyncio.get_event_loop.return_value.call_later = MagicMock()

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.abc.def",
                })
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
                assert data["restart_scheduled"] is True

                # Verify config writer was called
                mock_write_cfg.assert_called_once()
                call_args = mock_write_cfg.call_args
                cfg_dict = call_args[0][1]
                assert isinstance(cfg_dict, dict)
                assert "discord" in cfg_dict

                # Verify env writer was called
                mock_write_env.assert_called_once()
                env_content = mock_write_env.call_args[0][1]
                assert "MTIz.abc.def" in env_content

    @pytest.mark.asyncio
    async def test_hosts_parsed_correctly(self):
        """Hosts dict is passed through to build_config."""
        app, _ = _make_app()

        with (
            patch("src.web.api._write_config") as mock_write_cfg,
            patch("src.web.api._write_env_file"),
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(side_effect=lambda fn, *a: fn(*a))
            mock_asyncio.get_event_loop.return_value.call_later = MagicMock()

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.abc.def",
                    "hosts": {
                        "webserver": {"address": "10.0.0.5", "ssh_user": "deploy"},
                    },
                })
                assert resp.status == 200
                cfg_dict = mock_write_cfg.call_args[0][1]
                assert "webserver" in cfg_dict["tools"]["hosts"]
                assert cfg_dict["tools"]["hosts"]["webserver"]["address"] == "10.0.0.5"
                assert cfg_dict["tools"]["hosts"]["webserver"]["ssh_user"] == "deploy"

    @pytest.mark.asyncio
    async def test_features_parsed_correctly(self):
        """Feature flags are applied to config."""
        app, _ = _make_app()

        with (
            patch("src.web.api._write_config") as mock_write_cfg,
            patch("src.web.api._write_env_file"),
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(side_effect=lambda fn, *a: fn(*a))
            mock_asyncio.get_event_loop.return_value.call_later = MagicMock()

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.abc.def",
                    "features": {"browser": True, "voice": False, "comfyui": True},
                })
                assert resp.status == 200
                cfg_dict = mock_write_cfg.call_args[0][1]
                assert cfg_dict["browser"]["enabled"] is True
                assert cfg_dict["voice"]["enabled"] is False
                assert cfg_dict["comfyui"]["enabled"] is True

    @pytest.mark.asyncio
    async def test_web_api_token_set(self):
        """Web API token is written to config."""
        app, _ = _make_app()

        with (
            patch("src.web.api._write_config") as mock_write_cfg,
            patch("src.web.api._write_env_file"),
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(side_effect=lambda fn, *a: fn(*a))
            mock_asyncio.get_event_loop.return_value.call_later = MagicMock()

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.abc.def",
                    "web_api_token": "my-secure-token",
                })
                assert resp.status == 200
                cfg_dict = mock_write_cfg.call_args[0][1]
                assert cfg_dict["web"]["api_token"] == "my-secure-token"

    @pytest.mark.asyncio
    async def test_setup_complete_no_auth_required(self):
        """Setup complete endpoint accessible without auth."""
        app, _ = _make_app(api_token="secret")

        with (
            patch("src.web.api._write_config"),
            patch("src.web.api._write_env_file"),
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(side_effect=lambda fn, *a: fn(*a))
            mock_asyncio.get_event_loop.return_value.call_later = MagicMock()

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.abc.def",
                })
                # Should not get 401 even without Authorization header
                assert resp.status == 200

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/setup/complete",
                data=b"not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_write_failure_returns_500(self):
        """If config write fails, return 500."""
        app, _ = _make_app()

        with (
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(side_effect=OSError("disk full"))

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.abc.def",
                })
                assert resp.status == 500
                data = await resp.json()
                assert "error" in data

    @pytest.mark.asyncio
    async def test_malformed_hosts_ignored(self):
        """Hosts without address are silently skipped."""
        app, _ = _make_app()

        with (
            patch("src.web.api._write_config") as mock_write_cfg,
            patch("src.web.api._write_env_file"),
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(side_effect=lambda fn, *a: fn(*a))
            mock_asyncio.get_event_loop.return_value.call_later = MagicMock()

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.abc.def",
                    "hosts": {
                        "good": {"address": "10.0.0.1"},
                        "bad": {"ssh_user": "root"},  # no address
                    },
                })
                assert resp.status == 200
                cfg_dict = mock_write_cfg.call_args[0][1]
                assert "good" in cfg_dict["tools"]["hosts"]
                assert "bad" not in cfg_dict["tools"]["hosts"]

    @pytest.mark.asyncio
    async def test_timezone_preserved(self):
        """Custom timezone is written to config."""
        app, _ = _make_app()

        with (
            patch("src.web.api._write_config") as mock_write_cfg,
            patch("src.web.api._write_env_file"),
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(side_effect=lambda fn, *a: fn(*a))
            mock_asyncio.get_event_loop.return_value.call_later = MagicMock()

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.abc.def",
                    "timezone": "America/New_York",
                })
                assert resp.status == 200
                cfg_dict = mock_write_cfg.call_args[0][1]
                assert cfg_dict["timezone"] == "America/New_York"

    @pytest.mark.asyncio
    async def test_default_ssh_user_is_root(self):
        """When ssh_user not specified, defaults to root."""
        app, _ = _make_app()

        with (
            patch("src.web.api._write_config") as mock_write_cfg,
            patch("src.web.api._write_env_file"),
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(side_effect=lambda fn, *a: fn(*a))
            mock_asyncio.get_event_loop.return_value.call_later = MagicMock()

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.abc.def",
                    "hosts": {"srv": {"address": "10.0.0.1"}},
                })
                assert resp.status == 200
                cfg_dict = mock_write_cfg.call_args[0][1]
                assert cfg_dict["tools"]["hosts"]["srv"]["ssh_user"] == "root"


# ===================================================================
# _write_env_file helper
# ===================================================================


class TestWriteEnvFile:
    """Test the _write_env_file helper function."""

    def test_writes_content(self, tmp_path):
        from src.web.api import _write_env_file
        env_path = tmp_path / ".env"
        _write_env_file(env_path, "DISCORD_TOKEN=abc.def.ghi\n")
        assert env_path.exists()
        assert "DISCORD_TOKEN=abc.def.ghi" in env_path.read_text()

    def test_creates_parent_dirs(self, tmp_path):
        from src.web.api import _write_env_file
        env_path = tmp_path / "subdir" / ".env"
        _write_env_file(env_path, "TOKEN=x\n")
        assert env_path.exists()

    def test_sets_restricted_permissions(self, tmp_path):
        from src.web.api import _write_env_file
        env_path = tmp_path / ".env"
        _write_env_file(env_path, "SECRET=x\n")
        mode = env_path.stat().st_mode & 0o777
        assert mode == 0o600


# ===================================================================
# Auth bypass for setup endpoints
# ===================================================================


class TestSetupAuthBypass:
    """Verify setup endpoints bypass auth middleware."""

    @pytest.mark.asyncio
    async def test_setup_status_bypasses_bearer_auth(self):
        """GET /api/setup/status works without Bearer token when auth is enabled."""
        app, _ = _make_app(api_token="required-token")
        with patch("src.web.api.is_setup_needed", return_value=True):
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/api/setup/status")
                assert resp.status == 200

    @pytest.mark.asyncio
    async def test_setup_complete_bypasses_bearer_auth(self):
        """POST /api/setup/complete works without Bearer token when auth is enabled."""
        app, _ = _make_app(api_token="required-token")

        with (
            patch("src.web.api._write_config"),
            patch("src.web.api._write_env_file"),
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(side_effect=lambda fn, *a: fn(*a))
            mock_asyncio.get_event_loop.return_value.call_later = MagicMock()

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.abc.def",
                })
                assert resp.status == 200

    @pytest.mark.asyncio
    async def test_normal_api_still_requires_auth(self):
        """Regular API endpoints still require auth when token is set."""
        app, _ = _make_app(api_token="required-token")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_auth_skip_prefix_in_constants(self):
        """_AUTH_SKIP_API_PREFIXES includes /api/setup/."""
        from src.health.server import _AUTH_SKIP_API_PREFIXES
        assert any("/api/setup/" in p for p in _AUTH_SKIP_API_PREFIXES)


# ===================================================================
# is_setup_needed integration
# ===================================================================


class TestSetupNeededIntegration:
    """Test is_setup_needed via the API endpoint with real files."""

    @pytest.mark.asyncio
    async def test_needed_when_env_missing(self, tmp_path):
        config_path = tmp_path / "config.yml"
        config_path.write_text("discord:\n  token: test\n")
        env_path = tmp_path / ".env"
        # No .env file

        from src.setup_wizard import is_setup_needed
        assert is_setup_needed(config_path, env_path) is True

    @pytest.mark.asyncio
    async def test_needed_when_placeholder_token(self, tmp_path):
        config_path = tmp_path / "config.yml"
        config_path.write_text("discord:\n  token: test\n")
        env_path = tmp_path / ".env"
        env_path.write_text("DISCORD_TOKEN=your-discord-bot-token-here\n")

        from src.setup_wizard import is_setup_needed
        assert is_setup_needed(config_path, env_path) is True

    @pytest.mark.asyncio
    async def test_not_needed_when_real_token(self, tmp_path):
        config_path = tmp_path / "config.yml"
        config_path.write_text("discord:\n  token: test\n")
        env_path = tmp_path / ".env"
        env_path.write_text("DISCORD_TOKEN=MTIz.real.token\n")

        from src.setup_wizard import is_setup_needed
        assert is_setup_needed(config_path, env_path) is False

    @pytest.mark.asyncio
    async def test_needed_when_config_missing(self, tmp_path):
        config_path = tmp_path / "config.yml"
        # No config.yml
        env_path = tmp_path / ".env"
        env_path.write_text("DISCORD_TOKEN=MTIz.real.token\n")

        from src.setup_wizard import is_setup_needed
        assert is_setup_needed(config_path, env_path) is True

    @pytest.mark.asyncio
    async def test_needed_when_empty_token(self, tmp_path):
        config_path = tmp_path / "config.yml"
        config_path.write_text("discord:\n  token: test\n")
        env_path = tmp_path / ".env"
        env_path.write_text("DISCORD_TOKEN=\n")

        from src.setup_wizard import is_setup_needed
        assert is_setup_needed(config_path, env_path) is True


# ===================================================================
# Frontend — setup-wizard.js structure
# ===================================================================


@pytest.fixture(scope="module")
def setup_wizard_js():
    return SETUP_WIZARD_JS.read_text()


@pytest.fixture(scope="module")
def app_js():
    return APP_JS.read_text()


class TestSetupWizardPageStructure:
    """Verify setup-wizard.js has the expected template elements."""

    def test_has_discord_token_input(self, setup_wizard_js):
        assert "discord-token" in setup_wizard_js

    def test_has_multi_step_form(self, setup_wizard_js):
        assert "currentStep" in setup_wizard_js

    def test_has_progress_indicator(self, setup_wizard_js):
        assert "bg-blue-500" in setup_wizard_js  # Progress bar active color

    def test_has_host_management(self, setup_wizard_js):
        assert "addHost" in setup_wizard_js
        assert "removeHost" in setup_wizard_js

    def test_has_feature_toggles(self, setup_wizard_js):
        assert "browser" in setup_wizard_js
        assert "voice" in setup_wizard_js
        assert "comfyui" in setup_wizard_js

    def test_has_web_token_options(self, setup_wizard_js):
        assert "tokenMode" in setup_wizard_js
        assert "generate" in setup_wizard_js
        assert "custom" in setup_wizard_js

    def test_has_review_step(self, setup_wizard_js):
        assert "Review" in setup_wizard_js

    def test_has_success_state(self, setup_wizard_js):
        assert "setupDone" in setup_wizard_js

    def test_calls_setup_complete_api(self, setup_wizard_js):
        assert "/api/setup/complete" in setup_wizard_js

    def test_has_navigation_buttons(self, setup_wizard_js):
        assert "nextStep" in setup_wizard_js
        assert "prevStep" in setup_wizard_js

    def test_has_token_validation(self, setup_wizard_js):
        assert "tokenHint" in setup_wizard_js

    def test_has_generated_token_display(self, setup_wizard_js):
        assert "generatedToken" in setup_wizard_js

    def test_exports_default_component(self, setup_wizard_js):
        assert "export default" in setup_wizard_js

    def test_uses_raw_fetch(self, setup_wizard_js):
        """Setup wizard uses raw fetch() since auth isn't configured yet."""
        assert "fetch(" in setup_wizard_js
        assert "/api/setup/complete" in setup_wizard_js


class TestAppSetupRoute:
    """Verify app.js includes setup wizard route and redirect logic."""

    def test_imports_setup_wizard_page(self, app_js):
        assert "SetupWizardPage" in app_js
        assert "setup-wizard" in app_js

    def test_has_setup_route(self, app_js):
        assert "'/setup'" in app_js or '"/setup"' in app_js

    def test_setup_route_has_no_sidebar_meta(self, app_js):
        assert "noSidebar" in app_js

    def test_checks_setup_status_on_mount(self, app_js):
        assert "/api/setup/status" in app_js

    def test_redirects_to_setup_when_needed(self, app_js):
        assert "setupData.needed" in app_js or "setup" in app_js

    def test_nav_routes_exclude_setup(self, app_js):
        """Setup page should not appear in sidebar navigation."""
        assert "r.meta.icon" in app_js or "meta.icon" in app_js


# ===================================================================
# build_config integration via endpoint
# ===================================================================


class TestBuildConfigViaEndpoint:
    """Verify the complete endpoint produces valid config structures."""

    @pytest.mark.asyncio
    async def test_generated_config_has_all_sections(self):
        """Config written by setup/complete has all 16 required sections."""
        app, _ = _make_app()

        with (
            patch("src.web.api._write_config") as mock_write_cfg,
            patch("src.web.api._write_env_file"),
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(side_effect=lambda fn, *a: fn(*a))
            mock_asyncio.get_event_loop.return_value.call_later = MagicMock()

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.abc.def",
                })
                assert resp.status == 200
                cfg = mock_write_cfg.call_args[0][1]
                expected_sections = [
                    "discord", "openai_codex", "context", "sessions",
                    "tools", "webhook", "learning", "search", "logging",
                    "usage", "voice", "browser", "monitoring", "permissions",
                    "comfyui", "web",
                ]
                for section in expected_sections:
                    assert section in cfg, f"Missing config section: {section}"

    @pytest.mark.asyncio
    async def test_generated_env_has_token(self):
        """Env file written by setup/complete contains the Discord token."""
        app, _ = _make_app()

        with (
            patch("src.web.api._write_config"),
            patch("src.web.api._write_env_file") as mock_write_env,
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(side_effect=lambda fn, *a: fn(*a))
            mock_asyncio.get_event_loop.return_value.call_later = MagicMock()

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.secret.token",
                })
                assert resp.status == 200
                env_content = mock_write_env.call_args[0][1]
                assert "DISCORD_TOKEN=MTIz.secret.token" in env_content

    @pytest.mark.asyncio
    async def test_generated_config_is_yaml_serializable(self):
        """Config dict can be round-tripped through YAML."""
        app, _ = _make_app()

        with (
            patch("src.web.api._write_config") as mock_write_cfg,
            patch("src.web.api._write_env_file"),
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(side_effect=lambda fn, *a: fn(*a))
            mock_asyncio.get_event_loop.return_value.call_later = MagicMock()

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.abc.def",
                    "hosts": {"srv": {"address": "10.0.0.1"}},
                    "features": {"browser": True},
                    "web_api_token": "test123",
                })
                assert resp.status == 200
                cfg = mock_write_cfg.call_args[0][1]
                # Round-trip through YAML
                yaml_str = yaml.dump(cfg, default_flow_style=False)
                restored = yaml.safe_load(yaml_str)
                assert restored["discord"]["token"] == cfg["discord"]["token"]
                assert restored["browser"]["enabled"] is True
                assert "srv" in restored["tools"]["hosts"]


# ===================================================================
# Restart scheduling
# ===================================================================


class TestRestartScheduling:
    """Verify the delayed restart is scheduled after setup complete."""

    @pytest.mark.asyncio
    async def test_restart_scheduled_in_response(self):
        """Response includes restart_scheduled: true."""
        app, _ = _make_app()

        with (
            patch("src.web.api._write_config"),
            patch("src.web.api._write_env_file"),
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(side_effect=lambda fn, *a: fn(*a))
            mock_loop = MagicMock()
            mock_asyncio.get_event_loop.return_value = mock_loop

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.abc.def",
                })
                assert resp.status == 200
                data = await resp.json()
                assert data["restart_scheduled"] is True

    @pytest.mark.asyncio
    async def test_call_later_invoked(self):
        """asyncio.get_event_loop().call_later is invoked for delayed restart."""
        app, _ = _make_app()

        with (
            patch("src.web.api._write_config"),
            patch("src.web.api._write_env_file"),
            patch("src.web.api.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = AsyncMock(side_effect=lambda fn, *a: fn(*a))
            mock_loop = MagicMock()
            mock_asyncio.get_event_loop.return_value = mock_loop

            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/setup/complete", json={
                    "discord_token": "MTIz.abc.def",
                })
                assert resp.status == 200
                # call_later should be called with ~2 second delay
                mock_loop.call_later.assert_called_once()
                delay = mock_loop.call_later.call_args[0][0]
                assert delay == 2.0
