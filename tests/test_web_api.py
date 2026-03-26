"""Tests for the web management API (src/web/api.py) and security middleware.

Covers: all 30 REST endpoints, auth middleware, rate limiting, input validation,
sensitive field redaction, security headers, WebSocket manager, and edge cases.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import aiohttp
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from src.config.schema import WebConfig
from src.health.server import (
    HealthServer,
    _make_auth_middleware,
    _make_rate_limit_middleware,
    _make_security_headers_middleware,
    _RATE_LIMIT_MAX,
)
from src.web.api import (
    _redact_config,
    _validate_string,
    _contains_blocked_fields,
    _deep_merge,
    _SENSITIVE_FIELDS,
    _MAX_NAME_LEN,
    _MAX_CODE_LEN,
    _MAX_CONTENT_LEN,
    _MAX_GOAL_LEN,
    _MAX_DESCRIPTION_LEN,
    create_api_routes,
    setup_api,
)
from src.web.websocket import WebSocketManager, setup_websocket


# ---------------------------------------------------------------------------
# Helper: build a mock LokiBot with all attributes the API routes access
# ---------------------------------------------------------------------------


def _make_bot():
    """Build a MagicMock that satisfies every attribute the API routes touch."""
    bot = MagicMock()

    # Discord attributes
    guild = MagicMock()
    guild.id = 111
    guild.name = "TestGuild"
    guild.member_count = 42
    bot.guilds = [guild]
    bot.is_ready = MagicMock(return_value=True)
    bot._start_time = time.monotonic() - 120  # 2 min uptime
    bot.get_channel = MagicMock(return_value=MagicMock())

    # Config (Pydantic-like)
    bot.config = MagicMock()
    bot.config.model_dump = MagicMock(return_value={
        "discord": {"token": "xyzzy-secret", "channels": ["general"]},
        "tools": {"ssh_key_path": "/key", "tool_packs": []},
        "web": {"api_token": "tok", "enabled": True},
    })
    bot.config.tools.tool_packs = []

    # Tools
    bot._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run a command", "input_schema": {}},
    ])
    bot._cached_merged_tools = None
    bot._cached_skills_text = None

    # Sessions
    session = MagicMock()
    session.messages = [
        MagicMock(role="user", content="hi", timestamp=1704067200.0, user_id="u1"),
    ]
    session.summary = ""
    session.created_at = 1704067200.0
    session.last_active = 1704153600.0
    session.last_user_id = "u1"
    bot.sessions = MagicMock()
    bot.sessions._sessions = {"chan1": session}
    bot.sessions.reset = MagicMock()

    # Skills
    bot.skill_manager = MagicMock()
    bot.skill_manager.list_skills = MagicMock(return_value=[
        {"name": "joke", "description": "Tell a joke"},
    ])
    skill_mock = MagicMock()
    skill_mock.file_path = MagicMock()
    skill_mock.file_path.exists = MagicMock(return_value=True)
    skill_mock.file_path.read_text = MagicMock(return_value="print('hi')")
    bot.skill_manager._skills = {"joke": skill_mock}
    bot.skill_manager.create_skill = MagicMock(return_value="Skill created")
    bot.skill_manager.edit_skill = MagicMock(return_value="Skill updated")
    bot.skill_manager.delete_skill = MagicMock(return_value="Skill deleted")
    bot.skill_manager.has_skill = MagicMock(side_effect=lambda n: n in bot.skill_manager._skills)
    bot.skill_manager.execute = AsyncMock(return_value="test output")

    # Knowledge
    store = MagicMock()
    store.available = True
    store.list_sources = MagicMock(return_value=[
        {"source": "doc1", "chunks": 3, "preview": "Hello world document...", "ingested_at": "2024-01-01T10:00:00"},
    ])
    store.ingest = AsyncMock(return_value=5)
    store.search_hybrid = AsyncMock(return_value=[
        {"source": "doc1", "content": "hello", "score": 0.9},
    ])
    store.delete_source = MagicMock(return_value=3)
    store.get_source_content = MagicMock(return_value="Hello world full content")
    bot._knowledge_store = store
    bot._embedder = None

    # Scheduler
    bot.scheduler = MagicMock()
    bot.scheduler._schedules = [
        {"id": "sch1", "description": "Daily backup", "channel_id": "c1",
         "action": "reminder", "cron": "0 9 * * *", "last_run": None},
    ]
    bot.scheduler.list_all = MagicMock(return_value=[
        {"id": "sch1", "description": "Daily backup", "channel_id": "c1"},
    ])
    bot.scheduler.add = MagicMock(return_value={
        "id": "sch2", "description": "New schedule",
    })
    bot.scheduler.delete = MagicMock(return_value=True)
    bot.scheduler._callback = AsyncMock()

    # Loops
    loop_info = MagicMock()
    loop_info.goal = "monitor disks"
    loop_info.mode = "notify"
    loop_info.interval_seconds = 60
    loop_info.stop_condition = None
    loop_info.max_iterations = 50
    loop_info.channel_id = "123456"
    loop_info.requester_id = "u1"
    loop_info.requester_name = "Alice"
    loop_info.iteration_count = 5
    loop_info.last_trigger = "2024-01-02T10:00:00"
    loop_info.created_at = "2024-01-01T10:00:00"
    loop_info.status = "running"
    loop_info._iteration_history = [
        "Iteration 1: Disk at 45%",
        "Iteration 2: Disk at 46%",
        "Iteration 3: Disk at 47%",
    ]
    bot.loop_manager = MagicMock()
    bot.loop_manager.active_count = 1
    bot.loop_manager._loops = {"loop1": loop_info}
    bot.loop_manager.start_loop = MagicMock(return_value="new-loop-id")
    bot.loop_manager.stop_loop = MagicMock(return_value="Loop stopped")
    bot._run_loop_iteration = AsyncMock(return_value="done")

    # Processes
    proc = MagicMock()
    proc.command = "ls /"
    proc.host = "localhost"
    proc.status = "running"
    proc.exit_code = None
    proc.start_time = time.time() - 30
    from collections import deque
    proc.output_buffer = deque(["line1\n", "line2\n", "line3\n", "line4\n"], maxlen=500)
    registry = MagicMock()
    registry._processes = {1234: proc}
    registry.kill = AsyncMock(return_value="Process killed")
    bot.tool_executor = MagicMock()
    bot.tool_executor._process_registry = registry

    # Context / reload support
    bot.context_loader = MagicMock()
    bot._invalidate_prompt_caches = MagicMock()
    bot._build_system_prompt = MagicMock(return_value="system prompt")
    bot._system_prompt = "system prompt"

    # Audit
    bot.audit = MagicMock()
    bot.audit.search = AsyncMock(return_value=[
        {"timestamp": "2024-01-01", "tool_name": "run_command", "user": "u1"},
    ])
    bot.audit.count_by_tool = AsyncMock(return_value={
        "run_command": 42, "read_file": 10, "joke": 3,
    })

    # Memory
    bot.tool_executor._load_all_memory = MagicMock(return_value={
        "global": {"key1": "value1"},
        "user:u1": {"pref": "dark"},
    })
    bot.tool_executor._save_all_memory = MagicMock()

    return bot


def _make_app(bot=None, *, api_token="test-token"):
    """Create an aiohttp Application with API routes + middleware."""
    if bot is None:
        bot = _make_bot()
    web_config = WebConfig(api_token=api_token)
    app = web.Application(middlewares=[
        _make_security_headers_middleware(),
        _make_rate_limit_middleware(),
        _make_auth_middleware(web_config),
    ])
    setup_api(app, bot)
    return app, bot


def _auth_headers(token="test-token"):
    return {"Authorization": f"Bearer {token}"}


# ===================================================================
# Unit tests for helpers
# ===================================================================


class TestRedactConfig:
    def test_redacts_token(self):
        cfg = {"discord": {"token": "xyzzy", "channels": ["gen"]}}
        result = _redact_config(cfg)
        assert result["discord"]["token"] == "••••••••"
        assert result["discord"]["channels"] == ["gen"]

    def test_redacts_nested_keys(self):
        cfg = {"web": {"api_token": "secret123", "enabled": True}}
        result = _redact_config(cfg)
        assert result["web"]["api_token"] == "••••••••"
        assert result["web"]["enabled"] is True

    def test_empty_string_not_redacted(self):
        cfg = {"token": ""}
        result = _redact_config(cfg)
        assert result["token"] == ""

    def test_non_string_not_redacted(self):
        cfg = {"password": 12345}
        result = _redact_config(cfg)
        assert result["password"] == 12345

    def test_depth_limit(self):
        nested = {"a": {}}
        current = nested["a"]
        for _ in range(15):
            current["b"] = {}
            current = current["b"]
        result = _redact_config(nested)
        assert "..." in str(result)

    def test_list_recursion(self):
        cfg = [{"token": "secret"}]
        result = _redact_config(cfg)
        assert result[0]["token"] == "••••••••"

    def test_all_sensitive_fields(self):
        cfg = {f: "value" for f in _SENSITIVE_FIELDS}
        result = _redact_config(cfg)
        for f in _SENSITIVE_FIELDS:
            assert result[f] == "••••••••"


class TestValidateString:
    def test_within_limit(self):
        assert _validate_string("hello", "name", 10) is None

    def test_exceeds_limit(self):
        err = _validate_string("x" * 101, "name", 100)
        assert err is not None
        assert "100" in err

    def test_exact_limit(self):
        assert _validate_string("x" * 100, "name", 100) is None


# ===================================================================
# Auth middleware tests
# ===================================================================


class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_valid_token(self):
        app, _ = _make_app(api_token="secret")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/api/status",
                headers={"Authorization": "Bearer secret"},
            )
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_missing_token(self):
        app, _ = _make_app(api_token="secret")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_wrong_token(self):
        app, _ = _make_app(api_token="secret")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/api/status",
                headers={"Authorization": "Bearer wrong"},
            )
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_dev_mode_no_token(self):
        """When api_token is empty, auth is skipped (dev mode)."""
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            assert resp.status == 200


# ===================================================================
# Security headers tests
# ===================================================================


class TestSecurityHeaders:
    @pytest.mark.asyncio
    async def test_nosniff_header(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    @pytest.mark.asyncio
    async def test_frame_deny_header(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            assert resp.headers.get("X-Frame-Options") == "DENY"

    @pytest.mark.asyncio
    async def test_json_content_type(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            assert "application/json" in resp.headers.get("Content-Type", "")


# ===================================================================
# Status & Config endpoint tests
# ===================================================================


class TestStatusEndpoint:
    @pytest.mark.asyncio
    async def test_status_returns_all_fields(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            body = await resp.json()
            assert body["status"] == "online"
            assert body["guild_count"] == 1
            assert body["user_count"] == 42
            assert body["tool_count"] == 1
            assert "uptime_seconds" in body
            assert "session_count" in body
            assert "loop_count" in body
            assert "schedule_count" in body

    @pytest.mark.asyncio
    async def test_status_starting(self):
        bot = _make_bot()
        bot.is_ready = MagicMock(return_value=False)
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            body = await resp.json()
            assert body["status"] == "starting"


class TestConfigEndpoint:
    @pytest.mark.asyncio
    async def test_config_redacts_sensitive(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/config")
            body = await resp.json()
            assert body["discord"]["token"] == "••••••••"
            assert body["tools"]["ssh_key_path"] == "••••••••"
            assert body["web"]["api_token"] == "••••••••"

    @pytest.mark.asyncio
    async def test_config_preserves_non_sensitive(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/config")
            body = await resp.json()
            assert body["discord"]["channels"] == ["general"]
            assert body["web"]["enabled"] is True

    @pytest.mark.asyncio
    async def test_config_put_partial_update(self):
        """PUT /api/config with valid partial update succeeds."""
        from src.config.schema import Config
        real_config = Config(discord={"token": "tok"}, tools={"tool_packs": []}, web={"enabled": True})
        bot = _make_bot()
        bot.config = real_config
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            with patch("src.web.api._write_config"):
                resp = await client.put(
                    "/api/config",
                    json={"tools": {"tool_packs": ["docker", "git"]}},
                )
            assert resp.status == 200
            body = await resp.json()
            assert body["tools"]["tool_packs"] == ["docker", "git"]
            # Verify the bot's config was updated
            assert bot.config.tools.tool_packs == ["docker", "git"]

    @pytest.mark.asyncio
    async def test_config_put_blocks_sensitive_fields(self):
        """PUT /api/config rejects sensitive field updates with 403."""
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/config",
                json={"discord": {"token": "hacked"}},
            )
            assert resp.status == 403
            body = await resp.json()
            assert "sensitive" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_config_put_blocks_nested_sensitive(self):
        """PUT /api/config rejects nested sensitive fields."""
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/config",
                json={"web": {"api_token": "new-secret"}},
            )
            assert resp.status == 403

    @pytest.mark.asyncio
    async def test_config_put_invalid_config(self):
        """PUT /api/config rejects invalid config with 400."""
        from src.config.schema import Config
        real_config = Config(discord={"token": "tok"})
        bot = _make_bot()
        bot.config = real_config
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            # discord is required and must be a dict — setting it to a string should fail
            resp = await client.put(
                "/api/config",
                json={"discord": "not-a-dict"},
            )
            assert resp.status == 400
            body = await resp.json()
            assert "invalid config" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_config_put_invalid_json(self):
        """PUT /api/config rejects malformed JSON."""
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/config",
                data=b"not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_config_put_preserves_unrelated_fields(self):
        """PUT /api/config merges correctly without overwriting unrelated fields."""
        from src.config.schema import Config
        real_config = Config(
            discord={"token": "tok"},
            tools={"tool_packs": ["docker"]},
            timezone="US/Eastern",
        )
        bot = _make_bot()
        bot.config = real_config
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            with patch("src.web.api._write_config"):
                resp = await client.put(
                    "/api/config",
                    json={"tools": {"tool_packs": ["git"]}},
                )
            assert resp.status == 200
            # timezone should be preserved
            assert bot.config.timezone == "US/Eastern"
            # tool_packs should be updated
            assert bot.config.tools.tool_packs == ["git"]

    @pytest.mark.asyncio
    async def test_config_put_writes_to_disk(self):
        """PUT /api/config writes updated config to disk."""
        from src.config.schema import Config
        real_config = Config(discord={"token": "tok"})
        bot = _make_bot()
        bot.config = real_config
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            with patch("src.web.api._write_config") as mock_write:
                resp = await client.put(
                    "/api/config",
                    json={"timezone": "US/Pacific"},
                )
            assert resp.status == 200
            mock_write.assert_called_once()
            written_data = mock_write.call_args[0][1]
            assert written_data["timezone"] == "US/Pacific"


class TestContainsBlockedFields:
    def test_top_level_blocked(self):
        assert _contains_blocked_fields({"token": "x"}, _SENSITIVE_FIELDS) is True

    def test_nested_blocked(self):
        assert _contains_blocked_fields({"discord": {"api_key": "x"}}, _SENSITIVE_FIELDS) is True

    def test_no_blocked(self):
        assert _contains_blocked_fields({"timezone": "UTC"}, _SENSITIVE_FIELDS) is False

    def test_empty_dict(self):
        assert _contains_blocked_fields({}, _SENSITIVE_FIELDS) is False


class TestDeepMerge:
    def test_shallow_merge(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 3, "c": 4})
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"tools": {"packs": [], "enabled": True}}
        _deep_merge(base, {"tools": {"packs": ["docker"]}})
        assert base["tools"]["packs"] == ["docker"]
        assert base["tools"]["enabled"] is True

    def test_overwrite_non_dict(self):
        base = {"a": "old"}
        _deep_merge(base, {"a": "new"})
        assert base["a"] == "new"


# ===================================================================
# Sessions endpoint tests
# ===================================================================


class TestSessionsEndpoint:
    @pytest.mark.asyncio
    async def test_list_sessions(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions")
            body = await resp.json()
            assert len(body) == 1
            assert body[0]["channel_id"] == "chan1"
            assert body[0]["message_count"] == 1

    @pytest.mark.asyncio
    async def test_get_session(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions/chan1")
            body = await resp.json()
            assert body["channel_id"] == "chan1"
            assert len(body["messages"]) == 1
            assert body["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions/nonexistent")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_delete_session(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/sessions/chan1")
            body = await resp.json()
            assert body["status"] == "cleared"
            bot.sessions.reset.assert_called_once_with("chan1")

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/sessions/nope")
            assert resp.status == 404


# ===================================================================
# Tools endpoint tests
# ===================================================================


class TestToolsEndpoint:
    @pytest.mark.asyncio
    async def test_list_tools(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/tools")
            body = await resp.json()
            assert len(body) > 0
            # All tools have required keys
            for tool in body:
                assert "name" in tool
                assert "description" in tool
                assert "is_core" in tool

    @pytest.mark.asyncio
    async def test_list_packs(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/tools/packs")
            body = await resp.json()
            assert "packs" in body
            assert "enabled_packs" in body
            assert "all_packs_loaded" in body

    @pytest.mark.asyncio
    async def test_update_packs_valid(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/tools/packs",
                json={"packs": ["systemd", "prometheus"]},
            )
            body = await resp.json()
            assert body["status"] == "updated"
            assert bot.config.tools.tool_packs == ["systemd", "prometheus"]

    @pytest.mark.asyncio
    async def test_update_packs_invalid(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/tools/packs",
                json={"packs": ["nonexistent_pack"]},
            )
            assert resp.status == 400
            body = await resp.json()
            assert "unknown packs" in body["error"]


# ===================================================================
# Skills endpoint tests
# ===================================================================


class TestSkillsEndpoint:
    @pytest.mark.asyncio
    async def test_list_skills(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/skills")
            body = await resp.json()
            assert len(body) == 1
            assert body[0]["name"] == "joke"
            assert body[0]["code"] == "print('hi')"

    @pytest.mark.asyncio
    async def test_create_skill(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/skills",
                json={"name": "greet", "code": "def run(): return 'hi'"},
            )
            assert resp.status == 201
            body = await resp.json()
            assert body["result"] == "Skill created"

    @pytest.mark.asyncio
    async def test_create_skill_missing_fields(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/skills", json={"name": "x"})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_create_skill_name_too_long(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/skills",
                json={"name": "x" * (_MAX_NAME_LEN + 1), "code": "pass"},
            )
            assert resp.status == 400
            body = await resp.json()
            assert "maximum length" in body["error"]

    @pytest.mark.asyncio
    async def test_create_skill_code_too_long(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/skills",
                json={"name": "big", "code": "x" * (_MAX_CODE_LEN + 1)},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_create_skill_error(self):
        bot = _make_bot()
        bot.skill_manager.create_skill = MagicMock(return_value="Error: syntax error")
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/skills",
                json={"name": "bad", "code": "def bad("},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_update_skill(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/skills/joke",
                json={"code": "print('new')"},
            )
            assert resp.status == 200
            bot.skill_manager.edit_skill.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_skill_empty_code(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.put("/api/skills/joke", json={"code": ""})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_delete_skill(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/skills/joke")
            assert resp.status == 200
            bot.skill_manager.delete_skill.assert_called_once_with("joke")

    @pytest.mark.asyncio
    async def test_delete_skill_not_found(self):
        bot = _make_bot()
        bot.skill_manager.delete_skill = MagicMock(return_value="Error: not found")
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/skills/nope")
            assert resp.status == 404


# ===================================================================
# Knowledge endpoint tests
# ===================================================================


class TestKnowledgeEndpoint:
    @pytest.mark.asyncio
    async def test_list_knowledge(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge")
            body = await resp.json()
            assert len(body) == 1
            assert body[0]["source"] == "doc1"

    @pytest.mark.asyncio
    async def test_knowledge_unavailable(self):
        bot = _make_bot()
        bot._knowledge_store = None
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge")
            assert resp.status == 503

    @pytest.mark.asyncio
    async def test_ingest_knowledge(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/knowledge",
                json={"source": "test-doc", "content": "Hello world"},
            )
            assert resp.status == 201
            body = await resp.json()
            assert body["chunks"] == 5

    @pytest.mark.asyncio
    async def test_ingest_missing_fields(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/knowledge", json={"source": "x"})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_ingest_source_too_long(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/knowledge",
                json={"source": "x" * (_MAX_NAME_LEN + 1), "content": "data"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_delete_knowledge(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/knowledge/doc1")
            body = await resp.json()
            assert body["chunks_removed"] == 3

    @pytest.mark.asyncio
    async def test_delete_knowledge_not_found(self):
        bot = _make_bot()
        bot._knowledge_store.delete_source = MagicMock(return_value=0)
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/knowledge/nope")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_search_knowledge(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/search?q=hello")
            body = await resp.json()
            assert len(body) == 1

    @pytest.mark.asyncio
    async def test_search_knowledge_no_query(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/search")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_search_knowledge_invalid_limit(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/search?q=test&limit=abc")
            assert resp.status == 400
            body = await resp.json()
            assert "integer" in body["error"]


# ===================================================================
# Schedules endpoint tests
# ===================================================================


class TestSchedulesEndpoint:
    @pytest.mark.asyncio
    async def test_list_schedules(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/schedules")
            body = await resp.json()
            assert len(body) == 1

    @pytest.mark.asyncio
    async def test_create_schedule(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/schedules",
                json={
                    "description": "Backup DB",
                    "channel_id": "chan1",
                    "cron": "0 3 * * *",
                },
            )
            assert resp.status == 201

    @pytest.mark.asyncio
    async def test_create_schedule_missing_fields(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/schedules", json={"description": "x"}
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_create_schedule_description_too_long(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/schedules",
                json={
                    "description": "x" * (_MAX_DESCRIPTION_LEN + 1),
                    "channel_id": "c1",
                },
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_delete_schedule(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/schedules/sch1")
            body = await resp.json()
            assert body["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_schedule_not_found(self):
        bot = _make_bot()
        bot.scheduler.delete = MagicMock(return_value=False)
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/schedules/nope")
            assert resp.status == 404


# ===================================================================
# Loops endpoint tests
# ===================================================================


class TestLoopsEndpoint:
    @pytest.mark.asyncio
    async def test_list_loops(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/loops")
            body = await resp.json()
            assert len(body) == 1
            assert body[0]["id"] == "loop1"
            assert body[0]["goal"] == "monitor disks"

    @pytest.mark.asyncio
    async def test_start_loop(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/loops",
                json={"goal": "check disk", "channel_id": "123"},
            )
            assert resp.status == 201
            body = await resp.json()
            assert body["loop_id"] == "new-loop-id"

    @pytest.mark.asyncio
    async def test_start_loop_missing_goal(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/loops", json={"channel_id": "123"}
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_loop_missing_channel(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/loops", json={"goal": "test"}
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_loop_goal_too_long(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/loops",
                json={"goal": "x" * (_MAX_GOAL_LEN + 1), "channel_id": "123"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_loop_channel_not_found(self):
        bot = _make_bot()
        bot.get_channel = MagicMock(return_value=None)
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/loops",
                json={"goal": "test", "channel_id": "999"},
            )
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_start_loop_error(self):
        bot = _make_bot()
        bot.loop_manager.start_loop = MagicMock(
            return_value="Error: max loops reached"
        )
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/loops",
                json={"goal": "test", "channel_id": "123"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_stop_loop(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/loops/loop1")
            body = await resp.json()
            assert body["result"] == "Loop stopped"

    @pytest.mark.asyncio
    async def test_stop_loop_not_found(self):
        bot = _make_bot()
        bot.loop_manager.stop_loop = MagicMock(return_value="Loop not found")
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/loops/nope")
            assert resp.status == 404


# ===================================================================
# Processes endpoint tests
# ===================================================================


class TestProcessesEndpoint:
    @pytest.mark.asyncio
    async def test_list_processes(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/processes")
            body = await resp.json()
            assert len(body) == 1
            assert body[0]["pid"] == 1234
            assert body[0]["command"] == "ls /"

    @pytest.mark.asyncio
    async def test_list_processes_no_registry(self):
        bot = _make_bot()
        bot.tool_executor._process_registry = None
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/processes")
            body = await resp.json()
            assert body == []

    @pytest.mark.asyncio
    async def test_kill_process(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/processes/1234")
            body = await resp.json()
            assert body["result"] == "Process killed"

    @pytest.mark.asyncio
    async def test_kill_process_invalid_pid(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/processes/notanumber")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_kill_process_no_registry(self):
        bot = _make_bot()
        bot.tool_executor._process_registry = None
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/processes/1")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_kill_process_not_found(self):
        bot = _make_bot()
        bot.tool_executor._process_registry.kill = AsyncMock(
            return_value="No process with that PID"
        )
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/processes/9999")
            assert resp.status == 404


# ===================================================================
# Audit endpoint tests
# ===================================================================


class TestAuditEndpoint:
    @pytest.mark.asyncio
    async def test_search_audit(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/audit")
            body = await resp.json()
            assert len(body) == 1

    @pytest.mark.asyncio
    async def test_search_audit_with_filters(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/api/audit?tool=run_command&user=u1&limit=10"
            )
            assert resp.status == 200
            bot.audit.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_audit_invalid_limit(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/audit?limit=abc")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_search_audit_limit_capped(self):
        """Limit is capped at 200."""
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/audit?limit=999")
            assert resp.status == 200
            call_kwargs = bot.audit.search.call_args[1]
            assert call_kwargs["limit"] == 200


# ===================================================================
# Memory endpoint tests
# ===================================================================


class TestMemoryEndpoint:
    @pytest.mark.asyncio
    async def test_list_memory(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/memory")
            body = await resp.json()
            assert "global" in body
            assert body["global"]["count"] == 1

    @pytest.mark.asyncio
    async def test_get_memory(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/memory/global/key1")
            body = await resp.json()
            assert body["value"] == "value1"

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/memory/global/nonexistent")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_set_memory(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/memory/global/newkey",
                json={"value": "newval"},
            )
            body = await resp.json()
            assert body["status"] == "saved"
            bot.tool_executor._save_all_memory.assert_called()

    @pytest.mark.asyncio
    async def test_set_memory_missing_value(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.put("/api/memory/global/k", json={})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_set_memory_new_scope(self):
        """Setting a key in a new scope should create the scope."""
        bot = _make_bot()
        bot.tool_executor._load_all_memory = MagicMock(return_value={})
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/memory/newscope/k1", json={"value": "v1"}
            )
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_delete_memory(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/memory/global/key1")
            body = await resp.json()
            assert body["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/memory/global/nope")
            assert resp.status == 404


# ===================================================================
# Rate limiting tests
# ===================================================================


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self):
        """After _RATE_LIMIT_MAX requests, the next should get 429."""
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            # Send _RATE_LIMIT_MAX requests (should all succeed)
            for _ in range(_RATE_LIMIT_MAX):
                resp = await client.get("/api/status")
                assert resp.status == 200
            # Next request should be rate limited
            resp = await client.get("/api/status")
            assert resp.status == 429
            body = await resp.json()
            assert "rate limit" in body["error"]


# ===================================================================
# JSON error handling tests
# ===================================================================


class TestJSONErrorHandling:
    @pytest.mark.asyncio
    async def test_malformed_json_body(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/skills",
                data=b"not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            body = await resp.json()
            assert "invalid JSON" in body["error"]


# ===================================================================
# WebSocket manager tests
# ===================================================================


class TestWebSocketManager:
    def test_initial_state(self):
        bot = _make_bot()
        mgr = WebSocketManager(bot)
        assert mgr.client_count == 0

    def test_stores_api_token(self):
        bot = _make_bot()
        mgr = WebSocketManager(bot, api_token="secret-ws")
        assert mgr._api_token == "secret-ws"

    def test_empty_token_allows_all(self):
        """When api_token is empty, no auth is enforced (dev mode)."""
        bot = _make_bot()
        mgr = WebSocketManager(bot, api_token="")
        assert mgr._api_token == ""

    @pytest.mark.asyncio
    async def test_broadcast_no_subscribers(self):
        """broadcast_event with no subscribers should be a no-op."""
        bot = _make_bot()
        mgr = WebSocketManager(bot)
        await mgr.broadcast_event({"action": "tool_call", "tool": "run_command"})
        # No error — just verifying it doesn't crash

    @pytest.mark.asyncio
    async def test_broadcast_to_subscriber(self):
        """Events are sent to subscribed clients."""
        bot = _make_bot()
        mgr = WebSocketManager(bot)
        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mgr._clients.add(mock_ws)
        mgr._event_subscribers.add(mock_ws)

        await mgr.broadcast_event({"action": "tool_call"})
        mock_ws.send_json.assert_called_once()
        msg = mock_ws.send_json.call_args[0][0]
        assert msg["type"] == "event"
        assert msg["payload"]["action"] == "tool_call"

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_clients(self):
        """Dead clients are cleaned up during broadcast."""
        bot = _make_bot()
        mgr = WebSocketManager(bot)
        dead_ws = AsyncMock()
        dead_ws.send_json = AsyncMock(side_effect=ConnectionError("gone"))
        mgr._clients.add(dead_ws)
        mgr._event_subscribers.add(dead_ws)

        await mgr.broadcast_event({"action": "test"})
        assert dead_ws not in mgr._event_subscribers
        assert dead_ws not in mgr._clients


# ===================================================================
# WebSocket authentication tests
# ===================================================================


def _make_ws_app(bot=None, *, api_token="ws-secret"):
    """Create an aiohttp Application with a WebSocket endpoint for testing."""
    if bot is None:
        bot = _make_bot()
    app = web.Application()
    mgr = WebSocketManager(bot, api_token=api_token)
    app.router.add_get("/api/ws", mgr.handle)
    return app, mgr


class TestWebSocketAuth:
    @pytest.mark.asyncio
    async def test_ws_rejects_missing_token(self):
        """WebSocket connection without token is closed with 4001."""
        app, mgr = _make_ws_app(api_token="my-secret")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            msg = await ws.receive()
            assert msg.type == aiohttp.WSMsgType.CLOSE
            assert ws.close_code == 4001
            # Client was never added to the active set
            assert mgr.client_count == 0

    @pytest.mark.asyncio
    async def test_ws_rejects_wrong_token(self):
        """WebSocket connection with wrong token is closed with 4001."""
        app, mgr = _make_ws_app(api_token="my-secret")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws?token=wrong-token")
            msg = await ws.receive()
            assert msg.type == aiohttp.WSMsgType.CLOSE
            assert ws.close_code == 4001
            assert mgr.client_count == 0

    @pytest.mark.asyncio
    async def test_ws_accepts_valid_token(self):
        """WebSocket connection with correct token is accepted."""
        app, mgr = _make_ws_app(api_token="my-secret")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws?token=my-secret")
            # Connection should be open and client registered
            assert mgr.client_count == 1
            # Send a subscribe message to verify the connection works
            await ws.send_json({"subscribe": "events"})
            msg = await ws.receive_json()
            assert msg["type"] == "subscribed"
            assert msg["channel"] == "events"
            await ws.close()

    @pytest.mark.asyncio
    async def test_ws_allows_all_when_no_token(self):
        """When api_token is empty (dev mode), all connections allowed."""
        app, mgr = _make_ws_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            assert mgr.client_count == 1
            await ws.send_json({"subscribe": "events"})
            msg = await ws.receive_json()
            assert msg["type"] == "subscribed"
            await ws.close()

    @pytest.mark.asyncio
    async def test_ws_allows_no_token_param_in_dev_mode(self):
        """Dev mode: connection without any token param succeeds."""
        app, mgr = _make_ws_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws?other=param")
            assert mgr.client_count == 1
            await ws.close()


# ===================================================================
# HealthServer integration tests
# ===================================================================


class TestHealthServerWebIntegration:
    def test_set_bot_registers_api(self):
        """set_bot() should register API + WebSocket routes."""
        server = HealthServer(
            port=0,
            web_config=WebConfig(enabled=True, api_token="test-tok"),
        )
        bot = _make_bot()
        with patch("src.web.api.setup_api") as mock_setup_api, \
             patch("src.web.websocket.setup_websocket") as mock_setup_ws:
            mock_setup_ws.return_value = MagicMock()
            server.set_bot(bot)
            mock_setup_api.assert_called_once()
            mock_setup_ws.assert_called_once()
            # Verify api_token is passed through to WebSocket setup
            ws_call_kwargs = mock_setup_ws.call_args
            assert ws_call_kwargs[1]["api_token"] == "test-tok"

    def test_set_bot_disabled(self):
        """When web is disabled, set_bot() should be a no-op."""
        server = HealthServer(
            port=0,
            web_config=WebConfig(enabled=False),
        )
        bot = _make_bot()
        # set_bot returns early when web is disabled — no imports happen
        server.set_bot(bot)
        assert not hasattr(server, "_ws_manager")

    @pytest.mark.asyncio
    async def test_health_endpoint_ready(self):
        """Health endpoint returns 200 when ready."""
        server = HealthServer(port=0, web_config=WebConfig(enabled=True))
        server.set_ready(True)
        req = MagicMock()
        resp = await server._health(req)
        body = json.loads(resp.body)
        assert body["status"] == "ok"
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_health_endpoint_not_ready(self):
        """Health endpoint returns 503 when not ready."""
        server = HealthServer(port=0, web_config=WebConfig(enabled=True))
        server.set_ready(False)
        req = MagicMock()
        resp = await server._health(req)
        assert resp.status == 503


# ===================================================================
# Sensitive field exposure verification
# ===================================================================


class TestSensitiveFieldProtection:
    @pytest.mark.asyncio
    async def test_no_raw_token_in_config_response(self):
        """Ensure no sensitive values leak through GET /api/config."""
        bot = _make_bot()
        bot.config.model_dump = MagicMock(return_value={
            "discord": {"token": "MY_SECRET_TOKEN"},
            "tools": {"ssh_key_path": "/root/.ssh/id_rsa", "password": "dbpass"},
            "web": {"api_token": "web-secret", "api_key": "ak_123"},
            "openai_codex": {"credentials_path": "/creds.json"},
        })
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/config")
            raw = await resp.text()
            # None of these secrets should appear in the response body
            assert "MY_SECRET_TOKEN" not in raw
            assert "/root/.ssh/id_rsa" not in raw
            assert "dbpass" not in raw
            assert "web-secret" not in raw
            assert "ak_123" not in raw
            assert "/creds.json" not in raw

    @pytest.mark.asyncio
    async def test_status_does_not_leak_tokens(self):
        """Status endpoint should not include any config/token data."""
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            raw = await resp.text()
            assert "token" not in raw.lower() or "tool_count" in raw
            body = await resp.json()
            # Verify only expected keys are present
            expected_keys = {
                "status", "uptime_seconds", "guilds", "guild_count",
                "user_count", "tool_count", "skill_count", "session_count",
                "loop_count", "schedule_count",
            }
            assert set(body.keys()) == expected_keys


# ===================================================================
# Quick actions endpoint tests (Round 7)
# ===================================================================


class TestQuickActions:
    @pytest.mark.asyncio
    async def test_clear_all_sessions(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/sessions/clear-all")
            body = await resp.json()
            assert body["status"] == "cleared"
            assert body["count"] == 1
            bot.sessions.reset.assert_called_once_with("chan1")

    @pytest.mark.asyncio
    async def test_clear_all_sessions_empty(self):
        bot = _make_bot()
        bot.sessions._sessions = {}
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/sessions/clear-all")
            body = await resp.json()
            assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_reload_config(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/reload")
            body = await resp.json()
            assert body["status"] == "reloaded"
            bot.context_loader.reload.assert_called_once()
            bot._invalidate_prompt_caches.assert_called_once()
            bot._build_system_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_all_loops(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/loops/stop-all")
            body = await resp.json()
            assert "result" in body
            bot.loop_manager.stop_loop.assert_called_once_with("all")


class TestStatusGuildMemberCount:
    @pytest.mark.asyncio
    async def test_guild_includes_member_count(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            body = await resp.json()
            assert body["guilds"][0]["member_count"] == 42

    @pytest.mark.asyncio
    async def test_guild_member_count_zero_when_none(self):
        bot = _make_bot()
        bot.guilds[0].member_count = None
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            body = await resp.json()
            assert body["guilds"][0]["member_count"] == 0


class TestAuditErrorFilter:
    @pytest.mark.asyncio
    async def test_error_only_filter(self):
        bot = _make_bot()
        bot.audit.search = AsyncMock(return_value=[
            {"timestamp": "2024-01-01", "tool_name": "run_command", "error": True, "error_message": "fail"},
            {"timestamp": "2024-01-01", "tool_name": "read_file", "error": False},
        ])
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/audit?error_only=1")
            body = await resp.json()
            assert len(body) == 1
            assert body[0]["tool_name"] == "run_command"

    @pytest.mark.asyncio
    async def test_error_only_false_returns_all(self):
        bot = _make_bot()
        bot.audit.search = AsyncMock(return_value=[
            {"timestamp": "2024-01-01", "tool_name": "run_command", "error": True},
            {"timestamp": "2024-01-01", "tool_name": "read_file", "error": False},
        ])
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/audit?error_only=0")
            body = await resp.json()
            assert len(body) == 2


# ===================================================================
# Sessions — preview, export, bulk clear (Round 8)
# ===================================================================


class TestSessionPreview:
    @pytest.mark.asyncio
    async def test_list_includes_preview(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions")
            body = await resp.json()
            assert len(body) == 1
            s = body[0]
            assert "preview" in s
            assert len(s["preview"]) == 1
            assert s["preview"][0]["role"] == "user"
            assert s["preview"][0]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_list_includes_source_discord(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions")
            body = await resp.json()
            assert body[0]["source"] == "discord"

    @pytest.mark.asyncio
    async def test_list_includes_source_web(self):
        bot = _make_bot()
        session = MagicMock()
        session.messages = [MagicMock(role="user", content="test", timestamp=1704067200.0, user_id="web-user")]
        session.summary = ""
        session.created_at = 1704067200.0
        session.last_active = 1704067200.0
        session.last_user_id = "web-user"
        bot.sessions._sessions = {"web-chat": session}
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions")
            body = await resp.json()
            assert body[0]["source"] == "web"

    @pytest.mark.asyncio
    async def test_list_includes_last_user_id(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions")
            body = await resp.json()
            assert body[0]["last_user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_preview_truncates_long_content(self):
        bot = _make_bot()
        session = MagicMock()
        long_content = "x" * 200
        session.messages = [MagicMock(role="assistant", content=long_content, timestamp=1704067200.0, user_id=None)]
        session.summary = ""
        session.created_at = 1704067200.0
        session.last_active = 1704067200.0
        session.last_user_id = None
        bot.sessions._sessions = {"chan1": session}
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions")
            body = await resp.json()
            preview_content = body[0]["preview"][0]["content"]
            assert len(preview_content) <= 124  # 120 + "..."


class TestSessionExport:
    @pytest.mark.asyncio
    async def test_export_json(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions/chan1/export?format=json")
            assert resp.status == 200
            body = await resp.json()
            assert body["channel_id"] == "chan1"
            assert len(body["messages"]) == 1
            assert "exported_at" in body
            assert "attachment" in resp.headers.get("Content-Disposition", "")

    @pytest.mark.asyncio
    async def test_export_text(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions/chan1/export?format=text")
            assert resp.status == 200
            text = await resp.text()
            assert "Messages" in text
            assert "USER" in text
            assert resp.content_type == "text/plain"
            assert "attachment" in resp.headers.get("Content-Disposition", "")

    @pytest.mark.asyncio
    async def test_export_not_found(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions/nope/export?format=json")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_export_default_format_is_json(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions/chan1/export")
            assert resp.status == 200
            body = await resp.json()
            assert "messages" in body

    @pytest.mark.asyncio
    async def test_export_with_query_token_auth(self):
        app, _ = _make_app(api_token="secret123")
        async with TestClient(TestServer(app)) as client:
            # Without auth should fail
            resp = await client.get("/api/sessions/chan1/export")
            assert resp.status == 401
            # With query token should succeed
            resp = await client.get("/api/sessions/chan1/export?token=secret123")
            assert resp.status == 200


class TestSessionBulkClear:
    @pytest.mark.asyncio
    async def test_bulk_clear(self):
        bot = _make_bot()
        session2 = MagicMock()
        session2.messages = [MagicMock(role="user", content="bye", timestamp=1704067200.0, user_id="u2")]
        session2.summary = ""
        session2.created_at = 1704067200.0
        session2.last_active = 1704067200.0
        session2.last_user_id = "u2"
        bot.sessions._sessions["chan2"] = session2
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/sessions/clear-bulk",
                json={"channel_ids": ["chan1", "chan2"]},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["status"] == "cleared"
            assert body["count"] == 2

    @pytest.mark.asyncio
    async def test_bulk_clear_partial(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/sessions/clear-bulk",
                json={"channel_ids": ["chan1", "nonexistent"]},
            )
            body = await resp.json()
            assert body["count"] == 1

    @pytest.mark.asyncio
    async def test_bulk_clear_empty_list(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/sessions/clear-bulk",
                json={"channel_ids": []},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_bulk_clear_invalid_json(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/sessions/clear-bulk",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_bulk_clear_missing_field(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/sessions/clear-bulk",
                json={"wrong_field": ["chan1"]},
            )
            assert resp.status == 400


class TestQueryTokenAuth:
    @pytest.mark.asyncio
    async def test_query_token_accepted(self):
        app, _ = _make_app(api_token="mytoken")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status?token=mytoken")
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_query_token_rejected(self):
        app, _ = _make_app(api_token="mytoken")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status?token=wrongtoken")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_bearer_still_works(self):
        app, _ = _make_app(api_token="mytoken")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status", headers=_auth_headers("mytoken"))
            assert resp.status == 200


# ===================================================================
# Audit event callback + broadcast wiring (Round 10)
# ===================================================================


class TestAuditEventCallback:
    @pytest.mark.asyncio
    async def test_audit_logger_fires_event_callback(self):
        """AuditLogger.log_execution calls the event callback with the entry."""
        import tempfile, os
        from src.audit.logger import AuditLogger

        captured = []
        async def on_event(entry):
            captured.append(entry)

        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(os.path.join(tmp, "audit.jsonl"))
            logger.set_event_callback(on_event)
            await logger.log_execution(
                user_id="u1", user_name="tester", channel_id="c1",
                tool_name="run_command", tool_input={"command": "ls"},
                approved=True, result_summary="ok", execution_time_ms=100,
            )
            assert len(captured) == 1
            assert captured[0]["tool_name"] == "run_command"
            assert captured[0]["user_name"] == "tester"

    @pytest.mark.asyncio
    async def test_audit_logger_no_callback_does_not_crash(self):
        """Without a callback set, log_execution still works."""
        import tempfile, os
        from src.audit.logger import AuditLogger

        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(os.path.join(tmp, "audit.jsonl"))
            await logger.log_execution(
                user_id="u1", user_name="tester", channel_id="c1",
                tool_name="read_file", tool_input={},
                approved=True, result_summary="ok", execution_time_ms=50,
            )
            # Just verify it didn't crash
            assert logger.path.exists()

    @pytest.mark.asyncio
    async def test_audit_logger_callback_error_does_not_break_logging(self):
        """Callback errors must not prevent the audit entry from being written."""
        import tempfile, os
        from src.audit.logger import AuditLogger

        async def bad_callback(entry):
            raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(os.path.join(tmp, "audit.jsonl"))
            logger.set_event_callback(bad_callback)
            await logger.log_execution(
                user_id="u1", user_name="tester", channel_id="c1",
                tool_name="run_command", tool_input={},
                approved=True, result_summary="ok", execution_time_ms=10,
            )
            # Entry still written despite callback error
            assert logger.path.exists()
            content = logger.path.read_text()
            assert "run_command" in content

    @pytest.mark.asyncio
    async def test_audit_logger_error_entry_callback(self):
        """Error entries include the error field in the callback."""
        import tempfile, os
        from src.audit.logger import AuditLogger

        captured = []
        async def on_event(entry):
            captured.append(entry)

        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(os.path.join(tmp, "audit.jsonl"))
            logger.set_event_callback(on_event)
            await logger.log_execution(
                user_id="u1", user_name="tester", channel_id="c1",
                tool_name="run_command", tool_input={},
                approved=True, result_summary="failed",
                execution_time_ms=100, error="command not found",
            )
            assert len(captured) == 1
            assert captured[0]["error"] == "command not found"


class TestAuditCountByTool:
    @pytest.mark.asyncio
    async def test_count_by_tool_returns_counts(self):
        import tempfile, os
        from src.audit.logger import AuditLogger

        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(os.path.join(tmp, "audit.jsonl"))
            for _ in range(3):
                await logger.log_execution(
                    user_id="u1", user_name="tester", channel_id="c1",
                    tool_name="run_command", tool_input={},
                    approved=True, result_summary="ok", execution_time_ms=10,
                )
            await logger.log_execution(
                user_id="u1", user_name="tester", channel_id="c1",
                tool_name="read_file", tool_input={},
                approved=True, result_summary="ok", execution_time_ms=5,
            )
            counts = await logger.count_by_tool()
            assert counts["run_command"] == 3
            assert counts["read_file"] == 1
            # Sorted by count descending
            keys = list(counts.keys())
            assert keys[0] == "run_command"

    @pytest.mark.asyncio
    async def test_count_by_tool_empty_log(self):
        import tempfile, os
        from src.audit.logger import AuditLogger

        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(os.path.join(tmp, "audit.jsonl"))
            counts = await logger.count_by_tool()
            assert counts == {}

    @pytest.mark.asyncio
    async def test_count_by_tool_no_file(self):
        import tempfile, os
        from src.audit.logger import AuditLogger

        with tempfile.TemporaryDirectory() as tmp:
            # Logger created but no entries written — file doesn't exist
            path = os.path.join(tmp, "empty_audit.jsonl")
            logger = AuditLogger(path)
            os.remove(path) if os.path.exists(path) else None
            counts = await logger.count_by_tool()
            assert counts == {}


class TestBroadcastEventPayloadFormat:
    @pytest.mark.asyncio
    async def test_broadcast_wraps_event_in_payload(self):
        """broadcast_event wraps the event dict under a 'payload' key."""
        bot = _make_bot()
        mgr = WebSocketManager(bot)
        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mgr._clients.add(mock_ws)
        mgr._event_subscribers.add(mock_ws)

        await mgr.broadcast_event({
            "tool_name": "run_command",
            "error": None,
            "timestamp": "2024-01-01T00:00:00",
        })
        msg = mock_ws.send_json.call_args[0][0]
        assert msg["type"] == "event"
        assert "payload" in msg
        assert msg["payload"]["tool_name"] == "run_command"
        assert msg["payload"]["timestamp"] == "2024-01-01T00:00:00"


# ---------------------------------------------------------------------------
# Round 11: Tool stats endpoint
# ---------------------------------------------------------------------------


class TestToolStats:
    @pytest.mark.asyncio
    async def test_tool_stats_returns_counts(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/tools/stats")
            assert resp.status == 200
            body = await resp.json()
            assert body["run_command"] == 42
            assert body["read_file"] == 10
            bot.audit.count_by_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_stats_empty(self):
        bot = _make_bot()
        bot.audit.count_by_tool = AsyncMock(return_value={})
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/tools/stats")
            assert resp.status == 200
            body = await resp.json()
            assert body == {}

    @pytest.mark.asyncio
    async def test_tool_stats_requires_auth(self):
        app, _ = _make_app(api_token="secret")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/tools/stats")
            assert resp.status == 401
            resp2 = await client.get("/api/tools/stats", headers=_auth_headers("secret"))
            assert resp2.status == 200


# ---------------------------------------------------------------------------
# Round 11: Skill execution stats in list
# ---------------------------------------------------------------------------


class TestSkillExecutionStats:
    @pytest.mark.asyncio
    async def test_skills_list_includes_execution_count(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/skills")
            assert resp.status == 200
            body = await resp.json()
            assert len(body) == 1
            assert body[0]["name"] == "joke"
            assert body[0]["execution_count"] == 3  # from count_by_tool mock

    @pytest.mark.asyncio
    async def test_skills_list_zero_executions(self):
        bot = _make_bot()
        bot.audit.count_by_tool = AsyncMock(return_value={})
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/skills")
            body = await resp.json()
            assert body[0]["execution_count"] == 0


# ---------------------------------------------------------------------------
# Round 11: Skill test endpoint
# ---------------------------------------------------------------------------


class TestSkillTestEndpoint:
    @pytest.mark.asyncio
    async def test_test_skill_success(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/skills/joke/test")
            assert resp.status == 200
            body = await resp.json()
            assert body["result"] == "test output"
            assert body["is_error"] is False

    @pytest.mark.asyncio
    async def test_test_skill_not_found(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/skills/nonexistent/test")
            assert resp.status == 404
            body = await resp.json()
            assert "not found" in body["error"]

    @pytest.mark.asyncio
    async def test_test_skill_error_result(self):
        bot = _make_bot()
        bot.skill_manager.execute = AsyncMock(return_value="Skill error: syntax error")
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/skills/joke/test")
            assert resp.status == 200
            body = await resp.json()
            assert body["is_error"] is True

    @pytest.mark.asyncio
    async def test_test_skill_exception(self):
        bot = _make_bot()
        bot.skill_manager.execute = AsyncMock(side_effect=RuntimeError("boom"))
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/skills/joke/test")
            assert resp.status == 500
            body = await resp.json()
            assert body["is_error"] is True
            assert "boom" in body["result"]

    @pytest.mark.asyncio
    async def test_test_skill_passes_empty_input(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/skills/joke/test")
            bot.skill_manager.execute.assert_called_once_with("joke", {})


# ===================================================================
# Knowledge preview + re-ingest tests (Round 12)
# ===================================================================


class TestKnowledgePreview:
    @pytest.mark.asyncio
    async def test_list_includes_preview(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge")
            body = await resp.json()
            assert body[0]["preview"] == "Hello world document..."

    @pytest.mark.asyncio
    async def test_list_includes_ingested_at(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge")
            body = await resp.json()
            assert body[0]["ingested_at"] == "2024-01-01T10:00:00"


class TestKnowledgeReingest:
    @pytest.mark.asyncio
    async def test_reingest_success(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/knowledge/doc1/reingest")
            assert resp.status == 200
            body = await resp.json()
            assert body["source"] == "doc1"
            assert body["chunks"] == 5
            bot._knowledge_store.get_source_content.assert_called_once_with("doc1")
            bot._knowledge_store.ingest.assert_called()

    @pytest.mark.asyncio
    async def test_reingest_not_found(self):
        bot = _make_bot()
        bot._knowledge_store.get_source_content = MagicMock(return_value=None)
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/knowledge/nope/reingest")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_reingest_unavailable(self):
        bot = _make_bot()
        bot._knowledge_store = None
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/knowledge/doc1/reingest")
            assert resp.status == 503

    @pytest.mark.asyncio
    async def test_reingest_uses_web_reingest_uploader(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/knowledge/doc1/reingest")
            call_kwargs = bot._knowledge_store.ingest.call_args
            assert call_kwargs.kwargs.get("uploader") == "web-reingest"


# ===================================================================
# Memory bulk delete tests (Round 12)
# ===================================================================


class TestMemoryBulkDelete:
    @pytest.mark.asyncio
    async def test_bulk_delete_success(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/memory/bulk-delete",
                json={"entries": [{"scope": "global", "key": "key1"}]},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["count"] == 1
            bot.tool_executor._save_all_memory.assert_called()

    @pytest.mark.asyncio
    async def test_bulk_delete_partial(self):
        """Entries that don't exist are silently skipped."""
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/memory/bulk-delete",
                json={"entries": [
                    {"scope": "global", "key": "key1"},
                    {"scope": "global", "key": "nonexistent"},
                ]},
            )
            body = await resp.json()
            assert body["count"] == 1

    @pytest.mark.asyncio
    async def test_bulk_delete_empty_list(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/memory/bulk-delete",
                json={"entries": []},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_bulk_delete_invalid_json(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/memory/bulk-delete",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_bulk_delete_missing_entries(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/memory/bulk-delete", json={})
            assert resp.status == 400


# ===================================================================
# Knowledge store unit tests (Round 12)
# ===================================================================


class TestKnowledgeStorePreview:
    def test_list_sources_includes_preview(self):
        """list_sources returns preview from first chunk."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE knowledge_chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                uploader TEXT NOT NULL DEFAULT 'system',
                ingested_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "INSERT INTO knowledge_chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("id0", "First chunk content here", "test-doc", 0, 2, "system", "2024-01-01T00:00:00"),
        )
        conn.execute(
            "INSERT INTO knowledge_chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("id1", "Second chunk content", "test-doc", 1, 2, "system", "2024-01-01T00:00:00"),
        )
        conn.commit()

        from src.knowledge.store import KnowledgeStore
        store = KnowledgeStore.__new__(KnowledgeStore)
        store._conn = conn
        store._has_vec = False
        store._fts = None

        sources = store.list_sources()
        assert len(sources) == 1
        assert sources[0]["preview"] == "First chunk content here"
        assert sources[0]["chunks"] == 2

    def test_list_sources_preview_truncated(self):
        """Preview truncated to 200 chars with ellipsis."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE knowledge_chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                uploader TEXT NOT NULL DEFAULT 'system',
                ingested_at TEXT NOT NULL
            )
        """)
        long_content = "x" * 300
        conn.execute(
            "INSERT INTO knowledge_chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("id0", long_content, "long-doc", 0, 1, "system", "2024-01-01T00:00:00"),
        )
        conn.commit()

        from src.knowledge.store import KnowledgeStore
        store = KnowledgeStore.__new__(KnowledgeStore)
        store._conn = conn
        store._has_vec = False
        store._fts = None

        sources = store.list_sources()
        assert sources[0]["preview"].endswith("...")
        assert len(sources[0]["preview"]) == 203  # 200 + "..."

    def test_get_source_content(self):
        """get_source_content returns concatenated chunks."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE knowledge_chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                uploader TEXT NOT NULL DEFAULT 'system',
                ingested_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "INSERT INTO knowledge_chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("id0", "chunk one", "doc", 0, 2, "system", "2024-01-01"),
        )
        conn.execute(
            "INSERT INTO knowledge_chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("id1", "chunk two", "doc", 1, 2, "system", "2024-01-01"),
        )
        conn.commit()

        from src.knowledge.store import KnowledgeStore
        store = KnowledgeStore.__new__(KnowledgeStore)
        store._conn = conn
        store._has_vec = False
        store._fts = None

        content = store.get_source_content("doc")
        assert content == "chunk one\n\nchunk two"

    def test_get_source_content_not_found(self):
        """get_source_content returns None for missing source."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE knowledge_chunks (
                chunk_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                uploader TEXT NOT NULL DEFAULT 'system',
                ingested_at TEXT NOT NULL
            )
        """)
        conn.commit()

        from src.knowledge.store import KnowledgeStore
        store = KnowledgeStore.__new__(KnowledgeStore)
        store._conn = conn
        store._has_vec = False
        store._fts = None

        assert store.get_source_content("nope") is None


# ===================================================================
# Round 13: Loops — iteration history in list + restart endpoint
# ===================================================================


class TestLoopIterationHistory:
    @pytest.mark.asyncio
    async def test_list_includes_iteration_history(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/loops")
            body = await resp.json()
            assert len(body) == 1
            assert "iteration_history" in body[0]
            assert len(body[0]["iteration_history"]) == 3
            assert "Iteration 1" in body[0]["iteration_history"][0]

    @pytest.mark.asyncio
    async def test_list_empty_iteration_history(self):
        bot = _make_bot()
        bot.loop_manager._loops["loop1"]._iteration_history = []
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/loops")
            body = await resp.json()
            assert body[0]["iteration_history"] == []

    @pytest.mark.asyncio
    async def test_list_caps_iteration_history_at_5(self):
        bot = _make_bot()
        bot.loop_manager._loops["loop1"]._iteration_history = [
            f"Iteration {i}: output" for i in range(10)
        ]
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/loops")
            body = await resp.json()
            assert len(body[0]["iteration_history"]) == 5
            # Should be the last 5
            assert "Iteration 5" in body[0]["iteration_history"][0]


class TestLoopRestart:
    @pytest.mark.asyncio
    async def test_restart_loop_success(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/loops/loop1/restart")
            assert resp.status == 201
            body = await resp.json()
            assert body["old_id"] == "loop1"
            assert body["new_id"] == "new-loop-id"
            # Should have stopped the old loop
            bot.loop_manager.stop_loop.assert_called_once_with("loop1")
            # Should have started a new one with same config
            bot.loop_manager.start_loop.assert_called_once()
            call_kwargs = bot.loop_manager.start_loop.call_args[1]
            assert call_kwargs["goal"] == "monitor disks"
            assert call_kwargs["mode"] == "notify"
            assert call_kwargs["interval_seconds"] == 60

    @pytest.mark.asyncio
    async def test_restart_loop_not_found(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/loops/nonexistent/restart")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_restart_loop_channel_not_found(self):
        bot = _make_bot()
        bot.get_channel = MagicMock(return_value=None)
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/loops/loop1/restart")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_restart_stopped_loop(self):
        bot = _make_bot()
        bot.loop_manager._loops["loop1"].status = "stopped"
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/loops/loop1/restart")
            assert resp.status == 201
            # Should NOT call stop_loop since already stopped
            bot.loop_manager.stop_loop.assert_not_called()


# ===================================================================
# Round 13: Processes — output preview in list
# ===================================================================


class TestProcessOutputPreview:
    @pytest.mark.asyncio
    async def test_list_includes_output_preview(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/processes")
            body = await resp.json()
            assert len(body) == 1
            assert "output_preview" in body[0]
            # Last 3 lines of 4
            assert len(body[0]["output_preview"]) == 3
            assert body[0]["output_preview"][0] == "line2"
            assert body[0]["output_preview"][2] == "line4"

    @pytest.mark.asyncio
    async def test_list_empty_output_preview(self):
        bot = _make_bot()
        from collections import deque
        bot.tool_executor._process_registry._processes[1234].output_buffer = deque(maxlen=500)
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/processes")
            body = await resp.json()
            assert body[0]["output_preview"] == []


# ===================================================================
# Round 13: Schedules — run now + cron validation
# ===================================================================


class TestScheduleRunNow:
    @pytest.mark.asyncio
    async def test_run_now_success(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/schedules/sch1/run")
            assert resp.status == 200
            body = await resp.json()
            assert body["status"] == "triggered"
            assert body["schedule_id"] == "sch1"
            bot.scheduler._callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_now_not_found(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/schedules/nonexistent/run")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_run_now_no_callback(self):
        bot = _make_bot()
        bot.scheduler._callback = None
        app, _ = _make_app(bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/schedules/sch1/run")
            assert resp.status == 503

    @pytest.mark.asyncio
    async def test_run_now_updates_last_run(self):
        app, bot = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/schedules/sch1/run")
            # last_run should be updated
            sch = bot.scheduler._schedules[0]
            assert sch["last_run"] is not None


class TestScheduleValidateCron:
    @pytest.mark.asyncio
    async def test_validate_cron_valid(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/schedules/validate-cron",
                json={"expression": "0 */6 * * *"},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["valid"] is True
            assert len(body["next_runs"]) == 5

    @pytest.mark.asyncio
    async def test_validate_cron_invalid(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/schedules/validate-cron",
                json={"expression": "not a cron"},
            )
            body = await resp.json()
            assert body["valid"] is False
            assert "error" in body

    @pytest.mark.asyncio
    async def test_validate_cron_empty(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/schedules/validate-cron",
                json={"expression": ""},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_validate_cron_invalid_json(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/schedules/validate-cron",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
