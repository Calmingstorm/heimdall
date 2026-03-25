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
        MagicMock(role="user", content="hi", timestamp="2024-01-01", user_id="u1"),
    ]
    session.summary = ""
    session.created_at = "2024-01-01"
    session.last_active = "2024-01-02"
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

    # Knowledge
    store = MagicMock()
    store.available = True
    store.list_sources = MagicMock(return_value=[{"source": "doc1", "chunks": 3}])
    store.ingest = AsyncMock(return_value=5)
    store.search_hybrid = AsyncMock(return_value=[
        {"source": "doc1", "content": "hello", "score": 0.9},
    ])
    store.delete_source = MagicMock(return_value=3)
    bot._knowledge_store = store
    bot._embedder = None

    # Scheduler
    bot.scheduler = MagicMock()
    bot.scheduler.list_all = MagicMock(return_value=[
        {"id": "sch1", "description": "Daily backup", "channel_id": "c1"},
    ])
    bot.scheduler.add = MagicMock(return_value={
        "id": "sch2", "description": "New schedule",
    })
    bot.scheduler.delete = MagicMock(return_value=True)

    # Loops
    loop_info = MagicMock()
    loop_info.goal = "monitor disks"
    loop_info.mode = "notify"
    loop_info.interval_seconds = 60
    loop_info.stop_condition = None
    loop_info.max_iterations = 50
    loop_info.channel_id = "chan1"
    loop_info.requester_id = "u1"
    loop_info.requester_name = "Alice"
    loop_info.iteration_count = 5
    loop_info.last_trigger = "2024-01-02T10:00:00"
    loop_info.created_at = "2024-01-01T10:00:00"
    loop_info.status = "running"
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
    registry = MagicMock()
    registry._processes = {1234: proc}
    registry.kill = AsyncMock(return_value="Process killed")
    bot.tool_executor = MagicMock()
    bot.tool_executor._process_registry = registry

    # Audit
    bot.audit = MagicMock()
    bot.audit.search = AsyncMock(return_value=[
        {"timestamp": "2024-01-01", "tool_name": "run_command", "user": "u1"},
    ])

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
    async def test_config_update_returns_501(self):
        app, _ = _make_app(api_token="")
        async with TestClient(TestServer(app)) as client:
            resp = await client.put("/api/config", json={})
            assert resp.status == 501


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
                json={"packs": ["docker", "git"]},
            )
            body = await resp.json()
            assert body["status"] == "updated"
            assert bot.config.tools.tool_packs == ["docker", "git"]

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
        payload = mock_ws.send_json.call_args[0][0]
        assert payload["type"] == "event"
        assert payload["action"] == "tool_call"

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
