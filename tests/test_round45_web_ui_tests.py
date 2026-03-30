"""Round 45 — Comprehensive Web UI Tests.

Covers gaps in web UI test coverage:
- WebSocket protocol-level message handling (subscribe, unsubscribe, ping/pong,
  invalid JSON, unknown commands, ERROR/CLOSE types)
- WebSocket _tail_logs (initial tail, polling, file rotation, missing file)
- WebSocket disconnect cleanup (finally block)
- WebSocket client_count property
- setup_websocket() and setup_api() helpers
- Skill API endpoints (enable, disable, validate, detail, config GET/PUT)
- Knowledge chunks endpoint
- Agent endpoints (list, kill)
- Config PUT edge cases (no config.yml, invalid JSON body)
- _deep_merge depth limit
- _redact_config with lists and nested depth
- schedule creation with various fields
- reload endpoint behavior
- stop-all-loops endpoint behavior
- process list with output preview edge cases
- audit error_only filter with mixed results
- memory scope creation and multi-scope operations
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from src.config.schema import WebConfig
from src.health.server import (
    SessionManager,
    _make_auth_middleware,
    _make_rate_limit_middleware,
    _make_security_headers_middleware,
)
from src.web.api import (
    _deep_merge,
    _redact_config,
    _safe_filename,
    _sanitize_error,
    _validate_string,
    _contains_blocked_fields,
    _SENSITIVE_FIELDS,
    _MAX_NAME_LEN,
    _MAX_CODE_LEN,
    create_api_routes,
    setup_api,
)
from src.web.chat import (
    MAX_CHAT_CONTENT_LEN,
    WebMessage,
    _WebAuthor,
    _WebChannel,
    _WebSentMessage,
    process_web_chat,
)
from src.web.websocket import (
    WebSocketManager,
    _LOG_TAIL_LINES,
    _LOG_POLL_INTERVAL,
    setup_websocket,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bot():
    """Build a mock bot with all attributes the API routes access."""
    bot = MagicMock()
    guild = MagicMock()
    guild.id = 111
    guild.name = "TestGuild"
    guild.member_count = 42
    bot.guilds = [guild]
    bot.is_ready = MagicMock(return_value=True)
    bot._start_time = time.monotonic() - 120
    bot.get_channel = MagicMock(return_value=MagicMock())

    bot.config = MagicMock()
    bot.config.model_dump = MagicMock(return_value={
        "discord": {"token": "xyzzy-secret", "channels": ["general"]},
        "tools": {"ssh_key_path": "/key"},
        "web": {"api_token": "tok", "enabled": True},
    })
    bot.config.web.api_token = "test-token"

    bot._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run cmd", "input_schema": {}},
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
    bot.skill_manager.get_skill_info = MagicMock(return_value={
        "name": "joke",
        "description": "Tell a joke",
        "metadata": {"config_schema": {"timeout": {"type": "int", "default": 30}}},
    })
    bot.skill_manager.validate_skill_code = MagicMock(return_value={
        "valid": True, "errors": [],
    })
    bot.skill_manager.enable_skill = MagicMock(return_value="Skill 'joke' enabled")
    bot.skill_manager.disable_skill = MagicMock(return_value="Skill 'joke' disabled")
    bot.skill_manager.get_skill_config = MagicMock(return_value={"timeout": 30})
    bot.skill_manager.set_skill_config = MagicMock(return_value=[])

    # Knowledge
    store = MagicMock()
    store.available = True
    store.list_sources = MagicMock(return_value=[
        {"source": "doc1", "chunks": 3, "preview": "Hello world"},
    ])
    store.ingest = AsyncMock(return_value=5)
    store.search_hybrid = AsyncMock(return_value=[
        {"source": "doc1", "content": "hello", "score": 0.9},
    ])
    store.delete_source = MagicMock(return_value=3)
    store.get_source_content = MagicMock(return_value="Full content here")
    store.get_source_chunks = MagicMock(return_value=[
        {"id": 1, "content": "chunk1", "source": "doc1"},
        {"id": 2, "content": "chunk2", "source": "doc1"},
    ])
    bot._knowledge_store = store
    bot._embedder = None

    # Scheduler
    bot.scheduler = MagicMock()
    bot.scheduler._schedules = [
        {"id": "sch1", "description": "Daily backup", "channel_id": "c1",
         "action": "reminder", "cron": "0 9 * * *", "last_run": None},
    ]
    bot.scheduler.list_all = MagicMock(return_value=[
        {"id": "sch1", "description": "Daily backup"},
    ])
    bot.scheduler.add = MagicMock(return_value={"id": "sch2", "description": "New"})
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
    loop_info._iteration_history = ["Iter 1", "Iter 2", "Iter 3"]
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
    proc.output_buffer = deque(["line1\n", "line2\n", "line3\n", "line4\n"], maxlen=500)
    registry = MagicMock()
    registry._processes = {1234: proc}
    registry.kill = AsyncMock(return_value="Process killed")
    bot.tool_executor = MagicMock()
    bot.tool_executor._process_registry = registry

    # Agents
    agent_info = MagicMock()
    agent_info.label = "disk-checker"
    agent_info.goal = "Check disk usage"
    agent_info.status = "running"
    agent_info.channel_id = "123456"
    agent_info.requester_name = "Alice"
    agent_info.iteration_count = 3
    agent_info.tools_used = ["run_command", "read_file"]
    agent_info.created_at = time.time() - 60
    agent_info.ended_at = None
    agent_info.result = ""
    agent_info.error = ""
    bot.agent_manager = MagicMock()
    bot.agent_manager._agents = {"abc123": agent_info}
    bot.agent_manager.kill = MagicMock(return_value="Kill signal sent")

    # Monitoring
    bot.infra_watcher = MagicMock()
    bot.infra_watcher.get_status = MagicMock(return_value={
        "enabled": True, "checks": 3, "running": 2, "active_alerts": 0,
    })

    # Context / reload
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
    if bot is None:
        bot = _make_bot()
    wc = WebConfig(api_token=api_token)
    app = web.Application(middlewares=[
        _make_security_headers_middleware(),
        _make_rate_limit_middleware(),
        _make_auth_middleware(wc, SessionManager()),
    ])
    setup_api(app, bot)
    return app, bot


def _auth(token="test-token"):
    return {"Authorization": f"Bearer {token}"}


# ===================================================================
# WebSocket protocol-level tests
# ===================================================================


class TestWebSocketProtocol:
    """Test WebSocket subscribe, unsubscribe, ping/pong, invalid JSON, unknown commands."""

    async def test_subscribe_logs(self):
        bot = _make_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"subscribe": "logs"})
            resp = await ws.receive_json()
            assert resp["type"] == "subscribed"
            assert resp["channel"] == "logs"
            assert len(mgr._log_subscribers) == 1
            await ws.close()

    async def test_subscribe_events(self):
        bot = _make_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"subscribe": "events"})
            resp = await ws.receive_json()
            assert resp["type"] == "subscribed"
            assert resp["channel"] == "events"
            assert len(mgr._event_subscribers) == 1
            await ws.close()

    async def test_unsubscribe_logs(self):
        bot = _make_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            # Subscribe first
            await ws.send_json({"subscribe": "logs"})
            await ws.receive_json()  # consume subscribed msg
            # Unsubscribe
            await ws.send_json({"unsubscribe": "logs"})
            resp = await ws.receive_json()
            assert resp["type"] == "unsubscribed"
            assert resp["channel"] == "logs"
            await ws.close()

    async def test_unsubscribe_events(self):
        bot = _make_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"subscribe": "events"})
            await ws.receive_json()
            await ws.send_json({"unsubscribe": "events"})
            resp = await ws.receive_json()
            assert resp["type"] == "unsubscribed"
            assert resp["channel"] == "events"
            await ws.close()

    async def test_ping_pong(self):
        bot = _make_bot()
        app = web.Application()
        setup_websocket(app, bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"type": "ping", "ts": 1234567890})
            resp = await ws.receive_json()
            assert resp["type"] == "pong"
            assert resp["ts"] == 1234567890
            await ws.close()

    async def test_ping_pong_no_ts(self):
        bot = _make_bot()
        app = web.Application()
        setup_websocket(app, bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"type": "ping"})
            resp = await ws.receive_json()
            assert resp["type"] == "pong"
            assert resp["ts"] is None
            await ws.close()

    async def test_invalid_json(self):
        bot = _make_bot()
        app = web.Application()
        setup_websocket(app, bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_str("not json{{{")
            resp = await ws.receive_json()
            assert "error" in resp
            assert resp["error"] == "invalid JSON"
            await ws.close()

    async def test_unknown_command(self):
        bot = _make_bot()
        app = web.Application()
        setup_websocket(app, bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"foo": "bar"})
            resp = await ws.receive_json()
            assert "error" in resp
            assert resp["error"] == "unknown command"
            await ws.close()

    async def test_subscribe_both_channels(self):
        bot = _make_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"subscribe": "logs"})
            r1 = await ws.receive_json()
            assert r1["channel"] == "logs"
            await ws.send_json({"subscribe": "events"})
            r2 = await ws.receive_json()
            assert r2["channel"] == "events"
            assert len(mgr._log_subscribers) == 1
            assert len(mgr._event_subscribers) == 1
            await ws.close()


class TestWebSocketClientCount:
    async def test_client_count_tracks_connections(self):
        bot = _make_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot, api_token="")
        assert mgr.client_count == 0
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            assert mgr.client_count == 1
            await ws.close()
            # Allow server-side cleanup
            await asyncio.sleep(0.1)
            assert mgr.client_count == 0


class TestWebSocketDisconnectCleanup:
    async def test_cleanup_removes_from_all_sets(self):
        bot = _make_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"subscribe": "logs"})
            await ws.receive_json()
            await ws.send_json({"subscribe": "events"})
            await ws.receive_json()
            assert mgr.client_count == 1
            assert len(mgr._log_subscribers) == 1
            assert len(mgr._event_subscribers) == 1
            await ws.close()
            await asyncio.sleep(0.1)
            assert mgr.client_count == 0
            assert len(mgr._log_subscribers) == 0
            assert len(mgr._event_subscribers) == 0


class TestWebSocketTailLogs:
    """Test _tail_logs: initial tail from file, polling for new lines, rotation."""

    async def test_subscribe_logs_sends_existing_lines(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        # Write more than _LOG_TAIL_LINES lines
        lines = [json.dumps({"line": i}) for i in range(60)]
        log_path.write_text("\n".join(lines))

        bot = _make_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot, api_token="")

        with patch("src.web.websocket.Path") as mock_path_cls:
            mock_path_cls.return_value = log_path
            async with TestClient(TestServer(app)) as client:
                ws = await client.ws_connect("/api/ws")
                await ws.send_json({"subscribe": "logs"})
                # First message is the subscribed confirmation
                sub_resp = await ws.receive_json()
                assert sub_resp["type"] == "subscribed"

                # Read the tail lines (last 50 of 60)
                received = []
                for _ in range(_LOG_TAIL_LINES):
                    try:
                        msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                        if msg.get("type") == "log":
                            received.append(msg)
                    except asyncio.TimeoutError:
                        break
                assert len(received) == _LOG_TAIL_LINES
                # First received should be line 10 (60-50=10)
                assert json.loads(received[0]["line"])["line"] == 10
                await ws.close()

    async def test_subscribe_logs_empty_file(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        log_path.write_text("")

        bot = _make_bot()
        app = web.Application()
        setup_websocket(app, bot, api_token="")

        with patch("src.web.websocket.Path") as mock_path_cls:
            mock_path_cls.return_value = log_path
            async with TestClient(TestServer(app)) as client:
                ws = await client.ws_connect("/api/ws")
                await ws.send_json({"subscribe": "logs"})
                sub = await ws.receive_json()
                assert sub["type"] == "subscribed"
                # No log lines should follow immediately
                import aiohttp
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.5)
                    # If we get a message, it shouldn't be a log line
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        assert data.get("type") != "log"
                except asyncio.TimeoutError:
                    pass  # Expected — no lines to send
                await ws.close()

    async def test_subscribe_logs_no_file(self):
        """When audit.jsonl doesn't exist, just polls without error."""
        bot = _make_bot()
        app = web.Application()
        setup_websocket(app, bot, api_token="")

        with patch("src.web.websocket.Path") as mock_path_cls:
            mock_p = MagicMock()
            mock_p.exists.return_value = False
            mock_path_cls.return_value = mock_p
            async with TestClient(TestServer(app)) as client:
                ws = await client.ws_connect("/api/ws")
                await ws.send_json({"subscribe": "logs"})
                sub = await ws.receive_json()
                assert sub["type"] == "subscribed"
                await ws.close()


class TestSetupWebSocket:
    def test_setup_registers_route(self):
        bot = _make_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot, api_token="sekrit")
        assert isinstance(mgr, WebSocketManager)
        assert mgr._api_token == "sekrit"
        # Route should be registered
        routes = [r.resource.canonical for r in app.router.routes() if hasattr(r, "resource") and r.resource]
        assert "/api/ws" in routes

    def test_setup_no_token(self):
        bot = _make_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot, api_token="")
        assert mgr._api_token == ""


class TestSetupApi:
    def test_registers_routes(self):
        bot = _make_bot()
        app = web.Application()
        setup_api(app, bot)
        paths = set()
        for route in app.router.routes():
            if hasattr(route, "resource") and route.resource:
                paths.add(route.resource.canonical)
        # Check key endpoint paths
        assert "/api/status" in paths
        assert "/api/config" in paths
        assert "/api/sessions" in paths
        assert "/api/tools" in paths
        assert "/api/skills" in paths
        assert "/api/knowledge" in paths
        assert "/api/schedules" in paths
        assert "/api/loops" in paths
        assert "/api/agents" in paths
        assert "/api/processes" in paths
        assert "/api/audit" in paths
        assert "/api/memory" in paths
        assert "/api/chat" in paths


# ===================================================================
# Skill API endpoint tests (enable, disable, validate, detail, config)
# ===================================================================


class TestSkillEnableEndpoint:
    async def test_enable_skill_success(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/skills/joke/enable", headers=_auth())
            assert resp.status == 200
            data = await resp.json()
            assert "enabled" in data["result"].lower()

    async def test_enable_skill_not_found(self):
        app, bot = _make_app()
        bot.skill_manager.enable_skill = MagicMock(return_value="Skill 'nope' not found")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/skills/nope/enable", headers=_auth())
            assert resp.status == 404

    async def test_enable_invalidates_caches(self):
        app, bot = _make_app()
        bot._cached_merged_tools = "stale"
        bot._cached_skills_text = "stale"
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/skills/joke/enable", headers=_auth())
            assert bot._cached_merged_tools is None
            assert bot._cached_skills_text is None


class TestSkillDisableEndpoint:
    async def test_disable_skill_success(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/skills/joke/disable", headers=_auth())
            assert resp.status == 200
            data = await resp.json()
            assert "disabled" in data["result"].lower()

    async def test_disable_skill_not_found(self):
        app, bot = _make_app()
        bot.skill_manager.disable_skill = MagicMock(return_value="Skill 'nope' not found")
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/skills/nope/disable", headers=_auth())
            assert resp.status == 404

    async def test_disable_invalidates_caches(self):
        app, bot = _make_app()
        bot._cached_merged_tools = "stale"
        bot._cached_skills_text = "stale"
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/skills/joke/disable", headers=_auth())
            assert bot._cached_merged_tools is None
            assert bot._cached_skills_text is None


class TestSkillValidateEndpoint:
    async def test_validate_success(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/skills/validate",
                json={"code": "print('hello')"},
                headers=_auth(),
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["valid"] is True
            assert data["errors"] == []

    async def test_validate_empty_code(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/skills/validate",
                json={"code": ""},
                headers=_auth(),
            )
            assert resp.status == 400
            data = await resp.json()
            assert "code is required" in data["error"]

    async def test_validate_code_too_long(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/skills/validate",
                json={"code": "x" * (_MAX_CODE_LEN + 1)},
                headers=_auth(),
            )
            assert resp.status == 400
            data = await resp.json()
            assert "exceeds" in data["error"]

    async def test_validate_returns_errors(self):
        app, bot = _make_app()
        bot.skill_manager.validate_skill_code = MagicMock(return_value={
            "valid": False, "errors": ["syntax error on line 3"],
        })
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/skills/validate",
                json={"code": "def foo(:\n  pass"},
                headers=_auth(),
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["valid"] is False
            assert len(data["errors"]) == 1


class TestSkillDetailEndpoint:
    async def test_get_skill_detail(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/skills/joke", headers=_auth())
            assert resp.status == 200
            data = await resp.json()
            assert data["name"] == "joke"
            assert "metadata" in data

    async def test_get_skill_detail_not_found(self):
        app, bot = _make_app()
        bot.skill_manager.get_skill_info = MagicMock(return_value=None)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/skills/nonexistent", headers=_auth())
            assert resp.status == 404
            data = await resp.json()
            assert "not found" in data["error"]


class TestSkillConfigEndpoint:
    async def test_get_skill_config(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/skills/joke/config", headers=_auth())
            assert resp.status == 200
            data = await resp.json()
            assert data["config"] == {"timeout": 30}
            assert "schema" in data
            assert "timeout" in data["schema"]

    async def test_get_skill_config_not_found(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/skills/nonexistent/config", headers=_auth())
            assert resp.status == 404

    async def test_set_skill_config(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/skills/joke/config",
                json={"config": {"timeout": 60}},
                headers=_auth(),
            )
            assert resp.status == 200
            data = await resp.json()
            assert "config" in data
            bot.skill_manager.set_skill_config.assert_called_with("joke", {"timeout": 60})

    async def test_set_skill_config_not_found(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/skills/nonexistent/config",
                json={"config": {"timeout": 60}},
                headers=_auth(),
            )
            assert resp.status == 404

    async def test_set_skill_config_not_dict(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/skills/joke/config",
                json={"config": "not a dict"},
                headers=_auth(),
            )
            assert resp.status == 400
            data = await resp.json()
            assert "dict" in data["error"]

    async def test_set_skill_config_validation_errors(self):
        app, bot = _make_app()
        bot.skill_manager.set_skill_config = MagicMock(
            return_value=["timeout must be an integer"]
        )
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/skills/joke/config",
                json={"config": {"timeout": "bad"}},
                headers=_auth(),
            )
            assert resp.status == 400
            data = await resp.json()
            assert "errors" in data


# ===================================================================
# Knowledge chunks endpoint
# ===================================================================


class TestKnowledgeChunksEndpoint:
    async def test_list_chunks(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/doc1/chunks", headers=_auth())
            assert resp.status == 200
            data = await resp.json()
            assert len(data) == 2
            assert data[0]["content"] == "chunk1"

    async def test_list_chunks_not_found(self):
        app, bot = _make_app()
        bot._knowledge_store.get_source_chunks = MagicMock(return_value=[])
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/nonexistent/chunks", headers=_auth())
            assert resp.status == 404

    async def test_list_chunks_unavailable(self):
        app, bot = _make_app()
        bot._knowledge_store.available = False
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/doc1/chunks", headers=_auth())
            assert resp.status == 503

    async def test_list_chunks_no_store(self):
        app, bot = _make_app()
        bot._knowledge_store = None
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/doc1/chunks", headers=_auth())
            assert resp.status == 503


# ===================================================================
# Agent endpoints
# ===================================================================


class TestAgentEndpoints:
    async def test_list_agents(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents", headers=_auth())
            assert resp.status == 200
            data = await resp.json()
            assert len(data) == 1
            agent = data[0]
            assert agent["id"] == "abc123"
            assert agent["label"] == "disk-checker"
            assert agent["status"] == "running"
            assert agent["tools_used"] == ["run_command", "read_file"]
            assert agent["runtime_seconds"] > 0

    async def test_list_agents_empty(self):
        app, bot = _make_app()
        bot.agent_manager._agents = {}
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents", headers=_auth())
            assert resp.status == 200
            data = await resp.json()
            assert data == []

    async def test_list_agents_no_manager(self):
        app, bot = _make_app()
        bot.agent_manager._agents = "not a dict"
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents", headers=_auth())
            assert resp.status == 200
            data = await resp.json()
            assert data == []

    async def test_kill_agent(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/agents/abc123", headers=_auth())
            assert resp.status == 200
            data = await resp.json()
            assert "Kill signal sent" in data["result"]

    async def test_kill_agent_not_found(self):
        app, bot = _make_app()
        bot.agent_manager.kill = MagicMock(return_value="Agent 'xyz' not found")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/agents/xyz", headers=_auth())
            assert resp.status == 404

    async def test_kill_agent_no_manager(self):
        app, bot = _make_app()
        bot.agent_manager._agents = "not a dict"
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/agents/abc123", headers=_auth())
            assert resp.status == 404

    async def test_agent_goal_truncated(self):
        app, bot = _make_app()
        bot.agent_manager._agents["abc123"].goal = "x" * 300
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents", headers=_auth())
            data = await resp.json()
            assert len(data[0]["goal"]) == 200

    async def test_agent_result_truncated(self):
        app, bot = _make_app()
        bot.agent_manager._agents["abc123"].result = "r" * 300
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents", headers=_auth())
            data = await resp.json()
            assert len(data[0]["result"]) == 200

    async def test_agent_error_truncated(self):
        app, bot = _make_app()
        bot.agent_manager._agents["abc123"].error = "e" * 300
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents", headers=_auth())
            data = await resp.json()
            assert len(data[0]["error"]) == 200

    async def test_agent_with_ended_at(self):
        app, bot = _make_app()
        agent = bot.agent_manager._agents["abc123"]
        agent.ended_at = agent.created_at + 30
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents", headers=_auth())
            data = await resp.json()
            assert data[0]["runtime_seconds"] == 30.0

    async def test_agent_tools_capped_at_10(self):
        app, bot = _make_app()
        bot.agent_manager._agents["abc123"].tools_used = [f"tool_{i}" for i in range(20)]
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/agents", headers=_auth())
            data = await resp.json()
            assert len(data[0]["tools_used"]) == 10
            assert data[0]["tools_used"][0] == "tool_10"  # last 10


# ===================================================================
# Config PUT edge cases
# ===================================================================


class TestConfigPutEdgeCases:
    async def test_config_put_no_config_file(self):
        """Config update works even when config.yml doesn't exist on disk."""
        app, bot = _make_app()
        from src.config.schema import Config
        with patch("src.web.api.Config") as MockConfig, \
             patch("src.web.api.Path") as MockPath:
            MockConfig.return_value = bot.config
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            MockPath.return_value = mock_path
            async with TestClient(TestServer(app)) as client:
                resp = await client.put(
                    "/api/config",
                    json={"tools": {"tool_timeout_seconds": 600}},
                    headers=_auth(),
                )
                assert resp.status == 200

    async def test_config_put_not_json_object(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/config",
                json=["not", "an", "object"],
                headers=_auth(),
            )
            assert resp.status == 400
            data = await resp.json()
            assert "expected JSON object" in data["error"]

    async def test_config_put_disk_write_failure(self):
        """Config is applied in memory even if disk write fails."""
        app, bot = _make_app()
        from src.config.schema import Config
        with patch("src.web.api.Config") as MockConfig, \
             patch("src.web.api._write_config", side_effect=OSError("disk full")):
            MockConfig.return_value = bot.config
            async with TestClient(TestServer(app)) as client:
                resp = await client.put(
                    "/api/config",
                    json={"tools": {"tool_timeout_seconds": 600}},
                    headers=_auth(),
                )
                # Should still succeed (applied in memory)
                assert resp.status == 200


# ===================================================================
# Utility function edge cases
# ===================================================================


class TestDeepMergeEdgeCases:
    def test_depth_limit_prevents_infinite_recursion(self):
        """_deep_merge stops at depth 10."""
        base = {"a": {"b": {}}}
        d = base["a"]["b"]
        for i in range(15):
            d[f"l{i}"] = {}
            d = d[f"l{i}"]
        # Should not raise regardless of depth
        _deep_merge(base, {"a": {"b": {"l0": {"l1": {"l2": {"l3": {"l4": {
            "l5": {"l6": {"l7": {"l8": {"l9": {"l10": {"deep": "value"}}}}}}
        }}}}}}}})

    def test_merge_overwrites_with_non_dict(self):
        base = {"a": {"b": "old"}}
        _deep_merge(base, {"a": "replaced"})
        assert base["a"] == "replaced"

    def test_merge_adds_new_keys(self):
        base = {"a": 1}
        _deep_merge(base, {"b": 2})
        assert base == {"a": 1, "b": 2}


class TestRedactConfigEdgeCases:
    def test_redact_list_of_dicts(self):
        cfg = {"items": [{"token": "secret"}, {"name": "ok"}]}
        result = _redact_config(cfg)
        assert result["items"][0]["token"] == "••••••••"
        assert result["items"][1]["name"] == "ok"

    def test_redact_deeply_nested_stops_at_limit(self):
        nested = {}
        current = nested
        for i in range(15):
            current[f"l{i}"] = {}
            current = current[f"l{i}"]
        current["token"] = "secret"
        result = _redact_config(nested)
        # Beyond depth 10, returns "..."
        # The nesting is 15 levels deep, so token won't be individually redacted
        assert isinstance(result, dict)

    def test_redact_preserves_non_sensitive_int_values(self):
        cfg = {"password": 42, "port": 8080}
        result = _redact_config(cfg)
        assert result["password"] == 42  # int, not redacted
        assert result["port"] == 8080


class TestContainsBlockedFieldsEdgeCases:
    def test_depth_limit(self):
        """Returns False when depth exceeds 10."""
        deep = {}
        d = deep
        for _ in range(15):
            d["x"] = {}
            d = d["x"]
        d["token"] = "secret"
        # Should return False because depth limit prevents reaching the key
        assert not _contains_blocked_fields(deep, _SENSITIVE_FIELDS)


class TestSafeFilenameEdgeCases:
    def test_normal_filename(self):
        assert _safe_filename("my-session_123") == "my-session_123"

    def test_special_chars_replaced(self):
        assert _safe_filename("hello world!@#") == "hello_world___"

    def test_empty_returns_export(self):
        assert _safe_filename("") == "export"

    def test_max_length_truncation(self):
        long = "a" * 100
        result = _safe_filename(long, max_len=50)
        assert len(result) == 50


# ===================================================================
# Schedule creation with various fields
# ===================================================================


class TestScheduleCreationFields:
    async def test_create_with_cron(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/schedules",
                json={
                    "description": "Backup",
                    "channel_id": "c1",
                    "action": "tool",
                    "cron": "0 */6 * * *",
                    "tool_name": "run_command",
                    "tool_input": {"command": "backup.sh"},
                },
                headers=_auth(),
            )
            assert resp.status == 201
            bot.scheduler.add.assert_called_once()
            call_kwargs = bot.scheduler.add.call_args
            assert call_kwargs.kwargs.get("cron") == "0 */6 * * *"
            assert call_kwargs.kwargs.get("tool_name") == "run_command"

    async def test_create_with_run_at(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/schedules",
                json={
                    "description": "One-time task",
                    "channel_id": "c1",
                    "action": "message",
                    "run_at": "2024-06-15T10:00:00",
                    "message": "Time's up!",
                },
                headers=_auth(),
            )
            assert resp.status == 201

    async def test_create_schedule_value_error(self):
        app, bot = _make_app()
        bot.scheduler.add = MagicMock(side_effect=ValueError("invalid cron"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/schedules",
                json={"description": "bad", "channel_id": "c1"},
                headers=_auth(),
            )
            assert resp.status == 400


class TestScheduleRunNowEdgeCases:
    async def test_run_now_callback_exception(self):
        app, bot = _make_app()
        bot.scheduler._callback = AsyncMock(side_effect=RuntimeError("boom"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/schedules/sch1/run", headers=_auth())
            assert resp.status == 500


# ===================================================================
# Quick actions edge cases
# ===================================================================


class TestReloadEndpoint:
    async def test_reload_calls_context_loader(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/reload", headers=_auth())
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "reloaded"
            bot.context_loader.reload.assert_called_once()
            bot._invalidate_prompt_caches.assert_called_once()
            bot._build_system_prompt.assert_called()


class TestStopAllLoops:
    async def test_stop_all_returns_result(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/loops/stop-all", headers=_auth())
            assert resp.status == 200
            data = await resp.json()
            assert "result" in data
            bot.loop_manager.stop_loop.assert_called_with("all")


# ===================================================================
# Process endpoint edge cases
# ===================================================================


class TestProcessEndpointEdgeCases:
    async def test_process_output_preview_last_3_lines(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/processes", headers=_auth())
            data = await resp.json()
            # Bot has 4 lines, preview should be last 3
            assert len(data[0]["output_preview"]) == 3
            assert data[0]["output_preview"] == ["line2", "line3", "line4"]

    async def test_process_output_preview_few_lines(self):
        app, bot = _make_app()
        proc = bot.tool_executor._process_registry._processes[1234]
        proc.output_buffer = deque(["only\n"], maxlen=500)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/processes", headers=_auth())
            data = await resp.json()
            assert data[0]["output_preview"] == ["only"]

    async def test_process_with_exit_code(self):
        app, bot = _make_app()
        proc = bot.tool_executor._process_registry._processes[1234]
        proc.status = "exited"
        proc.exit_code = 1
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/processes", headers=_auth())
            data = await resp.json()
            assert data[0]["status"] == "exited"
            assert data[0]["exit_code"] == 1


# ===================================================================
# Audit endpoint edge cases
# ===================================================================


class TestAuditEndpointEdgeCases:
    async def test_error_only_filters_correctly(self):
        app, bot = _make_app()
        bot.audit.search = AsyncMock(return_value=[
            {"tool_name": "run_command", "error": None},
            {"tool_name": "read_file", "error": "permission denied"},
            {"tool_name": "write_file", "error": ""},
        ])
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/audit?error_only=true", headers=_auth())
            data = await resp.json()
            # Only entries with truthy error field
            assert len(data) == 1
            assert data[0]["tool_name"] == "read_file"

    async def test_error_only_false_returns_all(self):
        app, bot = _make_app()
        bot.audit.search = AsyncMock(return_value=[
            {"tool_name": "a", "error": None},
            {"tool_name": "b", "error": "fail"},
        ])
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/audit?error_only=false", headers=_auth())
            data = await resp.json()
            assert len(data) == 2

    async def test_audit_all_filter_params(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/api/audit?tool=run_command&user=alice&host=srv1&q=disk&date=2024-01-01&limit=5",
                headers=_auth(),
            )
            assert resp.status == 200
            call_kwargs = bot.audit.search.call_args.kwargs
            assert call_kwargs["tool_name"] == "run_command"
            assert call_kwargs["user"] == "alice"
            assert call_kwargs["host"] == "srv1"
            assert call_kwargs["keyword"] == "disk"
            assert call_kwargs["date"] == "2024-01-01"
            assert call_kwargs["limit"] == 5


# ===================================================================
# Memory endpoint edge cases
# ===================================================================


class TestMemoryEdgeCases:
    async def test_set_creates_new_scope(self):
        app, bot = _make_app()
        bot.tool_executor._load_all_memory = MagicMock(return_value={})
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/memory/new-scope/mykey",
                json={"value": "myval"},
                headers=_auth(),
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["scope"] == "new-scope"
            bot.tool_executor._save_all_memory.assert_called_once()

    async def test_set_value_coerced_to_string(self):
        app, bot = _make_app()
        bot.tool_executor._load_all_memory = MagicMock(return_value={"global": {}})
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/memory/global/number",
                json={"value": 42},
                headers=_auth(),
            )
            assert resp.status == 200
            # Verify the saved value was str(42)
            saved = bot.tool_executor._save_all_memory.call_args[0][0]
            assert saved["global"]["number"] == "42"

    async def test_bulk_delete_saves_only_when_deleted(self):
        app, bot = _make_app()
        bot.tool_executor._load_all_memory = MagicMock(return_value={"global": {}})
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/memory/bulk-delete",
                json={"entries": [{"scope": "nope", "key": "nope"}]},
                headers=_auth(),
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["count"] == 0
            # Should NOT call save when nothing was deleted
            bot.tool_executor._save_all_memory.assert_not_called()

    async def test_list_memory_multiple_scopes(self):
        app, bot = _make_app()
        bot.tool_executor._load_all_memory = MagicMock(return_value={
            "global": {"k1": "v1", "k2": "v2"},
            "user:alice": {"pref": "dark"},
            "user:bob": {},
        })
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/memory", headers=_auth())
            data = await resp.json()
            assert data["global"]["count"] == 2
            assert data["user:alice"]["count"] == 1
            assert data["user:bob"]["count"] == 0


# ===================================================================
# Session export edge cases
# ===================================================================


class TestSessionExportEdgeCases:
    async def test_export_text_with_summary(self):
        app, bot = _make_app()
        session = bot.sessions._sessions["chan1"]
        session.summary = "This is a session summary"
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/api/sessions/chan1/export?format=text", headers=_auth()
            )
            assert resp.status == 200
            text = await resp.text()
            assert "=== Summary ===" in text
            assert "This is a session summary" in text

    async def test_export_json_includes_exported_at(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/api/sessions/chan1/export?format=json", headers=_auth()
            )
            assert resp.status == 200
            data = await resp.json()
            assert "exported_at" in data
            assert data["exported_at"] > 0


# ===================================================================
# Loop endpoint edge cases
# ===================================================================


class TestLoopEndpointEdgeCases:
    async def test_loop_with_empty_history(self):
        app, bot = _make_app()
        bot.loop_manager._loops["loop1"]._iteration_history = []
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/loops", headers=_auth())
            data = await resp.json()
            assert data[0]["iteration_history"] == []

    async def test_loop_history_capped_at_5(self):
        app, bot = _make_app()
        bot.loop_manager._loops["loop1"]._iteration_history = [
            f"Iter {i}" for i in range(10)
        ]
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/loops", headers=_auth())
            data = await resp.json()
            history = data[0]["iteration_history"]
            assert len(history) == 5
            assert history[0] == "Iter 5"  # last 5

    async def test_start_loop_with_options(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/loops",
                json={
                    "goal": "monitor disk",
                    "channel_id": "123456",
                    "interval_seconds": 120,
                    "mode": "silent",
                    "stop_condition": "disk > 90%",
                    "max_iterations": 10,
                    "requester_id": "web-user",
                    "requester_name": "Web Admin",
                },
                headers=_auth(),
            )
            assert resp.status == 201
            call_kwargs = bot.loop_manager.start_loop.call_args.kwargs
            assert call_kwargs["interval_seconds"] == 120
            assert call_kwargs["mode"] == "silent"
            assert call_kwargs["stop_condition"] == "disk > 90%"
            assert call_kwargs["max_iterations"] == 10

    async def test_stop_loop_not_running(self):
        app, bot = _make_app()
        bot.loop_manager.stop_loop = MagicMock(return_value="Loop not running")
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/loops/loop1", headers=_auth())
            assert resp.status == 404


# ===================================================================
# Chat edge cases
# ===================================================================


class TestProcessWebChatEdgeCases:
    async def test_save_failure_does_not_crash(self):
        """Session save failure is logged but doesn't fail the response."""
        bot = MagicMock()
        bot.codex_client = MagicMock()
        bot.sessions = MagicMock()
        bot.sessions.add_message = MagicMock()
        bot.sessions.get_task_history = AsyncMock(return_value=[])
        bot.sessions.prune = MagicMock()
        bot.sessions.save = MagicMock(side_effect=OSError("disk full"))
        bot.sessions.remove_last_message = MagicMock()
        bot._build_system_prompt = MagicMock(return_value="prompt")
        bot._inject_tool_hints = AsyncMock(return_value="prompt")
        bot._process_with_tools = AsyncMock(
            return_value=("reply", False, False, ["run_command"], False)
        )
        result = await process_web_chat(bot, "hello", "web-test")
        assert result["response"] == "reply"
        assert result["is_error"] is False

    async def test_error_with_tools_saves_context(self):
        """Error path with tools saves abbreviated context."""
        bot = MagicMock()
        bot.codex_client = MagicMock()
        bot.sessions = MagicMock()
        bot.sessions.add_message = MagicMock()
        bot.sessions.get_task_history = AsyncMock(return_value=[])
        bot.sessions.prune = MagicMock()
        bot.sessions.save = MagicMock()
        bot.sessions.remove_last_message = MagicMock()
        bot._build_system_prompt = MagicMock(return_value="prompt")
        bot._inject_tool_hints = AsyncMock(return_value="prompt")
        bot._process_with_tools = AsyncMock(
            return_value=("error occurred", False, True, ["run_command", "read_file"], False)
        )
        result = await process_web_chat(bot, "hello", "web-test")
        assert result["is_error"] is True
        # Session should contain sanitized error reference with tool names
        calls = bot.sessions.add_message.call_args_list
        last_call = calls[-1]
        saved_msg = last_call[0][2]
        assert "run_command" in saved_msg
        assert "read_file" in saved_msg

    async def test_error_without_tools_saves_generic_context(self):
        bot = MagicMock()
        bot.codex_client = MagicMock()
        bot.sessions = MagicMock()
        bot.sessions.add_message = MagicMock()
        bot.sessions.get_task_history = AsyncMock(return_value=[])
        bot.sessions.prune = MagicMock()
        bot.sessions.save = MagicMock()
        bot.sessions.remove_last_message = MagicMock()
        bot._build_system_prompt = MagicMock(return_value="prompt")
        bot._inject_tool_hints = AsyncMock(return_value="prompt")
        bot._process_with_tools = AsyncMock(
            return_value=("error occurred", False, True, [], False)
        )
        result = await process_web_chat(bot, "hello", "web-test")
        assert result["is_error"] is True
        calls = bot.sessions.add_message.call_args_list
        last_call = calls[-1]
        saved_msg = last_call[0][2]
        assert "before tool execution" in saved_msg

    async def test_handoff_triggers_save(self):
        """When handoff=True but no tools, session should still be saved."""
        bot = MagicMock()
        bot.codex_client = MagicMock()
        bot.sessions = MagicMock()
        bot.sessions.add_message = MagicMock()
        bot.sessions.get_task_history = AsyncMock(return_value=[])
        bot.sessions.prune = MagicMock()
        bot.sessions.save = MagicMock()
        bot.sessions.remove_last_message = MagicMock()
        bot._build_system_prompt = MagicMock(return_value="prompt")
        bot._inject_tool_hints = AsyncMock(return_value="prompt")
        bot._process_with_tools = AsyncMock(
            return_value=("reply", False, False, [], True)
        )
        result = await process_web_chat(bot, "hello", "web-test")
        assert result["is_error"] is False
        # Should save because handoff=True
        assert bot.sessions.add_message.call_count == 2  # user + assistant
        bot.sessions.prune.assert_called_once()

    async def test_no_tools_no_handoff_skips_save(self):
        bot = MagicMock()
        bot.codex_client = MagicMock()
        bot.sessions = MagicMock()
        bot.sessions.add_message = MagicMock()
        bot.sessions.get_task_history = AsyncMock(return_value=[])
        bot.sessions.prune = MagicMock()
        bot.sessions.save = MagicMock()
        bot.sessions.remove_last_message = MagicMock()
        bot._build_system_prompt = MagicMock(return_value="prompt")
        bot._inject_tool_hints = AsyncMock(return_value="prompt")
        bot._process_with_tools = AsyncMock(
            return_value=("just chat", False, False, [], False)
        )
        result = await process_web_chat(bot, "hello", "web-test")
        assert result["is_error"] is False
        # Only user message added, no assistant message (tools_used=[] and handoff=False)
        assert bot.sessions.add_message.call_count == 1  # user only

    async def test_error_path_with_more_than_5_tools(self):
        """Error path truncates tool list to 5."""
        bot = MagicMock()
        bot.codex_client = MagicMock()
        bot.sessions = MagicMock()
        bot.sessions.add_message = MagicMock()
        bot.sessions.get_task_history = AsyncMock(return_value=[])
        bot.sessions.prune = MagicMock()
        bot.sessions.save = MagicMock()
        bot.sessions.remove_last_message = MagicMock()
        bot._build_system_prompt = MagicMock(return_value="prompt")
        bot._inject_tool_hints = AsyncMock(return_value="prompt")
        tools = [f"tool_{i}" for i in range(8)]
        bot._process_with_tools = AsyncMock(
            return_value=("error", False, True, tools, False)
        )
        result = await process_web_chat(bot, "hello", "web-test")
        calls = bot.sessions.add_message.call_args_list
        saved_msg = calls[-1][0][2]
        # Should only include first 5 tools
        assert "tool_0" in saved_msg
        assert "tool_4" in saved_msg
        assert "tool_5" not in saved_msg


class TestWebMessageEdgeCases:
    def test_webhook_id_is_none(self):
        msg = WebMessage("ch1", "u1", "TestUser")
        assert msg.webhook_id is None

    def test_attachments_is_empty_list(self):
        msg = WebMessage("ch1", "u1", "TestUser")
        assert msg.attachments == []

    def test_author_bot_is_false(self):
        msg = WebMessage("ch1", "u1", "TestUser")
        assert msg.author.bot is False

    def test_author_name_and_display_name(self):
        msg = WebMessage("ch1", "u1", "Alice")
        assert msg.author.name == "Alice"
        assert msg.author.display_name == "Alice"

    async def test_web_channel_send(self):
        ch = _WebChannel("ch1")
        result = await ch.send("hello")
        assert isinstance(result, _WebSentMessage)

    async def test_web_sent_message_edit_is_noop(self):
        msg = _WebSentMessage()
        await msg.edit(content="new")  # Should not raise

    def test_message_ids_are_sequential(self):
        m1 = WebMessage("ch1", "u1", "User")
        m2 = WebMessage("ch1", "u1", "User")
        assert m2.id > m1.id


# ===================================================================
# WebSocket broadcast edge cases
# ===================================================================


class TestBroadcastEdgeCases:
    async def test_broadcast_skips_when_no_subscribers(self):
        """broadcast_event returns immediately with no subscribers."""
        bot = _make_bot()
        mgr = WebSocketManager(bot, api_token="")
        assert len(mgr._event_subscribers) == 0
        # Should not raise
        await mgr.broadcast_event({"action": "test"})

    async def test_broadcast_removes_dead_client(self):
        bot = _make_bot()
        mgr = WebSocketManager(bot, api_token="")
        dead_ws = MagicMock()
        dead_ws.send_json = AsyncMock(side_effect=ConnectionError("gone"))
        mgr._event_subscribers.add(dead_ws)
        mgr._clients.add(dead_ws)
        await mgr.broadcast_event({"action": "test"})
        assert dead_ws not in mgr._event_subscribers
        assert dead_ws not in mgr._clients

    async def test_broadcast_removes_runtime_error_client(self):
        bot = _make_bot()
        mgr = WebSocketManager(bot, api_token="")
        dead_ws = MagicMock()
        dead_ws.send_json = AsyncMock(side_effect=RuntimeError("closed"))
        mgr._event_subscribers.add(dead_ws)
        mgr._clients.add(dead_ws)
        await mgr.broadcast_event({"action": "test"})
        assert dead_ws not in mgr._event_subscribers


# ===================================================================
# Status endpoint edge cases
# ===================================================================


class TestStatusEndpointEdgeCases:
    async def test_status_with_no_watcher(self):
        app, bot = _make_app()
        bot.infra_watcher = None
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status", headers=_auth())
            data = await resp.json()
            assert data["monitoring"]["enabled"] is False

    async def test_status_with_broken_watcher(self):
        app, bot = _make_app()
        bot.infra_watcher = MagicMock()
        bot.infra_watcher.get_status = MagicMock(side_effect=TypeError("bad"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status", headers=_auth())
            data = await resp.json()
            assert data["monitoring"]["enabled"] is False

    async def test_status_agent_manager_not_dict(self):
        app, bot = _make_app()
        bot.agent_manager._agents = 42  # not a dict
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status", headers=_auth())
            data = await resp.json()
            assert data["agent_count"] == 0

    async def test_status_process_registry_not_dict(self):
        app, bot = _make_app()
        bot.tool_executor._process_registry._processes = "bad"
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status", headers=_auth())
            data = await resp.json()
            assert data["process_count"] == 0

    async def test_status_no_process_registry(self):
        app, bot = _make_app()
        bot.tool_executor._process_registry = None
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status", headers=_auth())
            assert resp.status == 200
            data = await resp.json()
            assert data["process_count"] == 0

    async def test_status_watcher_returns_non_dict(self):
        app, bot = _make_app()
        bot.infra_watcher.get_status = MagicMock(return_value="not a dict")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status", headers=_auth())
            data = await resp.json()
            assert data["monitoring"]["enabled"] is False


# ===================================================================
# Knowledge endpoint edge cases
# ===================================================================


class TestKnowledgeEndpointEdgeCases:
    async def test_search_limit_capped_at_50(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/api/knowledge/search?q=test&limit=100", headers=_auth()
            )
            assert resp.status == 200
            call_kwargs = bot._knowledge_store.search_hybrid.call_args.kwargs
            assert call_kwargs["limit"] == 50

    async def test_ingest_content_too_long(self):
        app, bot = _make_app()
        from src.web.api import _MAX_CONTENT_LEN
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/knowledge",
                json={"source": "doc", "content": "x" * (_MAX_CONTENT_LEN + 1)},
                headers=_auth(),
            )
            assert resp.status == 400

    async def test_search_no_store(self):
        app, bot = _make_app()
        bot._knowledge_store = None
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/search?q=test", headers=_auth())
            assert resp.status == 503

    async def test_ingest_uses_web_api_uploader(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/knowledge",
                json={"source": "new-doc", "content": "some content"},
                headers=_auth(),
            )
            assert resp.status == 201
            call_kwargs = bot._knowledge_store.ingest.call_args.kwargs
            assert call_kwargs["uploader"] == "web-api"


# ===================================================================
# Tools endpoint edge cases
# ===================================================================


class TestToolsEndpointEdgeCases:
    async def test_tool_stats_returns_counts(self):
        app, bot = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/tools/stats", headers=_auth())
            assert resp.status == 200
            data = await resp.json()
            assert data["run_command"] == 42


# ===================================================================
# WebSocket chat error handling
# ===================================================================


class TestWebSocketChatErrors:
    async def test_ws_chat_exception(self):
        """WebSocket chat gracefully handles process_web_chat exceptions."""
        bot = _make_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot, api_token="")
        # Make process_web_chat raise
        with patch("src.web.websocket.process_web_chat", side_effect=RuntimeError("boom")):
            async with TestClient(TestServer(app)) as client:
                ws = await client.ws_connect("/api/ws")
                await ws.send_json({
                    "type": "chat",
                    "content": "hello",
                })
                resp = await ws.receive_json()
                assert resp["type"] == "chat_error"
                assert "boom" in resp["error"]
                await ws.close()

    async def test_ws_chat_content_whitespace_only(self):
        """Whitespace-only content is rejected."""
        bot = _make_bot()
        app = web.Application()
        setup_websocket(app, bot, api_token="")
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({
                "type": "chat",
                "content": "   ",
            })
            resp = await ws.receive_json()
            assert resp["type"] == "chat_error"
            assert "required" in resp["error"]
            await ws.close()


# ===================================================================
# Session list edge cases
# ===================================================================


class TestSessionListEdgeCases:
    async def test_list_sessions_sorted_by_last_active(self):
        app, bot = _make_app()
        s1 = MagicMock()
        s1.messages = []
        s1.summary = ""
        s1.created_at = 100.0
        s1.last_active = 100.0
        s1.last_user_id = "u1"
        s2 = MagicMock()
        s2.messages = []
        s2.summary = ""
        s2.created_at = 200.0
        s2.last_active = 200.0
        s2.last_user_id = "u2"
        bot.sessions._sessions = {"old": s1, "new": s2}
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions", headers=_auth())
            data = await resp.json()
            assert data[0]["channel_id"] == "new"
            assert data[1]["channel_id"] == "old"

    async def test_list_empty_sessions(self):
        app, bot = _make_app()
        bot.sessions._sessions = {}
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions", headers=_auth())
            data = await resp.json()
            assert data == []

    async def test_session_web_source_detection(self):
        app, bot = _make_app()
        s = MagicMock()
        s.messages = []
        s.summary = ""
        s.created_at = 100.0
        s.last_active = 200.0
        s.last_user_id = "u1"
        bot.sessions._sessions = {"web-default": s}
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions", headers=_auth())
            data = await resp.json()
            assert data[0]["source"] == "web"

    async def test_session_discord_source_detection(self):
        app, bot = _make_app()
        s = MagicMock()
        s.messages = []
        s.summary = ""
        s.created_at = 100.0
        s.last_active = 200.0
        s.last_user_id = "u1"
        bot.sessions._sessions = {"123456789": s}
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions", headers=_auth())
            data = await resp.json()
            assert data[0]["source"] == "discord"
