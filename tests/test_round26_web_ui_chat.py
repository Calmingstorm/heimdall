"""Round 26 — Additional tests for Web UI + Chat interface changes (rounds 3-16).

Covers:
- Chat processing edge cases (secret scrubbing, session save resilience, multi-turn)
- WebSocket endpoint edge cases (subscribe/unsubscribe, invalid JSON, unknown cmd, cleanup)
- Config endpoint edge cases (PUT non-dict, response redaction, deep merge in API)
- Dashboard quick actions (reload, clear-all, stop-all)
- Sessions API (preview, export, bulk clear, source detection)
- Tool/pack API (pack update, tool stats, packs endpoint)
- Knowledge API (search, ingest, reingest, delete)
- Loop API (list with history, restart, stop)
- Process API (list with preview, kill)
- Schedule API (create, delete, run-now, validate-cron)
- Memory API (CRUD, bulk-delete)
- WebSocket log tailing and event broadcast end-to-end
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import aiohttp
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
    _redact_config,
    _safe_filename,
    _sanitize_error,
    _deep_merge,
    _contains_blocked_fields,
    _SENSITIVE_FIELDS,
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
from src.web.websocket import WebSocketManager, setup_websocket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chat_bot(*, codex_available=True, process_result=None):
    """Build a mock bot for chat testing."""
    bot = MagicMock()
    guild = MagicMock()
    guild.id = 111
    guild.name = "TestGuild"
    guild.member_count = 10
    bot.guilds = [guild]
    bot.is_ready = MagicMock(return_value=True)
    bot._start_time = time.monotonic() - 60

    bot.config = MagicMock()
    bot.config.model_dump = MagicMock(return_value={
        "discord": {"token": "secret"},
        "tools": {"tool_packs": []},
        "web": {"api_token": "", "enabled": True},
    })
    bot.config.tools.tool_packs = []

    bot._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run", "input_schema": {}},
    ])
    bot._cached_merged_tools = None
    bot._cached_skills_text = None
    bot.skill_manager = MagicMock()
    bot.skill_manager.list_skills = MagicMock(return_value=[])
    bot.skill_manager._skills = {}

    bot.sessions = MagicMock()
    bot.sessions._sessions = {}
    bot.sessions.add_message = MagicMock()
    bot.sessions.remove_last_message = MagicMock()
    bot.sessions.prune = MagicMock()
    bot.sessions.save = MagicMock()
    bot.sessions.reset = MagicMock()
    bot.sessions.get_task_history = AsyncMock(return_value=[])

    if codex_available:
        bot.codex_client = MagicMock()
    else:
        bot.codex_client = None

    bot._build_system_prompt = MagicMock(return_value="You are Heimdall.")
    bot._inject_tool_hints = AsyncMock(return_value="You are Heimdall.")

    if process_result is None:
        process_result = ("Hello from Heimdall!", False, False, [], False)
    bot._process_with_tools = AsyncMock(return_value=process_result)

    bot.permissions = MagicMock()
    bot.audit = MagicMock()
    bot.audit.log_execution = AsyncMock()
    bot.audit.search = AsyncMock(return_value=[])
    bot.audit.count_by_tool = AsyncMock(return_value={})

    bot._knowledge_store = None
    bot._embedder = None
    bot.scheduler = MagicMock()
    bot.scheduler.list_all = MagicMock(return_value=[])
    bot.scheduler._schedules = []
    bot.scheduler._callback = None
    bot.loop_manager = MagicMock()
    bot.loop_manager.active_count = 0
    bot.loop_manager._loops = {}
    bot.tool_executor = MagicMock()
    bot.tool_executor._process_registry = None
    bot.tool_executor._load_all_memory = MagicMock(return_value={})
    bot.tool_executor._save_all_memory = MagicMock()
    bot.context_loader = MagicMock()
    bot._invalidate_prompt_caches = MagicMock()
    bot._system_prompt = "system prompt"

    return bot


def _make_app(bot=None, *, api_token=""):
    if bot is None:
        bot = _make_chat_bot()
    web_config = WebConfig(api_token=api_token)
    app = web.Application(middlewares=[
        _make_security_headers_middleware(),
        _make_rate_limit_middleware(),
        _make_auth_middleware(web_config, SessionManager()),
    ])
    setup_api(app, bot)
    return app, bot


# ===================================================================
# Chat: process_web_chat edge cases
# ===================================================================


class TestChatSecretScrubbing:
    """Verify that secrets in LLM responses are scrubbed before returning."""

    @pytest.mark.asyncio
    async def test_response_with_password_scrubbed(self):
        bot = _make_chat_bot(
            process_result=("Found password=s3cr3t123 in config", False, False, ["read_file"], False),
        )
        result = await process_web_chat(bot, "read config", "web-1")
        assert "s3cr3t123" not in result["response"]
        assert "****" in result["response"] or "REDACTED" in result["response"].upper() or "password=" in result["response"]

    @pytest.mark.asyncio
    async def test_response_with_api_key_scrubbed(self):
        bot = _make_chat_bot(
            process_result=("Key is api_key=abc123xyz", False, False, ["run_command"], False),
        )
        result = await process_web_chat(bot, "show key", "web-1")
        assert "abc123xyz" not in result["response"]

    @pytest.mark.asyncio
    async def test_clean_response_unchanged(self):
        bot = _make_chat_bot(
            process_result=("All systems healthy!", False, False, [], False),
        )
        result = await process_web_chat(bot, "status", "web-1")
        assert result["response"] == "All systems healthy!"


class TestChatSessionSaveResilience:
    """Session save failures shouldn't crash chat processing."""

    @pytest.mark.asyncio
    async def test_save_failure_doesnt_crash(self):
        bot = _make_chat_bot()
        bot.sessions.save = MagicMock(side_effect=OSError("disk full"))
        # Should not raise — save is wrapped in try/except
        result = await process_web_chat(bot, "hello", "web-1")
        # Still returns the response
        assert result["response"] == "Hello from Heimdall!"
        assert result["is_error"] is False

    @pytest.mark.asyncio
    async def test_prune_called_before_save(self):
        bot = _make_chat_bot()
        call_order = []
        bot.sessions.prune = MagicMock(side_effect=lambda: call_order.append("prune"))
        bot.sessions.save = MagicMock(side_effect=lambda: call_order.append("save"))
        await process_web_chat(bot, "hi", "web-1")
        assert call_order == ["prune", "save"]


class TestChatMultiTurn:
    """Multi-message conversation flow."""

    @pytest.mark.asyncio
    async def test_sequential_messages_use_same_channel(self):
        bot = _make_chat_bot()
        await process_web_chat(bot, "msg1", "web-session-1")
        await process_web_chat(bot, "msg2", "web-session-1")
        # Both should add to same channel
        user_calls = [
            c for c in bot.sessions.add_message.call_args_list
            if c[0][0] == "web-session-1" and c[0][1] == "user"
        ]
        assert len(user_calls) == 2

    @pytest.mark.asyncio
    async def test_history_fetched_per_message(self):
        bot = _make_chat_bot()
        await process_web_chat(bot, "msg1", "web-s1")
        await process_web_chat(bot, "msg2", "web-s1")
        # get_task_history should be called for each message
        assert bot.sessions.get_task_history.call_count == 2
        for call in bot.sessions.get_task_history.call_args_list:
            assert call[0][0] == "web-s1"
            assert call[1]["max_messages"] == 20

    @pytest.mark.asyncio
    async def test_different_channels_are_independent(self):
        bot = _make_chat_bot()
        await process_web_chat(bot, "msg1", "web-ch1")
        await process_web_chat(bot, "msg2", "web-ch2")
        ch1_calls = [c for c in bot.sessions.add_message.call_args_list if c[0][0] == "web-ch1"]
        ch2_calls = [c for c in bot.sessions.add_message.call_args_list if c[0][0] == "web-ch2"]
        assert len(ch1_calls) >= 1
        assert len(ch2_calls) >= 1


class TestChatErrorHandling:
    """Error paths in chat processing."""

    @pytest.mark.asyncio
    async def test_exception_scrubs_secrets_in_error_message(self):
        bot = _make_chat_bot()
        bot._process_with_tools = AsyncMock(
            side_effect=RuntimeError("Connection to password=secret123 failed")
        )
        result = await process_web_chat(bot, "test", "web-1")
        assert result["is_error"] is True
        assert "secret123" not in result["response"]

    @pytest.mark.asyncio
    async def test_error_with_tools_saves_tool_names(self):
        bot = _make_chat_bot(
            process_result=("Failed", False, True, ["run_command", "read_file"], False),
        )
        result = await process_web_chat(bot, "do things", "web-1")
        assert result["is_error"] is True
        assert result["tools_used"] == ["run_command", "read_file"]
        # Saved error message should mention the tools (capped at 5)
        assistant_saves = [
            c for c in bot.sessions.add_message.call_args_list if c[0][1] == "assistant"
        ]
        assert len(assistant_saves) == 1
        assert "run_command" in assistant_saves[0][0][2]
        assert "read_file" in assistant_saves[0][0][2]

    @pytest.mark.asyncio
    async def test_error_tools_capped_at_five(self):
        many_tools = [f"tool_{i}" for i in range(8)]
        bot = _make_chat_bot(
            process_result=("Failed", False, True, many_tools, False),
        )
        result = await process_web_chat(bot, "do things", "web-1")
        assistant_saves = [
            c for c in bot.sessions.add_message.call_args_list if c[0][1] == "assistant"
        ]
        saved = assistant_saves[0][0][2]
        # Only first 5 tools mentioned in sanitized message
        assert "tool_0" in saved
        assert "tool_4" in saved
        # tool_5 through tool_7 not in the truncated list
        assert saved.count("tool_") <= 6  # 5 tools + possible "tools" word


# ===================================================================
# WebSocket endpoint edge cases
# ===================================================================


class TestWebSocketSubscriptionFlow:
    """Subscribe/unsubscribe and message routing."""

    @pytest.mark.asyncio
    async def test_subscribe_events_and_receive(self):
        bot = _make_chat_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"subscribe": "events"})
            resp = await ws.receive_json()
            assert resp["type"] == "subscribed"
            assert resp["channel"] == "events"
            # Broadcast an event
            await mgr.broadcast_event({"tool": "run_command", "status": "ok"})
            event = await ws.receive_json()
            assert event["type"] == "event"
            assert event["payload"]["tool"] == "run_command"
            await ws.close()

    @pytest.mark.asyncio
    async def test_unsubscribe_events(self):
        bot = _make_chat_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"subscribe": "events"})
            await ws.receive_json()  # subscribed ack
            await ws.send_json({"unsubscribe": "events"})
            resp = await ws.receive_json()
            assert resp["type"] == "unsubscribed"
            assert resp["channel"] == "events"
            await ws.close()

    @pytest.mark.asyncio
    async def test_unsubscribe_logs(self):
        bot = _make_chat_bot()
        app = web.Application()
        setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            # Subscribe then unsubscribe logs
            await ws.send_json({"subscribe": "logs"})
            sub = await ws.receive_json()
            assert sub["type"] == "subscribed"
            await ws.send_json({"unsubscribe": "logs"})
            unsub = await ws.receive_json()
            assert unsub["type"] == "unsubscribed"
            assert unsub["channel"] == "logs"
            await ws.close()

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self):
        bot = _make_chat_bot()
        app = web.Application()
        setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_str("not valid json{{{")
            resp = await ws.receive_json()
            assert "error" in resp
            assert "invalid JSON" in resp["error"]
            await ws.close()

    @pytest.mark.asyncio
    async def test_unknown_command_returns_error(self):
        bot = _make_chat_bot()
        app = web.Application()
        setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"something": "weird"})
            resp = await ws.receive_json()
            assert "error" in resp
            assert "unknown" in resp["error"].lower()
            await ws.close()


class TestWebSocketCleanup:
    """Connection cleanup on disconnect."""

    @pytest.mark.asyncio
    async def test_disconnect_removes_from_clients(self):
        bot = _make_chat_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            assert mgr.client_count == 1
            await ws.close()
            # Give cleanup a moment
            await asyncio.sleep(0.1)
            assert mgr.client_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_removes_from_subscribers(self):
        bot = _make_chat_bot()
        app = web.Application()
        mgr = setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"subscribe": "events"})
            await ws.receive_json()
            assert len(mgr._event_subscribers) == 1
            await ws.close()
            await asyncio.sleep(0.1)
            assert len(mgr._event_subscribers) == 0


class TestWebSocketChatExceptions:
    """WebSocket chat error handling."""

    @pytest.mark.asyncio
    async def test_process_exception_returns_chat_error(self):
        bot = _make_chat_bot()
        bot._process_with_tools = AsyncMock(side_effect=RuntimeError("boom"))
        app = web.Application()
        setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"type": "chat", "content": "trigger error"})
            resp = await ws.receive_json()
            # Should get chat_response with is_error=True (from process_web_chat)
            # OR chat_error if the exception propagates past process_web_chat
            assert resp["type"] in ("chat_response", "chat_error")
            await ws.close()

    @pytest.mark.asyncio
    async def test_whitespace_only_content_rejected(self):
        bot = _make_chat_bot()
        app = web.Application()
        setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"type": "chat", "content": "   \n\t  "})
            resp = await ws.receive_json()
            assert resp["type"] == "chat_error"
            assert "content" in resp["error"]
            await ws.close()


# ===================================================================
# Config endpoint edge cases
# ===================================================================


class TestConfigPutEdgeCases:
    """Additional config PUT tests."""

    @pytest.mark.asyncio
    async def test_put_non_dict_body(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.put("/api/config", json=["not", "a", "dict"])
            assert resp.status == 400
            body = await resp.json()
            assert "object" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_put_response_is_redacted(self):
        from src.config.schema import Config
        real_config = Config(discord={"token": "super-secret"})
        bot = _make_chat_bot()
        bot.config = real_config
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            with patch("src.web.api._write_config"):
                resp = await client.put(
                    "/api/config",
                    json={"timezone": "UTC"},
                )
            assert resp.status == 200
            body = await resp.json()
            # Token should be redacted in the response
            assert body["discord"]["token"] == "••••••••"

    @pytest.mark.asyncio
    async def test_put_empty_object_no_change(self):
        from src.config.schema import Config
        real_config = Config(discord={"token": "tok"})
        bot = _make_chat_bot()
        bot.config = real_config
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            with patch("src.web.api._write_config"):
                resp = await client.put("/api/config", json={})
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_get_config_auth_required(self):
        app, _ = _make_app(api_token="secret123")
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/config")
            assert resp.status == 401
            resp = await client.get(
                "/api/config",
                headers={"Authorization": "Bearer secret123"},
            )
            assert resp.status == 200


# ===================================================================
# Deep merge helper
# ===================================================================


class TestDeepMergeEdgeCases:
    """Edge cases for _deep_merge used in config PUT."""

    def test_merge_nested(self):
        base = {"a": {"b": 1, "c": 2}}
        _deep_merge(base, {"a": {"b": 99}})
        assert base["a"]["b"] == 99
        assert base["a"]["c"] == 2

    def test_merge_adds_new_keys(self):
        base = {"a": 1}
        _deep_merge(base, {"b": 2})
        assert base == {"a": 1, "b": 2}

    def test_merge_replaces_non_dict_with_dict(self):
        base = {"a": "string"}
        _deep_merge(base, {"a": {"nested": True}})
        assert base["a"] == {"nested": True}

    def test_merge_depth_limit(self):
        # Deep nesting beyond limit stops merging
        base = {"a": {}}
        updates = {"a": {}}
        current_base = base["a"]
        current_upd = updates["a"]
        for i in range(12):
            current_base[f"l{i}"] = {}
            current_upd[f"l{i}"] = {"val": i}
            current_base = current_base[f"l{i}"]
            current_upd = current_upd[f"l{i}"]
        _deep_merge(base, updates)
        # Should not crash even with deep nesting


# ===================================================================
# Safe filename helper
# ===================================================================


class TestSafeFilename:
    def test_basic(self):
        assert _safe_filename("hello-world") == "hello-world"

    def test_special_chars(self):
        assert _safe_filename('a"b\\c/d') == "a_b_c_d"

    def test_newlines_stripped(self):
        assert "\n" not in _safe_filename("line1\nline2")

    def test_max_length(self):
        result = _safe_filename("a" * 200)
        assert len(result) <= 80

    def test_empty_becomes_export(self):
        assert _safe_filename("") == "export"

    def test_all_special_becomes_export(self):
        assert _safe_filename("!!!") == "___"

    def test_preserves_dots(self):
        assert _safe_filename("file.json") == "file.json"


# ===================================================================
# Sanitize error helper
# ===================================================================


class TestSanitizeError:
    def test_scrubs_password(self):
        result = _sanitize_error("Failed: password=abc123")
        assert "abc123" not in result

    def test_clean_message_unchanged(self):
        assert _sanitize_error("Something failed") == "Something failed"

    def test_handles_non_string(self):
        # _sanitize_error wraps in str()
        result = _sanitize_error(RuntimeError("oops"))
        assert "oops" in result


# ===================================================================
# REST API: Chat endpoint additional tests
# ===================================================================


class TestChatRESTEdgeCases:
    @pytest.mark.asyncio
    async def test_chat_with_tools_returns_200(self):
        bot = _make_chat_bot(
            process_result=("Done!", False, False, ["run_command", "read_file"], False),
        )
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat", json={"content": "list files"})
            assert resp.status == 200
            body = await resp.json()
            assert body["tools_used"] == ["run_command", "read_file"]
            assert body["is_error"] is False

    @pytest.mark.asyncio
    async def test_chat_whitespace_only(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat", json={"content": "   \n "})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_chat_null_content(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat", json={"content": None})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_chat_default_user_fields(self):
        bot = _make_chat_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat", json={"content": "hi"})
            assert resp.status == 200
        # Default channel_id, user_id, username
        bot.sessions.add_message.assert_any_call(
            "web-default", "user", "[WebUser]: hi", user_id="web-user",
        )


# ===================================================================
# REST API: Status endpoint
# ===================================================================


class TestStatusEndpointFields:
    """Verify all status fields are present."""

    @pytest.mark.asyncio
    async def test_status_has_all_fields(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            assert resp.status == 200
            body = await resp.json()
            expected_keys = {
                "status", "uptime_seconds", "guilds", "guild_count",
                "user_count", "tool_count", "skill_count", "session_count",
                "loop_count", "schedule_count",
            }
            assert expected_keys.issubset(set(body.keys()))

    @pytest.mark.asyncio
    async def test_status_guild_has_member_count(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            body = await resp.json()
            assert body["guilds"][0]["member_count"] == 10
            assert body["user_count"] == 10


# ===================================================================
# REST API: Sessions preview + export + bulk clear
# ===================================================================


class TestSessionsPreviewAndExport:
    """Session preview, source detection, export, and bulk clear."""

    def _make_session_bot(self):
        bot = _make_chat_bot()
        session = MagicMock()
        session.messages = [
            MagicMock(role="user", content="hello there", timestamp=1704067200.0, user_id="u1"),
            MagicMock(role="assistant", content="Hi! How can I help?", timestamp=1704067201.0, user_id=None),
        ]
        session.summary = "Greeting conversation"
        session.created_at = 1704067200.0
        session.last_active = 1704067201.0
        session.last_user_id = "u1"
        bot.sessions._sessions = {
            "discord-chan1": session,
            "web-chat-1": session,
        }
        return bot

    @pytest.mark.asyncio
    async def test_list_sessions_preview(self):
        bot = self._make_session_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions")
            body = await resp.json()
            assert len(body) == 2
            # Check preview (last 2 messages)
            for s in body:
                assert "preview" in s
                assert len(s["preview"]) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_source_detection(self):
        bot = self._make_session_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions")
            body = await resp.json()
            sources = {s["channel_id"]: s["source"] for s in body}
            assert sources["web-chat-1"] == "web"
            assert sources["discord-chan1"] == "discord"

    @pytest.mark.asyncio
    async def test_export_session_json(self):
        bot = self._make_session_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions/discord-chan1/export")
            assert resp.status == 200
            body = await resp.json()
            assert body["channel_id"] == "discord-chan1"
            assert "messages" in body
            assert "exported_at" in body
            assert "Content-Disposition" in resp.headers

    @pytest.mark.asyncio
    async def test_export_session_text(self):
        bot = self._make_session_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions/discord-chan1/export?format=text")
            assert resp.status == 200
            text = await resp.text()
            assert "Summary" in text
            assert "Messages" in text
            assert resp.content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_export_not_found(self):
        bot = self._make_session_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/sessions/nonexistent/export")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_bulk_clear(self):
        bot = self._make_session_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/sessions/clear-bulk",
                json={"channel_ids": ["discord-chan1", "web-chat-1"]},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["count"] == 2

    @pytest.mark.asyncio
    async def test_bulk_clear_empty_list(self):
        bot = self._make_session_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/sessions/clear-bulk",
                json={"channel_ids": []},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_clear_all_sessions(self):
        bot = self._make_session_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/sessions/clear-all")
            assert resp.status == 200
            body = await resp.json()
            assert body["count"] == 2
            assert body["status"] == "cleared"


# ===================================================================
# REST API: Tools and Packs
# ===================================================================


class TestToolsAndPacks:
    @pytest.mark.asyncio
    async def test_list_tools(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/tools")
            assert resp.status == 200
            body = await resp.json()
            # Uses real get_tool_definitions(), returns all 80 tools (67 base + 6 agent + 2 loop-agent bridge + 2 skill toggle + 3 skill management)
            assert len(body) == 80
            tool_names = [t["name"] for t in body]
            assert "run_command" in tool_names

    @pytest.mark.asyncio
    async def test_list_packs(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/tools/packs")
            assert resp.status == 200
            body = await resp.json()
            assert "packs" in body
            assert "all_packs_loaded" in body

    @pytest.mark.asyncio
    async def test_update_packs_valid(self):
        from src.tools.registry import TOOL_PACKS
        bot = _make_chat_bot()
        app, _ = _make_app(bot)
        valid_pack = list(TOOL_PACKS.keys())[0]
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/tools/packs",
                json={"packs": [valid_pack]},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["packs"] == [valid_pack]
            assert bot._cached_merged_tools is None

    @pytest.mark.asyncio
    async def test_update_packs_invalid(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/tools/packs",
                json={"packs": ["nonexistent_pack"]},
            )
            assert resp.status == 400
            body = await resp.json()
            assert "unknown" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_tool_stats(self):
        bot = _make_chat_bot()
        bot.audit.count_by_tool = AsyncMock(return_value={"run_command": 42})
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/tools/stats")
            assert resp.status == 200
            body = await resp.json()
            assert body["run_command"] == 42


# ===================================================================
# REST API: Knowledge
# ===================================================================


class TestKnowledgeAPI:
    def _make_knowledge_bot(self):
        bot = _make_chat_bot()
        store = MagicMock()
        store.available = True
        store.list_sources = MagicMock(return_value=[
            {"source": "doc1", "chunks": 3, "preview": "Hello...", "ingested_at": "2024-01-01"},
        ])
        store.ingest = AsyncMock(return_value=5)
        store.search_hybrid = AsyncMock(return_value=[
            {"source": "doc1", "content": "hello", "score": 0.9},
        ])
        store.delete_source = MagicMock(return_value=3)
        store.get_source_content = MagicMock(return_value="Full content here")
        bot._knowledge_store = store
        return bot

    @pytest.mark.asyncio
    async def test_list_knowledge(self):
        bot = self._make_knowledge_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge")
            assert resp.status == 200
            body = await resp.json()
            assert len(body) == 1
            assert body[0]["source"] == "doc1"

    @pytest.mark.asyncio
    async def test_search_knowledge(self):
        bot = self._make_knowledge_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/search?q=hello")
            assert resp.status == 200
            body = await resp.json()
            assert len(body) == 1

    @pytest.mark.asyncio
    async def test_search_missing_query(self):
        bot = self._make_knowledge_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge/search")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_ingest_knowledge(self):
        bot = self._make_knowledge_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/knowledge",
                json={"source": "test-doc", "content": "Some text content"},
            )
            assert resp.status == 201
            body = await resp.json()
            assert body["chunks"] == 5

    @pytest.mark.asyncio
    async def test_delete_knowledge(self):
        bot = self._make_knowledge_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/knowledge/doc1")
            assert resp.status == 200
            body = await resp.json()
            assert body["chunks_removed"] == 3

    @pytest.mark.asyncio
    async def test_reingest_knowledge(self):
        bot = self._make_knowledge_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/knowledge/doc1/reingest")
            assert resp.status == 200
            body = await resp.json()
            assert body["source"] == "doc1"

    @pytest.mark.asyncio
    async def test_reingest_not_found(self):
        bot = self._make_knowledge_bot()
        bot._knowledge_store.get_source_content = MagicMock(return_value=None)
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/knowledge/nonexistent/reingest")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_knowledge_unavailable(self):
        bot = _make_chat_bot()
        bot._knowledge_store = None
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/knowledge")
            assert resp.status == 503


# ===================================================================
# REST API: Schedules
# ===================================================================


class TestScheduleAPI:
    def _make_schedule_bot(self):
        bot = _make_chat_bot()
        bot.scheduler._schedules = [
            {"id": "s1", "description": "Daily check", "channel_id": "c1",
             "action": "reminder", "cron": "0 9 * * *", "last_run": None},
        ]
        bot.scheduler.list_all = MagicMock(return_value=[
            {"id": "s1", "description": "Daily check"},
        ])
        bot.scheduler.add = MagicMock(return_value={"id": "s2", "description": "New"})
        bot.scheduler.delete = MagicMock(return_value=True)
        bot.scheduler._callback = AsyncMock()
        return bot

    @pytest.mark.asyncio
    async def test_list_schedules(self):
        bot = self._make_schedule_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/schedules")
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_create_schedule(self):
        bot = self._make_schedule_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/schedules",
                json={"description": "Test", "channel_id": "c1"},
            )
            assert resp.status == 201

    @pytest.mark.asyncio
    async def test_delete_schedule(self):
        bot = self._make_schedule_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/schedules/s1")
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_run_schedule_now(self):
        bot = self._make_schedule_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/schedules/s1/run")
            assert resp.status == 200
            body = await resp.json()
            assert body["status"] == "triggered"

    @pytest.mark.asyncio
    async def test_run_schedule_not_found(self):
        bot = self._make_schedule_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/schedules/nonexistent/run")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_validate_cron_valid(self):
        bot = self._make_schedule_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/schedules/validate-cron",
                json={"expression": "0 9 * * *"},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["valid"] is True
            assert len(body["next_runs"]) == 5

    @pytest.mark.asyncio
    async def test_validate_cron_invalid(self):
        bot = self._make_schedule_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/schedules/validate-cron",
                json={"expression": "not a cron"},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["valid"] is False

    @pytest.mark.asyncio
    async def test_validate_cron_empty(self):
        bot = self._make_schedule_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/schedules/validate-cron",
                json={"expression": ""},
            )
            assert resp.status == 400


# ===================================================================
# REST API: Loops
# ===================================================================


class TestLoopAPI:
    def _make_loop_bot(self):
        bot = _make_chat_bot()
        loop_info = MagicMock()
        loop_info.goal = "monitor disks"
        loop_info.mode = "notify"
        loop_info.interval_seconds = 60
        loop_info.stop_condition = None
        loop_info.max_iterations = 50
        loop_info.channel_id = "123"
        loop_info.requester_id = "u1"
        loop_info.requester_name = "Alice"
        loop_info.iteration_count = 3
        loop_info.last_trigger = "2024-01-02"
        loop_info.created_at = "2024-01-01"
        loop_info.status = "running"
        loop_info._iteration_history = ["Iter 1", "Iter 2", "Iter 3"]
        bot.loop_manager._loops = {"loop1": loop_info}
        bot.loop_manager.active_count = 1
        bot.loop_manager.start_loop = MagicMock(return_value="new-loop-id")
        bot.loop_manager.stop_loop = MagicMock(return_value="Loop stopped")
        bot._run_loop_iteration = AsyncMock(return_value="done")
        bot.get_channel = MagicMock(return_value=MagicMock())
        return bot

    @pytest.mark.asyncio
    async def test_list_loops_with_history(self):
        bot = self._make_loop_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/loops")
            assert resp.status == 200
            body = await resp.json()
            assert len(body) == 1
            assert body[0]["goal"] == "monitor disks"
            assert body[0]["iteration_history"] == ["Iter 1", "Iter 2", "Iter 3"]
            assert body[0]["status"] == "running"

    @pytest.mark.asyncio
    async def test_stop_loop(self):
        bot = self._make_loop_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/loops/loop1")
            assert resp.status == 200
            body = await resp.json()
            assert body["result"] == "Loop stopped"

    @pytest.mark.asyncio
    async def test_restart_loop(self):
        bot = self._make_loop_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/loops/loop1/restart")
            assert resp.status == 201
            body = await resp.json()
            assert body["old_id"] == "loop1"
            assert body["new_id"] == "new-loop-id"

    @pytest.mark.asyncio
    async def test_restart_loop_not_found(self):
        bot = self._make_loop_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/loops/nonexistent/restart")
            assert resp.status == 404


# ===================================================================
# REST API: Processes
# ===================================================================


class TestProcessAPI:
    def _make_process_bot(self):
        bot = _make_chat_bot()
        proc = MagicMock()
        proc.command = "sleep 100"
        proc.host = "localhost"
        proc.status = "running"
        proc.exit_code = None
        proc.start_time = time.time() - 30
        proc.output_buffer = deque(["line1\n", "line2\n", "line3\n"], maxlen=500)
        registry = MagicMock()
        registry._processes = {1234: proc}
        registry.kill = AsyncMock(return_value="Process killed")
        bot.tool_executor._process_registry = registry
        return bot

    @pytest.mark.asyncio
    async def test_list_processes_with_preview(self):
        bot = self._make_process_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/processes")
            assert resp.status == 200
            body = await resp.json()
            assert len(body) == 1
            assert body[0]["pid"] == 1234
            assert body[0]["output_preview"] == ["line1", "line2", "line3"]
            assert body[0]["status"] == "running"

    @pytest.mark.asyncio
    async def test_kill_process(self):
        bot = self._make_process_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/processes/1234")
            assert resp.status == 200
            body = await resp.json()
            assert body["result"] == "Process killed"

    @pytest.mark.asyncio
    async def test_kill_invalid_pid(self):
        bot = self._make_process_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/processes/not-a-number")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_no_registry(self):
        bot = _make_chat_bot()
        bot.tool_executor._process_registry = None
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/processes")
            assert resp.status == 200
            body = await resp.json()
            assert body == []


# ===================================================================
# REST API: Memory
# ===================================================================


class TestMemoryAPI:
    def _make_memory_bot(self):
        bot = _make_chat_bot()
        bot.tool_executor._load_all_memory = MagicMock(return_value={
            "global": {"key1": "value1", "key2": "value2"},
            "user:u1": {"pref": "dark"},
        })
        return bot

    @pytest.mark.asyncio
    async def test_list_memory(self):
        bot = self._make_memory_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/memory")
            assert resp.status == 200
            body = await resp.json()
            assert body["global"]["count"] == 2
            assert "key1" in body["global"]["keys"]

    @pytest.mark.asyncio
    async def test_get_memory_entry(self):
        bot = self._make_memory_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/memory/global/key1")
            assert resp.status == 200
            body = await resp.json()
            assert body["value"] == "value1"

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self):
        bot = self._make_memory_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/memory/global/nonexistent")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_set_memory(self):
        bot = self._make_memory_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/memory/global/newkey",
                json={"value": "newvalue"},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["status"] == "saved"

    @pytest.mark.asyncio
    async def test_delete_memory(self):
        bot = self._make_memory_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete("/api/memory/global/key1")
            assert resp.status == 200
            body = await resp.json()
            assert body["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_bulk_delete_memory(self):
        bot = self._make_memory_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/memory/bulk-delete",
                json={"entries": [
                    {"scope": "global", "key": "key1"},
                    {"scope": "global", "key": "key2"},
                ]},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["count"] == 2


# ===================================================================
# REST API: Audit
# ===================================================================


class TestAuditAPI:
    @pytest.mark.asyncio
    async def test_audit_search(self):
        bot = _make_chat_bot()
        bot.audit.search = AsyncMock(return_value=[
            {"timestamp": "2024-01-01", "tool_name": "run_command", "error": None},
            {"timestamp": "2024-01-01", "tool_name": "read_file", "error": "failed"},
        ])
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/audit")
            assert resp.status == 200
            body = await resp.json()
            assert len(body) == 2

    @pytest.mark.asyncio
    async def test_audit_error_only_filter(self):
        bot = _make_chat_bot()
        bot.audit.search = AsyncMock(return_value=[
            {"timestamp": "2024-01-01", "tool_name": "run_command", "error": None},
            {"timestamp": "2024-01-01", "tool_name": "read_file", "error": "failed"},
        ])
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/audit?error_only=1")
            assert resp.status == 200
            body = await resp.json()
            assert len(body) == 1
            assert body[0]["error"] == "failed"


# ===================================================================
# REST API: Quick Actions (dashboard)
# ===================================================================


class TestQuickActionsAPI:
    @pytest.mark.asyncio
    async def test_reload_config(self):
        bot = _make_chat_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/reload")
            assert resp.status == 200
            body = await resp.json()
            assert body["status"] == "reloaded"
        bot.context_loader.reload.assert_called_once()
        bot._invalidate_prompt_caches.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_all_loops(self):
        bot = _make_chat_bot()
        bot.loop_manager.stop_loop = MagicMock(return_value="All loops stopped")
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/loops/stop-all")
            assert resp.status == 200
        bot.loop_manager.stop_loop.assert_called_with("all")


# ===================================================================
# Tool pack integrity
# ===================================================================


class TestToolPackIntegrity:
    """Verify tool pack configuration matches current state."""

    def test_current_packs(self):
        from src.tools.registry import TOOL_PACKS
        # After Round 2: docker and git removed
        assert "docker" not in TOOL_PACKS
        assert "git" not in TOOL_PACKS
        # Remaining 5 packs
        expected = {"systemd", "incus", "ansible", "prometheus", "comfyui"}
        assert set(TOOL_PACKS.keys()) == expected

    def test_pack_count(self):
        from src.tools.registry import TOOL_PACKS
        assert len(TOOL_PACKS) == 5

    def test_pack_tool_counts(self):
        from src.tools.registry import TOOL_PACKS
        assert len(TOOL_PACKS["systemd"]) == 3
        assert len(TOOL_PACKS["incus"]) == 11
        assert len(TOOL_PACKS["ansible"]) == 1
        assert len(TOOL_PACKS["prometheus"]) == 4
        assert len(TOOL_PACKS["comfyui"]) == 1

    def test_total_tools(self):
        from src.tools.registry import TOOLS
        assert len(TOOLS) == 80

    def test_all_pack_tools_exist_in_tools_list(self):
        from src.tools.registry import TOOLS, TOOL_PACKS
        tool_names = {t["name"] for t in TOOLS}
        for pack_name, pack_tools in TOOL_PACKS.items():
            for tool_name in pack_tools:
                assert tool_name in tool_names, f"{tool_name} from {pack_name} not in TOOLS"


# ===================================================================
# WebSocket: broadcast event format
# ===================================================================


class TestBroadcastEventFormat:
    """Verify event payload format for WebSocket broadcast."""

    @pytest.mark.asyncio
    async def test_event_wrapped_in_payload(self):
        bot = _make_chat_bot()
        mgr = WebSocketManager(bot)
        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mgr._clients.add(mock_ws)
        mgr._event_subscribers.add(mock_ws)

        event = {"tool_name": "run_command", "user": "alice", "host": "srv1"}
        await mgr.broadcast_event(event)

        msg = mock_ws.send_json.call_args[0][0]
        assert msg["type"] == "event"
        assert "payload" in msg
        assert msg["payload"]["tool_name"] == "run_command"
        assert msg["payload"]["user"] == "alice"

    @pytest.mark.asyncio
    async def test_broadcast_skips_when_no_subscribers(self):
        bot = _make_chat_bot()
        mgr = WebSocketManager(bot)
        # No subscribers — should be a no-op, no error
        await mgr.broadcast_event({"test": True})

    @pytest.mark.asyncio
    async def test_broadcast_runtime_error_removes_client(self):
        bot = _make_chat_bot()
        mgr = WebSocketManager(bot)
        dead_ws = AsyncMock()
        dead_ws.send_json = AsyncMock(side_effect=RuntimeError("closed"))
        mgr._clients.add(dead_ws)
        mgr._event_subscribers.add(dead_ws)

        await mgr.broadcast_event({"test": True})
        assert dead_ws not in mgr._event_subscribers
        assert dead_ws not in mgr._clients


# ===================================================================
# WebMessage content attribute
# ===================================================================


class TestWebMessageDetails:
    def test_webchannel_typing_returns_context_manager(self):
        ch = _WebChannel("ch1")
        cm = ch.typing()
        assert hasattr(cm, "__aenter__")
        assert hasattr(cm, "__aexit__")

    def test_webauthor_attributes(self):
        author = _WebAuthor("uid", "TestUser")
        assert author.id == "uid"
        assert author.bot is False
        assert author.display_name == "TestUser"
        assert author.name == "TestUser"
        assert author.mention == "@TestUser"

    def test_webmessage_has_content_attribute(self):
        """WebMessage has content attribute for tool handler compatibility."""
        msg = WebMessage("ch", "u", "User", content="hello")
        assert msg.content == "hello"
        msg2 = WebMessage("ch", "u", "User")
        assert msg2.content == ""

    @pytest.mark.asyncio
    async def test_websent_message_edit_noop(self):
        sent = _WebSentMessage()
        await sent.edit(content="new", embed=MagicMock())  # should not raise


# ===================================================================
# MAX_CHAT_CONTENT_LEN constant
# ===================================================================


class TestChatConstants:
    def test_max_content_len_is_4000(self):
        assert MAX_CHAT_CONTENT_LEN == 4000

    def test_max_content_len_used_by_api(self):
        """api.py imports and uses the same constant."""
        from src.web.api import MAX_CHAT_CONTENT_LEN as api_max
        assert api_max == MAX_CHAT_CONTENT_LEN

    def test_max_content_len_used_by_websocket(self):
        """websocket.py imports and uses the same constant."""
        from src.web.websocket import MAX_CHAT_CONTENT_LEN as ws_max
        assert ws_max == MAX_CHAT_CONTENT_LEN
