"""Tests for web chat backend (src/web/chat.py, WebSocket chat, REST /api/chat).

Covers: WebMessage virtual message, process_web_chat(), POST /api/chat endpoint,
and WebSocket chat message handling.
"""
from __future__ import annotations

import asyncio
import json
import time
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
from src.web.api import create_api_routes, setup_api
from src.web.chat import (
    MAX_CHAT_CONTENT_LEN,
    WebMessage,
    _NoOpContextManager,
    _WebAuthor,
    _WebChannel,
    _WebSentMessage,
    process_web_chat,
)
from src.web.websocket import WebSocketManager, setup_websocket


# ---------------------------------------------------------------------------
# Helper: mock bot for chat tests
# ---------------------------------------------------------------------------


def _make_chat_bot(*, codex_available=True, process_result=None):
    """Build a mock bot suitable for chat endpoint testing."""
    bot = MagicMock()

    # Discord attributes (minimal for API routes that aren't chat-specific)
    guild = MagicMock()
    guild.id = 111
    guild.name = "TestGuild"
    guild.member_count = 10
    bot.guilds = [guild]
    bot.is_ready = MagicMock(return_value=True)
    bot._start_time = time.monotonic() - 60

    # Config
    bot.config = MagicMock()
    bot.config.model_dump = MagicMock(return_value={
        "discord": {"token": "secret"},
        "tools": {},
        "web": {"api_token": "", "enabled": True},
    })
    bot.config.tools.enabled = True
    bot.config.tools.tool_timeout_seconds = 60
    bot.config.discord.respond_to_bots = False
    bot.config.discord.allowed_users = []

    # Tools / skills (for API routes)
    bot._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run a command", "input_schema": {}},
    ])
    bot._cached_merged_tools = None
    bot._cached_skills_text = None
    bot.skill_manager = MagicMock()
    bot.skill_manager.list_skills = MagicMock(return_value=[])
    bot.skill_manager._skills = {}

    # Sessions
    bot.sessions = MagicMock()
    bot.sessions._sessions = {}
    bot.sessions.add_message = MagicMock()
    bot.sessions.remove_last_message = MagicMock()
    bot.sessions.prune = MagicMock()
    bot.sessions.save = MagicMock()
    bot.sessions.get_task_history = AsyncMock(return_value=[
        {"role": "user", "content": "[WebUser]: hello"},
    ])

    # Codex client
    if codex_available:
        bot.codex_client = MagicMock()
    else:
        bot.codex_client = None

    # System prompt builders
    bot._build_system_prompt = MagicMock(return_value="You are Heimdall.")
    bot._inject_tool_hints = AsyncMock(return_value="You are Heimdall.")

    # _process_with_tools
    if process_result is None:
        # Default: simple text response, no tools, no error
        process_result = ("Hello from Heimdall!", False, False, [], False)
    bot._process_with_tools = AsyncMock(return_value=process_result)

    # Permissions
    bot.permissions = MagicMock()
    bot.permissions.is_guest = MagicMock(return_value=False)
    bot.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)

    # Tool memory / audit (for _process_with_tools internals - not needed for mock)
    bot.tool_memory = MagicMock()
    bot.audit = MagicMock()
    bot.audit.log_execution = AsyncMock()

    # Knowledge store (not needed but API routes check it)
    bot._knowledge_store = None
    bot._embedder = None

    # Scheduler / loops (for API routes)
    bot.scheduler = MagicMock()
    bot.scheduler.list_all = MagicMock(return_value=[])
    bot.loop_manager = MagicMock()
    bot.loop_manager.active_count = 0
    bot.loop_manager._loops = {}

    # Processes
    bot.tool_executor = MagicMock()
    bot.tool_executor._process_registry = None
    bot.tool_executor._load_all_memory = MagicMock(return_value={})
    bot.tool_executor._save_all_memory = MagicMock()

    return bot


def _make_app(bot=None, *, api_token=""):
    """Create an aiohttp Application with API routes + middleware."""
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
# WebMessage virtual message tests
# ===================================================================


class TestWebMessage:
    def test_creates_with_attributes(self):
        msg = WebMessage(channel_id="web-1", user_id="u1", username="Alice")
        assert msg.channel.id == "web-1"
        assert msg.author.id == "u1"
        assert msg.author.display_name == "Alice"
        assert msg.author.bot is False
        assert msg.webhook_id is None
        assert msg.attachments == []
        assert isinstance(msg.id, int)

    def test_unique_ids(self):
        m1 = WebMessage(channel_id="c", user_id="u", username="U")
        m2 = WebMessage(channel_id="c", user_id="u", username="U")
        assert m1.id != m2.id

    def test_author_str(self):
        msg = WebMessage(channel_id="c", user_id="u", username="Bob")
        assert str(msg.author) == "Bob"

    def test_author_mention(self):
        msg = WebMessage(channel_id="c", user_id="u", username="Bob")
        assert msg.author.mention == "@Bob"

    @pytest.mark.asyncio
    async def test_channel_typing_noop(self):
        msg = WebMessage(channel_id="c", user_id="u", username="U")
        async with msg.channel.typing():
            pass  # Should not raise

    @pytest.mark.asyncio
    async def test_channel_send_returns_editable(self):
        msg = WebMessage(channel_id="c", user_id="u", username="U")
        sent = await msg.channel.send("test")
        assert sent is not None
        await sent.edit(content="edited")  # Should not raise


class TestNoOpContextManager:
    @pytest.mark.asyncio
    async def test_enter_exit(self):
        cm = _NoOpContextManager()
        async with cm as val:
            assert val is cm


class TestWebSentMessage:
    @pytest.mark.asyncio
    async def test_edit_noop(self):
        m = _WebSentMessage()
        await m.edit(content="x", embed=None)


# ===================================================================
# process_web_chat tests
# ===================================================================


class TestProcessWebChat:
    @pytest.mark.asyncio
    async def test_basic_chat(self):
        bot = _make_chat_bot()
        result = await process_web_chat(bot, "hello", "web-1")
        assert result["response"] == "Hello from Heimdall!"
        assert result["tools_used"] == []
        assert result["is_error"] is False

        # Verify session management
        bot.sessions.add_message.assert_any_call(
            "web-1", "user", "[WebUser]: hello", user_id="web-user",
        )
        bot._process_with_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_user(self):
        bot = _make_chat_bot()
        result = await process_web_chat(
            bot, "hi", "web-2",
            user_id="alice", username="Alice",
        )
        assert result["is_error"] is False
        bot.sessions.add_message.assert_any_call(
            "web-2", "user", "[Alice]: hi", user_id="alice",
        )

    @pytest.mark.asyncio
    async def test_no_codex_client(self):
        bot = _make_chat_bot(codex_available=False)
        result = await process_web_chat(bot, "hello", "web-1")
        assert result["is_error"] is True
        assert "No LLM backend" in result["response"]
        bot.sessions.remove_last_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_tools_used_saved(self):
        bot = _make_chat_bot(
            process_result=("Disk is fine", False, False, ["check_disk"], False),
        )
        result = await process_web_chat(bot, "check disk", "web-1")
        assert result["tools_used"] == ["check_disk"]
        assert result["is_error"] is False
        # Should save assistant response when tools were used
        calls = [c for c in bot.sessions.add_message.call_args_list
                 if c[0][1] == "assistant"]
        assert len(calls) == 1
        assert calls[0][0][2] == "Disk is fine"

    @pytest.mark.asyncio
    async def test_no_tools_not_saved(self):
        """When no tools are used and no handoff, assistant response is not saved."""
        bot = _make_chat_bot(
            process_result=("Just chatting", False, False, [], False),
        )
        result = await process_web_chat(bot, "hello", "web-1")
        assert result["is_error"] is False
        # Only user message should be saved, not assistant
        assistant_saves = [c for c in bot.sessions.add_message.call_args_list
                          if c[0][1] == "assistant"]
        assert len(assistant_saves) == 0

    @pytest.mark.asyncio
    async def test_handoff_saved(self):
        bot = _make_chat_bot(
            process_result=("Skill result", False, False, [], True),
        )
        result = await process_web_chat(bot, "use skill", "web-1")
        assert result["is_error"] is False
        assistant_saves = [c for c in bot.sessions.add_message.call_args_list
                          if c[0][1] == "assistant"]
        assert len(assistant_saves) == 1

    @pytest.mark.asyncio
    async def test_error_saves_sanitized(self):
        bot = _make_chat_bot(
            process_result=("API failed", False, True, ["run_command"], False),
        )
        result = await process_web_chat(bot, "do thing", "web-1")
        assert result["is_error"] is True
        assistant_saves = [c for c in bot.sessions.add_message.call_args_list
                          if c[0][1] == "assistant"]
        assert len(assistant_saves) == 1
        saved_text = assistant_saves[0][0][2]
        assert "run_command" in saved_text
        assert "error" in saved_text.lower()

    @pytest.mark.asyncio
    async def test_error_no_tools_sanitized(self):
        bot = _make_chat_bot(
            process_result=("Failed", False, True, [], False),
        )
        result = await process_web_chat(bot, "do thing", "web-1")
        assert result["is_error"] is True
        assistant_saves = [c for c in bot.sessions.add_message.call_args_list
                          if c[0][1] == "assistant"]
        assert len(assistant_saves) == 1
        assert "error before tool execution" in assistant_saves[0][0][2].lower()

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        bot = _make_chat_bot()
        bot._process_with_tools = AsyncMock(side_effect=RuntimeError("boom"))
        result = await process_web_chat(bot, "hello", "web-1")
        assert result["is_error"] is True
        assert "boom" in result["response"]
        bot.sessions.remove_last_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_builds_system_prompt(self):
        bot = _make_chat_bot()
        await process_web_chat(bot, "hello", "web-1", user_id="u42")
        bot._build_system_prompt.assert_called_once_with(
            channel=None, user_id="u42", query="hello",
        )
        bot._inject_tool_hints.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_pruned_and_saved(self):
        bot = _make_chat_bot()
        await process_web_chat(bot, "hi", "web-1")
        bot.sessions.prune.assert_called_once()
        bot.sessions.save.assert_called_once()


# ===================================================================
# REST POST /api/chat endpoint tests
# ===================================================================


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_basic_chat(self):
        bot = _make_chat_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat",
                json={"content": "hello"},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["response"] == "Hello from Heimdall!"
            assert body["tools_used"] == []
            assert body["is_error"] is False

    @pytest.mark.asyncio
    async def test_with_custom_user(self):
        bot = _make_chat_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat",
                json={
                    "content": "hi",
                    "channel_id": "web-custom",
                    "user_id": "alice",
                    "username": "Alice",
                },
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["is_error"] is False

    @pytest.mark.asyncio
    async def test_empty_content(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat", json={"content": ""})
            assert resp.status == 400
            body = await resp.json()
            assert "content" in body["error"]

    @pytest.mark.asyncio
    async def test_missing_content(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat", json={})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_content_too_long(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat",
                json={"content": "x" * (MAX_CHAT_CONTENT_LEN + 1)},
            )
            assert resp.status == 400
            body = await resp.json()
            assert str(MAX_CHAT_CONTENT_LEN) in body["error"]

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        app, _ = _make_app()
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/chat",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_error_returns_502(self):
        bot = _make_chat_bot(
            process_result=("Error occurred", False, True, [], False),
        )
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat", json={"content": "fail"})
            assert resp.status == 502
            body = await resp.json()
            assert body["is_error"] is True

    @pytest.mark.asyncio
    async def test_no_codex_returns_502(self):
        bot = _make_chat_bot(codex_available=False)
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/chat", json={"content": "hello"})
            assert resp.status == 502
            body = await resp.json()
            assert body["is_error"] is True

    @pytest.mark.asyncio
    async def test_auth_required(self):
        bot = _make_chat_bot()
        app, _ = _make_app(bot, api_token="secret")
        async with TestClient(TestServer(app)) as client:
            # Without token
            resp = await client.post("/api/chat", json={"content": "hi"})
            assert resp.status == 401
            # With token
            resp = await client.post(
                "/api/chat",
                json={"content": "hi"},
                headers={"Authorization": "Bearer secret"},
            )
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_default_channel_id(self):
        bot = _make_chat_bot()
        app, _ = _make_app(bot)
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/chat", json={"content": "hi"})
        # Verify the default channel_id was used
        bot.sessions.add_message.assert_any_call(
            "web-default", "user", "[WebUser]: hi", user_id="web-user",
        )


# ===================================================================
# WebSocket chat message handling tests
# ===================================================================


class TestWebSocketChat:
    @pytest.mark.asyncio
    async def test_chat_message(self):
        bot = _make_chat_bot()
        app = web.Application()
        ws_manager = setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({
                "type": "chat",
                "content": "hello",
                "channel_id": "web-ws",
            })
            resp = await ws.receive_json()
            assert resp["type"] == "chat_response"
            assert resp["content"] == "Hello from Heimdall!"
            assert resp["tools_used"] == []
            assert resp["is_error"] is False
            await ws.close()

    @pytest.mark.asyncio
    async def test_chat_empty_content(self):
        bot = _make_chat_bot()
        app = web.Application()
        setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"type": "chat", "content": ""})
            resp = await ws.receive_json()
            assert resp["type"] == "chat_error"
            assert "content" in resp["error"]
            await ws.close()

    @pytest.mark.asyncio
    async def test_chat_missing_content(self):
        bot = _make_chat_bot()
        app = web.Application()
        setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"type": "chat"})
            resp = await ws.receive_json()
            assert resp["type"] == "chat_error"
            await ws.close()

    @pytest.mark.asyncio
    async def test_chat_content_too_long(self):
        bot = _make_chat_bot()
        app = web.Application()
        setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({
                "type": "chat",
                "content": "x" * (MAX_CHAT_CONTENT_LEN + 1),
            })
            resp = await ws.receive_json()
            assert resp["type"] == "chat_error"
            assert str(MAX_CHAT_CONTENT_LEN) in resp["error"]
            await ws.close()

    @pytest.mark.asyncio
    async def test_chat_with_custom_user(self):
        bot = _make_chat_bot()
        app = web.Application()
        setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({
                "type": "chat",
                "content": "hi",
                "channel_id": "my-channel",
                "user_id": "alice",
                "username": "Alice",
            })
            resp = await ws.receive_json()
            assert resp["type"] == "chat_response"
            assert resp["is_error"] is False
            await ws.close()
        bot.sessions.add_message.assert_any_call(
            "my-channel", "user", "[Alice]: hi", user_id="alice",
        )

    @pytest.mark.asyncio
    async def test_chat_default_channel(self):
        bot = _make_chat_bot()
        app = web.Application()
        setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"type": "chat", "content": "hi"})
            resp = await ws.receive_json()
            assert resp["type"] == "chat_response"
            await ws.close()
        bot.sessions.add_message.assert_any_call(
            "web-default", "user", "[WebUser]: hi", user_id="web-user",
        )

    @pytest.mark.asyncio
    async def test_chat_error_response(self):
        bot = _make_chat_bot(
            process_result=("LLM error", False, True, [], False),
        )
        app = web.Application()
        setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"type": "chat", "content": "fail"})
            resp = await ws.receive_json()
            assert resp["type"] == "chat_response"
            assert resp["is_error"] is True
            await ws.close()

    @pytest.mark.asyncio
    async def test_chat_with_tools(self):
        bot = _make_chat_bot(
            process_result=("Done!", False, False, ["run_command"], False),
        )
        app = web.Application()
        setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            await ws.send_json({"type": "chat", "content": "ls /"})
            resp = await ws.receive_json()
            assert resp["type"] == "chat_response"
            assert resp["tools_used"] == ["run_command"]
            await ws.close()

    @pytest.mark.asyncio
    async def test_subscribe_and_chat_coexist(self):
        """Chat and log subscription should work on the same connection."""
        bot = _make_chat_bot()
        app = web.Application()
        setup_websocket(app, bot)
        async with TestClient(TestServer(app)) as client:
            ws = await client.ws_connect("/api/ws")
            # Subscribe to events
            await ws.send_json({"subscribe": "events"})
            sub_resp = await ws.receive_json()
            assert sub_resp["type"] == "subscribed"
            # Send chat
            await ws.send_json({"type": "chat", "content": "hello"})
            chat_resp = await ws.receive_json()
            assert chat_resp["type"] == "chat_response"
            await ws.close()

    @pytest.mark.asyncio
    async def test_chat_auth_required(self):
        bot = _make_chat_bot()
        app = web.Application()
        ws_manager = setup_websocket(app, bot, api_token="secret")
        async with TestClient(TestServer(app)) as client:
            # Without token — should be closed
            ws = await client.ws_connect("/api/ws")
            msg = await ws.receive()
            assert msg.type == aiohttp.WSMsgType.CLOSE
