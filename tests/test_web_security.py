"""Tests for Round 43 — Web UI Security (CSP, CSRF, session timeout, login, audit trail).

Covers: CSP headers, CSRF Origin checking, session management, auth endpoints,
web audit middleware, and session timeout behaviour.
"""
from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from src.config.schema import WebConfig
from src.health.server import (
    HealthServer,
    SessionManager,
    _AUTH_SKIP_PATHS,
    _CSP_POLICY,
    _make_auth_middleware,
    _make_csrf_middleware,
    _make_security_headers_middleware,
    _make_web_audit_middleware,
)
from src.web.api import setup_api


# ---------------------------------------------------------------------------
# Helper: mock bot for API routes
# ---------------------------------------------------------------------------

def _make_bot():
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
        "discord": {"token": "secret"},
        "tools": {"ssh_key_path": "/key"},
        "web": {"api_token": "test-token", "enabled": True, "session_timeout_minutes": 0},
    })
    bot.config.web.api_token = "test-token"
    bot._merged_tool_definitions = MagicMock(return_value=[
        {"name": "run_command", "description": "Run a command", "input_schema": {}},
    ])
    bot._cached_merged_tools = None
    bot._cached_skills_text = None
    session = MagicMock()
    session.messages = []
    session.summary = ""
    session.created_at = 1704067200.0
    session.last_active = 1704153600.0
    session.last_user_id = "u1"
    bot.sessions = MagicMock()
    bot.sessions._sessions = {"chan1": session}
    bot.sessions.reset = MagicMock()
    bot.skill_manager = MagicMock()
    bot.skill_manager.list_skills = MagicMock(return_value=[])
    bot.skill_manager._skills = {}
    bot.audit = MagicMock()
    bot.audit.count_by_tool = AsyncMock(return_value={})
    bot.audit.search = AsyncMock(return_value=[])
    bot.audit.log_web_action = AsyncMock()
    store = MagicMock()
    store.available = True
    store.list_sources = MagicMock(return_value=[])
    bot._knowledge_store = store
    bot._embedder = MagicMock()
    bot.scheduler = MagicMock()
    bot.scheduler.list_all = MagicMock(return_value=[])
    bot.loop_manager = MagicMock()
    bot.loop_manager._loops = {}
    bot.loop_manager.active_count = 0
    bot.agent_manager = MagicMock()
    bot.agent_manager._agents = {}
    bot.tool_executor = MagicMock()
    bot.tool_executor._process_registry = MagicMock()
    bot.tool_executor._process_registry._processes = {}
    bot.tool_executor._load_all_memory = MagicMock(return_value={})
    bot.tool_executor._save_all_memory = MagicMock()
    bot.infra_watcher = None
    bot.context_loader = MagicMock()
    bot._invalidate_prompt_caches = MagicMock()
    bot._build_system_prompt = MagicMock(return_value="test prompt")
    return bot


def _make_app(web_config=None, include_routes=True):
    """Create a test app with all security middlewares."""
    wc = web_config or WebConfig(api_token="test-token")
    sm = SessionManager(timeout_minutes=wc.session_timeout_minutes)
    app = web.Application(middlewares=[
        _make_security_headers_middleware(),
        _make_csrf_middleware(),
        _make_auth_middleware(wc, sm),
        _make_web_audit_middleware(),
    ])
    app["session_manager"] = sm
    if include_routes:
        bot = _make_bot()
        bot.config.web.api_token = wc.api_token
        app["audit_logger"] = bot.audit
        setup_api(app, bot)
    return app, sm


# ===========================================================================
# Session Manager
# ===========================================================================

class TestSessionManager:
    def test_create_returns_session_id(self):
        sm = SessionManager()
        sid, timeout = sm.create()
        assert isinstance(sid, str)
        assert len(sid) > 20
        assert timeout == 0

    def test_validate_known_session(self):
        sm = SessionManager()
        sid, _ = sm.create()
        assert sm.validate(sid) is True

    def test_validate_unknown_session(self):
        sm = SessionManager()
        assert sm.validate("nonexistent") is False

    def test_destroy_session(self):
        sm = SessionManager()
        sid, _ = sm.create()
        assert sm.destroy(sid) is True
        assert sm.validate(sid) is False

    def test_destroy_unknown_returns_false(self):
        sm = SessionManager()
        assert sm.destroy("nonexistent") is False

    def test_active_count(self):
        sm = SessionManager()
        assert sm.active_count == 0
        sid1, _ = sm.create()
        assert sm.active_count == 1
        sid2, _ = sm.create()
        assert sm.active_count == 2
        sm.destroy(sid1)
        assert sm.active_count == 1

    def test_timeout_expires_session(self):
        sm = SessionManager(timeout_minutes=1)
        sid, timeout = sm.create()
        assert timeout == 60
        sm._sessions[sid] = time.monotonic() - 120
        assert sm.validate(sid) is False

    def test_timeout_refreshes_on_validate(self):
        sm = SessionManager(timeout_minutes=1)
        sid, _ = sm.create()
        sm._sessions[sid] = time.monotonic() - 30
        assert sm.validate(sid) is True
        assert (time.monotonic() - sm._sessions[sid]) < 2

    def test_cleanup_removes_expired(self):
        sm = SessionManager(timeout_minutes=1)
        sid1, _ = sm.create()
        sid2, _ = sm.create()
        sm._sessions[sid1] = time.monotonic() - 120
        removed = sm.cleanup()
        assert removed == 1
        assert sm.active_count == 1
        assert sm.validate(sid2) is True

    def test_cleanup_noop_without_timeout(self):
        sm = SessionManager(timeout_minutes=0)
        sm.create()
        removed = sm.cleanup()
        assert removed == 0

    def test_timeout_seconds_property(self):
        sm = SessionManager(timeout_minutes=30)
        assert sm.timeout_seconds == 1800


# ===========================================================================
# CSP Headers
# ===========================================================================

class TestCSPHeaders:
    async def test_csp_header_present(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            assert resp.status == 200
            csp = resp.headers.get("Content-Security-Policy")
            assert csp is not None
            assert "default-src 'self'" in csp
            assert "frame-ancestors 'none'" in csp

    async def test_csp_allows_cdn_scripts(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            csp = resp.headers.get("Content-Security-Policy", "")
            assert "cdn.tailwindcss.com" in csp
            assert "unpkg.com" in csp
            assert "cdn.jsdelivr.net" in csp

    async def test_csp_allows_google_fonts(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            csp = resp.headers.get("Content-Security-Policy", "")
            assert "fonts.googleapis.com" in csp
            assert "fonts.gstatic.com" in csp

    async def test_csp_allows_websockets(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            csp = resp.headers.get("Content-Security-Policy", "")
            assert "ws:" in csp
            assert "wss:" in csp

    async def test_xframe_options_deny(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            assert resp.headers.get("X-Frame-Options") == "DENY"

    async def test_nosniff_header(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    async def test_referrer_policy(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    async def test_permissions_policy(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            pp = resp.headers.get("Permissions-Policy", "")
            assert "camera=()" in pp
            assert "microphone=()" in pp

    async def test_csp_policy_constant(self):
        assert "default-src 'self'" in _CSP_POLICY
        assert "script-src" in _CSP_POLICY
        assert "style-src" in _CSP_POLICY
        assert "font-src" in _CSP_POLICY
        assert "base-uri 'self'" in _CSP_POLICY
        assert "form-action 'self'" in _CSP_POLICY


# ===========================================================================
# CSRF Protection
# ===========================================================================

class TestCSRFProtection:
    async def test_get_requests_skip_csrf(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            assert resp.status == 200

    async def test_post_with_matching_origin_allowed(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            host = f"{client.host}:{client.port}"
            resp = await client.post(
                "/api/reload",
                headers={"Origin": f"http://{host}"},
            )
            assert resp.status == 200

    async def test_post_with_mismatched_origin_blocked(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/reload",
                headers={"Origin": "http://evil.com"},
            )
            assert resp.status == 403
            data = await resp.json()
            assert "cross-origin" in data["error"]

    async def test_post_with_mismatched_referer_blocked(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/reload",
                headers={"Referer": "http://evil.com/page"},
            )
            assert resp.status == 403

    async def test_post_without_origin_or_referer_allowed(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/reload")
            assert resp.status == 200

    async def test_login_endpoint_skips_csrf(self):
        app, _ = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/auth/login",
                json={"token": "test-token"},
                headers={"Origin": "http://evil.com"},
            )
            assert resp.status == 200

    async def test_delete_with_mismatched_origin_blocked(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.delete(
                "/api/sessions/chan1",
                headers={"Origin": "http://evil.com"},
            )
            assert resp.status == 403

    async def test_put_with_mismatched_origin_blocked(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.put(
                "/api/config",
                json={},
                headers={"Origin": "http://evil.com"},
            )
            assert resp.status == 403


# ===========================================================================
# Auth Endpoints
# ===========================================================================

class TestAuthEndpoints:
    async def test_login_success(self):
        app, sm = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/auth/login", json={"token": "test-token"})
            assert resp.status == 200
            data = await resp.json()
            assert "session_id" in data
            assert "timeout_seconds" in data
            assert sm.validate(data["session_id"]) is True

    async def test_login_invalid_token(self):
        app, _ = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/auth/login", json={"token": "wrong"})
            assert resp.status == 401
            data = await resp.json()
            assert "invalid" in data["error"]

    async def test_login_empty_token(self):
        app, _ = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/auth/login", json={"token": ""})
            assert resp.status == 400

    async def test_login_missing_token(self):
        app, _ = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/auth/login", json={})
            assert resp.status == 400

    async def test_login_no_auth_configured(self):
        app, _ = _make_app(WebConfig(api_token=""))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/auth/login", json={"token": "anything"})
            assert resp.status == 200
            data = await resp.json()
            assert "session_id" in data

    async def test_login_with_timeout(self):
        app, _ = _make_app(WebConfig(api_token="test-token", session_timeout_minutes=30))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/auth/login", json={"token": "test-token"})
            data = await resp.json()
            assert data["timeout_seconds"] == 1800

    async def test_logout(self):
        app, sm = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/auth/login", json={"token": "test-token"})
            sid = (await resp.json())["session_id"]
            assert sm.validate(sid) is True
            resp = await client.post(
                "/api/auth/logout",
                headers={"Authorization": f"Bearer {sid}"},
            )
            assert resp.status == 200
            assert sm.validate(sid) is False

    async def test_session_check(self):
        app, _ = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/api/auth/session",
                headers={"Authorization": "Bearer test-token"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["authenticated"] is True

    async def test_session_token_auth(self):
        app, _ = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/auth/login", json={"token": "test-token"})
            sid = (await resp.json())["session_id"]
            resp = await client.get(
                "/api/status",
                headers={"Authorization": f"Bearer {sid}"},
            )
            assert resp.status == 200

    async def test_expired_session_rejected(self):
        app, sm = _make_app(WebConfig(api_token="test-token", session_timeout_minutes=1))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/auth/login", json={"token": "test-token"})
            sid = (await resp.json())["session_id"]
            sm._sessions[sid] = time.monotonic() - 120
            resp = await client.get(
                "/api/status",
                headers={"Authorization": f"Bearer {sid}"},
            )
            assert resp.status == 401

    async def test_raw_api_token_still_works(self):
        app, _ = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/api/status",
                headers={"Authorization": "Bearer test-token"},
            )
            assert resp.status == 200

    async def test_invalid_session_token_rejected(self):
        app, _ = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/api/status",
                headers={"Authorization": "Bearer invalid-session-id"},
            )
            assert resp.status == 401


# ===========================================================================
# Web Audit Middleware
# ===========================================================================

class TestWebAuditMiddleware:
    async def test_state_changing_requests_logged(self):
        app, _ = _make_app(WebConfig(api_token=""))
        mock_audit = AsyncMock()
        app["audit_logger"] = MagicMock()
        app["audit_logger"].log_web_action = mock_audit
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/reload")
            assert mock_audit.called
            kw = mock_audit.call_args.kwargs
            assert kw["method"] == "POST"
            assert kw["path"] == "/api/reload"
            assert kw["status"] == 200

    async def test_get_requests_not_logged(self):
        app, _ = _make_app(WebConfig(api_token=""))
        mock_audit = AsyncMock()
        app["audit_logger"] = MagicMock()
        app["audit_logger"].log_web_action = mock_audit
        async with TestClient(TestServer(app)) as client:
            await client.get("/api/status")
            assert not mock_audit.called

    async def test_delete_requests_logged(self):
        app, _ = _make_app(WebConfig(api_token=""))
        mock_audit = AsyncMock()
        app["audit_logger"] = MagicMock()
        app["audit_logger"].log_web_action = mock_audit
        async with TestClient(TestServer(app)) as client:
            await client.delete("/api/sessions/chan1")
            assert mock_audit.called
            kw = mock_audit.call_args.kwargs
            assert kw["method"] == "DELETE"

    async def test_audit_logs_execution_time(self):
        app, _ = _make_app(WebConfig(api_token=""))
        mock_audit = AsyncMock()
        app["audit_logger"] = MagicMock()
        app["audit_logger"].log_web_action = mock_audit
        async with TestClient(TestServer(app)) as client:
            await client.post("/api/reload")
            kw = mock_audit.call_args.kwargs
            assert "execution_time_ms" in kw
            assert isinstance(kw["execution_time_ms"], int)

    async def test_audit_failure_doesnt_block_response(self):
        app, _ = _make_app(WebConfig(api_token=""))
        mock_audit = AsyncMock(side_effect=Exception("audit broken"))
        app["audit_logger"] = MagicMock()
        app["audit_logger"].log_web_action = mock_audit
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/reload")
            assert resp.status == 200

    async def test_audit_without_logger(self):
        app, _ = _make_app(WebConfig(api_token=""))
        app.pop("audit_logger", None)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/reload")
            assert resp.status == 200


# ===========================================================================
# Session Timeout
# ===========================================================================

class TestSessionTimeout:
    def test_zero_timeout_means_no_expiry(self):
        sm = SessionManager(timeout_minutes=0)
        sid, timeout = sm.create()
        assert timeout == 0
        sm._sessions[sid] = time.monotonic() - 999999
        assert sm.validate(sid) is True

    def test_session_expires_after_timeout(self):
        sm = SessionManager(timeout_minutes=5)
        sid, timeout = sm.create()
        assert timeout == 300
        sm._sessions[sid] = time.monotonic() - 200
        assert sm.validate(sid) is True
        sm._sessions[sid] = time.monotonic() - 400
        assert sm.validate(sid) is False

    def test_config_session_timeout_default(self):
        wc = WebConfig()
        assert wc.session_timeout_minutes == 0

    def test_config_session_timeout_custom(self):
        wc = WebConfig(session_timeout_minutes=60)
        assert wc.session_timeout_minutes == 60

    def test_health_server_creates_session_manager(self):
        hs = HealthServer(
            port=9999,
            web_config=WebConfig(session_timeout_minutes=30),
        )
        sm = hs._session_manager
        assert isinstance(sm, SessionManager)
        assert sm.timeout_seconds == 1800


# ===========================================================================
# Audit Logger — log_web_action
# ===========================================================================

class TestAuditLogWebAction:
    async def test_log_web_action_writes_entry(self, tmp_path):
        from src.audit.logger import AuditLogger
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(path=str(log_path))
        await logger.log_web_action(
            method="POST",
            path="/api/reload",
            status=200,
            ip="127.0.0.1",
            execution_time_ms=42,
        )
        content = log_path.read_text()
        entry = json.loads(content.strip())
        assert entry["type"] == "web_action"
        assert entry["method"] == "POST"
        assert entry["path"] == "/api/reload"
        assert entry["status"] == 200
        assert entry["ip"] == "127.0.0.1"
        assert entry["execution_time_ms"] == 42
        assert "timestamp" in entry

    async def test_log_web_action_fires_callback(self, tmp_path):
        from src.audit.logger import AuditLogger
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(path=str(log_path))
        callback = AsyncMock()
        logger.set_event_callback(callback)
        await logger.log_web_action(
            method="DELETE",
            path="/api/sessions/test",
            status=200,
        )
        assert callback.called
        entry = callback.call_args[0][0]
        assert entry["type"] == "web_action"
        assert entry["method"] == "DELETE"


# ===========================================================================
# Auth Skip Paths
# ===========================================================================

class TestAuthSkipPaths:
    def test_login_in_skip_paths(self):
        assert "/api/auth/login" in _AUTH_SKIP_PATHS

    async def test_login_accessible_without_auth(self):
        app, _ = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/auth/login", json={"token": "test-token"})
            assert resp.status == 200

    async def test_other_endpoints_require_auth(self):
        app, _ = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/status")
            assert resp.status == 401


# ===========================================================================
# Integration: Full auth flow
# ===========================================================================

class TestFullAuthFlow:
    async def test_login_use_session_logout(self):
        app, sm = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            # 1. Login
            resp = await client.post("/api/auth/login", json={"token": "test-token"})
            assert resp.status == 200
            sid = (await resp.json())["session_id"]

            # 2. Use session token
            resp = await client.get(
                "/api/status",
                headers={"Authorization": f"Bearer {sid}"},
            )
            assert resp.status == 200

            # 3. Logout
            resp = await client.post(
                "/api/auth/logout",
                headers={"Authorization": f"Bearer {sid}"},
            )
            assert resp.status == 200

            # 4. Session token should now be rejected
            resp = await client.get(
                "/api/status",
                headers={"Authorization": f"Bearer {sid}"},
            )
            assert resp.status == 401

    async def test_session_query_param_auth(self):
        app, _ = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/auth/login", json={"token": "test-token"})
            sid = (await resp.json())["session_id"]
            resp = await client.get(f"/api/status?token={sid}")
            assert resp.status == 200

    async def test_multiple_sessions(self):
        app, sm = _make_app(WebConfig(api_token="test-token"))
        async with TestClient(TestServer(app)) as client:
            resp1 = await client.post("/api/auth/login", json={"token": "test-token"})
            sid1 = (await resp1.json())["session_id"]
            resp2 = await client.post("/api/auth/login", json={"token": "test-token"})
            sid2 = (await resp2.json())["session_id"]

            assert sid1 != sid2
            assert sm.active_count >= 2

            for sid in (sid1, sid2):
                resp = await client.get(
                    "/api/status",
                    headers={"Authorization": f"Bearer {sid}"},
                )
                assert resp.status == 200

            await client.post(
                "/api/auth/logout",
                headers={"Authorization": f"Bearer {sid1}"},
            )

            resp = await client.get(
                "/api/status",
                headers={"Authorization": f"Bearer {sid1}"},
            )
            assert resp.status == 401

            resp = await client.get(
                "/api/status",
                headers={"Authorization": f"Bearer {sid2}"},
            )
            assert resp.status == 200
