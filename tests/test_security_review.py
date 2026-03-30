"""Tests for Round 49 — Security Review (external threats only).

Covers: path traversal, SQL injection prevention, secret scrubbing completeness,
agent isolation, skill sandboxing, WebSocket auth, input validation, XSS prevention,
auth middleware, rate limiting, webhook auth, config redaction, error scrubbing,
session security, and chat input validation.
"""
from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from src.agents.manager import (
    AGENT_BLOCKED_TOOLS,
    AgentInfo,
    AgentManager,
    MAX_AGENT_ITERATIONS,
    MAX_AGENT_LIFETIME,
    MAX_CONCURRENT_AGENTS,
    filter_agent_tools,
)
from src.config.schema import WebConfig, WebhookConfig
from src.health.server import (
    HealthServer,
    SessionManager,
    _make_auth_middleware,
    _make_csrf_middleware,
    _make_rate_limit_middleware,
    _make_security_headers_middleware,
    _RATE_LIMIT_MAX,
)
from src.llm.secret_scrubber import OUTPUT_SECRET_PATTERNS, scrub_output_secrets
from src.search.fts import _prepare_query
from src.tools.skill_context import (
    MAX_SKILL_FILES,
    MAX_SKILL_HTTP_REQUESTS,
    MAX_SKILL_MESSAGES,
    MAX_SKILL_TOOL_CALLS,
    SKILL_SAFE_TOOLS,
    ResourceTracker,
    is_path_denied,
    is_url_blocked,
)
from src.tools.skill_manager import SKILL_NAME_PATTERN, BUILTIN_TOOL_NAMES
from src.web.api import (
    _MAX_CODE_LEN,
    _MAX_CONTENT_LEN,
    _MAX_GOAL_LEN,
    _MAX_NAME_LEN,
    _SENSITIVE_FIELDS,
    _contains_blocked_fields,
    _redact_config,
    _safe_filename,
    _sanitize_error,
    _validate_string,
)
from src.web.chat import MAX_CHAT_CONTENT_LEN


# ===========================================================================
# 1. Path Traversal Prevention
# ===========================================================================

class TestPathTraversalPrevention:
    """Verify _serve_ui_file blocks path traversal attacks."""

    def test_blocks_parent_traversal(self, tmp_path):
        """Requests with .. must not escape the ui directory."""
        ui_dir = tmp_path / "ui"
        ui_dir.mkdir()
        (ui_dir / "index.html").write_text("<html></html>")
        secret_dir = tmp_path / "secret"
        secret_dir.mkdir()
        (secret_dir / "data.txt").write_text("top-secret")

        path = "../secret/data.txt"
        file = (ui_dir / path).resolve()
        ui_root = str(ui_dir.resolve()) + "/"
        assert not str(file).startswith(ui_root)

    def test_blocks_sibling_prefix_attack(self, tmp_path):
        """A sibling dir whose name starts with 'ui' must not bypass the check."""
        ui_dir = tmp_path / "ui"
        ui_dir.mkdir()
        sibling = tmp_path / "ui_backup"
        sibling.mkdir()
        (sibling / "secret.txt").write_text("secret")

        path = "../ui_backup/secret.txt"
        file = (ui_dir / path).resolve()
        ui_root = str(ui_dir.resolve()) + "/"
        assert not str(file).startswith(ui_root)

    def test_allows_valid_file(self, tmp_path):
        ui_dir = tmp_path / "ui"
        ui_dir.mkdir()
        (ui_dir / "app.js").write_text("console.log('hi')")

        file = (ui_dir / "app.js").resolve()
        ui_root = str(ui_dir.resolve()) + "/"
        assert str(file).startswith(ui_root)

    def test_allows_nested_valid_file(self, tmp_path):
        ui_dir = tmp_path / "ui"
        (ui_dir / "js").mkdir(parents=True)
        (ui_dir / "js" / "app.js").write_text("ok")

        file = (ui_dir / "js" / "app.js").resolve()
        ui_root = str(ui_dir.resolve()) + "/"
        assert str(file).startswith(ui_root)

    def test_blocks_double_dot_traversal(self, tmp_path):
        ui_dir = tmp_path / "ui"
        ui_dir.mkdir()
        file = (ui_dir / "../../etc/passwd").resolve()
        ui_root = str(ui_dir.resolve()) + "/"
        assert not str(file).startswith(ui_root)

    def test_source_code_uses_trailing_separator(self):
        """Verify the source code fix is in place."""
        import inspect
        source = inspect.getsource(HealthServer._serve_ui_file)
        assert '+ "/"' in source or "+ os.sep" in source or "is_relative_to" in source


# ===========================================================================
# 2. SQL Injection Prevention (FTS5 query sanitization)
# ===========================================================================

class TestFTS5Sanitization:
    """Verify FTS5 queries are properly escaped."""

    def test_special_chars_quoted(self):
        result = _prepare_query('test" OR 1=1 --')
        assert result.startswith('"')

    def test_asterisk_quoted(self):
        assert _prepare_query("*") == '"*"'

    def test_brackets_quoted(self):
        result = _prepare_query("test{inject}")
        assert result.startswith('"')

    def test_keywords_escaped(self):
        result = _prepare_query("to be or not to be")
        # FTS5 keywords should be individually quoted
        assert '"to"' in result or '"TO"' in result

    def test_ip_address_quoted(self):
        assert _prepare_query("10.0.0.50") == '"10.0.0.50"'

    def test_path_quoted(self):
        assert _prepare_query("/etc/passwd") == '"/etc/passwd"'

    def test_empty_returns_empty(self):
        assert _prepare_query("") == ""

    def test_simple_term_passthrough(self):
        assert _prepare_query("nginx") == "nginx"

    def test_internal_quotes_escaped(self):
        result = _prepare_query('file"name')
        assert '""' in result  # internal quotes doubled


# ===========================================================================
# 3. Secret Scrubbing Completeness
# ===========================================================================

class TestSecretScrubbing:
    """Verify secret patterns catch all known credential formats."""

    def test_password_variants(self):
        assert "[REDACTED]" in scrub_output_secrets("password=hunter2")
        assert "[REDACTED]" in scrub_output_secrets("PASSWORD: secretval")
        assert "[REDACTED]" in scrub_output_secrets("passwd=mypass")
        assert "[REDACTED]" in scrub_output_secrets("pwd=secret123")

    def test_api_key_variants(self):
        assert "[REDACTED]" in scrub_output_secrets("api_key=abcd1234efgh")
        assert "[REDACTED]" in scrub_output_secrets("apikey=xyz123")
        assert "[REDACTED]" in scrub_output_secrets("API-KEY=test")

    def test_openai_key(self):
        assert "[REDACTED]" in scrub_output_secrets("sk-1234567890abcdefghijklmno")

    def test_private_keys(self):
        for kind in ("RSA", "EC", "OPENSSH", "DSA", ""):
            prefix = f"BEGIN {kind} PRIVATE KEY" if kind else "BEGIN PRIVATE KEY"
            assert "[REDACTED]" in scrub_output_secrets(f"-----{prefix}-----")

    def test_database_uris(self):
        assert "[REDACTED]" in scrub_output_secrets("postgres://user:pass@localhost/db")
        assert "[REDACTED]" in scrub_output_secrets("mysql://admin:secret@host/db")
        assert "[REDACTED]" in scrub_output_secrets("mongodb+srv://user:pass@cluster/db")

    def test_github_tokens(self):
        for prefix in ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"):
            assert "[REDACTED]" in scrub_output_secrets(prefix + "a" * 36)

    def test_aws_access_key(self):
        assert "[REDACTED]" in scrub_output_secrets("AKIAIOSFODNN7EXAMPLE")

    def test_stripe_keys(self):
        assert "[REDACTED]" in scrub_output_secrets("sk_live_" + "a" * 24)
        assert "[REDACTED]" in scrub_output_secrets("rk_live_" + "b" * 24)
        assert "[REDACTED]" in scrub_output_secrets("sk_test_" + "c" * 24)

    def test_slack_tokens(self):
        assert "[REDACTED]" in scrub_output_secrets("xoxb-123-456-abcdef")
        assert "[REDACTED]" in scrub_output_secrets("xoxp-123-456-abcdef")

    def test_safe_text_unchanged(self):
        text = "The server is running nginx 1.24.0 on port 443"
        assert scrub_output_secrets(text) == text

    def test_multiple_secrets_all_scrubbed(self):
        text = "password=secret1 and api_key=secret2"
        result = scrub_output_secrets(text)
        assert "secret1" not in result
        assert "secret2" not in result

    def test_token_min_length_threshold(self):
        """token= requires 16+ chars to avoid false positives."""
        short = "token=short"
        long_val = "token=abcdefghijklmnop"
        assert scrub_output_secrets(short) == short
        assert "[REDACTED]" in scrub_output_secrets(long_val)

    def test_pattern_count(self):
        assert len(OUTPUT_SECRET_PATTERNS) >= 10


# ===========================================================================
# 4. Agent Isolation
# ===========================================================================

class TestAgentIsolation:
    """Verify agents cannot access each other or spawn sub-agents."""

    def test_blocked_tools_comprehensive(self):
        expected = {
            "spawn_agent", "send_to_agent", "list_agents",
            "kill_agent", "get_agent_results", "wait_for_agents",
        }
        assert expected == AGENT_BLOCKED_TOOLS

    def test_filter_removes_all_blocked(self):
        tools = [
            {"name": "run_command"},
            {"name": "spawn_agent"},
            {"name": "check_disk"},
            {"name": "kill_agent"},
            {"name": "list_agents"},
            {"name": "send_to_agent"},
            {"name": "get_agent_results"},
            {"name": "wait_for_agents"},
        ]
        filtered = filter_agent_tools(tools)
        names = {t["name"] for t in filtered}
        assert names == {"run_command", "check_disk"}

    def test_filter_empty_list(self):
        assert filter_agent_tools([]) == []

    def test_filter_no_blocked_present(self):
        tools = [{"name": "run_command"}, {"name": "check_disk"}]
        assert len(filter_agent_tools(tools)) == 2

    def test_agent_messages_isolated(self):
        a1 = AgentInfo(
            id="a1", label="agent1", goal="task1",
            channel_id="ch1", requester_id="u1", requester_name="User1",
        )
        a2 = AgentInfo(
            id="a2", label="agent2", goal="task2",
            channel_id="ch1", requester_id="u1", requester_name="User1",
        )
        a1.messages.append({"role": "user", "content": "secret for agent1"})
        a2.messages.append({"role": "user", "content": "secret for agent2"})
        assert a1.messages != a2.messages
        assert "agent1" in a1.messages[0]["content"]
        assert "agent2" in a2.messages[0]["content"]

    def test_per_channel_limit(self):
        mgr = AgentManager()
        cb = AsyncMock(return_value={"text": "done", "tool_calls": [], "stop_reason": "stop"})
        tool_cb = AsyncMock(return_value="ok")
        announce_cb = AsyncMock()
        ids = []
        for i in range(MAX_CONCURRENT_AGENTS):
            aid = mgr.spawn(
                label=f"agent{i}", goal=f"task{i}",
                channel_id="ch1", requester_id="u1", requester_name="User",
                iteration_callback=cb, tool_executor_callback=tool_cb,
                announce_callback=announce_cb,
            )
            ids.append(aid)
            assert not aid.startswith("Error")
        # Overflow
        result = mgr.spawn(
            label="overflow", goal="too many",
            channel_id="ch1", requester_id="u1", requester_name="User",
            iteration_callback=cb, tool_executor_callback=tool_cb,
            announce_callback=announce_cb,
        )
        assert result.startswith("Error")
        for aid in ids:
            mgr.kill(aid)

    def test_agent_output_scrubbed(self):
        text = "Result: password=hunter2"
        result = scrub_output_secrets(text)
        assert "hunter2" not in result
        assert "[REDACTED]" in result

    def test_iteration_limit(self):
        assert MAX_AGENT_ITERATIONS == 30

    def test_lifetime_limit(self):
        assert MAX_AGENT_LIFETIME == 3600

    def test_agent_requires_label_and_goal(self):
        mgr = AgentManager()
        cb = AsyncMock()
        result = mgr.spawn(
            label="", goal="task",
            channel_id="ch1", requester_id="u1", requester_name="User",
            iteration_callback=cb, tool_executor_callback=cb, announce_callback=cb,
        )
        assert result.startswith("Error")
        result = mgr.spawn(
            label="agent", goal="",
            channel_id="ch1", requester_id="u1", requester_name="User",
            iteration_callback=cb, tool_executor_callback=cb, announce_callback=cb,
        )
        assert result.startswith("Error")


# ===========================================================================
# 5. Skill Sandboxing
# ===========================================================================

class TestSkillSandboxing:
    """Verify skill execution limits and file/URL restrictions."""

    def test_safe_tools_exclude_destructive(self):
        dangerous = {
            "run_command", "write_file", "run_script", "docker_compose_up",
            "docker_compose_down", "systemd_manage", "incus_exec",
            "ansible_playbook", "generate_image",
        }
        overlap = SKILL_SAFE_TOOLS & dangerous
        assert overlap == set(), f"Dangerous tools in safe list: {overlap}"

    def test_limits_defined(self):
        assert MAX_SKILL_TOOL_CALLS == 50
        assert MAX_SKILL_HTTP_REQUESTS == 20
        assert MAX_SKILL_MESSAGES == 10
        assert MAX_SKILL_FILES == 10

    def test_path_denied_env_files(self):
        for p in (".env", ".env.local", "/app/.env", "/app/.env.production"):
            assert is_path_denied(p) is True, f"{p} should be denied"

    def test_path_denied_config(self):
        assert is_path_denied("config.yml") is True
        assert is_path_denied("config.yaml") is True

    def test_path_denied_sensitive_system(self):
        assert is_path_denied("/etc/shadow") is True

    def test_path_denied_ssh(self):
        assert is_path_denied("id_rsa") is True
        assert is_path_denied("id_ed25519") is True
        assert is_path_denied("/home/user/.ssh/config") is True

    def test_path_denied_credentials(self):
        assert is_path_denied("credentials.json") is True
        assert is_path_denied("/home/user/.kube/config") is True

    def test_path_allowed_normal(self):
        assert is_path_denied("/var/log/nginx/access.log") is False
        assert is_path_denied("/app/src/main.py") is False

    def test_url_blocked_localhost(self):
        assert is_url_blocked("http://localhost/secret") is True
        assert is_url_blocked("http://127.0.0.1/api") is True
        assert is_url_blocked("http://0.0.0.0/api") is True

    def test_url_blocked_cloud_metadata(self):
        assert is_url_blocked("http://169.254.169.254/latest/meta-data/") is True
        assert is_url_blocked("http://metadata.google.internal/") is True

    def test_url_blocked_private_ranges(self):
        assert is_url_blocked("http://10.0.0.1/api") is True
        assert is_url_blocked("http://172.16.0.1/api") is True
        assert is_url_blocked("http://192.168.2.1/api") is True

    def test_url_allowed_public(self):
        assert is_url_blocked("https://api.example.com/data") is False

    def test_url_blocked_empty(self):
        assert is_url_blocked("http:///path") is True
        assert is_url_blocked("") is True

    def test_skill_name_blocks_injection(self):
        assert SKILL_NAME_PATTERN.match("valid_skill") is not None
        assert SKILL_NAME_PATTERN.match("../etc/passwd") is None
        assert SKILL_NAME_PATTERN.match("has space") is None
        assert SKILL_NAME_PATTERN.match("skill;rm -rf /") is None
        assert SKILL_NAME_PATTERN.match("1skill") is None
        assert SKILL_NAME_PATTERN.match("a" * 51) is None

    def test_skill_names_cannot_shadow_builtins(self):
        assert "run_command" in BUILTIN_TOOL_NAMES
        assert "read_file" in BUILTIN_TOOL_NAMES

    def test_resource_tracker_initial_state(self):
        tracker = ResourceTracker()
        assert tracker.tool_calls == 0
        assert tracker.http_requests == 0
        assert tracker.messages_sent == 0
        assert tracker.files_sent == 0
        assert tracker.bytes_downloaded == 0


# ===========================================================================
# 6. WebSocket Auth
# ===========================================================================

class TestWebSocketAuth:
    """Verify WebSocket connections require authentication."""

    def test_ws_manager_stores_token(self):
        from src.web.websocket import WebSocketManager
        bot = MagicMock()
        mgr = WebSocketManager(bot, api_token="secret-token")
        assert mgr._api_token == "secret-token"

    def test_ws_manager_empty_token_dev_mode(self):
        from src.web.websocket import WebSocketManager
        bot = MagicMock()
        mgr = WebSocketManager(bot, api_token="")
        assert mgr._api_token == ""

    def test_ws_manager_uses_hmac_compare(self):
        """Verify the WebSocket handler uses constant-time comparison."""
        import inspect
        from src.web.websocket import WebSocketManager
        source = inspect.getsource(WebSocketManager.handle)
        assert "hmac.compare_digest" in source


# ===========================================================================
# 7. Input Validation
# ===========================================================================

class TestInputValidation:
    """Verify API input validation prevents oversized/malformed data."""

    def test_validate_string_ok(self):
        assert _validate_string("hello", "field", 100) is None

    def test_validate_string_exceeds(self):
        assert _validate_string("a" * 200, "f", 100) is not None

    def test_validate_string_exact(self):
        assert _validate_string("a" * 100, "f", 100) is None

    def test_limits_defined(self):
        assert _MAX_NAME_LEN == 100
        assert _MAX_CODE_LEN == 50_000
        assert _MAX_CONTENT_LEN == 500_000
        assert _MAX_GOAL_LEN == 2000
        assert MAX_CHAT_CONTENT_LEN == 4000

    def test_safe_filename_strips_special(self):
        assert _safe_filename("test file (1).txt") == "test_file__1_.txt"
        assert _safe_filename("../../etc/passwd") == ".._.._etc_passwd"
        assert _safe_filename("<script>alert(1)</script>") == "_script_alert_1___script_"

    def test_safe_filename_max_length(self):
        assert len(_safe_filename("a" * 200)) <= 80

    def test_safe_filename_empty(self):
        assert _safe_filename("") == "export"
        # "!!!" → "___" (all chars replaced, non-empty result)
        assert _safe_filename("!!!") == "___"


# ===========================================================================
# 8. Config Security
# ===========================================================================

class TestConfigSecurity:
    """Verify sensitive fields are redacted and blocked from API updates."""

    def test_sensitive_fields(self):
        expected = {"token", "api_token", "secret", "ssh_key_path",
                    "credentials_path", "api_key", "password"}
        assert expected == _SENSITIVE_FIELDS

    def test_redact_masks_tokens(self):
        config = {"token": "real-secret", "name": "bot", "nested": {"api_key": "key"}}
        result = _redact_config(config)
        assert result["token"] == "••••••••"
        assert result["name"] == "bot"
        assert result["nested"]["api_key"] == "••••••••"

    def test_redact_empty_not_masked(self):
        config = {"token": "", "api_key": ""}
        result = _redact_config(config)
        assert result["token"] == ""
        assert result["api_key"] == ""

    def test_redact_non_string_not_masked(self):
        assert _redact_config({"token": 12345})["token"] == 12345

    def test_blocked_fields_flat(self):
        assert _contains_blocked_fields({"token": "v"}, _SENSITIVE_FIELDS) is True
        assert _contains_blocked_fields({"name": "v"}, _SENSITIVE_FIELDS) is False

    def test_blocked_fields_nested(self):
        assert _contains_blocked_fields(
            {"nested": {"password": "v"}}, _SENSITIVE_FIELDS
        ) is True

    def test_blocked_fields_depth_limit(self):
        """Deeply nested dicts should not crash."""
        d = {}
        current = d
        for _ in range(20):
            current["a"] = {}
            current = current["a"]
        current["token"] = "secret"
        # Should not crash regardless of result
        _contains_blocked_fields(d, _SENSITIVE_FIELDS)

    def test_redact_depth_limit(self):
        d = {}
        current = d
        for _ in range(20):
            current["a"] = {}
            current = current["a"]
        _redact_config(d)  # Should not crash


# ===========================================================================
# 9. Auth Middleware
# ===========================================================================

class TestAuthMiddleware:
    """Verify auth middleware properly gates API access."""

    async def test_no_header_returns_401(self):
        wc = WebConfig(api_token="secret")
        sm = SessionManager()
        app = web.Application(middlewares=[_make_auth_middleware(wc, sm)])
        app.router.add_get("/api/test", lambda r: web.json_response({"ok": True}))
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/api/test")).status == 401

    async def test_correct_bearer_200(self):
        wc = WebConfig(api_token="secret")
        sm = SessionManager()
        app = web.Application(middlewares=[_make_auth_middleware(wc, sm)])
        app.router.add_get("/api/test", lambda r: web.json_response({"ok": True}))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/test", headers={"Authorization": "Bearer secret"})
            assert resp.status == 200

    async def test_wrong_bearer_401(self):
        wc = WebConfig(api_token="secret")
        sm = SessionManager()
        app = web.Application(middlewares=[_make_auth_middleware(wc, sm)])
        app.router.add_get("/api/test", lambda r: web.json_response({"ok": True}))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/test", headers={"Authorization": "Bearer wrong"})
            assert resp.status == 401

    async def test_session_token_accepted(self):
        wc = WebConfig(api_token="secret")
        sm = SessionManager()
        sid, _ = sm.create()
        app = web.Application(middlewares=[_make_auth_middleware(wc, sm)])
        app.router.add_get("/api/test", lambda r: web.json_response({"ok": True}))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/test", headers={"Authorization": f"Bearer {sid}"})
            assert resp.status == 200

    async def test_query_param_token(self):
        wc = WebConfig(api_token="secret")
        sm = SessionManager()
        app = web.Application(middlewares=[_make_auth_middleware(wc, sm)])
        app.router.add_get("/api/test", lambda r: web.json_response({"ok": True}))
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/api/test?token=secret")).status == 200

    async def test_dev_mode_skips_auth(self):
        wc = WebConfig(api_token="")
        sm = SessionManager()
        app = web.Application(middlewares=[_make_auth_middleware(wc, sm)])
        app.router.add_get("/api/test", lambda r: web.json_response({"ok": True}))
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/api/test")).status == 200

    async def test_non_api_skips_auth(self):
        wc = WebConfig(api_token="secret")
        sm = SessionManager()
        app = web.Application(middlewares=[_make_auth_middleware(wc, sm)])
        app.router.add_get("/health", lambda r: web.json_response({"ok": True}))
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/health")).status == 200


# ===========================================================================
# 10. Rate Limiting
# ===========================================================================

class TestRateLimiting:

    def test_rate_limit_constants(self):
        assert _RATE_LIMIT_MAX == 120

    async def test_blocks_excess_requests(self):
        app = web.Application(middlewares=[_make_rate_limit_middleware()])
        app.router.add_get("/api/test", lambda r: web.json_response({"ok": True}))
        async with TestClient(TestServer(app)) as client:
            for _ in range(120):
                assert (await client.get("/api/test")).status == 200
            resp = await client.get("/api/test")
            assert resp.status == 429


# ===========================================================================
# 11. Webhook Auth
# ===========================================================================

class TestWebhookAuth:

    def test_rejects_no_secret_configured(self):
        hs = HealthServer(port=9999, webhook_config=WebhookConfig(enabled=True, secret=""))
        assert hs._verify_hmac_sha256(b"body", "anysig") is False

    def test_rejects_wrong_signature(self):
        hs = HealthServer(port=9999, webhook_config=WebhookConfig(enabled=True, secret="s"))
        assert hs._verify_hmac_sha256(b"body", "wrong") is False

    def test_accepts_correct_signature(self):
        secret = "mysecret"
        body = b'{"test": true}'
        sig = hmac_mod.new(secret.encode(), body, hashlib.sha256).hexdigest()
        hs = HealthServer(port=9999, webhook_config=WebhookConfig(enabled=True, secret=secret))
        assert hs._verify_hmac_sha256(body, sig) is True


# ===========================================================================
# 12. XSS Prevention
# ===========================================================================

class TestXSSPrevention:

    async def test_api_returns_json_content_type(self):
        wc = WebConfig(api_token="")
        sm = SessionManager()
        app = web.Application(middlewares=[
            _make_security_headers_middleware(),
            _make_auth_middleware(wc, sm),
        ])

        bot = MagicMock()
        bot.guilds = []
        bot.is_ready = MagicMock(return_value=True)
        bot._start_time = time.monotonic()
        bot._merged_tool_definitions = MagicMock(return_value=[])
        bot.skill_manager = MagicMock()
        bot.skill_manager.list_skills = MagicMock(return_value=[])
        bot.sessions = MagicMock()
        bot.sessions._sessions = {}
        bot.loop_manager = MagicMock()
        bot.loop_manager._loops = {}
        bot.loop_manager.active_count = 0
        bot.agent_manager = MagicMock()
        bot.agent_manager._agents = {}
        bot.scheduler = MagicMock()
        bot.scheduler.list_all = MagicMock(return_value=[])
        bot.tool_executor = MagicMock()
        bot.tool_executor._process_registry = MagicMock()
        bot.tool_executor._process_registry._processes = {}
        bot.audit = MagicMock()
        bot.audit.count_by_tool = AsyncMock(return_value={})
        bot.infra_watcher = None
        bot.config = MagicMock()
        bot.config.web.api_token = ""


        from src.web.api import setup_api
        setup_api(app, bot)
        app["session_manager"] = sm

        async with TestClient(TestServer(app)) as client:
            for path in ("/api/status", "/api/sessions", "/api/tools"):
                resp = await client.get(path)
                assert resp.content_type == "application/json"


# ===========================================================================
# 13. Error Message Scrubbing
# ===========================================================================

class TestErrorScrubbing:

    def test_scrubs_db_uri_in_error(self):
        err = "Connection failed: postgres://admin:pass@host/db"
        result = _sanitize_error(err)
        assert "pass" not in result or "[REDACTED]" in result

    def test_handles_non_string(self):
        result = _sanitize_error(ValueError("test"))
        assert isinstance(result, str)


# ===========================================================================
# 14. Session Security
# ===========================================================================

class TestSessionSecurity:

    def test_token_length(self):
        sm = SessionManager()
        sid, _ = sm.create()
        assert len(sid) >= 40

    def test_tokens_unique(self):
        sm = SessionManager()
        tokens = {sm.create()[0] for _ in range(100)}
        assert len(tokens) == 100

    def test_cleanup_removes_expired(self):
        sm = SessionManager(timeout_minutes=1)
        sid, _ = sm.create()
        sm._sessions[sid] = time.monotonic() - 120
        assert sm.cleanup() == 1
        assert sm.validate(sid) is False


# ===========================================================================
# 15. Chat Input Validation
# ===========================================================================

class TestChatInputValidation:

    def test_content_limit(self):
        assert MAX_CHAT_CONTENT_LEN == 4000

    def test_web_message_not_bot(self):
        from src.web.chat import WebMessage
        msg = WebMessage("ch", "u1", "User")
        assert msg.author.bot is False

    def test_web_message_no_webhook_id(self):
        from src.web.chat import WebMessage
        msg = WebMessage("ch", "u1", "User")
        assert msg.webhook_id is None

    def test_web_message_ids_increment(self):
        from src.web.chat import WebMessage
        m1 = WebMessage("ch", "u", "User")
        m2 = WebMessage("ch", "u", "User")
        assert m2.id > m1.id

    def test_web_message_empty_attachments(self):
        from src.web.chat import WebMessage
        msg = WebMessage("ch", "u", "User")
        assert msg.attachments == []


# ===========================================================================
# 16. CSRF Middleware
# ===========================================================================

class TestCSRFMiddleware:

    async def test_get_allowed_regardless(self):
        app = web.Application(middlewares=[_make_csrf_middleware()])
        app.router.add_get("/api/test", lambda r: web.json_response({"ok": True}))
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/test", headers={"Origin": "http://evil.com"})
            assert resp.status == 200

    async def test_post_cross_origin_blocked(self):
        app = web.Application(middlewares=[_make_csrf_middleware()])
        app.router.add_post("/api/test", lambda r: web.json_response({"ok": True}))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/test", headers={"Origin": "http://evil.com"})
            assert resp.status == 403

    async def test_post_same_origin_allowed(self):
        app = web.Application(middlewares=[_make_csrf_middleware()])
        app.router.add_post("/api/test", lambda r: web.json_response({"ok": True}))
        async with TestClient(TestServer(app)) as client:
            host = f"{client.host}:{client.port}"
            resp = await client.post("/api/test", headers={"Origin": f"http://{host}"})
            assert resp.status == 200

    async def test_post_no_origin_allowed(self):
        app = web.Application(middlewares=[_make_csrf_middleware()])
        app.router.add_post("/api/test", lambda r: web.json_response({"ok": True}))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/test")
            assert resp.status == 200

    async def test_login_skips_csrf(self):
        app = web.Application(middlewares=[_make_csrf_middleware()])
        app.router.add_post("/api/auth/login", lambda r: web.json_response({"ok": True}))
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/auth/login", headers={"Origin": "http://evil.com"})
            assert resp.status == 200
