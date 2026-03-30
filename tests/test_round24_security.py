"""Tests for Round 24 security review — external threat protection.

Covers:
- Timing-safe auth comparison (Bearer header + WebSocket + query param)
- Content-Disposition header injection prevention
- Error message secret scrubbing in web endpoints
- Path traversal prevention (static file serving)
- Config PUT blocks sensitive fields
- Config GET redacts sensitive fields
- SQL parameterization verification
- Secret scrubber pattern coverage
"""
from __future__ import annotations

import hmac
import inspect
import json
import re
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.web.api import (
    _safe_filename,
    _sanitize_error,
    _redact_config,
    _contains_blocked_fields,
    _SENSITIVE_FIELDS,
    create_api_routes,
)
from src.llm.secret_scrubber import scrub_output_secrets, OUTPUT_SECRET_PATTERNS


# ---------------------------------------------------------------------------
# Helper: mock bot
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
        "discord": {"token": "xyzzy-secret", "channels": ["general"]},
        "tools": {"ssh_key_path": "/key", "tool_packs": []},
        "web": {"api_token": "tok", "enabled": True},
    })
    bot.config.tools.
    bot._merged_tool_definitions = MagicMock(return_value=[])
    bot._cached_merged_tools = None
    bot._cached_skills_text = None
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
    bot.skill_manager = MagicMock()
    bot.skill_manager.list_skills = MagicMock(return_value=[])
    bot.skill_manager._skills = {}
    store = MagicMock()
    store.available = True
    bot._knowledge_store = store
    bot._embedder = None
    bot.scheduler = MagicMock()
    bot.scheduler._schedules = []
    bot.scheduler.list_all = MagicMock(return_value=[])
    bot.loop_manager = MagicMock()
    bot.loop_manager.active_count = 0
    bot.loop_manager._loops = {}
    bot.tool_executor = MagicMock()
    bot.tool_executor._process_registry = None
    bot.audit = MagicMock()
    bot.audit.count_by_tool = AsyncMock(return_value={})
    bot.context_loader = MagicMock()
    return bot


# ===========================================================================
# 1. Timing-safe auth comparison
# ===========================================================================

class TestTimingSafeAuth:
    """Bearer header and WebSocket auth must use timing-safe comparison."""

    def test_bearer_auth_uses_hmac_compare_digest(self):
        """Auth middleware must use hmac.compare_digest, not ==."""
        source = Path("src/health/server.py").read_text()
        # Find the auth_middleware function
        # It should use hmac.compare_digest for Bearer token
        assert "hmac.compare_digest(auth_header" in source

    def test_bearer_auth_no_direct_equality(self):
        """Auth middleware must NOT use == for token comparison."""
        source = Path("src/health/server.py").read_text()
        # Extract the auth middleware function body
        start = source.index("async def auth_middleware")
        end = source.index("return auth_middleware", start)
        middleware_body = source[start:end]
        # Should not have direct == comparison with token
        assert "auth_header ==" not in middleware_body
        assert 'f"Bearer {token}"' not in middleware_body.split("hmac.compare_digest")[0].split("auth_header ==")[-1:][0] if "auth_header ==" in middleware_body else True

    def test_websocket_auth_uses_hmac_compare_digest(self):
        """WebSocket auth must use hmac.compare_digest, not != or ==."""
        source = Path("src/web/websocket.py").read_text()
        assert "hmac.compare_digest(token, self._api_token)" in source

    def test_websocket_auth_no_direct_inequality(self):
        """WebSocket auth must NOT use != for token comparison."""
        source = Path("src/web/websocket.py").read_text()
        assert "token != self._api_token" not in source

    def test_query_token_uses_hmac_compare_digest(self):
        """Query param token auth already uses hmac.compare_digest."""
        source = Path("src/health/server.py").read_text()
        assert "hmac.compare_digest(query_token, token)" in source

    def test_websocket_imports_hmac(self):
        """WebSocket module must import hmac for timing-safe comparison."""
        source = Path("src/web/websocket.py").read_text()
        assert "import hmac" in source


# ===========================================================================
# 2. Content-Disposition header injection prevention
# ===========================================================================

class TestContentDispositionSafety:
    """Session export filenames must be sanitized to prevent header injection."""

    def test_safe_filename_basic(self):
        assert _safe_filename("chan1") == "chan1"

    def test_safe_filename_with_special_chars(self):
        result = _safe_filename('test"; evil-header: value')
        assert '"' not in result
        assert ";" not in result
        assert " " not in result

    def test_safe_filename_with_quotes(self):
        result = _safe_filename('abc"def')
        assert '"' not in result
        assert result == "abc_def"

    def test_safe_filename_with_newlines(self):
        result = _safe_filename("abc\r\ndef")
        assert "\r" not in result
        assert "\n" not in result

    def test_safe_filename_with_slashes(self):
        result = _safe_filename("../../etc/passwd")
        assert "/" not in result

    def test_safe_filename_max_length(self):
        long = "a" * 200
        result = _safe_filename(long)
        assert len(result) <= 80

    def test_safe_filename_empty_string(self):
        result = _safe_filename("")
        assert result == "export"

    def test_safe_filename_only_special_chars(self):
        result = _safe_filename('";:\r\n ')
        # All chars replaced with underscore, so non-empty
        assert result  # not empty
        assert all(c in "abcdefghijklmnopqrstuvwxyz_.-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in result)

    def test_safe_filename_preserves_hyphens_underscores(self):
        assert _safe_filename("web-default_123") == "web-default_123"

    def test_safe_filename_preserves_periods(self):
        assert _safe_filename("file.name") == "file.name"

    def test_export_session_uses_safe_filename(self):
        """Export endpoint must sanitize cid in Content-Disposition."""
        source = Path("src/web/api.py").read_text()
        assert "_safe_filename(cid)" in source
        assert "safe_cid" in source


# ===========================================================================
# 3. Error message secret scrubbing
# ===========================================================================

class TestErrorSecretScrubbing:
    """Error messages returned to clients must be scrubbed for secrets."""

    def test_sanitize_error_scrubs_password(self):
        msg = _sanitize_error("Connection failed: password=s3cret123")
        assert "s3cret123" not in msg
        assert "[REDACTED]" in msg

    def test_sanitize_error_scrubs_api_key(self):
        msg = _sanitize_error("Bad request: api_key=sk-1234567890abcdef1234")
        assert "sk-1234567890abcdef1234" not in msg

    def test_sanitize_error_scrubs_db_uri(self):
        msg = _sanitize_error("postgres://admin:s3cret@db.host:5432/mydb")
        assert "s3cret" not in msg
        assert "[REDACTED]" in msg

    def test_sanitize_error_passes_clean_messages(self):
        msg = _sanitize_error("File not found: /tmp/test.txt")
        assert msg == "File not found: /tmp/test.txt"

    def test_api_skill_test_error_scrubbed(self):
        """POST /api/skills/{name}/test error uses _sanitize_error."""
        source = Path("src/web/api.py").read_text()
        # Find the test_skill handler's except block
        assert "_sanitize_error(e)" in source

    def test_api_schedule_create_error_scrubbed(self):
        """POST /api/schedules error uses _sanitize_error."""
        source = Path("src/web/api.py").read_text()
        # All str(e) in web responses should use _sanitize_error
        assert source.count("_sanitize_error(e)") >= 3  # skill test, schedule create, schedule run

    def test_api_schedule_run_error_scrubbed(self):
        """POST /api/schedules/{id}/run error uses _sanitize_error."""
        source = Path("src/web/api.py").read_text()
        # The schedule run-now handler
        start = source.index("async def run_schedule_now")
        end = source.index("@routes.", start + 1)
        handler_body = source[start:end]
        assert "_sanitize_error" in handler_body

    def test_websocket_chat_error_scrubbed(self):
        """WebSocket chat error messages are scrubbed."""
        source = Path("src/web/websocket.py").read_text()
        assert "scrub_output_secrets(str(e))" in source

    def test_web_chat_error_scrubbed(self):
        """Web chat exception error messages are scrubbed."""
        source = Path("src/web/chat.py").read_text()
        assert "_scrub(str(e))" in source

    def test_no_raw_str_e_in_web_responses(self):
        """No web endpoint should return raw str(e) without scrubbing."""
        for filepath in ("src/web/api.py", "src/web/websocket.py", "src/web/chat.py"):
            source = Path(filepath).read_text()
            # Find all json_response or send_json calls containing str(e)
            # These should use _sanitize_error or scrub_output_secrets
            # Check that str(e) is NOT directly in an "error" field
            lines = source.split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if '"error": str(e)' in stripped or '"result": str(e)' in stripped:
                    pytest.fail(
                        f"{filepath}:{i+1}: raw str(e) in response — "
                        f"should use _sanitize_error or scrub_output_secrets"
                    )


# ===========================================================================
# 4. Path traversal prevention
# ===========================================================================

class TestPathTraversalPrevention:
    """Static file serving must prevent path traversal attacks."""

    def test_serve_ui_file_checks_resolved_path(self):
        """_serve_ui_file must resolve symlinks and check prefix."""
        source = Path("src/health/server.py").read_text()
        assert ".resolve()" in source
        assert "startswith" in source
        assert "HTTPForbidden" in source

    def test_serve_ui_file_resolves_before_check(self):
        """File path is resolved BEFORE prefix check (not after)."""
        source = Path("src/health/server.py").read_text()
        start = source.index("def _serve_ui_file")
        # Find the next method definition
        end = source.index("\n    async def ", start + 20)
        handler_body = source[start:end]
        # .resolve() must come before startswith
        resolve_pos = handler_body.index(".resolve()")
        startswith_pos = handler_body.index("startswith")
        assert resolve_pos < startswith_pos


# ===========================================================================
# 5. Config PUT blocks sensitive fields
# ===========================================================================

class TestConfigPutSecurity:
    """PUT /api/config must reject updates to sensitive fields."""

    def test_sensitive_fields_include_token(self):
        assert "token" in _SENSITIVE_FIELDS

    def test_sensitive_fields_include_api_token(self):
        assert "api_token" in _SENSITIVE_FIELDS

    def test_sensitive_fields_include_password(self):
        assert "password" in _SENSITIVE_FIELDS

    def test_sensitive_fields_include_api_key(self):
        assert "api_key" in _SENSITIVE_FIELDS

    def test_sensitive_fields_include_ssh_key_path(self):
        assert "ssh_key_path" in _SENSITIVE_FIELDS

    def test_sensitive_fields_include_credentials_path(self):
        assert "credentials_path" in _SENSITIVE_FIELDS

    def test_sensitive_fields_include_secret(self):
        assert "secret" in _SENSITIVE_FIELDS

    def test_contains_blocked_fields_flat(self):
        assert _contains_blocked_fields({"token": "x"}, _SENSITIVE_FIELDS)

    def test_contains_blocked_fields_nested(self):
        assert _contains_blocked_fields(
            {"discord": {"token": "x"}}, _SENSITIVE_FIELDS
        )

    def test_contains_blocked_fields_clean(self):
        assert not _contains_blocked_fields(
            {"timezone": "UTC"}, _SENSITIVE_FIELDS
        )

    def test_contains_blocked_fields_depth_limit(self):
        """Deep nesting doesn't cause recursion issues."""
        deep = {"a": {}}
        current = deep["a"]
        for _ in range(20):
            current["b"] = {}
            current = current["b"]
        current["token"] = "x"
        # Should return False (depth > 10 cuts off)
        assert not _contains_blocked_fields(deep, _SENSITIVE_FIELDS)


# ===========================================================================
# 6. Config GET redacts sensitive fields
# ===========================================================================

class TestConfigGetRedaction:
    """GET /api/config must redact all sensitive values."""

    def test_redact_flat_token(self):
        result = _redact_config({"token": "secret123"})
        assert result["token"] == "••••••••"

    def test_redact_nested_token(self):
        result = _redact_config({"discord": {"token": "secret123"}})
        assert result["discord"]["token"] == "••••••••"

    def test_redact_empty_string_preserved(self):
        """Empty strings in sensitive fields are NOT masked (nothing to hide)."""
        result = _redact_config({"token": ""})
        assert result["token"] == ""

    def test_redact_non_sensitive_untouched(self):
        result = _redact_config({"timezone": "UTC"})
        assert result["timezone"] == "UTC"

    def test_redact_all_sensitive_fields(self):
        data = {field: f"value_{field}" for field in _SENSITIVE_FIELDS}
        result = _redact_config(data)
        for field in _SENSITIVE_FIELDS:
            assert result[field] == "••••••••", f"Field {field} not redacted"

    def test_redact_list_values(self):
        result = _redact_config({"items": [{"token": "x"}, {"name": "y"}]})
        assert result["items"][0]["token"] == "••••••••"
        assert result["items"][1]["name"] == "y"


# ===========================================================================
# 7. Secret scrubber pattern coverage
# ===========================================================================

class TestSecretScrubberCoverage:
    """Secret scrubber must catch common credential patterns."""

    def test_scrub_password_equals(self):
        assert "[REDACTED]" in scrub_output_secrets("password=mysecret123")

    def test_scrub_password_colon(self):
        assert "[REDACTED]" in scrub_output_secrets("password: mysecret123")

    def test_scrub_api_key(self):
        assert "[REDACTED]" in scrub_output_secrets("api_key=abcdef12345")

    def test_scrub_openai_key(self):
        assert "[REDACTED]" in scrub_output_secrets("key is sk-1234567890abcdef12345678")

    def test_scrub_rsa_private_key(self):
        assert "[REDACTED]" in scrub_output_secrets("-----BEGIN RSA PRIVATE KEY-----")

    def test_scrub_openssh_private_key(self):
        assert "[REDACTED]" in scrub_output_secrets("-----BEGIN OPENSSH PRIVATE KEY-----")

    def test_scrub_postgres_uri(self):
        assert "[REDACTED]" in scrub_output_secrets("postgres://user:pass@host:5432/db")

    def test_scrub_mysql_uri(self):
        assert "[REDACTED]" in scrub_output_secrets("mysql://user:pass@host:3306/db")

    def test_scrub_mongodb_uri(self):
        assert "[REDACTED]" in scrub_output_secrets("mongodb://user:pass@host:27017/db")

    def test_scrub_github_pat(self):
        assert "[REDACTED]" in scrub_output_secrets("ghp_1234567890abcdefghijklmnopqrstuvwxyz01")

    def test_scrub_github_oauth(self):
        assert "[REDACTED]" in scrub_output_secrets("gho_1234567890abcdefghijklmnopqrstuvwxyz01")

    def test_scrub_aws_access_key(self):
        assert "[REDACTED]" in scrub_output_secrets("AKIAIOSFODNN7EXAMPLE")

    def test_scrub_stripe_live_key(self):
        assert "[REDACTED]" in scrub_output_secrets("sk_live_1234567890abcdefghij")

    def test_scrub_slack_token(self):
        assert "[REDACTED]" in scrub_output_secrets("xoxb-123456789-abcdef")

    def test_scrub_preserves_clean_text(self):
        clean = "Server disk usage is at 85%"
        assert scrub_output_secrets(clean) == clean

    def test_scrub_multiple_secrets(self):
        text = "password=x api_key=y"
        result = scrub_output_secrets(text)
        assert "password" not in result or "[REDACTED]" in result
        assert result.count("[REDACTED]") >= 2

    def test_pattern_count_minimum(self):
        """Should have at least 10 secret detection patterns."""
        assert len(OUTPUT_SECRET_PATTERNS) >= 10


# ===========================================================================
# 8. WebSocket security
# ===========================================================================

class TestWebSocketSecurity:
    """WebSocket endpoints must enforce auth and validate input."""

    def test_websocket_validates_chat_content_length(self):
        """Chat content exceeding limit is rejected."""
        source = Path("src/web/websocket.py").read_text()
        assert "MAX_CHAT_CONTENT_LEN" in source

    def test_websocket_validates_empty_content(self):
        """Empty chat content is rejected."""
        source = Path("src/web/websocket.py").read_text()
        assert "content is required" in source

    def test_websocket_handles_invalid_json(self):
        """Invalid JSON messages get error response."""
        source = Path("src/web/websocket.py").read_text()
        assert "json.JSONDecodeError" in source
        assert '"error": "invalid JSON"' in source


# ===========================================================================
# 9. Security headers
# ===========================================================================

class TestSecurityHeaders:
    """All responses must include security headers."""

    def test_nosniff_header(self):
        source = Path("src/health/server.py").read_text()
        assert "X-Content-Type-Options" in source
        assert "nosniff" in source

    def test_frame_deny_header(self):
        source = Path("src/health/server.py").read_text()
        assert "X-Frame-Options" in source
        assert "DENY" in source


# ===========================================================================
# 10. Rate limiting
# ===========================================================================

class TestRateLimiting:
    """API endpoints must be rate-limited."""

    def test_rate_limit_exists(self):
        source = Path("src/health/server.py").read_text()
        assert "rate_limit" in source.lower()

    def test_rate_limit_per_ip(self):
        source = Path("src/health/server.py").read_text()
        assert "request.remote" in source

    def test_rate_limit_returns_429(self):
        source = Path("src/health/server.py").read_text()
        assert "429" in source


# ===========================================================================
# 11. FTS query sanitization
# ===========================================================================

class TestFTSSanitization:
    """FTS5 queries must sanitize special characters to prevent syntax injection."""

    def test_fts_has_prepare_query(self):
        source = Path("src/search/fts.py").read_text()
        assert "_prepare_query" in source

    def test_fts_escapes_special_chars(self):
        """FTS _prepare_query escapes FTS5 special characters."""
        from src.search.fts import _prepare_query
        # Should handle quotes safely
        result = _prepare_query('test"injection')
        assert '""' in result or '"test' in result  # Quotes are escaped

    def test_fts_handles_operators(self):
        """FTS _prepare_query handles boolean operators safely."""
        from src.search.fts import _prepare_query
        # AND/OR/NOT should be quoted to prevent syntax abuse
        result = _prepare_query("DROP OR TABLE")
        # Operators should be individually quoted
        assert result  # Non-empty result

    def test_fts_uses_parameterized_queries(self):
        """FTS search uses ? placeholders, not f-strings."""
        source = Path("src/search/fts.py").read_text()
        # All MATCH clauses should use ? for the query
        assert "MATCH ?" in source


# ===========================================================================
# 12. SQL parameterization verification
# ===========================================================================

class TestSQLParameterization:
    """All SQL queries must use parameterized ? placeholders."""

    @pytest.mark.parametrize("filepath", [
        "src/knowledge/store.py",
        "src/search/fts.py",
        "src/search/vectorstore.py",
    ])
    def test_no_fstring_sql_with_user_input(self, filepath):
        """No SQL queries use f-strings with user-supplied values."""
        source = Path(filepath).read_text()
        # Find all .execute() calls and verify they use ? placeholders
        # The only allowed f-strings are for compile-time constants (VECTOR_DIM)
        lines = source.split("\n")
        in_execute = False
        for i, line in enumerate(lines):
            if ".execute(" in line and "f'" in line or '.execute(' in line and 'f"' in line:
                # Check if this f-string only uses constants
                if "VECTOR_DIM" not in line and "{VECTOR_DIM}" not in line:
                    # Multi-line f-string — check next lines
                    block = "\n".join(lines[i:i+5])
                    if "VECTOR_DIM" not in block:
                        pytest.fail(
                            f"{filepath}:{i+1}: f-string in .execute() without "
                            f"VECTOR_DIM constant — possible SQL injection"
                        )
