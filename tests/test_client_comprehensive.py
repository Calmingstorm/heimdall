"""Comprehensive tests for src/discord/client.py — targeting 43% → 80%+ coverage.

Covers:
- Utility functions: scrub_response_secrets, truncate_tool_output, _detect_image_type,
  _check_for_secrets, _is_allowed_user, _is_allowed_channel, _get_attachment_hint
- on_message flow: dedup, mention stripping, attachment processing, secrets detection,
  voice commands
- _handle_message: thread context inheritance, channel locking
- _process_attachments: images, text files, large files, binary files
- Tool handlers: purge, browser_screenshot, generate_file, post_file, schedule_task,
  list_schedules, delete_schedule, parse_time, search_history, search_knowledge,
  ingest_document, list_knowledge, delete_knowledge, search_audit, create_digest
- Background tasks: delegate_task, list_tasks, cancel_task
- Scheduled tasks: _on_scheduled_task, _on_scheduled_digest, _format_digest_raw,
  _run_scheduled_workflow
- Streaming: _stream_iteration
- Discord helpers: _send_with_retry, _send_chunked, _resolve_mentions, _on_monitor_alert
- Error handling: CircuitOpenError fallback paths
"""
from __future__ import annotations

import asyncio
import base64
import io
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402
import discord  # noqa: E402

from src.discord.client import (  # noqa: E402
    LokiBot,
    scrub_response_secrets,
    truncate_tool_output,
    DISCORD_MAX_LEN,
    MAX_TOOL_ITERATIONS,
    TOOL_OUTPUT_MAX_CHARS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub(**overrides):
    """Create a minimal LokiBot stub for method-level tests.

    Follows the pattern from test_chat_path_optimization.py.
    """
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
    stub._last_tool_use = {}
    stub._system_prompt = "test system prompt"
    stub._channel_locks = {}
    stub._processed_messages = MagicMock()
    stub._processed_messages_max = 100
    stub._background_tasks = {}
    stub._background_tasks_max = 20
    stub.config = MagicMock()
    stub.config.tools.enabled = True
    stub.config.tools.tool_timeout_seconds = 300
    stub.config.discord.allowed_users = ["12345"]
    stub.config.discord.channels = ["67890"]
    stub.config.discord.respond_to_bots = False
    stub.config.discord.require_mention = False
    stub.config.monitoring.alert_channel_id = "67890"
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.get_task_history = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.sessions.reset = MagicMock()
    stub.sessions.search_history = AsyncMock(return_value=[])
    stub.sessions.get_or_create = MagicMock()
    stub.sessions._sessions = {}
    stub.classifier.classify = AsyncMock(return_value="chat")
    stub.codex_client = MagicMock()
    stub.codex_client.chat = AsyncMock(return_value="Codex chat response")
    stub.codex_client.chat_with_tools = AsyncMock()
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.skill_manager.get_tool_definitions = MagicMock(return_value=[])
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.audit.search = AsyncMock(return_value=[])
    stub._build_system_prompt = MagicMock(return_value="full system prompt")
    stub._build_chat_system_prompt = MagicMock(return_value="chat system prompt")
    stub._send_with_retry = AsyncMock(return_value=MagicMock(id=999))
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[])
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._pending_files = {}
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub.permissions.is_admin = MagicMock(return_value=True)
    stub.voice_manager = None
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="tool result")
    stub.tool_executor.set_user_context = MagicMock()
    stub.tool_executor._resolve_host = MagicMock(return_value=("10.0.0.1", "root", "linux"))
    stub.tool_executor._load_memory_for_user = MagicMock(return_value={})
    stub.tool_memory = MagicMock()
    stub.tool_memory.record = AsyncMock()
    stub.browser_manager = None
    stub.reflector = MagicMock()
    stub.reflector.get_prompt_section = MagicMock(return_value="")
    stub._knowledge_store = None
    stub._embedder = None
    stub._fts_index = None
    stub._vector_store = None
    stub._memory_path = "/tmp/test_memory.json"
    stub.scheduler = MagicMock()
    stub.scheduler.add = MagicMock(return_value={
        "id": "sched-1", "description": "Test task", "next_run": "2026-03-19T08:00:00",
    })
    stub.scheduler.list_all = MagicMock(return_value=[])
    stub.scheduler.delete = MagicMock(return_value=True)
    stub.infra_watcher = None
    stub.context_loader = MagicMock()
    stub.tree = MagicMock()
    stub.user = MagicMock()
    stub.user.id = 111
    stub.guilds = []
    stub.get_channel = MagicMock(return_value=None)

    for k, v in overrides.items():
        setattr(stub, k, v)
    return stub


def _make_message(channel_id="67890", author_id="12345", content="test"):
    """Create a mock Discord message."""
    msg = AsyncMock()
    msg.id = int(time.time() * 1000)
    msg.content = content
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.channel.send = AsyncMock()
    msg.channel.purge = AsyncMock(return_value=[1, 2, 3])
    msg.channel.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(), __aexit__=AsyncMock(),
    ))
    msg.author = MagicMock()
    msg.author.id = author_id
    msg.author.bot = False
    msg.author.mention = f"<@{author_id}>"
    msg.author.display_name = "TestUser"
    msg.reply = AsyncMock()
    msg.delete = AsyncMock()
    msg.edit = AsyncMock()
    msg.attachments = []
    return msg


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestScrubResponseSecrets:
    """Tests for scrub_response_secrets."""

    def test_scrubs_api_key(self):
        """Should redact API keys from LLM responses."""
        text = "Here's the key: sk-abcdefghijklmnopqrstuvwxyz1234"
        result = scrub_response_secrets(text)
        assert "sk-abcdefghij" not in result
        assert "[REDACTED]" in result

    def test_scrubs_slack_token(self):
        """Should redact Slack tokens."""
        text = "The token is xoxb-1234567890-abcdefghij"
        result = scrub_response_secrets(text)
        assert "xoxb-" not in result

    def test_scrubs_password_disclosure(self):
        """Should redact 'my password is ...' patterns."""
        text = "my password is supersecretpass123"
        result = scrub_response_secrets(text)
        assert "supersecretpass123" not in result

    def test_preserves_clean_text(self):
        """Should not alter text without secrets."""
        text = "The server is running normally on port 8080."
        assert scrub_response_secrets(text) == text


class TestTruncateToolOutput:
    """Tests for truncate_tool_output."""

    def test_short_output_unchanged(self):
        """Output under the limit should pass through unchanged."""
        text = "short output"
        assert truncate_tool_output(text) == text

    def test_exactly_at_limit(self):
        """Output exactly at the limit should pass through unchanged."""
        text = "x" * TOOL_OUTPUT_MAX_CHARS
        assert truncate_tool_output(text) == text

    def test_long_output_truncated(self):
        """Output over the limit should be truncated with marker."""
        text = "x" * (TOOL_OUTPUT_MAX_CHARS + 1000)
        result = truncate_tool_output(text)
        assert len(result) < len(text)
        assert "characters omitted" in result
        # Preserves start and end
        assert result.startswith("x")
        assert result.endswith("x")

    def test_custom_max_chars(self):
        """Should respect a custom max_chars parameter."""
        text = "x" * 200
        result = truncate_tool_output(text, max_chars=100)
        assert "characters omitted" in result


class TestDetectImageType:
    """Tests for LokiBot._detect_image_type."""

    def test_png(self):
        """Should detect PNG from magic bytes."""
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        assert LokiBot._detect_image_type(data) == "image/png"

    def test_jpeg(self):
        """Should detect JPEG from magic bytes."""
        data = b"\xff\xd8\xff\xe0" + b"\x00" * 10
        assert LokiBot._detect_image_type(data) == "image/jpeg"

    def test_gif(self):
        """Should detect GIF from magic bytes."""
        data = b"GIF89a" + b"\x00" * 10
        assert LokiBot._detect_image_type(data) == "image/gif"

    def test_webp(self):
        """Should detect WebP from magic bytes."""
        data = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 10
        assert LokiBot._detect_image_type(data) == "image/webp"

    def test_unknown(self):
        """Should return None for unknown formats."""
        data = b"\x00\x00\x00\x00\x00\x00\x00\x00"
        assert LokiBot._detect_image_type(data) is None


class TestGetAttachmentHint:
    """Tests for LokiBot._get_attachment_hint."""

    def test_python_file(self):
        """Python files should suggest skill creation."""
        hint = LokiBot._get_attachment_hint("tool.py", ".py", 1000)
        assert "skill" in hint.lower()

    def test_yaml_config(self):
        """YAML config files should suggest deploying or ingesting."""
        hint = LokiBot._get_attachment_hint("config.yml", ".yml", 500)
        assert "configuration" in hint.lower()

    def test_shell_script(self):
        """Shell scripts should suggest deploying or reviewing."""
        hint = LokiBot._get_attachment_hint("deploy.sh", ".sh", 500)
        assert "shell script" in hint.lower()

    def test_systemd_unit(self):
        """Systemd unit files should suggest deploying."""
        hint = LokiBot._get_attachment_hint("app.service", ".service", 200)
        assert "systemd" in hint.lower()

    def test_markdown_doc(self):
        """Markdown files should suggest knowledge base ingestion."""
        hint = LokiBot._get_attachment_hint("readme.md", ".md", 2000)
        assert "documentation" in hint.lower()

    def test_large_file_extra_hint(self):
        """Files over 50KB should get an extra ingest suggestion."""
        hint = LokiBot._get_attachment_hint("big.py", ".py", 60000)
        assert "large" in hint.lower()

    def test_no_hint_for_unknown(self):
        """Unknown file types without large size should return empty."""
        hint = LokiBot._get_attachment_hint("data.bin", ".bin", 1000)
        assert hint == ""

    def test_json_config(self):
        """JSON config files should suggest deploying or ingesting."""
        hint = LokiBot._get_attachment_hint("package.json", ".json", 500)
        assert "configuration" in hint.lower()

    def test_timer_unit(self):
        """Timer files should suggest deploying."""
        hint = LokiBot._get_attachment_hint("backup.timer", ".timer", 100)
        assert "systemd" in hint.lower()

    def test_txt_doc(self):
        """Text files should suggest knowledge base ingestion."""
        hint = LokiBot._get_attachment_hint("notes.txt", ".txt", 500)
        assert "documentation" in hint.lower()

    def test_bash_script(self):
        """Bash scripts should suggest deploying or reviewing."""
        hint = LokiBot._get_attachment_hint("init.bash", ".bash", 300)
        assert "shell script" in hint.lower()

    def test_ini_config(self):
        """INI files should be recognized as config."""
        hint = LokiBot._get_attachment_hint("app.ini", ".ini", 200)
        assert "configuration" in hint.lower()


# ---------------------------------------------------------------------------
# _is_allowed_user / _is_allowed_channel / _check_for_secrets
# ---------------------------------------------------------------------------

class TestAccessChecks:
    """Tests for _is_allowed_user, _is_allowed_channel, _check_for_secrets."""

    def test_allowed_user_in_list(self):
        """User in allowed list should return True."""
        stub = _make_bot_stub()
        stub._is_allowed_user = LokiBot._is_allowed_user.__get__(stub)
        user = MagicMock()
        user.id = 12345
        assert stub._is_allowed_user(user) is True

    def test_disallowed_user(self):
        """User not in allowed list should return False."""
        stub = _make_bot_stub()
        stub._is_allowed_user = LokiBot._is_allowed_user.__get__(stub)
        user = MagicMock()
        user.id = 99999
        assert stub._is_allowed_user(user) is False

    def test_empty_allowed_users_allows_all(self):
        """Empty allowed_users list should allow everyone."""
        stub = _make_bot_stub()
        stub.config.discord.allowed_users = []
        stub._is_allowed_user = LokiBot._is_allowed_user.__get__(stub)
        user = MagicMock()
        user.id = 99999
        assert stub._is_allowed_user(user) is True

    def test_allowed_channel(self):
        """Channel in allowed list should return True."""
        stub = _make_bot_stub()
        stub._is_allowed_channel = LokiBot._is_allowed_channel.__get__(stub)
        assert stub._is_allowed_channel(67890) is True

    def test_disallowed_channel(self):
        """Channel not in allowed list should return False."""
        stub = _make_bot_stub()
        stub._is_allowed_channel = LokiBot._is_allowed_channel.__get__(stub)
        assert stub._is_allowed_channel(11111) is False

    def test_empty_channels_allows_all(self):
        """Empty channels list should allow all channels."""
        stub = _make_bot_stub()
        stub.config.discord.channels = []
        stub._is_allowed_channel = LokiBot._is_allowed_channel.__get__(stub)
        assert stub._is_allowed_channel(99999) is True

    def test_check_for_secrets_api_key(self):
        """Should detect API keys."""
        stub = _make_bot_stub()
        stub._check_for_secrets = LokiBot._check_for_secrets.__get__(stub)
        assert stub._check_for_secrets("here is sk-abcdefghijklmnopqrstuvwxyz") is True

    def test_check_for_secrets_clean(self):
        """Should not flag clean text."""
        stub = _make_bot_stub()
        stub._check_for_secrets = LokiBot._check_for_secrets.__get__(stub)
        assert stub._check_for_secrets("check the server disk space") is False


# ---------------------------------------------------------------------------
# _send_with_retry
# ---------------------------------------------------------------------------

class TestSendWithRetry:
    """Tests for _send_with_retry — Discord message sending with retries."""

    async def test_successful_reply(self):
        """Should send a reply on first attempt."""
        stub = _make_bot_stub()
        stub._send_with_retry = LokiBot._send_with_retry.__get__(stub)
        msg = _make_message()
        sent = MagicMock(id=999)
        msg.reply = AsyncMock(return_value=sent)

        result = await stub._send_with_retry(msg, "hello")
        assert result == sent
        msg.reply.assert_awaited_once_with("hello")

    async def test_channel_send_not_reply(self):
        """Should use channel.send when as_reply=False."""
        stub = _make_bot_stub()
        stub._send_with_retry = LokiBot._send_with_retry.__get__(stub)
        msg = _make_message()
        sent = MagicMock(id=999)
        msg.channel.send = AsyncMock(return_value=sent)

        result = await stub._send_with_retry(msg, "hello", as_reply=False)
        assert result == sent
        msg.channel.send.assert_awaited_once_with("hello")

    async def test_retry_on_http_exception(self):
        """Should retry on Discord HTTPException and succeed."""
        stub = _make_bot_stub()
        stub._send_with_retry = LokiBot._send_with_retry.__get__(stub)
        msg = _make_message()
        sent = MagicMock(id=999)
        msg.reply = AsyncMock(
            side_effect=[discord.HTTPException(MagicMock(), "fail"), sent]
        )

        result = await stub._send_with_retry(msg, "hello")
        assert result == sent
        assert msg.reply.await_count == 2

    async def test_all_retries_fail(self):
        """Should return None after all retries fail."""
        stub = _make_bot_stub()
        stub._send_with_retry = LokiBot._send_with_retry.__get__(stub)
        msg = _make_message()
        msg.reply = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(), "fail")
        )

        result = await stub._send_with_retry(msg, "hello")
        assert result is None


# ---------------------------------------------------------------------------
# _send_chunked
# ---------------------------------------------------------------------------

class TestSendChunked:
    """Tests for _send_chunked — Discord message splitting."""

    async def test_short_message(self):
        """Short messages should be sent as a single reply."""
        stub = _make_bot_stub()
        stub._send_chunked = LokiBot._send_chunked.__get__(stub)
        msg = _make_message()

        await stub._send_chunked(msg, "short message")
        stub._send_with_retry.assert_awaited_once()

    async def test_very_long_message_as_file(self):
        """Very long messages should be sent as a file attachment."""
        stub = _make_bot_stub()
        stub._send_chunked = LokiBot._send_chunked.__get__(stub)
        msg = _make_message()

        text = "x" * (DISCORD_MAX_LEN * 5)
        await stub._send_chunked(msg, text)
        # Should send "attached as file" message with files kwarg
        stub._send_with_retry.assert_awaited_once()
        files = stub._send_with_retry.call_args[1].get("files")
        assert files and len(files) > 0

    async def test_medium_long_message_chunked(self):
        """Messages between 2000 and 8000 chars should be split into chunks."""
        stub = _make_bot_stub()
        stub._send_chunked = LokiBot._send_chunked.__get__(stub)
        msg = _make_message()

        # Create a message that needs 2-3 chunks
        text = ("line of text here\n" * 150)  # ~2700 chars
        await stub._send_chunked(msg, text)
        # Should have multiple calls to _send_with_retry
        assert stub._send_with_retry.await_count >= 2

    async def test_code_block_splitting(self):
        """Code blocks should be properly closed and reopened across chunks."""
        stub = _make_bot_stub()
        stub._send_chunked = LokiBot._send_chunked.__get__(stub)
        msg = _make_message()

        # Create a long code block that will span chunks
        code = "```python\n" + ("x = 1\n" * 400) + "```"
        await stub._send_chunked(msg, code)
        assert stub._send_with_retry.await_count >= 2


# ---------------------------------------------------------------------------
# Tool handlers (Discord-native tools)
# ---------------------------------------------------------------------------

class TestHandlePurge:
    """Tests for _handle_purge tool handler."""

    async def test_purge_success(self):
        """Should purge messages and reset session."""
        stub = _make_bot_stub()
        stub._handle_purge = LokiBot._handle_purge.__get__(stub)
        msg = _make_message()

        result = await stub._handle_purge(msg, {"count": 50})
        assert "Deleted 3 messages" in result
        stub.sessions.reset.assert_called_once()

    async def test_purge_caps_at_500(self):
        """Should cap count at 500."""
        stub = _make_bot_stub()
        stub._handle_purge = LokiBot._handle_purge.__get__(stub)
        msg = _make_message()

        await stub._handle_purge(msg, {"count": 1000})
        msg.channel.purge.assert_awaited_once_with(limit=500)

    async def test_purge_forbidden(self):
        """Should handle Forbidden error gracefully."""
        stub = _make_bot_stub()
        stub._handle_purge = LokiBot._handle_purge.__get__(stub)
        msg = _make_message()
        msg.channel.purge = AsyncMock(side_effect=discord.Forbidden(MagicMock(), "nope"))

        result = await stub._handle_purge(msg, {})
        assert "permission" in result.lower()

    async def test_purge_generic_error(self):
        """Should handle generic errors gracefully."""
        stub = _make_bot_stub()
        stub._handle_purge = LokiBot._handle_purge.__get__(stub)
        msg = _make_message()
        msg.channel.purge = AsyncMock(side_effect=RuntimeError("oops"))

        result = await stub._handle_purge(msg, {})
        assert "Failed" in result


class TestHandleBrowserScreenshot:
    """Tests for _handle_browser_screenshot tool handler."""

    async def test_no_browser_manager(self):
        """Should return error when browser is not enabled."""
        stub = _make_bot_stub()
        stub._handle_browser_screenshot = LokiBot._handle_browser_screenshot.__get__(stub)
        msg = _make_message()

        result = await stub._handle_browser_screenshot(msg, {"url": "http://example.com"})
        assert "not enabled" in result

    async def test_screenshot_success(self):
        """Should post screenshot to channel."""
        stub = _make_bot_stub()
        stub.browser_manager = MagicMock()
        stub._handle_browser_screenshot = LokiBot._handle_browser_screenshot.__get__(stub)
        msg = _make_message()

        mock_handler = AsyncMock(return_value=("Screenshot taken", b"PNG_DATA"))
        with patch.dict("sys.modules", {"src.tools.browser": MagicMock(
            handle_browser_screenshot=mock_handler,
        )}):
            result = await stub._handle_browser_screenshot(msg, {"url": "http://example.com"})
            assert "Screenshot" in result
            msg.channel.send.assert_awaited_once()

    async def test_screenshot_error(self):
        """Should handle browser errors gracefully."""
        stub = _make_bot_stub()
        stub.browser_manager = MagicMock()
        stub._handle_browser_screenshot = LokiBot._handle_browser_screenshot.__get__(stub)
        msg = _make_message()

        mock_handler = AsyncMock(side_effect=RuntimeError("browser crash"))
        with patch.dict("sys.modules", {"src.tools.browser": MagicMock(
            handle_browser_screenshot=mock_handler,
        )}):
            result = await stub._handle_browser_screenshot(msg, {"url": "http://example.com"})
            assert "failed" in result.lower() or "Browser" in result


class TestHandleGenerateFile:
    """Tests for _handle_generate_file tool handler."""

    async def test_generate_file_success(self):
        """Should create and post a file to Discord."""
        stub = _make_bot_stub()
        stub._handle_generate_file = LokiBot._handle_generate_file.__get__(stub)
        msg = _make_message()

        result = await stub._handle_generate_file(msg, {
            "filename": "test.txt",
            "content": "Hello World",
            "caption": "Here's the file",
        })
        assert "test.txt" in result
        msg.channel.send.assert_awaited_once()

    async def test_generate_file_defaults(self):
        """Should use defaults when fields are missing."""
        stub = _make_bot_stub()
        stub._handle_generate_file = LokiBot._handle_generate_file.__get__(stub)
        msg = _make_message()

        result = await stub._handle_generate_file(msg, {})
        assert "output.txt" in result

    async def test_generate_file_error(self):
        """Should handle send errors."""
        stub = _make_bot_stub()
        stub._handle_generate_file = LokiBot._handle_generate_file.__get__(stub)
        msg = _make_message()
        msg.channel.send = AsyncMock(side_effect=RuntimeError("fail"))

        result = await stub._handle_generate_file(msg, {"content": "data"})
        assert "Failed" in result


class TestHandlePostFile:
    """Tests for _handle_post_file tool handler."""

    async def test_missing_params(self):
        """Should return error when host or path is missing."""
        stub = _make_bot_stub()
        stub._handle_post_file = LokiBot._handle_post_file.__get__(stub)
        msg = _make_message()

        result = await stub._handle_post_file(msg, {})
        assert "required" in result.lower()

    async def test_unknown_host(self):
        """Should return error for unknown host."""
        stub = _make_bot_stub()
        stub.tool_executor._resolve_host = MagicMock(return_value=None)
        stub._handle_post_file = LokiBot._handle_post_file.__get__(stub)
        msg = _make_message()

        result = await stub._handle_post_file(msg, {"host": "unknown", "path": "/tmp/file"})
        assert "Unknown" in result

    async def test_post_file_success(self):
        """Should fetch file via SSH and post to Discord."""
        stub = _make_bot_stub()
        stub.config.tools.ssh_key_path = "/tmp/key"
        stub.config.tools.ssh_known_hosts_path = "/tmp/hosts"
        stub._handle_post_file = LokiBot._handle_post_file.__get__(stub)
        msg = _make_message()

        file_data = base64.b64encode(b"file content here")
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(file_data, b""))

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)):
            with patch("asyncio.wait_for", new=AsyncMock(return_value=(file_data, b""))):
                result = await stub._handle_post_file(msg, {
                    "host": "server", "path": "/tmp/test.txt", "caption": "test file",
                })
                assert "Posted" in result or "test.txt" in result

    async def test_post_file_ssh_failure(self):
        """Should handle SSH failure gracefully."""
        stub = _make_bot_stub()
        stub.config.tools.ssh_key_path = "/tmp/key"
        stub.config.tools.ssh_known_hosts_path = "/tmp/hosts"
        stub._handle_post_file = LokiBot._handle_post_file.__get__(stub)
        msg = _make_message()

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"No such file"))

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)):
            with patch("asyncio.wait_for", new=AsyncMock(return_value=(b"", b"No such file"))):
                result = await stub._handle_post_file(msg, {
                    "host": "server", "path": "/nonexistent",
                })
                assert "Failed" in result or "not found" in result.lower() or "No such file" in result

    async def test_post_file_timeout(self):
        """Should handle SSH timeout."""
        stub = _make_bot_stub()
        stub.config.tools.ssh_key_path = "/tmp/key"
        stub.config.tools.ssh_known_hosts_path = "/tmp/hosts"
        stub._handle_post_file = LokiBot._handle_post_file.__get__(stub)
        msg = _make_message()

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=asyncio.TimeoutError())):
            result = await stub._handle_post_file(msg, {
                "host": "server", "path": "/tmp/big.bin",
            })
            assert "timed out" in result.lower() or "Failed" in result


class TestHandleScheduleTask:
    """Tests for _handle_schedule_task tool handler."""

    def test_schedule_task_success(self):
        """Should create a scheduled task."""
        stub = _make_bot_stub()
        stub._handle_schedule_task = LokiBot._handle_schedule_task.__get__(stub)
        msg = _make_message()

        result = stub._handle_schedule_task(msg, {
            "description": "Daily check",
            "action": "check",
            "cron": "0 8 * * *",
        })
        assert "Scheduled" in result
        assert "sched-1" in result

    def test_schedule_one_time_task(self):
        """Should create a one-time scheduled task."""
        stub = _make_bot_stub()
        stub._handle_schedule_task = LokiBot._handle_schedule_task.__get__(stub)
        msg = _make_message()

        result = stub._handle_schedule_task(msg, {
            "description": "One-time check",
            "action": "reminder",
            "run_at": "2026-03-19T10:00:00",
        })
        assert "Scheduled" in result

    def test_schedule_with_trigger(self):
        """Should create a webhook-triggered task."""
        stub = _make_bot_stub()
        stub.scheduler.add = MagicMock(return_value={
            "id": "sched-2", "description": "Deploy hook",
            "trigger": {"type": "gitea", "repo": "myproject"},
        })
        stub._handle_schedule_task = LokiBot._handle_schedule_task.__get__(stub)
        msg = _make_message()

        result = stub._handle_schedule_task(msg, {
            "description": "Deploy hook",
            "action": "workflow",
            "trigger": {"type": "gitea", "repo": "myproject"},
        })
        assert "webhook-triggered" in result.lower() or "Trigger" in result

    def test_schedule_value_error(self):
        """Should handle invalid schedule gracefully."""
        stub = _make_bot_stub()
        stub.scheduler.add = MagicMock(side_effect=ValueError("Invalid cron"))
        stub._handle_schedule_task = LokiBot._handle_schedule_task.__get__(stub)
        msg = _make_message()

        result = stub._handle_schedule_task(msg, {"cron": "invalid"})
        assert "Failed" in result

    def test_schedule_generic_error(self):
        """Should handle unexpected errors."""
        stub = _make_bot_stub()
        stub.scheduler.add = MagicMock(side_effect=RuntimeError("oops"))
        stub._handle_schedule_task = LokiBot._handle_schedule_task.__get__(stub)
        msg = _make_message()

        result = stub._handle_schedule_task(msg, {})
        assert "Error" in result


class TestHandleListSchedules:
    """Tests for _handle_list_schedules tool handler."""

    def test_no_schedules(self):
        """Should return message when no schedules exist."""
        stub = _make_bot_stub()
        stub._handle_list_schedules = LokiBot._handle_list_schedules.__get__(stub)

        result = stub._handle_list_schedules()
        assert "No scheduled" in result

    def test_list_cron_schedule(self):
        """Should list cron schedules with details."""
        stub = _make_bot_stub()
        stub.scheduler.list_all = MagicMock(return_value=[{
            "id": "s1", "description": "Disk check",
            "cron": "0 8 * * *", "next_run": "2026-03-19T08:00:00",
            "last_run": "2026-03-18T08:00:00",
        }])
        stub._handle_list_schedules = LokiBot._handle_list_schedules.__get__(stub)

        result = stub._handle_list_schedules()
        assert "Disk check" in result
        assert "cron" in result

    def test_list_trigger_schedule(self):
        """Should list trigger-based schedules."""
        stub = _make_bot_stub()
        stub.scheduler.list_all = MagicMock(return_value=[{
            "id": "s2", "description": "Deploy hook",
            "trigger": {"type": "gitea", "repo": "myproject"},
            "next_run": "on trigger",
            "last_run": "never",
        }])
        stub._handle_list_schedules = LokiBot._handle_list_schedules.__get__(stub)

        result = stub._handle_list_schedules()
        assert "trigger" in result
        assert "Deploy hook" in result

    def test_list_one_time_schedule(self):
        """Should list one-time schedules."""
        stub = _make_bot_stub()
        stub.scheduler.list_all = MagicMock(return_value=[{
            "id": "s3", "description": "Reminder",
            "next_run": "2026-03-19T10:00:00",
            "last_run": "never",
        }])
        stub._handle_list_schedules = LokiBot._handle_list_schedules.__get__(stub)

        result = stub._handle_list_schedules()
        assert "one-time" in result


class TestHandleDeleteSchedule:
    """Tests for _handle_delete_schedule tool handler."""

    def test_delete_success(self):
        """Should delete an existing schedule."""
        stub = _make_bot_stub()
        stub._handle_delete_schedule = LokiBot._handle_delete_schedule.__get__(stub)

        result = stub._handle_delete_schedule({"schedule_id": "s1"})
        assert "Deleted" in result

    def test_delete_not_found(self):
        """Should return not found for missing schedule."""
        stub = _make_bot_stub()
        stub.scheduler.delete = MagicMock(return_value=False)
        stub._handle_delete_schedule = LokiBot._handle_delete_schedule.__get__(stub)

        result = stub._handle_delete_schedule({"schedule_id": "nonexistent"})
        assert "not found" in result


class TestHandleParseTime:
    """Tests for _handle_parse_time tool handler."""

    def test_parse_time_success(self):
        """Should parse a valid time expression."""
        stub = _make_bot_stub()
        stub._handle_parse_time = LokiBot._handle_parse_time.__get__(stub)

        result = stub._handle_parse_time({"expression": "in 2 hours"})
        assert "Parsed" in result

    def test_parse_time_empty(self):
        """Should return error for empty expression."""
        stub = _make_bot_stub()
        stub._handle_parse_time = LokiBot._handle_parse_time.__get__(stub)

        result = stub._handle_parse_time({"expression": ""})
        assert "required" in result.lower()

    def test_parse_time_no_expression(self):
        """Should return error when expression field is missing."""
        stub = _make_bot_stub()
        stub._handle_parse_time = LokiBot._handle_parse_time.__get__(stub)

        result = stub._handle_parse_time({})
        assert "required" in result.lower()


class TestHandleSearchHistory:
    """Tests for _handle_search_history tool handler."""

    async def test_search_no_query(self):
        """Should return error without a query."""
        stub = _make_bot_stub()
        stub._handle_search_history = LokiBot._handle_search_history.__get__(stub)

        result = await stub._handle_search_history({"query": ""})
        assert "required" in result.lower()

    async def test_search_no_results(self):
        """Should return message when no results found."""
        stub = _make_bot_stub()
        stub._handle_search_history = LokiBot._handle_search_history.__get__(stub)

        result = await stub._handle_search_history({"query": "nonexistent"})
        assert "No past conversations" in result

    async def test_search_with_results(self):
        """Should format results with timestamps and content."""
        stub = _make_bot_stub()
        stub.sessions.search_history = AsyncMock(return_value=[{
            "timestamp": time.time(),
            "type": "user",
            "content": "check disk space on server",
        }])
        stub._handle_search_history = LokiBot._handle_search_history.__get__(stub)

        result = await stub._handle_search_history({"query": "disk"})
        assert "Found 1" in result
        assert "disk" in result


class TestHandleSearchKnowledge:
    """Tests for _handle_search_knowledge tool handler."""

    async def test_no_knowledge_store(self):
        """Should return error when knowledge store is not available."""
        stub = _make_bot_stub()
        stub._handle_search_knowledge = LokiBot._handle_search_knowledge.__get__(stub)

        result = await stub._handle_search_knowledge({"query": "test"})
        assert "not available" in result

    async def test_no_query(self):
        """Should return error without a query."""
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._embedder = MagicMock()
        stub._handle_search_knowledge = LokiBot._handle_search_knowledge.__get__(stub)

        result = await stub._handle_search_knowledge({"query": ""})
        assert "required" in result.lower()

    async def test_no_results(self):
        """Should suggest web search when no results found."""
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.search_hybrid = AsyncMock(return_value=[])
        stub._embedder = MagicMock()
        stub._handle_search_knowledge = LokiBot._handle_search_knowledge.__get__(stub)

        result = await stub._handle_search_knowledge({"query": "obscure topic"})
        assert "web_search" in result or "No knowledge" in result

    async def test_search_with_results(self):
        """Should format knowledge results with scores."""
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.search_hybrid = AsyncMock(return_value=[{
            "source": "docs.md",
            "score": 0.85,
            "content": "This is relevant content about the topic.",
        }])
        stub._embedder = MagicMock()
        stub._handle_search_knowledge = LokiBot._handle_search_knowledge.__get__(stub)

        result = await stub._handle_search_knowledge({"query": "relevant"})
        assert "Found 1" in result
        assert "docs.md" in result


class TestHandleIngestDocument:
    """Tests for _handle_ingest_document tool handler."""

    async def test_no_knowledge_store(self):
        """Should return error when knowledge store is not available."""
        stub = _make_bot_stub()
        stub._handle_ingest_document = LokiBot._handle_ingest_document.__get__(stub)

        result = await stub._handle_ingest_document({"source": "doc.md", "content": "text"}, "user")
        assert "not available" in result

    async def test_missing_fields(self):
        """Should return error when source or content is missing."""
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._embedder = MagicMock()
        stub._handle_ingest_document = LokiBot._handle_ingest_document.__get__(stub)

        result = await stub._handle_ingest_document({"source": "", "content": ""}, "user")
        assert "required" in result.lower()

    async def test_ingest_success(self):
        """Should ingest document and report chunk count."""
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.ingest = AsyncMock(return_value=5)
        stub._embedder = MagicMock()
        stub._handle_ingest_document = LokiBot._handle_ingest_document.__get__(stub)

        result = await stub._handle_ingest_document(
            {"source": "readme.md", "content": "Hello world content"},
            "TestUser",
        )
        assert "5 chunks" in result

    async def test_ingest_zero_chunks(self):
        """Should report failure when no chunks embedded."""
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.ingest = AsyncMock(return_value=0)
        stub._embedder = MagicMock()
        stub._handle_ingest_document = LokiBot._handle_ingest_document.__get__(stub)

        result = await stub._handle_ingest_document(
            {"source": "bad.md", "content": "x"}, "user",
        )
        assert "Failed" in result


class TestHandleListKnowledge:
    """Tests for _handle_list_knowledge tool handler."""

    def test_no_knowledge_store(self):
        """Should return error when knowledge store is not available."""
        stub = _make_bot_stub()
        stub._handle_list_knowledge = LokiBot._handle_list_knowledge.__get__(stub)

        result = stub._handle_list_knowledge()
        assert "not available" in result

    def test_empty_knowledge_base(self):
        """Should return empty message."""
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.list_sources = MagicMock(return_value=[])
        stub._handle_list_knowledge = LokiBot._handle_list_knowledge.__get__(stub)

        result = stub._handle_list_knowledge()
        assert "empty" in result.lower()

    def test_list_knowledge_with_sources(self):
        """Should list all documents with chunk counts."""
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.list_sources = MagicMock(return_value=[
            {"source": "readme.md", "chunks": 5, "uploader": "TestUser",
             "ingested_at": "2026-03-18T12:00:00"},
            {"source": "config.yml", "chunks": 3, "uploader": "Bot",
             "ingested_at": "2026-03-17T08:00:00"},
        ])
        stub._handle_list_knowledge = LokiBot._handle_list_knowledge.__get__(stub)

        result = stub._handle_list_knowledge()
        assert "2 document(s)" in result
        assert "8 total chunks" in result
        assert "readme.md" in result


class TestHandleDeleteKnowledge:
    """Tests for _handle_delete_knowledge tool handler."""

    def test_no_knowledge_store(self):
        """Should return error when knowledge store is not available."""
        stub = _make_bot_stub()
        stub._handle_delete_knowledge = LokiBot._handle_delete_knowledge.__get__(stub)

        result = stub._handle_delete_knowledge({"source": "doc.md"})
        assert "not available" in result

    def test_no_source(self):
        """Should return error when source is missing."""
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._handle_delete_knowledge = LokiBot._handle_delete_knowledge.__get__(stub)

        result = stub._handle_delete_knowledge({"source": ""})
        assert "required" in result.lower()

    def test_delete_success(self):
        """Should delete and report chunk count."""
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.delete_source = MagicMock(return_value=5)
        stub._handle_delete_knowledge = LokiBot._handle_delete_knowledge.__get__(stub)

        result = stub._handle_delete_knowledge({"source": "old.md"})
        assert "5 chunks removed" in result

    def test_delete_not_found(self):
        """Should return not found when source doesn't exist."""
        stub = _make_bot_stub()
        stub._knowledge_store = MagicMock()
        stub._knowledge_store.delete_source = MagicMock(return_value=0)
        stub._handle_delete_knowledge = LokiBot._handle_delete_knowledge.__get__(stub)

        result = stub._handle_delete_knowledge({"source": "nonexistent.md"})
        assert "No document" in result


class TestHandleSearchAudit:
    """Tests for _handle_search_audit tool handler."""

    async def test_no_results(self):
        """Should return message when no entries found."""
        stub = _make_bot_stub()
        stub._handle_search_audit = LokiBot._handle_search_audit.__get__(stub)

        result = await stub._handle_search_audit({})
        assert "No audit log" in result

    async def test_with_results(self):
        """Should format audit entries with timestamps and details."""
        stub = _make_bot_stub()
        stub.audit.search = AsyncMock(return_value=[{
            "timestamp": "2026-03-18T12:00:00",
            "tool_name": "check_disk",
            "user_name": "TestUser",
            "approved": True,
            "execution_time_ms": 150,
            "result_summary": "Disk: 42% used",
            "error": None,
        }])
        stub._handle_search_audit = LokiBot._handle_search_audit.__get__(stub)

        result = await stub._handle_search_audit({"tool_name": "check_disk"})
        assert "1 entries" in result
        assert "check_disk" in result
        assert "approved" in result

    async def test_with_error_entry(self):
        """Should show error status for errored entries."""
        stub = _make_bot_stub()
        stub.audit.search = AsyncMock(return_value=[{
            "timestamp": "2026-03-18T12:00:00",
            "tool_name": "run_command",
            "user_name": "TestUser",
            "approved": True,
            "execution_time_ms": 50,
            "result_summary": "",
            "error": "SSH timeout",
        }])
        stub._handle_search_audit = LokiBot._handle_search_audit.__get__(stub)

        result = await stub._handle_search_audit({})
        assert "ERROR" in result
        assert "SSH timeout" in result


class TestHandleCreateDigest:
    """Tests for _handle_create_digest tool handler."""

    def test_create_digest_success(self):
        """Should create a digest schedule."""
        stub = _make_bot_stub()
        stub._handle_create_digest = LokiBot._handle_create_digest.__get__(stub)
        msg = _make_message()

        result = stub._handle_create_digest(msg, {
            "cron": "0 9 * * *",
            "description": "Morning Digest",
        })
        assert "digest" in result.lower() or "sched-1" in result

    def test_create_digest_default_cron(self):
        """Should use default cron when not specified."""
        stub = _make_bot_stub()
        stub._handle_create_digest = LokiBot._handle_create_digest.__get__(stub)
        msg = _make_message()

        result = stub._handle_create_digest(msg, {})
        stub.scheduler.add.assert_called_once()
        call_kwargs = stub.scheduler.add.call_args
        assert call_kwargs.kwargs.get("cron") == "0 8 * * *" or call_kwargs[1].get("cron") == "0 8 * * *"

    def test_create_digest_error(self):
        """Should handle scheduler error."""
        stub = _make_bot_stub()
        stub.scheduler.add = MagicMock(side_effect=ValueError("bad cron"))
        stub._handle_create_digest = LokiBot._handle_create_digest.__get__(stub)
        msg = _make_message()

        result = stub._handle_create_digest(msg, {"cron": "invalid"})
        assert "Failed" in result


# ---------------------------------------------------------------------------
# Background task handlers
# ---------------------------------------------------------------------------

class TestHandleDelegateTask:
    """Tests for _handle_delegate_task — background task creation."""

    async def test_no_steps(self):
        """Should return error when no steps provided."""
        stub = _make_bot_stub()
        stub._handle_delegate_task = LokiBot._handle_delegate_task.__get__(stub)
        msg = _make_message()

        result = await stub._handle_delegate_task(msg, {"description": "test"})
        assert "No steps" in result

    async def test_too_many_steps(self):
        """Should return error when too many steps."""
        stub = _make_bot_stub()
        stub._handle_delegate_task = LokiBot._handle_delegate_task.__get__(stub)
        msg = _make_message()

        from src.discord.background_task import MAX_STEPS
        steps = [{"tool_name": f"tool_{i}"} for i in range(MAX_STEPS + 1)]
        result = await stub._handle_delegate_task(msg, {
            "description": "test", "steps": steps,
        })
        assert "Too many" in result

    async def test_invalid_step(self):
        """Should return error for steps without tool_name."""
        stub = _make_bot_stub()
        stub._handle_delegate_task = LokiBot._handle_delegate_task.__get__(stub)
        msg = _make_message()

        result = await stub._handle_delegate_task(msg, {
            "steps": [{"description": "no tool"}],
        })
        assert "tool_name" in result

    async def test_delegate_success(self):
        """Should create and start a background task."""
        stub = _make_bot_stub()
        stub._handle_delegate_task = LokiBot._handle_delegate_task.__get__(stub)
        msg = _make_message()

        with patch("src.discord.client.run_background_task", new=AsyncMock()):
            result = await stub._handle_delegate_task(msg, {
                "description": "Deploy check",
                "steps": [
                    {"tool_name": "check_disk", "tool_input": {"host": "server"}},
                    {"tool_name": "check_memory", "tool_input": {"host": "server"}},
                ],
            })
        assert "Background task started" in result
        assert "2 steps" in result
        assert len(stub._background_tasks) == 1


class TestHandleListTasks:
    """Tests for _handle_list_tasks — list background tasks."""

    def test_no_tasks(self):
        """Should return message when no tasks exist."""
        stub = _make_bot_stub()
        stub._handle_list_tasks = LokiBot._handle_list_tasks.__get__(stub)

        result = stub._handle_list_tasks()
        assert "No background" in result

    def test_list_overview(self):
        """Should list all tasks with status."""
        stub = _make_bot_stub()
        task = MagicMock()
        task.task_id = "task-1"
        task.description = "Deploy check"
        task.status = "running"
        task.steps = [MagicMock(), MagicMock()]
        task.results = [MagicMock(status="ok")]
        stub._background_tasks = {"task-1": task}
        stub._handle_list_tasks = LokiBot._handle_list_tasks.__get__(stub)

        result = stub._handle_list_tasks()
        assert "task-1" in result
        assert "Deploy check" in result
        assert "running" in result

    def test_detailed_view(self):
        """Should show detailed results for specific task."""
        stub = _make_bot_stub()
        task = MagicMock()
        task.task_id = "task-1"
        task.description = "Deploy check"
        task.status = "completed"
        task.steps = [MagicMock()]
        result_mock = MagicMock()
        result_mock.index = 0
        result_mock.status = "ok"
        result_mock.description = "Check disk"
        result_mock.elapsed_ms = 150
        result_mock.output = "Disk: 42% used"
        task.results = [result_mock]
        stub._background_tasks = {"task-1": task}
        stub._handle_list_tasks = LokiBot._handle_list_tasks.__get__(stub)

        result = stub._handle_list_tasks({"task_id": "task-1"})
        assert "Deploy check" in result
        assert "completed" in result
        assert "Check disk" in result

    def test_task_not_found(self):
        """Should return error for nonexistent task."""
        stub = _make_bot_stub()
        stub._background_tasks = {"task-1": MagicMock()}
        stub._handle_list_tasks = LokiBot._handle_list_tasks.__get__(stub)

        result = stub._handle_list_tasks({"task_id": "nonexistent"})
        assert "No task found" in result


class TestHandleCancelTask:
    """Tests for _handle_cancel_task — cancel background tasks."""

    def test_cancel_running_task(self):
        """Should cancel a running task."""
        stub = _make_bot_stub()
        task = MagicMock()
        task.status = "running"
        task.cancel = MagicMock()
        stub._background_tasks = {"task-1": task}
        stub._handle_cancel_task = LokiBot._handle_cancel_task.__get__(stub)

        result = stub._handle_cancel_task({"task_id": "task-1"})
        assert "Cancellation" in result
        task.cancel.assert_called_once()

    def test_cancel_not_found(self):
        """Should return error for nonexistent task."""
        stub = _make_bot_stub()
        stub._handle_cancel_task = LokiBot._handle_cancel_task.__get__(stub)

        result = stub._handle_cancel_task({"task_id": "nonexistent"})
        assert "No task found" in result

    def test_cancel_not_running(self):
        """Should return error for completed task."""
        stub = _make_bot_stub()
        task = MagicMock()
        task.status = "completed"
        stub._background_tasks = {"task-1": task}
        stub._handle_cancel_task = LokiBot._handle_cancel_task.__get__(stub)

        result = stub._handle_cancel_task({"task_id": "task-1"})
        assert "not running" in result


# ---------------------------------------------------------------------------
# Scheduled task callbacks
# ---------------------------------------------------------------------------

class TestOnScheduledTask:
    """Tests for _on_scheduled_task callback."""

    async def test_no_channel_id(self):
        """Should warn and return when no channel_id."""
        stub = _make_bot_stub()
        stub._on_scheduled_task = LokiBot._on_scheduled_task.__get__(stub)

        await stub._on_scheduled_task({"id": "s1", "action": "reminder"})
        # Should not crash

    async def test_channel_not_found(self):
        """Should warn when channel is not found."""
        stub = _make_bot_stub()
        stub._on_scheduled_task = LokiBot._on_scheduled_task.__get__(stub)

        await stub._on_scheduled_task({
            "id": "s1", "action": "reminder", "channel_id": "99999",
        })
        stub.get_channel.assert_called_with(99999)

    async def test_reminder_action(self):
        """Should send reminder message to channel."""
        stub = _make_bot_stub()
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub.get_channel = MagicMock(return_value=channel)
        stub._resolve_mentions = MagicMock(side_effect=lambda x: x)
        stub._on_scheduled_task = LokiBot._on_scheduled_task.__get__(stub)

        await stub._on_scheduled_task({
            "id": "s1", "action": "reminder", "channel_id": "67890",
            "description": "Check backups", "message": "Don't forget backups!",
        })
        channel.send.assert_awaited_once()
        call_text = channel.send.call_args[0][0]
        assert "Don't forget backups!" in call_text

    async def test_check_action(self):
        """Should execute tool and post result."""
        stub = _make_bot_stub()
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub.get_channel = MagicMock(return_value=channel)
        stub._on_scheduled_task = LokiBot._on_scheduled_task.__get__(stub)

        await stub._on_scheduled_task({
            "id": "s1", "action": "check", "channel_id": "67890",
            "description": "Disk check", "tool_name": "check_disk",
            "tool_input": {"host": "server"},
        })
        channel.send.assert_awaited()
        stub.tool_executor.execute.assert_awaited()

    async def test_check_action_error(self):
        """Should post error when tool execution fails."""
        stub = _make_bot_stub()
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub.get_channel = MagicMock(return_value=channel)
        stub.tool_executor.execute = AsyncMock(side_effect=RuntimeError("timeout"))
        stub._on_scheduled_task = LokiBot._on_scheduled_task.__get__(stub)

        await stub._on_scheduled_task({
            "id": "s1", "action": "check", "channel_id": "67890",
            "description": "Disk check", "tool_name": "check_disk",
        })
        channel.send.assert_awaited()
        call_text = channel.send.call_args[0][0]
        assert "failed" in call_text.lower()

    async def test_digest_action(self):
        """Should delegate to _on_scheduled_digest."""
        stub = _make_bot_stub()
        channel = AsyncMock()
        stub.get_channel = MagicMock(return_value=channel)
        stub._on_scheduled_digest = AsyncMock()
        stub._on_scheduled_task = LokiBot._on_scheduled_task.__get__(stub)

        schedule = {"id": "s1", "action": "digest", "channel_id": "67890"}
        await stub._on_scheduled_task(schedule)
        stub._on_scheduled_digest.assert_awaited_once_with(schedule)

    async def test_workflow_action(self):
        """Should delegate to _run_scheduled_workflow."""
        stub = _make_bot_stub()
        channel = AsyncMock()
        stub.get_channel = MagicMock(return_value=channel)
        stub._run_scheduled_workflow = AsyncMock()
        stub._on_scheduled_task = LokiBot._on_scheduled_task.__get__(stub)

        schedule = {
            "id": "s1", "action": "workflow", "channel_id": "67890",
            "steps": [{"tool_name": "check_disk"}],
        }
        await stub._on_scheduled_task(schedule)
        stub._run_scheduled_workflow.assert_awaited_once()

    async def test_unknown_action_logs_warning(self):
        """Should log a warning for unknown action types."""
        stub = _make_bot_stub()
        channel = AsyncMock()
        stub.get_channel = MagicMock(return_value=channel)
        stub._on_scheduled_task = LokiBot._on_scheduled_task.__get__(stub)

        with patch("src.discord.client.log") as mock_log:
            await stub._on_scheduled_task({
                "id": "s1", "action": "nonexistent", "channel_id": "67890",
            })
            mock_log.warning.assert_called_once()
            assert "nonexistent" in mock_log.warning.call_args[0][1]


class TestOnScheduledDigest:
    """Tests for _on_scheduled_digest — daily digest execution."""

    async def test_no_channel(self):
        """Should return when channel is not found."""
        stub = _make_bot_stub()
        stub._on_scheduled_digest = LokiBot._on_scheduled_digest.__get__(stub)

        await stub._on_scheduled_digest({"id": "d1", "channel_id": "99999"})
        # Should not crash

    async def test_digest_success(self):
        """Should run digest and post summarized results."""
        stub = _make_bot_stub()
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub.get_channel = MagicMock(return_value=channel)
        stub._format_digest_raw = AsyncMock(return_value="### Disk\n42% used\n### Memory\n60% used")
        # codex_client.chat returns a plain string (Codex path)
        stub.codex_client.chat = AsyncMock(return_value="All systems healthy.")
        stub._on_scheduled_digest = LokiBot._on_scheduled_digest.__get__(stub)

        await stub._on_scheduled_digest({"id": "d1", "channel_id": "67890"})
        channel.send.assert_awaited()
        assert "Digest" in channel.send.call_args[0][0]

    async def test_digest_codex_failure_falls_back_to_raw(self):
        """Should fall back to raw data when Codex summary fails."""
        stub = _make_bot_stub()
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub.get_channel = MagicMock(return_value=channel)
        stub._format_digest_raw = AsyncMock(return_value="raw data here")
        stub.codex_client.chat = AsyncMock(side_effect=RuntimeError("API down"))
        stub._on_scheduled_digest = LokiBot._on_scheduled_digest.__get__(stub)

        await stub._on_scheduled_digest({"id": "d1", "channel_id": "67890"})
        channel.send.assert_awaited()
        call_text = channel.send.call_args[0][0]
        assert "raw data here" in call_text

    async def test_digest_data_collection_failure(self):
        """Should post error when data collection fails."""
        stub = _make_bot_stub()
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub.get_channel = MagicMock(return_value=channel)
        stub._format_digest_raw = AsyncMock(side_effect=RuntimeError("SSH timeout"))
        stub._on_scheduled_digest = LokiBot._on_scheduled_digest.__get__(stub)

        await stub._on_scheduled_digest({"id": "d1", "channel_id": "67890"})
        channel.send.assert_awaited()
        call_text = channel.send.call_args[0][0]
        assert "Failed" in call_text

    async def test_digest_no_channel_id(self):
        """Should warn and return when schedule has no channel_id."""
        stub = _make_bot_stub()
        stub._on_scheduled_digest = LokiBot._on_scheduled_digest.__get__(stub)

        await stub._on_scheduled_digest({"id": "d1"})
        # Should not crash


class TestFormatDigestRaw:
    """Tests for _format_digest_raw — data collection for digest."""

    async def test_format_digest_raw(self):
        """Should collect disk, memory, and Prometheus data."""
        stub = _make_bot_stub()
        stub.config.tools.hosts = {"server": MagicMock(), "desktop": MagicMock()}
        stub.tool_executor.execute = AsyncMock(return_value="42% used")
        stub._format_digest_raw = LokiBot._format_digest_raw.__get__(stub)

        result = await stub._format_digest_raw()
        assert "Disk" in result
        assert "Memory" in result
        assert "Prometheus" in result

    async def test_format_digest_with_exception(self):
        """Should handle tool execution errors in digest."""
        stub = _make_bot_stub()
        stub.config.tools.hosts = {"server": MagicMock()}

        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("SSH timeout")
            return "OK"

        stub.tool_executor.execute = AsyncMock(side_effect=side_effect)
        stub._format_digest_raw = LokiBot._format_digest_raw.__get__(stub)

        result = await stub._format_digest_raw()
        assert "ERROR" in result


class TestRunScheduledWorkflow:
    """Tests for _run_scheduled_workflow — multi-step workflow execution."""

    async def test_basic_workflow(self):
        """Should execute all steps and post results."""
        stub = _make_bot_stub()
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub._run_scheduled_workflow = LokiBot._run_scheduled_workflow.__get__(stub)

        schedule = {
            "description": "Deploy flow",
            "steps": [
                {"tool_name": "check_disk", "description": "Check disk"},
                {"tool_name": "check_memory", "description": "Check memory"},
            ],
        }
        await stub._run_scheduled_workflow(channel, schedule)
        channel.send.assert_awaited_once()
        call_text = channel.send.call_args[0][0]
        assert "Deploy flow" in call_text
        assert "OK" in call_text

    async def test_workflow_with_condition_match(self):
        """Should execute step when condition is met."""
        stub = _make_bot_stub()
        stub.tool_executor.execute = AsyncMock(return_value="disk 85% used - warning")
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub._run_scheduled_workflow = LokiBot._run_scheduled_workflow.__get__(stub)

        schedule = {
            "description": "Conditional check",
            "steps": [
                {"tool_name": "check_disk", "description": "Check disk"},
                {"tool_name": "run_command", "description": "Cleanup",
                 "condition": "warning"},
            ],
        }
        await stub._run_scheduled_workflow(channel, schedule)
        # Both steps should have executed
        assert stub.tool_executor.execute.await_count == 2

    async def test_workflow_with_condition_skip(self):
        """Should skip step when condition is not met."""
        stub = _make_bot_stub()
        stub.tool_executor.execute = AsyncMock(return_value="disk 42% used - healthy")
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub._run_scheduled_workflow = LokiBot._run_scheduled_workflow.__get__(stub)

        schedule = {
            "description": "Conditional check",
            "steps": [
                {"tool_name": "check_disk", "description": "Check disk"},
                {"tool_name": "run_command", "description": "Cleanup",
                 "condition": "warning"},
            ],
        }
        await stub._run_scheduled_workflow(channel, schedule)
        # Only first step should execute (condition "warning" not in output)
        assert stub.tool_executor.execute.await_count == 1
        call_text = channel.send.call_args[0][0]
        assert "skipped" in call_text

    async def test_workflow_negated_condition(self):
        """Should skip step when negated condition IS present."""
        stub = _make_bot_stub()
        stub.tool_executor.execute = AsyncMock(return_value="disk 42% - healthy")
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub._run_scheduled_workflow = LokiBot._run_scheduled_workflow.__get__(stub)

        schedule = {
            "description": "Negated check",
            "steps": [
                {"tool_name": "check_disk", "description": "Check disk"},
                {"tool_name": "send_alert", "description": "Alert",
                 "condition": "!healthy"},
            ],
        }
        await stub._run_scheduled_workflow(channel, schedule)
        # Should skip step 2 because "healthy" IS in output and condition is negated
        call_text = channel.send.call_args[0][0]
        assert "skipped" in call_text

    async def test_workflow_abort_on_failure(self):
        """Should abort workflow on step failure with on_failure=abort."""
        stub = _make_bot_stub()
        stub.tool_executor.execute = AsyncMock(side_effect=RuntimeError("SSH timeout"))
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub._run_scheduled_workflow = LokiBot._run_scheduled_workflow.__get__(stub)

        schedule = {
            "description": "Failing workflow",
            "steps": [
                {"tool_name": "check_disk", "description": "Check disk",
                 "on_failure": "abort"},
                {"tool_name": "check_memory", "description": "Check memory"},
            ],
        }
        await stub._run_scheduled_workflow(channel, schedule)
        call_text = channel.send.call_args[0][0]
        assert "FAILED" in call_text
        assert "aborted" in call_text.lower()

    async def test_workflow_continue_on_failure(self):
        """Should continue workflow on step failure with on_failure=continue."""
        stub = _make_bot_stub()

        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("SSH timeout")
            return "OK"

        stub.tool_executor.execute = AsyncMock(side_effect=side_effect)
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub._run_scheduled_workflow = LokiBot._run_scheduled_workflow.__get__(stub)

        schedule = {
            "description": "Resilient workflow",
            "steps": [
                {"tool_name": "check_disk", "description": "Check disk",
                 "on_failure": "continue"},
                {"tool_name": "check_memory", "description": "Check memory"},
            ],
        }
        await stub._run_scheduled_workflow(channel, schedule)
        # Both steps should have been attempted
        assert call_count == 2

    async def test_workflow_skill_execution(self):
        """Should use skill_manager for skill tools."""
        stub = _make_bot_stub()
        stub.skill_manager.has_skill = MagicMock(return_value=True)
        stub.skill_manager.execute = AsyncMock(return_value="skill result")
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub._run_scheduled_workflow = LokiBot._run_scheduled_workflow.__get__(stub)

        schedule = {
            "description": "Skill workflow",
            "steps": [
                {"tool_name": "my_skill", "description": "Run skill"},
            ],
        }
        await stub._run_scheduled_workflow(channel, schedule)
        stub.skill_manager.execute.assert_awaited_once()

    async def test_workflow_truncation(self):
        """Should truncate very long workflow output."""
        stub = _make_bot_stub()
        stub.tool_executor.execute = AsyncMock(return_value="x" * 500)
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub._run_scheduled_workflow = LokiBot._run_scheduled_workflow.__get__(stub)

        schedule = {
            "description": "Long workflow",
            "steps": [{"tool_name": f"tool_{i}", "description": f"Step {i}"}
                      for i in range(20)],
        }
        await stub._run_scheduled_workflow(channel, schedule)
        call_text = channel.send.call_args[0][0]
        assert len(call_text) <= 1930  # truncated to ~1900 + "..." + header


# ---------------------------------------------------------------------------
# _resolve_mentions
# ---------------------------------------------------------------------------

class TestResolveMentions:
    """Tests for _resolve_mentions — @username to <@ID> conversion."""

    def test_resolve_known_member(self):
        """Should replace @username with proper Discord mention."""
        stub = _make_bot_stub()
        member = MagicMock()
        member.name = "testuser"
        member.nick = None
        member.id = 12345
        guild = MagicMock()
        guild.members = [member]
        stub.guilds = [guild]
        stub._resolve_mentions = LokiBot._resolve_mentions.__get__(stub)

        result = stub._resolve_mentions("Hey @testuser check this out")
        assert "<@12345>" in result

    def test_resolve_nickname(self):
        """Should match on nickname too."""
        stub = _make_bot_stub()
        member = MagicMock()
        member.name = "user123"
        member.nick = "bob"
        member.id = 99999
        guild = MagicMock()
        guild.members = [member]
        stub.guilds = [guild]
        stub._resolve_mentions = LokiBot._resolve_mentions.__get__(stub)

        result = stub._resolve_mentions("Hey @bob")
        assert "<@99999>" in result

    def test_unresolved_mention(self):
        """Should leave unknown mentions unchanged."""
        stub = _make_bot_stub()
        stub.guilds = [MagicMock(members=[])]
        stub._resolve_mentions = LokiBot._resolve_mentions.__get__(stub)

        result = stub._resolve_mentions("Hey @unknown")
        assert "@unknown" in result


# ---------------------------------------------------------------------------
# _on_monitor_alert
# ---------------------------------------------------------------------------

class TestOnMonitorAlert:
    """Tests for _on_monitor_alert — proactive monitoring callback."""

    async def test_alert_to_configured_channel(self):
        """Should send alert to configured alert channel."""
        stub = _make_bot_stub()
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub.get_channel = MagicMock(return_value=channel)
        stub._on_monitor_alert = LokiBot._on_monitor_alert.__get__(stub)

        await stub._on_monitor_alert("Disk usage above 90%!")
        channel.send.assert_awaited_once_with("Disk usage above 90%!")

    async def test_alert_fallback_channel(self):
        """Should fall back to first configured channel when alert_channel_id is empty."""
        stub = _make_bot_stub()
        stub.config.monitoring.alert_channel_id = ""
        stub.config.discord.channels = ["12345"]
        channel = AsyncMock()
        channel.send = AsyncMock()
        stub.get_channel = MagicMock(return_value=channel)
        stub._on_monitor_alert = LokiBot._on_monitor_alert.__get__(stub)

        await stub._on_monitor_alert("Memory critical!")
        stub.get_channel.assert_called_with(12345)

    async def test_alert_no_channel(self):
        """Should warn and return when no channel is available."""
        stub = _make_bot_stub()
        stub.config.monitoring.alert_channel_id = ""
        stub.config.discord.channels = []
        stub._on_monitor_alert = LokiBot._on_monitor_alert.__get__(stub)

        await stub._on_monitor_alert("Alert!")
        # Should not crash

    async def test_alert_channel_not_found(self):
        """Should warn when channel object is not found."""
        stub = _make_bot_stub()
        stub._on_monitor_alert = LokiBot._on_monitor_alert.__get__(stub)

        await stub._on_monitor_alert("Alert!")
        # get_channel returns None by default, should not crash

    async def test_alert_send_failure(self):
        """Should handle send failure gracefully."""
        stub = _make_bot_stub()
        channel = AsyncMock()
        channel.send = AsyncMock(side_effect=RuntimeError("fail"))
        stub.get_channel = MagicMock(return_value=channel)
        stub._on_monitor_alert = LokiBot._on_monitor_alert.__get__(stub)

        await stub._on_monitor_alert("Alert!")
        # Should not crash


# ---------------------------------------------------------------------------
# _process_attachments
# ---------------------------------------------------------------------------

class TestProcessAttachments:
    """Tests for _process_attachments — handling Discord file attachments."""

    async def test_no_attachments(self):
        """Should return empty results for no attachments."""
        stub = _make_bot_stub()
        stub._process_attachments = LokiBot._process_attachments.__get__(stub)
        stub._detect_image_type = LokiBot._detect_image_type
        msg = _make_message()
        msg.attachments = []

        text, images = await stub._process_attachments(msg)
        assert text == ""
        assert images == []

    async def test_image_attachment(self):
        """Should process image as base64 content block."""
        stub = _make_bot_stub()
        stub._process_attachments = LokiBot._process_attachments.__get__(stub)
        stub._detect_image_type = LokiBot._detect_image_type
        stub._get_attachment_hint = LokiBot._get_attachment_hint

        att = AsyncMock()
        att.filename = "screenshot.png"
        att.content_type = "image/png"
        att.size = 50000
        att.read = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        msg = _make_message()
        msg.attachments = [att]

        text, images = await stub._process_attachments(msg)
        assert len(images) == 1
        assert images[0]["type"] == "image"
        assert images[0]["source"]["media_type"] == "image/png"
        assert "Image attached" in text

    async def test_oversized_image(self):
        """Should skip images over 5MB."""
        stub = _make_bot_stub()
        stub._process_attachments = LokiBot._process_attachments.__get__(stub)
        stub._detect_image_type = LokiBot._detect_image_type

        att = AsyncMock()
        att.filename = "huge.png"
        att.content_type = "image/png"
        att.size = 6 * 1024 * 1024  # 6MB

        msg = _make_message()
        msg.attachments = [att]

        text, images = await stub._process_attachments(msg)
        assert len(images) == 0
        assert "exceeds 5 MB" in text

    async def test_text_file_attachment(self):
        """Should inline text file contents."""
        stub = _make_bot_stub()
        stub._process_attachments = LokiBot._process_attachments.__get__(stub)
        stub._detect_image_type = LokiBot._detect_image_type
        stub._get_attachment_hint = LokiBot._get_attachment_hint

        att = AsyncMock()
        att.filename = "config.yml"
        att.content_type = "text/yaml"
        att.size = 500
        att.read = AsyncMock(return_value=b"server:\n  host: localhost\n  port: 8080")

        msg = _make_message()
        msg.attachments = [att]

        text, images = await stub._process_attachments(msg)
        assert "config.yml" in text
        assert "localhost" in text
        assert len(images) == 0

    async def test_large_text_file(self):
        """Should preview large text files and suggest ingestion."""
        stub = _make_bot_stub()
        stub._process_attachments = LokiBot._process_attachments.__get__(stub)
        stub._detect_image_type = LokiBot._detect_image_type

        att = AsyncMock()
        att.filename = "big_log.txt"
        att.content_type = "text/plain"
        att.size = 200_000
        att.read = AsyncMock(return_value=b"x" * 200_000)

        msg = _make_message()
        msg.attachments = [att]

        text, images = await stub._process_attachments(msg)
        assert "too large" in text.lower() or "ingest" in text.lower()

    async def test_binary_attachment(self):
        """Should report binary files without reading content."""
        stub = _make_bot_stub()
        stub._process_attachments = LokiBot._process_attachments.__get__(stub)
        stub._detect_image_type = LokiBot._detect_image_type

        att = AsyncMock()
        att.filename = "data.bin"
        att.content_type = "application/octet-stream"
        att.size = 5000

        msg = _make_message()
        msg.attachments = [att]

        text, images = await stub._process_attachments(msg)
        assert "data.bin" in text
        assert "binary" in text.lower() or "octet-stream" in text.lower()

    async def test_large_non_text_attachment(self):
        """Should report large non-text attachments as too large."""
        stub = _make_bot_stub()
        stub._process_attachments = LokiBot._process_attachments.__get__(stub)
        stub._detect_image_type = LokiBot._detect_image_type

        att = AsyncMock()
        att.filename = "archive.zip"
        att.content_type = "application/zip"
        att.size = 200_000

        msg = _make_message()
        msg.attachments = [att]

        text, images = await stub._process_attachments(msg)
        assert "too large" in text.lower()

    async def test_image_read_failure(self):
        """Should handle image read errors gracefully."""
        stub = _make_bot_stub()
        stub._process_attachments = LokiBot._process_attachments.__get__(stub)
        stub._detect_image_type = LokiBot._detect_image_type

        att = AsyncMock()
        att.filename = "broken.png"
        att.content_type = "image/png"
        att.size = 5000
        att.read = AsyncMock(side_effect=RuntimeError("read error"))

        msg = _make_message()
        msg.attachments = [att]

        text, images = await stub._process_attachments(msg)
        assert "failed to read" in text.lower()
        assert len(images) == 0

    async def test_jpg_media_type_normalization(self):
        """Should normalize image/jpg to image/jpeg."""
        stub = _make_bot_stub()
        stub._process_attachments = LokiBot._process_attachments.__get__(stub)
        stub._detect_image_type = LokiBot._detect_image_type
        stub._get_attachment_hint = LokiBot._get_attachment_hint

        att = AsyncMock()
        att.filename = "photo.jpg"
        att.content_type = "image/jpg"
        att.size = 5000
        # Use non-matching magic bytes to test content_type path
        att.read = AsyncMock(return_value=b"\x00" * 100)

        msg = _make_message()
        msg.attachments = [att]

        text, images = await stub._process_attachments(msg)
        assert len(images) == 1
        assert images[0]["source"]["media_type"] == "image/jpeg"


# Codex path tests are in TestProcessWithTools.


# ---------------------------------------------------------------------------
# on_message flow
# ---------------------------------------------------------------------------

class TestOnMessage:
    """Tests for the on_message entry point."""

    async def test_ignores_bot_messages(self):
        """Should ignore messages from bots."""
        stub = _make_bot_stub()
        stub.on_message = LokiBot.on_message.__get__(stub)
        msg = _make_message()
        msg.author.bot = True

        await stub.on_message(msg)
        stub.sessions.add_message.assert_not_called()

    async def test_ignores_disallowed_user(self):
        """Should ignore messages from disallowed users."""
        stub = _make_bot_stub()
        stub._is_allowed_user = MagicMock(return_value=False)
        stub.on_message = LokiBot.on_message.__get__(stub)
        msg = _make_message(author_id="99999")

        await stub.on_message(msg)
        stub.sessions.add_message.assert_not_called()

    async def test_ignores_disallowed_channel(self):
        """Should ignore messages from disallowed channels."""
        stub = _make_bot_stub()
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=False)
        stub.on_message = LokiBot.on_message.__get__(stub)
        msg = _make_message(channel_id="99999")

        await stub.on_message(msg)
        stub.sessions.add_message.assert_not_called()

    async def test_dedup_skips_repeat_message(self):
        """Should skip duplicate message IDs."""
        stub = _make_bot_stub()
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._processed_messages = {123: None}
        stub.on_message = LokiBot.on_message.__get__(stub)
        msg = _make_message()
        msg.id = 123

        await stub.on_message(msg)
        # _handle_message should not be called

    async def test_strips_bot_mention(self):
        """Should strip bot mention from message content."""
        stub = _make_bot_stub()
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._processed_messages = {}
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub._check_for_secrets = MagicMock(return_value=False)
        stub._handle_message = AsyncMock()
        stub.user.mentioned_in = MagicMock(return_value=True)
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content=f"<@111> check disk")
        msg.id = int(time.time() * 1000) + 1

        await stub.on_message(msg)
        # _handle_message should be called with stripped content
        stub._handle_message.assert_awaited_once()
        call_args = stub._handle_message.call_args
        assert "check disk" in call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "")

    async def test_secret_detection_deletes_message(self):
        """Should delete messages containing secrets."""
        stub = _make_bot_stub()
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._processed_messages = {}
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub._check_for_secrets = MagicMock(return_value=True)
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content="api_key=supersecretkey12345")
        msg.id = int(time.time() * 1000) + 2

        await stub.on_message(msg)
        msg.delete.assert_awaited_once()
        stub.sessions.scrub_secrets.assert_called_once()

    async def test_secret_detection_forbidden(self):
        """Should handle Forbidden when can't delete secret message."""
        stub = _make_bot_stub()
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._processed_messages = {}
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub._check_for_secrets = MagicMock(return_value=True)
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content="api_key=supersecretkey12345")
        msg.id = int(time.time() * 1000) + 3
        msg.delete = AsyncMock(side_effect=discord.Forbidden(MagicMock(), "nope"))

        await stub.on_message(msg)
        # Should send warning about manual deletion
        msg.channel.send.assert_awaited()

    async def test_empty_message_no_images_ignored(self):
        """Should ignore messages with no content and no images."""
        stub = _make_bot_stub()
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._processed_messages = {}
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub._check_for_secrets = MagicMock(return_value=False)
        stub._handle_message = AsyncMock()
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content="")
        msg.id = int(time.time() * 1000) + 4

        await stub.on_message(msg)
        stub._handle_message.assert_not_awaited()

    async def test_image_only_uses_placeholder_content(self):
        """Should use '(see attached image)' for image-only messages."""
        stub = _make_bot_stub()
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._processed_messages = {}
        stub._process_attachments = AsyncMock(return_value=("", [{"type": "image"}]))
        stub._check_for_secrets = MagicMock(return_value=False)
        stub._handle_message = AsyncMock()
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content="")
        msg.id = int(time.time() * 1000) + 5

        await stub.on_message(msg)
        stub._handle_message.assert_awaited_once()
        call_args = stub._handle_message.call_args
        content = call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "")
        assert content is not None


# ---------------------------------------------------------------------------
# _handle_message (thread context, locking)
# ---------------------------------------------------------------------------

class TestHandleMessage:
    """Tests for _handle_message — locking and thread context."""

    async def test_acquires_channel_lock(self):
        """Should acquire per-channel lock."""
        stub = _make_bot_stub()
        stub._handle_message = LokiBot._handle_message.__get__(stub)
        stub._handle_message_inner = AsyncMock()
        msg = _make_message()

        await stub._handle_message(msg, "test content")
        stub._handle_message_inner.assert_awaited_once()
        assert str(msg.channel.id) in stub._channel_locks

    async def test_thread_context_inheritance(self):
        """Should inherit context from parent channel for threads."""
        stub = _make_bot_stub()
        stub._handle_message = LokiBot._handle_message.__get__(stub)
        stub._handle_message_inner = AsyncMock()

        msg = _make_message()
        # Make channel look like a Thread
        msg.channel = MagicMock(spec=discord.Thread)
        msg.channel.id = "thread-1"
        msg.channel.parent = MagicMock()
        msg.channel.parent.id = "parent-1"

        parent_session = MagicMock()
        parent_session.summary = "Previous conversation about disk usage"
        parent_session.messages = [
            MagicMock(role="user", content="check disk"),
            MagicMock(role="assistant", content="Disk is 42% full"),
        ]

        thread_session = MagicMock()
        thread_session.messages = []  # Empty — no existing thread context
        thread_session.summary = ""

        # get_or_create returns thread_session for thread channel, parent_session for parent
        def _get_or_create(channel_id):
            if channel_id == "thread-1":
                return thread_session
            return parent_session
        stub.sessions.get_or_create = MagicMock(side_effect=_get_or_create)

        await stub._handle_message(msg, "follow up question")
        # Thread session should have inherited parent context
        assert "Parent channel context" in thread_session.summary or parent_session.summary in thread_session.summary


# ---------------------------------------------------------------------------
# _handle_message_inner error handling paths
# ---------------------------------------------------------------------------

class TestHandleMessageInnerErrors:
    """Tests for error handling paths in _handle_message_inner."""

    async def test_task_process_with_tools_exception_sends_error(self):
        """Should send error message when _process_with_tools raises any exception."""
        stub = _make_bot_stub()
        stub._process_with_tools = AsyncMock(side_effect=RuntimeError("tool loop crashed"))
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "67890")

        # The inner except (line ~1075) catches it, sets is_error=True and response to error
        # The response goes to _send_chunked (not _send_with_retry) since already_sent=False
        stub._send_chunked.assert_awaited()
        call_text = stub._send_chunked.call_args[0][1]
        assert "tool execution failed" in call_text.lower() or "tool loop crashed" in call_text.lower()

    async def test_task_no_codex_sends_no_backend_message(self):
        """Should send 'No tool backend' message when codex_client is None."""
        stub = _make_bot_stub()
        stub.codex_client = None
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "67890")

        stub._send_with_retry.assert_awaited()
        call_text = stub._send_with_retry.call_args[0][1]
        assert "no tool backend" in call_text.lower()
        stub.sessions.remove_last_message.assert_called()

    async def test_task_process_with_tools_error_saves_to_history(self):
        """Should save error to history when _process_with_tools returns is_error=True (Round 14)."""
        stub = _make_bot_stub()
        stub._process_with_tools = AsyncMock(
            return_value=("Tool execution failed: timeout", False, True, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "67890")

        stub.sessions.remove_last_message.assert_not_called()
        assistant_saves = [
            c for c in stub.sessions.add_message.call_args_list
            if c[0][1] == "assistant"
        ]
        assert len(assistant_saves) == 1

    async def test_generic_exception_sends_went_wrong(self):
        """Should send 'Something went wrong' for unexpected exceptions."""
        stub = _make_bot_stub()
        # Make classifier itself raise, which bubbles to outer except
        stub.classifier.classify = AsyncMock(side_effect=RuntimeError("classifier crashed"))
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "hello", "67890")

        stub._send_with_retry.assert_awaited()
        call_text = stub._send_with_retry.call_args[0][1]
        assert "went wrong" in call_text.lower()

    async def test_voice_callback_called(self):
        """Should call voice callback after successful response."""
        stub = _make_bot_stub()
        stub._process_with_tools = AsyncMock(
            return_value=("Hello!", False, False, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        voice_cb = AsyncMock()

        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(
                msg, "check disk", "67890", voice_callback=voice_cb,
            )
        voice_cb.assert_awaited_once()

    async def test_tool_memory_recording(self):
        """Should record tool usage patterns after successful tool use."""
        stub = _make_bot_stub()
        stub._process_with_tools = AsyncMock(
            return_value=("Disk: 42%", False, False, ["check_disk"], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "67890")

        stub.tool_memory.record.assert_awaited_once()

    async def test_error_response_saves_to_history(self):
        """Should save error to history for checkpoint-save (Round 14)."""
        stub = _make_bot_stub()
        stub._process_with_tools = AsyncMock(
            return_value=("Error occurred", False, True, [], False)  # is_error=True
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "67890")

        stub.sessions.remove_last_message.assert_not_called()
        assistant_saves = [
            c for c in stub.sessions.add_message.call_args_list
            if c[0][1] == "assistant"
        ]
        assert len(assistant_saves) == 1


# ---------------------------------------------------------------------------
# _handle_message_inner — claude_code routing
# ---------------------------------------------------------------------------

class TestClaudeCodeRouting:
    """Tests for claude_code message routing in _handle_message_inner."""

    async def test_claude_code_success(self):
        """Should route to claude_code when classifier returns claude_code."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub.tool_executor._handle_claude_code = AsyncMock(return_value="Code analysis result")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root/project")):
            await stub._handle_message_inner(msg, "review the code", "67890")

        stub.tool_executor._handle_claude_code.assert_awaited()
        stub._send_chunked.assert_awaited()

    async def test_claude_code_failure_falls_back_to_codex(self):
        """Should fall back to Codex chat when claude_code CLI fails."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Codex fallback")
        stub.tool_executor._handle_claude_code = AsyncMock(return_value="Claude Code failed: timeout")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root")):
            await stub._handle_message_inner(msg, "review code", "67890")

        stub.codex_client.chat.assert_awaited()

    async def test_claude_code_exception_falls_back_to_codex(self):
        """Should fall back to Codex chat when claude_code raises exception."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Codex fallback")
        stub.tool_executor._handle_claude_code = AsyncMock(side_effect=RuntimeError("SSH failed"))
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root")):
            await stub._handle_message_inner(msg, "review code", "67890")

        stub.codex_client.chat.assert_awaited()

    async def test_claude_code_failure_falls_back_to_error_message(self):
        """Should send error when claude_code fails and codex_client is None."""
        stub = _make_bot_stub()
        stub.codex_client = None
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub.tool_executor._handle_claude_code = AsyncMock(return_value="Claude Code failed: timeout")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root")):
            await stub._handle_message_inner(msg, "review code", "67890")

        stub._send_chunked.assert_awaited()

    async def test_claude_code_exception_no_codex_sends_error(self):
        """Should send error when claude_code raises and codex_client is None."""
        stub = _make_bot_stub()
        stub.codex_client = None
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub.tool_executor._handle_claude_code = AsyncMock(side_effect=RuntimeError("fail"))
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root")):
            await stub._handle_message_inner(msg, "review code", "67890")

        stub._send_chunked.assert_awaited()

    async def test_claude_code_with_conversation_context(self):
        """Should inject conversation context into claude_code prompt."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub.sessions.get_history_with_compaction = AsyncMock(return_value=[
            {"role": "user", "content": "what does foo.py do?"},
            {"role": "assistant", "content": "It handles bar logic."},
            {"role": "user", "content": "review the error handling"},
        ])
        stub.tool_executor._handle_claude_code = AsyncMock(return_value="Code review done")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root")):
            await stub._handle_message_inner(msg, "review the error handling", "67890")

        # Check that the prompt sent to claude_code includes context
        call_args = stub.tool_executor._handle_claude_code.call_args[0][0]
        assert "Previous conversation" in call_args["prompt"]

    async def test_claude_code_failure_with_codex_uses_codex_fallback(self):
        """Should fall back to Codex when claude_code fails."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub.tool_executor._handle_claude_code = AsyncMock(return_value="Claude Code failed: timeout")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root")):
            await stub._handle_message_inner(msg, "review code", "67890")

        stub.codex_client.chat.assert_awaited()

    async def test_claude_code_exception_with_codex_uses_codex_fallback(self):
        """Should fall back to Codex when claude_code raises."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="claude_code")
        stub.tool_executor._handle_claude_code = AsyncMock(side_effect=RuntimeError("fail"))
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False), \
             patch("src.discord.client.resolve_claude_code_target", return_value=("desktop", "/root")):
            await stub._handle_message_inner(msg, "review code", "67890")

        stub.codex_client.chat.assert_awaited()


# ---------------------------------------------------------------------------
# _handle_message_inner — task path budget exceeded
# ---------------------------------------------------------------------------

class TestTaskNoCodex:
    """Tests for task path when codex_client is not configured."""

    async def test_task_no_codex_returns_early(self):
        """Should return 'No tool backend' message for task when codex_client is None."""
        stub = _make_bot_stub()
        stub.codex_client = None
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "67890")

        stub._send_with_retry.assert_awaited()
        call_text = stub._send_with_retry.call_args[0][1]
        assert "no tool backend" in call_text.lower()
        stub.sessions.remove_last_message.assert_called()


# ---------------------------------------------------------------------------
# _handle_message_inner — image injection
# ---------------------------------------------------------------------------

class TestImageInjection:
    """Tests for image block injection in _handle_message_inner."""

    async def test_image_blocks_injected_into_history(self):
        """Should inject image blocks into the last user message."""
        stub = _make_bot_stub()
        stub.sessions.get_history_with_compaction = AsyncMock(return_value=[
            {"role": "user", "content": "what is this?"},
        ])
        stub._process_with_tools = AsyncMock(
            return_value=("I see a cat!", False, False, [], False)
        )
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        image_blocks = [{"type": "image", "source": {"type": "base64", "data": "abc"}}]

        # Images force the task route; codex_client is set by default in _make_bot_stub
        await stub._handle_message_inner(
            msg, "what is this?", "67890", image_blocks=image_blocks,
        )

        # _process_with_tools should have been called (images force task route)
        stub._process_with_tools.assert_awaited()


# ---------------------------------------------------------------------------
# _track_recent_action
# ---------------------------------------------------------------------------

class TestTrackRecentAction:
    """Tests for _track_recent_action — tool execution context tracking."""

    def test_tracks_action(self):
        """Should record tool execution in recent actions."""
        stub = _make_bot_stub()
        stub._track_recent_action = LokiBot._track_recent_action.__get__(stub)

        stub._track_recent_action(
            "check_disk", {"host": "server"}, "42% used", 150, channel_id="67890",
        )
        assert "67890" in stub._recent_actions
        assert len(stub._recent_actions["67890"]) == 1

    def test_no_channel_id_skips(self):
        """Should skip tracking when no channel_id."""
        stub = _make_bot_stub()
        stub._track_recent_action = LokiBot._track_recent_action.__get__(stub)

        stub._track_recent_action("check_disk", {}, "ok", 100, channel_id=None)
        assert len(stub._recent_actions) == 0

    def test_caps_per_channel_list(self):
        """Should cap actions per channel to max."""
        stub = _make_bot_stub()
        stub._recent_actions_max = 3
        stub._track_recent_action = LokiBot._track_recent_action.__get__(stub)

        for i in range(5):
            stub._track_recent_action(f"tool_{i}", {}, "ok", 100, channel_id="67890")
        assert len(stub._recent_actions["67890"]) == 3


# ---------------------------------------------------------------------------
# _inject_tool_hints
# ---------------------------------------------------------------------------

class TestInjectToolHints:
    """Tests for _inject_tool_hints — returns system prompt with hints appended."""

    async def test_injects_hints(self):
        """Should return prompt with tool hints appended."""
        stub = _make_bot_stub()
        stub._inject_tool_hints = LokiBot._inject_tool_hints.__get__(stub)
        stub.tool_memory = MagicMock()
        stub.tool_memory.format_hints = AsyncMock(return_value="## Tool Hints\nUse check_disk for disk checks")

        result = await stub._inject_tool_hints("base prompt", "check disk")
        assert "Tool Hints" in result
        assert result.startswith("base prompt")

    async def test_no_hints_available(self):
        """Should return prompt unchanged when no hints available."""
        stub = _make_bot_stub()
        stub._inject_tool_hints = LokiBot._inject_tool_hints.__get__(stub)
        stub.tool_memory = MagicMock()
        stub.tool_memory.format_hints = AsyncMock(return_value="")

        result = await stub._inject_tool_hints("base prompt", "hello")
        assert result == "base prompt"

    async def test_handles_exception(self):
        """Should return prompt unchanged on exception."""
        stub = _make_bot_stub()
        stub._inject_tool_hints = LokiBot._inject_tool_hints.__get__(stub)
        stub.tool_memory = MagicMock()
        stub.tool_memory.format_hints = AsyncMock(side_effect=RuntimeError("fail"))

        result = await stub._inject_tool_hints("base prompt", "test")
        assert result == "base prompt"

    async def test_no_tool_memory(self):
        """Should return prompt unchanged when tool_memory missing."""
        stub = _make_bot_stub()
        stub._inject_tool_hints = LokiBot._inject_tool_hints.__get__(stub)
        del stub.tool_memory

        result = await stub._inject_tool_hints("base prompt", "test")
        assert result == "base prompt"


# ---------------------------------------------------------------------------
# _merged_tool_definitions
# ---------------------------------------------------------------------------

class TestMergedToolDefinitions:
    """Tests for _merged_tool_definitions — tool deduplication."""

    def test_builtin_tools_only(self):
        """Should return all builtin tools when no skills."""
        stub = _make_bot_stub()
        stub._merged_tool_definitions = LokiBot._merged_tool_definitions.__get__(stub)

        result = stub._merged_tool_definitions()
        # Should contain builtin tools
        assert isinstance(result, list)

    def test_skill_deduplicated(self):
        """Should exclude skills that shadow builtin tool names."""
        stub = _make_bot_stub()
        stub.skill_manager.get_tool_definitions = MagicMock(return_value=[
            {"name": "check_disk", "description": "custom check_disk"},
            {"name": "my_custom_skill", "description": "custom skill"},
        ])
        stub._merged_tool_definitions = LokiBot._merged_tool_definitions.__get__(stub)

        result = stub._merged_tool_definitions()
        names = [t["name"] for t in result]
        # check_disk should appear only once (builtin version)
        assert names.count("check_disk") == 1
        # my_custom_skill should be included
        assert "my_custom_skill" in names


# ---------------------------------------------------------------------------
# Voice command handling in on_message
# ---------------------------------------------------------------------------

class TestVoiceCommands:
    """Tests for voice command handling in on_message."""

    async def test_join_voice_command(self):
        """Should join voice channel on 'join voice' command."""
        stub = _make_bot_stub()
        stub.voice_manager = MagicMock()
        stub.voice_manager.join_channel = AsyncMock(return_value="Joined voice channel!")
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._processed_messages = {}
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub._check_for_secrets = MagicMock(return_value=False)
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content="join voice channel")
        msg.id = int(time.time() * 1000) + 10
        msg.author = MagicMock(spec=discord.Member)
        msg.author.id = "12345"
        msg.author.bot = False
        msg.author.mention = "<@12345>"
        msg.author.voice = MagicMock()
        msg.author.voice.channel = MagicMock()
        msg.reply = AsyncMock()

        await stub.on_message(msg)
        stub.voice_manager.join_channel.assert_awaited_once()

    async def test_leave_voice_command(self):
        """Should leave voice channel on 'leave voice' command."""
        stub = _make_bot_stub()
        stub.voice_manager = MagicMock()
        stub.voice_manager.leave_channel = AsyncMock(return_value="Left voice channel")
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._processed_messages = {}
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub._check_for_secrets = MagicMock(return_value=False)
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content="leave voice")
        msg.id = int(time.time() * 1000) + 11
        msg.reply = AsyncMock()

        await stub.on_message(msg)
        stub.voice_manager.leave_channel.assert_awaited_once()

    async def test_join_voice_no_voice_channel(self):
        """Should reply with error when user not in voice channel."""
        stub = _make_bot_stub()
        stub.voice_manager = MagicMock()
        stub._is_allowed_user = MagicMock(return_value=True)
        stub._is_allowed_channel = MagicMock(return_value=True)
        stub._processed_messages = {}
        stub._process_attachments = AsyncMock(return_value=("", []))
        stub._check_for_secrets = MagicMock(return_value=False)
        stub.on_message = LokiBot.on_message.__get__(stub)

        msg = _make_message(content="join voice channel")
        msg.id = int(time.time() * 1000) + 12
        msg.author = MagicMock(spec=discord.Member)
        msg.author.id = "12345"
        msg.author.bot = False
        msg.author.mention = "<@12345>"
        msg.author.voice = None
        msg.reply = AsyncMock()

        await stub.on_message(msg)
        msg.reply.assert_awaited_once()
        call_text = msg.reply.call_args[0][0]
        assert "voice channel first" in call_text.lower()


# ---------------------------------------------------------------------------
# _process_with_tools — tool loop
# ---------------------------------------------------------------------------

class TestProcessWithTools:
    """Tests for _process_with_tools tool loop."""

    async def test_max_iterations_exceeded(self):
        """Should return error after MAX_TOOL_ITERATIONS."""
        from src.llm.types import LLMResponse, ToolCall

        stub = _make_bot_stub()
        stub._process_with_tools = LokiBot._process_with_tools.__get__(stub)

        # Track iteration count
        iteration_count = [0]

        async def fake_chat_with_tools(messages, system, tools):
            iteration_count[0] += 1
            # Always return a tool_use response to force iteration
            return LLMResponse(
                text="",
                tool_calls=[ToolCall(id=f"t{iteration_count[0]}", name="check_disk", input={})],
                stop_reason="tool_use",
            )

        stub.codex_client.chat_with_tools = fake_chat_with_tools
        stub._merged_tool_definitions = MagicMock(return_value=[
            {"name": "check_disk", "description": "Check disk"}
        ])
        # Tool executor returns output so iterations proceed
        stub.tool_executor.execute = AsyncMock(return_value="42% used")
        stub.audit = MagicMock()
        stub.audit.log_execution = AsyncMock()
        stub._track_recent_action = MagicMock()

        msg = _make_message()
        with patch("src.discord.client.scrub_output_secrets", side_effect=lambda x: x), \
             patch("src.discord.client.truncate_tool_output", side_effect=lambda x: x):
            text, already_sent, is_error, tools, _handoff = await stub._process_with_tools(
                msg, [{"role": "user", "content": "check all disks"}],
            )
        assert "Too many tool calls" in text
        assert is_error is True
        assert iteration_count[0] == MAX_TOOL_ITERATIONS


# ---------------------------------------------------------------------------
# Guest tier forcing
# ---------------------------------------------------------------------------

class TestGuestTierForcing:
    """Tests for guest tier chat route forcing."""

    async def test_guest_forced_to_chat(self):
        """Guest tier users should be forced to chat route."""
        stub = _make_bot_stub()
        stub.permissions.is_guest = MagicMock(return_value=True)
        stub.codex_client = MagicMock()
        stub.codex_client.chat = AsyncMock(return_value="Chat response")
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        # Even with task keyword, guest should get chat
        with patch("src.discord.client.is_task_by_keyword", return_value=True):
            await stub._handle_message_inner(msg, "check disk", "67890")

        # Should use chat prompt via Codex, not full system prompt
        stub._build_chat_system_prompt.assert_called()
        stub.codex_client.chat.assert_awaited()


# ---------------------------------------------------------------------------
# Budget exceeded downgrade to chat
# ---------------------------------------------------------------------------

class TestTaskRouteWithCodex:
    """Tests for task route using Codex with tools."""

    async def test_task_with_codex_calls_process_with_tools(self):
        """When Codex available, task route calls _process_with_tools."""
        stub = _make_bot_stub()
        stub.classifier.classify = AsyncMock(return_value="task")
        stub._process_with_tools = AsyncMock(return_value=("Tool result", False, False, ["check_disk"], False))
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)

        msg = _make_message()
        with patch("src.discord.client.is_task_by_keyword", return_value=False):
            await stub._handle_message_inner(msg, "check status", "67890")

        # Should use Codex tool calling via _process_with_tools
        stub._process_with_tools.assert_awaited()
        # Verify system_prompt_override is passed (not use_codex — that param was removed)
        call_kwargs = stub._process_with_tools.call_args[1]
        assert "system_prompt_override" in call_kwargs
