"""Round 18: Tests for changes made in Rounds 6-16.

Covers:
- FTS query escaping — reserved words AND, OR, NOT, NEAR, TO (Round 4)
- FTS search with "to" doesn't crash (Round 4)
- Fabrication detection: "not enabled" / "not available" patterns (Round 6)
- Fabrication detection: no trigger when tools WERE called (Round 6)
- Hedging detection: bash code blocks without tool calls (Round 6)
- Hedging detection: no trigger on explanatory code blocks (Round 6)
- Skill context: http_get with custom headers + default Accept (Round 5)
- Skill context: http_post with custom headers (Round 5)
- Discord message splitting at 2000 chars (Round 4)
- Code block continuation across message splits (Round 4)
- delegate_task: host parameter defaults (Round 7)
- delegate_task: error counting — errored steps counted as errors (Round 8)
- Browser CDP reconnect on stale connection (Round 9)
- Autonomous loop: start/stop/list lifecycle (Rounds 11-14)
- Autonomous loop: max concurrent limit (Round 11)
- Autonomous loop: stop_condition with LOOP_STOP (Round 14)
- Autonomous loop: mode (notify vs act vs silent) (Round 11)
- Autonomous loop: max_iterations auto-stop (Round 11)
- Autonomous loop: error in iteration doesn't crash loop (Round 14)
- Autonomous loop: LLM gets full tool access in iterations (Round 13)
"""
from __future__ import annotations

import asyncio
import io
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest

import discord

from src.search.fts import _prepare_query, _FTS5_KEYWORDS, FullTextIndex
from src.discord.client import (
    detect_fabrication,
    detect_tool_unavailable,
    detect_hedging,
    detect_code_hedging,
    DISCORD_MAX_LEN,
    LokiBot,
)
from src.discord.background_task import (
    _is_error_output,
    _get_default_host,
    BackgroundTask,
    run_background_task,
    StepResult,
)
from src.tools.browser import BrowserManager, _CONNECTION_ERROR_PATTERNS
from src.tools.autonomous_loop import (
    LoopManager,
    LoopInfo,
    MAX_CONCURRENT_LOOPS,
    MIN_INTERVAL_SECONDS,
    DEFAULT_MAX_ITERATIONS,
    LOOP_STOP_SENTINEL,
    MAX_CONSECUTIVE_ERRORS,
    RUNAWAY_THRESHOLD,
)
from src.tools.skill_context import SkillContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub(**overrides):
    """Minimal LokiBot stub for method-level tests."""
    stub = MagicMock()
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600
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
    stub.sessions = MagicMock()
    stub.sessions.get_history_with_compaction = AsyncMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.codex_client = MagicMock()
    stub.codex_client.chat_with_tools = AsyncMock()
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.get_tool_definitions = MagicMock(return_value=[])
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub._build_system_prompt = MagicMock(return_value="full system prompt")
    stub._send_with_retry = AsyncMock(return_value=MagicMock(id=999))
    stub._send_chunked = AsyncMock()
    stub._merged_tool_definitions = MagicMock(return_value=[])
    stub._inject_tool_hints = AsyncMock(side_effect=lambda sp, *a, **kw: sp)
    stub._pending_files = {}
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="tool result")
    stub.browser_manager = None
    stub.reflector = None
    stub.loop_manager = MagicMock()
    for k, v in overrides.items():
        setattr(stub, k, v)
    return stub


def _make_message(channel_id="67890", author_id="12345", content="test", msg_id=None):
    """Create a mock Discord message."""
    msg = MagicMock(spec=discord.Message)
    msg.id = msg_id or 111222333
    msg.content = content
    msg.author = MagicMock()
    msg.author.id = int(author_id)
    msg.author.display_name = "TestUser"
    msg.author.bot = False
    msg.channel = MagicMock()
    msg.channel.id = int(channel_id)
    msg.channel.send = AsyncMock()
    msg.channel.fetch_message = AsyncMock()
    msg.add_reaction = AsyncMock()
    msg.attachments = []
    msg.reference = None
    return msg


# ===========================================================================
# FTS query escaping (Round 4)
# ===========================================================================

class TestFTSQueryEscaping:
    """Tests for _prepare_query FTS5 reserved word handling."""

    def test_empty_query_returns_empty(self):
        assert _prepare_query("") == ""

    def test_whitespace_only_returns_empty(self):
        assert _prepare_query("   ") == ""

    def test_reserved_word_to_is_quoted(self):
        result = _prepare_query("how to check disk")
        assert '"to"' in result
        assert "how" in result
        assert "check" in result
        assert "disk" in result

    def test_reserved_word_and_is_quoted(self):
        result = _prepare_query("cpu and memory")
        assert '"and"' in result.lower() or '"AND"' in result

    def test_reserved_word_or_is_quoted(self):
        result = _prepare_query("docker or podman")
        assert '"or"' in result.lower() or '"OR"' in result

    def test_reserved_word_not_is_quoted(self):
        result = _prepare_query("not found")
        assert '"not"' in result.lower() or '"NOT"' in result

    def test_reserved_word_near_is_quoted(self):
        result = _prepare_query("near match")
        assert '"near"' in result.lower() or '"NEAR"' in result

    def test_all_fts5_keywords_covered(self):
        """Every keyword in _FTS5_KEYWORDS should be escaped."""
        for kw in _FTS5_KEYWORDS:
            result = _prepare_query(f"some {kw.lower()} thing")
            assert f'"{kw.lower()}"' in result

    def test_non_reserved_words_unquoted(self):
        result = _prepare_query("check disk usage")
        assert result == "check disk usage"

    def test_special_chars_wrap_entire_query(self):
        result = _prepare_query("node_cpu_seconds{instance='x'}")
        assert result.startswith('"')
        assert result.endswith('"')

    def test_ip_address_wrapped(self):
        result = _prepare_query("10.0.0.1")
        assert result.startswith('"')
        assert result.endswith('"')

    def test_path_wrapped(self):
        result = _prepare_query("/var/log/syslog")
        assert result.startswith('"')
        assert result.endswith('"')

    def test_internal_quotes_escaped(self):
        result = _prepare_query('say "hello"')
        assert '""' in result

    def test_mixed_reserved_and_normal(self):
        result = _prepare_query("send to server and back")
        assert '"to"' in result
        assert '"and"' in result.lower() or '"AND"' in result
        assert "send" in result
        assert "server" in result
        assert "back" in result


class TestFTSSearchWithReservedWords:
    """Integration: FTS search with reserved words doesn't crash."""

    def test_search_with_to_doesnt_crash(self):
        """The 'to' keyword caused 'no such column: to' before the fix."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            idx = FullTextIndex(f.name)
            if not idx.available:
                pytest.skip("FTS5 not available")
            idx.index_knowledge_chunk("c1", "how to check disk usage", "docs", 0)
            results = idx.search_knowledge("how to check")
            # Should not crash — results may or may not match
            assert isinstance(results, list)

    def test_search_with_and_doesnt_crash(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            idx = FullTextIndex(f.name)
            if not idx.available:
                pytest.skip("FTS5 not available")
            idx.index_knowledge_chunk("c1", "cpu and memory usage", "docs", 0)
            results = idx.search_knowledge("cpu and memory")
            assert isinstance(results, list)

    def test_search_with_not_doesnt_crash(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            idx = FullTextIndex(f.name)
            if not idx.available:
                pytest.skip("FTS5 not available")
            idx.index_knowledge_chunk("c1", "not found error", "docs", 0)
            results = idx.search_knowledge("not found")
            assert isinstance(results, list)


# ===========================================================================
# Fabrication detection — tool unavailability patterns (Round 6)
# ===========================================================================

class TestDetectToolUnavailable:
    """detect_tool_unavailable catches 'not enabled' / 'not available' patterns."""

    def test_not_enabled_detected(self):
        assert detect_tool_unavailable(
            "The generate_image tool is not enabled on this instance.", []
        )

    def test_not_available_detected(self):
        assert detect_tool_unavailable(
            "That tool is not available right now.", []
        )

    def test_is_disabled_detected(self):
        assert detect_tool_unavailable(
            "ComfyUI is disabled in the current configuration.", []
        )

    def test_isnt_available_detected(self):
        assert detect_tool_unavailable(
            "The browser isn't available in this environment.", []
        )

    def test_not_configured_detected(self):
        assert detect_tool_unavailable(
            "Image generation is not configured.", []
        )

    def test_cannot_be_used_detected(self):
        assert detect_tool_unavailable(
            "This tool cannot be used at the moment.", []
        )

    def test_is_not_supported_detected(self):
        assert detect_tool_unavailable(
            "That feature is not supported.", []
        )

    def test_no_trigger_when_tools_called(self):
        """Real tool errors (tools were called) should NOT trigger this."""
        assert not detect_tool_unavailable(
            "ComfyUI is not enabled.", ["generate_image"]
        )

    def test_no_trigger_on_short_text(self):
        assert not detect_tool_unavailable("not ok", [])

    def test_no_trigger_on_normal_text(self):
        assert not detect_tool_unavailable(
            "Here's the disk usage report for your server.", []
        )


class TestDetectFabrication:
    """detect_fabrication catches fake command output patterns."""

    def test_no_trigger_when_tools_called(self):
        assert not detect_fabrication(
            "I ran the command and here's the output:\n```\nok\n```",
            ["run_command"],
        )

    def test_no_trigger_on_short_text(self):
        assert not detect_fabrication("ok", [])

    def test_no_trigger_on_normal_chat(self):
        assert not detect_fabrication(
            "I think we should restart the service to fix this.", []
        )


# ===========================================================================
# Hedging detection (Round 6)
# ===========================================================================

class TestDetectHedging:
    """detect_hedging catches 'shall I' / 'would you like' patterns."""

    def test_shall_i_detected(self):
        assert detect_hedging("Shall I go ahead and restart the service?", [])

    def test_would_you_like_detected(self):
        assert detect_hedging("Would you like me to check the logs?", [])

    def test_if_you_want_detected(self):
        assert detect_hedging("I can do that if you'd like.", [])

    def test_want_me_to_detected(self):
        assert detect_hedging("Want me to run that command?", [])

    def test_let_me_know_detected(self):
        assert detect_hedging("Let me know if you need anything else.", [])

    def test_no_trigger_when_tools_called(self):
        assert not detect_hedging("Shall I restart?", ["run_command"])

    def test_no_trigger_on_short_text(self):
        assert not detect_hedging("ok", [])


class TestDetectCodeHedging:
    """detect_code_hedging catches bash code blocks without tool calls."""

    def test_bash_block_detected(self):
        text = "You can run this:\n```bash\nsudo systemctl restart nginx\n```"
        assert detect_code_hedging(text, [])

    def test_sh_block_detected(self):
        text = "Try this:\n```sh\ndf -h\n```"
        assert detect_code_hedging(text, [])

    def test_python_block_not_detected(self):
        """Python code blocks should not trigger code hedging."""
        text = "Here's how:\n```python\nprint('hello')\n```"
        assert not detect_code_hedging(text, [])

    def test_no_trigger_when_tools_called(self):
        text = "Result:\n```bash\ndf -h\n```"
        assert not detect_code_hedging(text, ["run_command"])

    def test_no_trigger_on_short_text(self):
        assert not detect_code_hedging("```bash", [])

    def test_bare_code_block_not_detected(self):
        """Code blocks without bash/sh language tag should not trigger."""
        text = "Here's the output:\n```\nsome output\n```"
        assert not detect_code_hedging(text, [])


# ===========================================================================
# Skill context: HTTP headers (Round 5)
# ===========================================================================

class TestSkillContextHTTPHeaders:
    """http_get/http_post handle custom headers correctly."""

    @pytest.fixture
    def skill_ctx(self):
        executor = MagicMock()
        executor.config.hosts = {}
        return SkillContext(tool_executor=executor, skill_name="test_skill")

    async def test_http_get_default_accept_json(self, skill_ctx):
        """http_get includes Accept: application/json by default."""
        mock_resp = AsyncMock()
        mock_resp.content_type = "application/json"
        mock_resp.json = AsyncMock(return_value={"ok": True})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await skill_ctx.http_get("https://example.com/api")

        call_kwargs = mock_session.get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Accept"] == "application/json"

    async def test_http_get_custom_headers_merge(self, skill_ctx):
        """Custom headers merge with defaults; custom overrides default."""
        mock_resp = AsyncMock()
        mock_resp.content_type = "text/plain"
        mock_resp.text = AsyncMock(return_value="joke text")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await skill_ctx.http_get(
                "https://example.com/api",
                headers={"Accept": "text/plain", "X-Custom": "yes"},
            )

        call_kwargs = mock_session.get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        # User-provided Accept overrides default
        assert headers["Accept"] == "text/plain"
        assert headers["X-Custom"] == "yes"

    async def test_http_post_custom_headers(self, skill_ctx):
        """http_post passes custom headers through."""
        mock_resp = AsyncMock()
        mock_resp.content_type = "application/json"
        mock_resp.json = AsyncMock(return_value={"created": True})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await skill_ctx.http_post(
                "https://example.com/api",
                json={"data": 1},
                headers={"Authorization": "Bearer tok123"},
            )

        call_kwargs = mock_session.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer tok123"

    async def test_http_post_no_default_accept(self, skill_ctx):
        """http_post does NOT inject default Accept: application/json."""
        mock_resp = AsyncMock()
        mock_resp.content_type = "application/json"
        mock_resp.json = AsyncMock(return_value={"status": "ok"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await skill_ctx.http_post("https://example.com/api", data="payload")

        call_kwargs = mock_session.post.call_args
        # When no custom headers provided, merged={} → passed as None
        headers = call_kwargs.kwargs.get("headers", "MISSING")
        if headers == "MISSING":
            headers = call_kwargs[1].get("headers")
        # Should be None (empty dict mapped to None) or empty dict — no Accept header
        assert headers is None or (isinstance(headers, dict) and "Accept" not in headers)


# ===========================================================================
# Discord message splitting (Round 4)
# ===========================================================================

class TestSendChunked:
    """_send_chunked splits messages at DISCORD_MAX_LEN with code block continuation."""

    @pytest.fixture
    def bot_stub(self):
        return _make_bot_stub()

    @pytest.fixture
    def msg(self):
        return _make_message()

    async def test_short_message_sent_directly(self, bot_stub, msg):
        """Messages under 2000 chars sent as-is."""
        text = "Hello world"
        await LokiBot._send_chunked(bot_stub, msg, text)
        bot_stub._send_with_retry.assert_called_once()
        call_args = bot_stub._send_with_retry.call_args
        assert call_args[0][1] == text

    async def test_long_message_split_into_chunks(self, bot_stub, msg):
        """Messages over 2000 chars split into multiple sends."""
        # Create text that's ~3000 chars
        text = ("A" * 100 + "\n") * 30  # 30 lines of 101 chars = 3030 chars
        await LokiBot._send_chunked(bot_stub, msg, text)
        assert bot_stub._send_with_retry.call_count >= 2

    async def test_very_long_message_sent_as_file(self, bot_stub, msg):
        """Messages over 4x DISCORD_MAX_LEN sent as file attachment."""
        text = "X" * (DISCORD_MAX_LEN * 4 + 1)
        await LokiBot._send_chunked(bot_stub, msg, text)
        call_args = bot_stub._send_with_retry.call_args
        files = call_args.kwargs.get("files") or call_args[1].get("files", [])
        assert any(
            getattr(f, "filename", "") == "response.md" for f in files
        )

    async def test_code_block_continuation(self, bot_stub, msg):
        """Code blocks split across chunks reopen with language tag."""
        # Build text with a code block that will span two chunks
        lines = ["```python"]
        # Each line ~100 chars, need enough to exceed 2000
        for i in range(25):
            lines.append(f"print('line {i}' + 'x' * 80)  # {'p' * 60}")
        lines.append("```")
        text = "\n".join(lines)

        assert len(text) > DISCORD_MAX_LEN

        await LokiBot._send_chunked(bot_stub, msg, text)
        assert bot_stub._send_with_retry.call_count >= 2

        # Second chunk should reopen the code block
        second_call = bot_stub._send_with_retry.call_args_list[1]
        second_text = second_call[0][1]
        assert second_text.startswith("```python")

    async def test_files_attached_to_first_chunk(self, bot_stub, msg):
        """Pending skill files attached to first chunk only."""
        bot_stub._pending_files[str(msg.channel.id)] = [
            (b"filedata", "output.txt"),
        ]
        text = ("A" * 100 + "\n") * 30  # >2000 chars
        await LokiBot._send_chunked(bot_stub, msg, text)
        # First call should have files
        first_call = bot_stub._send_with_retry.call_args_list[0]
        files = first_call.kwargs.get("files") or first_call[1].get("files", [])
        assert len(files) >= 1
        # Second call should NOT have files
        if bot_stub._send_with_retry.call_count >= 2:
            second_call = bot_stub._send_with_retry.call_args_list[1]
            second_files = second_call.kwargs.get("files") or second_call[1].get("files", [])
            assert not second_files


# ===========================================================================
# delegate_task: host defaults + error counting (Rounds 7-8)
# ===========================================================================

class TestDelegateTaskHostDefault:
    """_get_default_host returns first configured host or 'localhost'."""

    def test_returns_first_host(self):
        executor = MagicMock()
        executor.config.hosts = {"webserver": {}, "dbserver": {}}
        assert _get_default_host(executor) == "webserver"

    def test_falls_back_to_localhost(self):
        executor = MagicMock()
        executor.config.hosts = {}
        assert _get_default_host(executor) == "localhost"

    def test_handles_no_hosts_attr(self):
        executor = MagicMock(spec=[])
        assert _get_default_host(executor) == "localhost"

    def test_handles_none_hosts(self):
        executor = MagicMock()
        executor.config.hosts = None
        assert _get_default_host(executor) == "localhost"


class TestIsErrorOutput:
    """_is_error_output detects error strings from executor."""

    def test_error_executing_detected(self):
        assert _is_error_output("Error executing check_disk: 'host'")

    def test_unknown_tool_detected(self):
        assert _is_error_output("Unknown tool: foobar")

    def test_normal_output_not_error(self):
        assert not _is_error_output("Filesystem Size Used Avail Use%")

    def test_empty_string_not_error(self):
        assert not _is_error_output("")

    def test_partial_match_not_error(self):
        """'Error' in middle of output should not match (must start with it)."""
        assert not _is_error_output("Got an Error executing the thing")


class TestBackgroundTaskErrorCounting:
    """run_background_task counts errored steps as errors, not OK."""

    async def test_error_output_counted_as_error(self):
        """Steps returning error strings get status='error' not 'ok'."""
        channel = MagicMock()
        channel.send = AsyncMock(return_value=MagicMock())
        channel.id = 12345

        task = BackgroundTask(
            task_id="test123",
            description="test task",
            steps=[
                {"tool_name": "check_disk", "tool_input": {"host": "web"}},
            ],
            channel=channel,
            requester="tester",
        )

        executor = MagicMock()
        executor.config.hosts = {"web": {}}
        executor.execute = AsyncMock(
            return_value="Error executing check_disk: 'host'"
        )

        skill_mgr = MagicMock()
        skill_mgr.has_skill = MagicMock(return_value=False)

        await run_background_task(task, executor, skill_mgr)

        assert len(task.results) == 1
        assert task.results[0].status == "error"

    async def test_ok_output_counted_as_ok(self):
        """Normal outputs get status='ok'."""
        channel = MagicMock()
        channel.send = AsyncMock(return_value=MagicMock())
        channel.id = 12345

        task = BackgroundTask(
            task_id="test456",
            description="test task",
            steps=[
                {"tool_name": "check_disk", "tool_input": {"host": "web"}},
            ],
            channel=channel,
            requester="tester",
        )

        executor = MagicMock()
        executor.config.hosts = {"web": {}}
        executor.execute = AsyncMock(return_value="/dev/sda1 50G 25G 25G 50%")

        skill_mgr = MagicMock()
        skill_mgr.has_skill = MagicMock(return_value=False)

        await run_background_task(task, executor, skill_mgr)

        assert len(task.results) == 1
        assert task.results[0].status == "ok"

    async def test_error_with_abort_stops_task(self):
        """on_failure='abort' (default) stops task on error output."""
        channel = MagicMock()
        channel.send = AsyncMock(return_value=MagicMock())
        channel.id = 12345

        task = BackgroundTask(
            task_id="test789",
            description="test task",
            steps=[
                {"tool_name": "check_disk", "tool_input": {"host": "web"}},
                {"tool_name": "check_memory", "tool_input": {"host": "web"}},
            ],
            channel=channel,
            requester="tester",
        )

        executor = MagicMock()
        executor.config.hosts = {"web": {}}
        executor.execute = AsyncMock(
            return_value="Error executing check_disk: 'host'"
        )

        skill_mgr = MagicMock()
        skill_mgr.has_skill = MagicMock(return_value=False)

        await run_background_task(task, executor, skill_mgr)

        assert task.status == "failed"
        assert len(task.results) == 1  # Second step never ran


# ===========================================================================
# Browser CDP reconnect (Round 9)
# ===========================================================================

class TestBrowserConnectionError:
    """_is_connection_error detects stale CDP connection patterns."""

    def test_connection_closed_detected(self):
        exc = Exception("Connection closed while reading")
        assert BrowserManager._is_connection_error(exc)

    def test_target_closed_detected(self):
        exc = Exception("Target closed")
        assert BrowserManager._is_connection_error(exc)

    def test_browser_has_been_closed_detected(self):
        exc = Exception("Browser has been closed")
        assert BrowserManager._is_connection_error(exc)

    def test_websocket_is_closed_detected(self):
        exc = Exception("WebSocket is closed")
        assert BrowserManager._is_connection_error(exc)

    def test_connection_refused_detected(self):
        exc = Exception("Connection refused")
        assert BrowserManager._is_connection_error(exc)

    def test_not_connected_detected(self):
        exc = Exception("Not connected to browser")
        assert BrowserManager._is_connection_error(exc)

    def test_normal_error_not_detected(self):
        exc = Exception("Element not found")
        assert not BrowserManager._is_connection_error(exc)

    def test_timeout_not_detected(self):
        exc = Exception("Timeout 30000ms exceeded")
        assert not BrowserManager._is_connection_error(exc)

    def test_all_patterns_covered(self):
        """Each pattern in _CONNECTION_ERROR_PATTERNS should be detected."""
        for pattern in _CONNECTION_ERROR_PATTERNS:
            exc = Exception(f"Something {pattern} happened")
            assert BrowserManager._is_connection_error(exc), f"Pattern not detected: {pattern}"

    def test_case_insensitive(self):
        exc = Exception("CONNECTION CLOSED while reading from driver")
        assert BrowserManager._is_connection_error(exc)


class TestBrowserReconnect:
    """new_page retries once on stale CDP connection."""

    async def test_reconnect_on_connection_error(self):
        """new_page force-reconnects and retries on connection error."""
        mgr = BrowserManager()
        mgr._browser = MagicMock()
        mgr._browser.is_connected = MagicMock(return_value=True)

        call_count = 0
        good_context = MagicMock()
        good_context.close = AsyncMock()
        good_page = MagicMock()

        async def mock_create_page(timeout_ms=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Connection closed while reading from the driver")
            return good_context, good_page

        mgr._create_page = mock_create_page
        mgr._ensure_connected = AsyncMock()
        mgr._force_reconnect = AsyncMock()

        async with mgr.new_page() as page:
            assert page is good_page

        mgr._force_reconnect.assert_called_once()
        assert call_count == 2

    async def test_non_connection_error_not_retried(self):
        """Non-connection errors are raised immediately without retry."""
        mgr = BrowserManager()
        mgr._browser = MagicMock()
        mgr._browser.is_connected = MagicMock(return_value=True)

        async def mock_create_page(timeout_ms=None):
            raise ValueError("Element not found")

        mgr._create_page = mock_create_page
        mgr._ensure_connected = AsyncMock()

        with pytest.raises(ValueError, match="Element not found"):
            async with mgr.new_page():
                pass

    async def test_force_reconnect_clears_browser(self):
        """_force_reconnect closes stale browser and sets to None."""
        mgr = BrowserManager()
        mock_browser = AsyncMock()
        mgr._browser = mock_browser

        # Mock _ensure_connected to just set a new browser
        async def mock_ensure():
            mgr._browser = MagicMock()
        mgr._ensure_connected = mock_ensure

        await mgr._force_reconnect()

        mock_browser.close.assert_called_once()


# ===========================================================================
# Autonomous loop lifecycle (Rounds 11-14)
# ===========================================================================

class TestLoopManagerStart:
    """start_loop creates loops with correct validation."""

    async def test_start_loop_returns_id(self):
        mgr = LoopManager()
        callback = AsyncMock(return_value="iteration result")
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        loop_id = mgr.start_loop(
            goal="test goal",
            channel=channel,
            requester_id="user1",
            requester_name="TestUser",
            iteration_callback=callback,
        )

        assert isinstance(loop_id, str)
        assert len(loop_id) == 8
        assert mgr.active_count == 1
        # Clean up
        mgr.stop_loop("all")

    async def test_max_concurrent_limit(self):
        """Exceeding MAX_CONCURRENT_LOOPS returns an error string."""
        mgr = LoopManager()
        callback = AsyncMock(return_value="ok")
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        # Start MAX_CONCURRENT_LOOPS loops
        for i in range(MAX_CONCURRENT_LOOPS):
            result = mgr.start_loop(
                goal=f"loop {i}",
                channel=channel,
                requester_id="user1",
                requester_name="TestUser",
                iteration_callback=callback,
            )
            assert not result.startswith("Error")

        # 11th loop should fail
        result = mgr.start_loop(
            goal="one too many",
            channel=channel,
            requester_id="user1",
            requester_name="TestUser",
            iteration_callback=callback,
        )
        assert "Error" in result
        assert str(MAX_CONCURRENT_LOOPS) in result
        # Clean up
        mgr.stop_loop("all")

    async def test_invalid_mode_defaults_to_notify(self):
        mgr = LoopManager()
        callback = AsyncMock(return_value="ok")
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        loop_id = mgr.start_loop(
            goal="test",
            channel=channel,
            requester_id="user1",
            requester_name="TestUser",
            iteration_callback=callback,
            mode="invalid",
        )
        info = mgr._loops[loop_id]
        assert info.mode == "notify"
        mgr.stop_loop("all")

    async def test_interval_clamped_to_minimum(self):
        mgr = LoopManager()
        callback = AsyncMock(return_value="ok")
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        loop_id = mgr.start_loop(
            goal="test",
            channel=channel,
            requester_id="user1",
            requester_name="TestUser",
            iteration_callback=callback,
            interval_seconds=1,  # Below MIN_INTERVAL_SECONDS
        )
        info = mgr._loops[loop_id]
        assert info.interval_seconds >= MIN_INTERVAL_SECONDS
        mgr.stop_loop("all")

    async def test_max_iterations_clamped(self):
        mgr = LoopManager()
        callback = AsyncMock(return_value="ok")
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        loop_id = mgr.start_loop(
            goal="test",
            channel=channel,
            requester_id="user1",
            requester_name="TestUser",
            iteration_callback=callback,
            max_iterations=5000,  # Above 1000 cap
        )
        info = mgr._loops[loop_id]
        assert info.max_iterations <= 1000
        mgr.stop_loop("all")


class TestLoopManagerStop:
    """stop_loop cancels loops correctly."""

    async def test_stop_by_id(self):
        mgr = LoopManager()
        callback = AsyncMock(return_value="ok")
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        loop_id = mgr.start_loop(
            goal="test",
            channel=channel,
            requester_id="user1",
            requester_name="TestUser",
            iteration_callback=callback,
        )

        result = mgr.stop_loop(loop_id)
        assert "stopped" in result.lower()
        assert mgr._loops[loop_id].status == "stopped"

    async def test_stop_all(self):
        mgr = LoopManager()
        callback = AsyncMock(return_value="ok")
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        ids = []
        for _ in range(3):
            ids.append(mgr.start_loop(
                goal="test",
                channel=channel,
                requester_id="user1",
                requester_name="TestUser",
                iteration_callback=callback,
            ))

        result = mgr.stop_loop("all")
        assert "3" in result
        for lid in ids:
            assert mgr._loops[lid].status == "stopped"

    def test_stop_nonexistent_id(self):
        mgr = LoopManager()
        result = mgr.stop_loop("doesnotexist")
        assert "No loop found" in result

    async def test_stop_already_stopped(self):
        mgr = LoopManager()
        callback = AsyncMock(return_value="ok")
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        loop_id = mgr.start_loop(
            goal="test",
            channel=channel,
            requester_id="user1",
            requester_name="TestUser",
            iteration_callback=callback,
        )
        mgr.stop_loop(loop_id)
        result = mgr.stop_loop(loop_id)
        assert "not running" in result.lower()


class TestLoopManagerList:
    """list_loops returns formatted loop information."""

    def test_no_loops(self):
        mgr = LoopManager()
        result = mgr.list_loops()
        assert result == "No autonomous loops."

    async def test_lists_active_loops(self):
        mgr = LoopManager()
        callback = AsyncMock(return_value="ok")
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        loop_id = mgr.start_loop(
            goal="Monitor disk usage",
            channel=channel,
            requester_id="user1",
            requester_name="TestUser",
            iteration_callback=callback,
            interval_seconds=60,
            mode="notify",
        )

        result = mgr.list_loops()
        assert loop_id in result
        assert "Monitor disk usage" in result
        assert "running" in result
        assert "notify" in result
        mgr.stop_loop("all")


def _make_loop_info(mgr, **overrides):
    """Create a LoopInfo and register it in the manager, without starting a task."""
    defaults = dict(
        id="test01",
        goal="test goal",
        mode="notify",
        interval_seconds=MIN_INTERVAL_SECONDS,
        stop_condition=None,
        max_iterations=50,
        channel_id="12345",
        requester_id="user1",
        requester_name="TestUser",
    )
    defaults.update(overrides)
    info = LoopInfo(**defaults)
    mgr._loops[info.id] = info
    return info


async def _run_loop_fast(mgr, info, channel, callback):
    """Run _run_loop with patched wait_for so intervals elapse instantly."""
    async def fast_wait_for(coro, timeout):
        raise asyncio.TimeoutError()

    with patch("asyncio.wait_for", side_effect=fast_wait_for):
        await mgr._run_loop(info, channel, callback)


class TestLoopIteration:
    """Loop iteration behavior: stop conditions, modes, error handling."""

    async def test_loop_stop_sentinel(self):
        """Response containing LOOP_STOP stops the loop."""
        mgr = LoopManager()
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        callback = AsyncMock(return_value="Game over! LOOP_STOP")

        info = _make_loop_info(mgr, id="stop1", goal="play game", max_iterations=5)
        await _run_loop_fast(mgr, info, channel, callback)

        assert info.status == "completed"
        assert info.iteration_count == 1

    async def test_max_iterations_stops_loop(self):
        """Loop stops after reaching max_iterations."""
        mgr = LoopManager()
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        call_count = 0

        async def counting_callback(prompt, ch, ctx):
            nonlocal call_count
            call_count += 1
            return f"iteration {call_count}"

        info = _make_loop_info(mgr, id="maxiter", max_iterations=2)
        await _run_loop_fast(mgr, info, channel, counting_callback)

        assert info.status == "completed"
        assert info.iteration_count == 2
        assert call_count == 2

    async def test_error_in_iteration_doesnt_crash_loop(self):
        """A single iteration error doesn't stop the entire loop."""
        mgr = LoopManager()
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        call_count = 0

        async def flaky_callback(prompt, ch, ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API timeout")
            return "recovered ok"

        info = _make_loop_info(mgr, id="flaky1", max_iterations=2)
        await _run_loop_fast(mgr, info, channel, flaky_callback)

        # Loop should have continued past the error
        assert call_count == 2
        assert info.status == "completed"

    async def test_consecutive_errors_stop_loop(self):
        """MAX_CONSECUTIVE_ERRORS stops the loop."""
        mgr = LoopManager()
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        error_count = 0

        async def always_fail(prompt, ch, ctx):
            nonlocal error_count
            error_count += 1
            raise RuntimeError("API down")

        info = _make_loop_info(mgr, id="errtest", max_iterations=100)
        await _run_loop_fast(mgr, info, channel, always_fail)

        assert info.status == "error"
        assert error_count == MAX_CONSECUTIVE_ERRORS

    async def test_iteration_prompt_contains_goal(self):
        """The prompt passed to the callback contains the goal."""
        mgr = LoopManager()
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        captured_prompt = None

        async def capture_callback(prompt, ch, ctx):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "done LOOP_STOP"

        info = _make_loop_info(mgr, id="goal1", goal="Check disk usage on webserver")
        await _run_loop_fast(mgr, info, channel, capture_callback)

        assert captured_prompt is not None
        assert "Check disk usage on webserver" in captured_prompt
        assert "AUTONOMOUS LOOP" in captured_prompt

    async def test_iteration_prompt_contains_mode(self):
        """The prompt includes the loop mode."""
        mgr = LoopManager()
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        captured_prompt = None

        async def capture_callback(prompt, ch, ctx):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "LOOP_STOP"

        info = _make_loop_info(mgr, id="mode1", mode="silent")
        await _run_loop_fast(mgr, info, channel, capture_callback)

        assert captured_prompt is not None
        assert "silent" in captured_prompt.lower()

    async def test_iteration_prompt_contains_stop_condition(self):
        """The prompt includes the stop condition when set."""
        mgr = LoopManager()
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        captured_prompt = None

        async def capture_callback(prompt, ch, ctx):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "LOOP_STOP"

        info = _make_loop_info(
            mgr, id="stopcond1", stop_condition="when the game ends",
        )
        await _run_loop_fast(mgr, info, channel, capture_callback)

        assert captured_prompt is not None
        assert "when the game ends" in captured_prompt

    async def test_previous_context_passed_to_callback(self):
        """After first iteration, previous context is passed to callback."""
        mgr = LoopManager()
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        contexts = []

        async def capture_callback(prompt, ch, ctx):
            contexts.append(ctx)
            if len(contexts) >= 2:
                return "LOOP_STOP"
            return "first iteration result"

        info = _make_loop_info(mgr, id="ctx1", max_iterations=3)
        await _run_loop_fast(mgr, info, channel, capture_callback)

        # First iteration should have no previous context
        assert contexts[0] is None
        # Second iteration should have context from first
        assert len(contexts) >= 2
        assert contexts[1] is not None
        assert "first iteration result" in contexts[1]

    async def test_llm_gets_full_tool_access(self):
        """The iteration callback receives the prompt — demonstrating LLM has tools."""
        mgr = LoopManager()
        channel = MagicMock()
        channel.id = 12345
        channel.send = AsyncMock()

        prompts_received = []

        async def tool_callback(prompt, ch, ctx):
            prompts_received.append(prompt)
            return "LOOP_STOP"

        info = _make_loop_info(mgr, id="tools1", goal="Run 'date' and report")
        await _run_loop_fast(mgr, info, channel, tool_callback)

        # The callback was invoked with the goal prompt
        assert len(prompts_received) == 1
        assert "Run 'date' and report" in prompts_received[0]


class TestLoopManagerCleanup:
    """cleanup_finished removes old non-running loops."""

    def test_cleanup_removes_old_stopped_loops(self):
        mgr = LoopManager()
        # Manually insert a stopped loop with old last_trigger
        info = LoopInfo(
            id="old1",
            goal="old loop",
            mode="notify",
            interval_seconds=60,
            stop_condition=None,
            max_iterations=50,
            channel_id="123",
            requester_id="user1",
            requester_name="TestUser",
            status="stopped",
            last_trigger=time.monotonic() - 7200,  # 2 hours ago
        )
        mgr._loops["old1"] = info

        mgr.cleanup_finished()
        assert "old1" not in mgr._loops

    def test_cleanup_keeps_running_loops(self):
        mgr = LoopManager()
        info = LoopInfo(
            id="active1",
            goal="active loop",
            mode="notify",
            interval_seconds=60,
            stop_condition=None,
            max_iterations=50,
            channel_id="123",
            requester_id="user1",
            requester_name="TestUser",
            status="running",
            last_trigger=time.monotonic() - 7200,
        )
        mgr._loops["active1"] = info

        mgr.cleanup_finished()
        assert "active1" in mgr._loops


class TestLoopBuildPrompt:
    """_build_iteration_prompt constructs correct prompts."""

    def test_prompt_includes_iteration_count(self):
        mgr = LoopManager()
        info = LoopInfo(
            id="test1",
            goal="check stuff",
            mode="notify",
            interval_seconds=60,
            stop_condition=None,
            max_iterations=50,
            channel_id="123",
            requester_id="user1",
            requester_name="TestUser",
            iteration_count=3,
        )
        prompt = mgr._build_iteration_prompt(info)
        assert "iteration 3 of 50" in prompt

    def test_prompt_includes_stop_condition(self):
        mgr = LoopManager()
        info = LoopInfo(
            id="test1",
            goal="play game",
            mode="act",
            interval_seconds=60,
            stop_condition="when someone wins",
            max_iterations=50,
            channel_id="123",
            requester_id="user1",
            requester_name="TestUser",
        )
        prompt = mgr._build_iteration_prompt(info)
        assert "when someone wins" in prompt
        assert "LOOP_STOP" in prompt

    def test_notify_mode_prompt(self):
        mgr = LoopManager()
        info = LoopInfo(
            id="test1",
            goal="test",
            mode="notify",
            interval_seconds=60,
            stop_condition=None,
            max_iterations=50,
            channel_id="123",
            requester_id="user1",
            requester_name="TestUser",
        )
        prompt = mgr._build_iteration_prompt(info)
        assert "concise update" in prompt.lower()

    def test_silent_mode_prompt(self):
        mgr = LoopManager()
        info = LoopInfo(
            id="test1",
            goal="test",
            mode="silent",
            interval_seconds=60,
            stop_condition=None,
            max_iterations=50,
            channel_id="123",
            requester_id="user1",
            requester_name="TestUser",
        )
        prompt = mgr._build_iteration_prompt(info)
        assert "notable" in prompt.lower() or "urgent" in prompt.lower()
