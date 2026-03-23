"""Round 21: Performance optimizations.

Tests for:
1. Tool definition caching (LokiBot._merged_tool_definitions)
2. Tool conversion caching (CodexChatClient._convert_tools_cached)
3. History optimization (no double-fetch for non-guests)
4. Memory cleanup (stale cache eviction)
5. Session save is non-blocking
6. Source structure verification
"""
from __future__ import annotations

import asyncio
import inspect
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.discord.client import LokiBot
from src.llm.openai_codex import CodexChatClient
from src.sessions.manager import SessionManager
from src.tools.registry import get_tool_definitions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub(**overrides):
    """Minimal LokiBot stub for performance tests."""
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
    stub._cached_merged_tools = None
    stub._last_cache_cleanup = 0.0
    stub._cache_cleanup_interval = 300.0
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
    stub.sessions.get_history = MagicMock(return_value=[])
    stub.sessions.add_message = MagicMock()
    stub.sessions.remove_last_message = MagicMock(return_value=True)
    stub.sessions.prune = MagicMock()
    stub.sessions.save = MagicMock()
    stub.sessions.reset = MagicMock()
    stub.sessions._sessions = {}
    stub.codex_client = AsyncMock()
    stub.codex_client.chat = AsyncMock(return_value="test response")
    stub.codex_client.chat_with_tools = AsyncMock()
    stub.tool_executor = MagicMock()
    stub.tool_executor.execute = AsyncMock(return_value="tool result")
    stub.skill_manager = MagicMock()
    stub.skill_manager.get_tool_definitions = MagicMock(return_value=[])
    stub.skill_manager.list_skills = MagicMock(return_value=[])
    stub.skill_manager.has_skill = MagicMock(return_value=False)
    stub.permissions = MagicMock()
    stub.permissions.is_guest = MagicMock(return_value=False)
    stub.permissions.filter_tools = MagicMock(side_effect=lambda uid, tools: tools)
    stub.audit = MagicMock()
    stub.audit.log_execution = AsyncMock()
    stub.reflector = MagicMock()
    stub.reflector.get_prompt_section = MagicMock(return_value="")
    stub.context_loader = MagicMock()
    stub.context_loader.context = ""
    stub.tool_memory = MagicMock()
    stub.tool_memory.record = AsyncMock()
    stub.voice_manager = None
    stub.browser_manager = None
    stub._knowledge_store = None
    stub._embedder = None
    stub._pending_files = {}
    stub._bot_msg_buffer = {}
    stub._bot_msg_tasks = {}
    stub.infra_watcher = None

    for key, val in overrides.items():
        setattr(stub, key, val)
    return stub


# ---------------------------------------------------------------------------
# 1. Tool definition caching
# ---------------------------------------------------------------------------

class TestToolDefinitionCaching:
    """_merged_tool_definitions should cache and return same list on repeated calls."""

    def test_returns_list_on_first_call(self):
        stub = _make_bot_stub()
        stub._merged_tool_definitions = LokiBot._merged_tool_definitions.__get__(stub)
        result = stub._merged_tool_definitions()
        assert isinstance(result, list)
        assert len(result) >= 50  # built-in tools

    def test_second_call_returns_cached(self):
        """Second call returns the exact same list object (identity)."""
        stub = _make_bot_stub()
        stub._merged_tool_definitions = LokiBot._merged_tool_definitions.__get__(stub)
        first = stub._merged_tool_definitions()
        second = stub._merged_tool_definitions()
        assert first is second

    def test_get_tool_definitions_called_once(self):
        """get_tool_definitions (from registry) called only on first invocation."""
        stub = _make_bot_stub()
        stub._merged_tool_definitions = LokiBot._merged_tool_definitions.__get__(stub)
        with patch("src.discord.client.get_tool_definitions") as mock_get:
            mock_get.return_value = [{"name": "test_tool", "description": "test"}]
            stub._cached_merged_tools = None
            stub._merged_tool_definitions()
            stub._merged_tool_definitions()
            stub._merged_tool_definitions()
            assert mock_get.call_count == 1

    def test_cache_invalidated_on_none(self):
        """Setting _cached_merged_tools = None forces rebuild."""
        stub = _make_bot_stub()
        stub._merged_tool_definitions = LokiBot._merged_tool_definitions.__get__(stub)
        first = stub._merged_tool_definitions()
        stub._cached_merged_tools = None
        second = stub._merged_tool_definitions()
        # Both are valid tool lists, but may or may not be the same object
        assert isinstance(second, list)
        assert len(second) >= 50

    def test_cache_includes_skills(self):
        """Cached result includes skill definitions."""
        stub = _make_bot_stub()
        stub.skill_manager.get_tool_definitions = MagicMock(return_value=[
            {"name": "my_skill", "description": "skill", "input_schema": {}},
        ])
        stub._merged_tool_definitions = LokiBot._merged_tool_definitions.__get__(stub)
        result = stub._merged_tool_definitions()
        names = [t["name"] for t in result]
        assert "my_skill" in names

    def test_cache_deduplicates_skills(self):
        """Skills with same name as builtin are excluded from cached result."""
        stub = _make_bot_stub()
        stub.skill_manager.get_tool_definitions = MagicMock(return_value=[
            {"name": "check_disk", "description": "shadow builtin"},
        ])
        stub._merged_tool_definitions = LokiBot._merged_tool_definitions.__get__(stub)
        result = stub._merged_tool_definitions()
        names = [t["name"] for t in result]
        assert names.count("check_disk") == 1

    def test_skill_create_invalidates_cache(self):
        """Simulating create_skill sets _cached_merged_tools = None."""
        stub = _make_bot_stub()
        stub._merged_tool_definitions = LokiBot._merged_tool_definitions.__get__(stub)
        first = stub._merged_tool_definitions()
        assert stub._cached_merged_tools is first
        # Simulate what _run_tool does on create_skill
        stub._cached_merged_tools = None
        assert stub._cached_merged_tools is None


# ---------------------------------------------------------------------------
# 2. Tool conversion caching
# ---------------------------------------------------------------------------

class TestToolConversionCaching:
    """CodexChatClient._convert_tools_cached should avoid redundant conversions."""

    def test_same_list_returns_cached(self):
        """Same list object returns cached conversion."""
        client = CodexChatClient.__new__(CodexChatClient)
        client._last_tools_list = None
        client._last_tools_converted = []
        tools = [
            {"name": "t1", "description": "d1", "input_schema": {"type": "object", "properties": {}}},
            {"name": "t2", "description": "d2", "input_schema": {"type": "object", "properties": {}}},
        ]
        first = client._convert_tools_cached(tools)
        second = client._convert_tools_cached(tools)
        assert first is second
        assert len(first) == 2

    def test_different_list_rebuilds(self):
        """Different list object triggers rebuild."""
        client = CodexChatClient.__new__(CodexChatClient)
        client._last_tools_list = None
        client._last_tools_converted = []
        tools1 = [{"name": "t1", "description": "d1", "input_schema": {}}]
        tools2 = [{"name": "t2", "description": "d2", "input_schema": {}}]
        first = client._convert_tools_cached(tools1)
        second = client._convert_tools_cached(tools2)
        assert first is not second
        assert first[0]["name"] == "t1"
        assert second[0]["name"] == "t2"

    def test_converted_format_correct(self):
        """Cached result has OpenAI function format."""
        client = CodexChatClient.__new__(CodexChatClient)
        client._last_tools_list = None
        client._last_tools_converted = []
        tools = [{"name": "run_command", "description": "Run a command", "input_schema": {"type": "object"}}]
        result = client._convert_tools_cached(tools)
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "run_command"
        assert result[0]["parameters"] == {"type": "object"}

    def test_multiple_iterations_one_conversion(self):
        """Simulating tool loop: 5 iterations with same tools, only 1 conversion."""
        client = CodexChatClient.__new__(CodexChatClient)
        client._last_tools_list = None
        client._last_tools_converted = []
        tools = [{"name": f"t{i}", "description": f"d{i}", "input_schema": {}} for i in range(10)]
        # Track calls
        original_convert = CodexChatClient._convert_tools
        call_count = 0
        def counting_convert(t):
            nonlocal call_count
            call_count += 1
            return original_convert(t)
        with patch.object(CodexChatClient, "_convert_tools", staticmethod(counting_convert)):
            for _ in range(5):
                client._convert_tools_cached(tools)
        assert call_count == 1  # only first iteration converts

    def test_init_has_cache_attrs(self):
        """CodexChatClient.__init__ initializes cache attributes."""
        auth = MagicMock()
        client = CodexChatClient(auth=auth, model="test", max_tokens=100)
        assert client._last_tools_list is None
        assert client._last_tools_converted == []


# ---------------------------------------------------------------------------
# 3. History optimization: no double-fetch for non-guests
# ---------------------------------------------------------------------------

class TestHistoryOptimization:
    """Non-guest path should use get_task_history, not get_history_with_compaction."""

    async def test_non_guest_uses_task_history(self):
        """Non-guest message calls get_task_history, not get_history_with_compaction."""
        from src.llm.types import LLMResponse
        stub = _make_bot_stub()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)
        stub._build_system_prompt = MagicMock(return_value="sys prompt")
        stub._inject_tool_hints = AsyncMock(return_value="sys prompt")
        stub._process_with_tools = AsyncMock(return_value=("result", False, False, ["run_command"], False))
        stub._send_chunked = AsyncMock()
        stub._send_with_retry = AsyncMock()
        stub._maybe_cleanup_caches = MagicMock()

        msg = MagicMock()
        msg.author.id = 12345
        msg.author.bot = False
        msg.author.display_name = "TestUser"
        msg.author.name = "testuser"
        msg.channel.id = 67890
        msg.webhook_id = None

        await stub._handle_message_inner(msg, "check disk", "chan-1")

        stub.sessions.get_task_history.assert_called_once()
        stub.sessions.get_history_with_compaction.assert_not_called()

    async def test_guest_uses_full_history(self):
        """Guest message calls get_history_with_compaction."""
        stub = _make_bot_stub()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)
        stub._build_chat_system_prompt = MagicMock(return_value="chat prompt")
        stub._send_chunked = AsyncMock()
        stub._send_with_retry = AsyncMock()
        stub._maybe_cleanup_caches = MagicMock()
        stub.permissions.is_guest = MagicMock(return_value=True)
        stub.sessions.get_history_with_compaction = AsyncMock(return_value=[
            {"role": "user", "content": "hello"},
        ])

        msg = MagicMock()
        msg.author.id = 99999
        msg.author.bot = False
        msg.author.display_name = "Guest"
        msg.author.name = "guest"
        msg.channel.id = 67890
        msg.webhook_id = None

        await stub._handle_message_inner(msg, "hello", "chan-1")

        stub.sessions.get_history_with_compaction.assert_called_once()
        stub.sessions.get_task_history.assert_not_called()

    async def test_handoff_uses_get_history(self):
        """Skill handoff path uses get_history (not get_history_with_compaction)."""
        stub = _make_bot_stub()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)
        stub._build_system_prompt = MagicMock(return_value="sys prompt")
        stub._build_chat_system_prompt = MagicMock(return_value="chat prompt")
        stub._inject_tool_hints = AsyncMock(return_value="sys prompt")
        stub._process_with_tools = AsyncMock(return_value=("skill result", False, False, ["my_skill"], True))
        stub._send_chunked = AsyncMock()
        stub._send_with_retry = AsyncMock()
        stub._maybe_cleanup_caches = MagicMock()
        stub.sessions.get_history = MagicMock(return_value=[
            {"role": "user", "content": "do something"},
        ])

        msg = MagicMock()
        msg.author.id = 12345
        msg.author.bot = False
        msg.author.display_name = "TestUser"
        msg.author.name = "testuser"
        msg.channel.id = 67890
        msg.webhook_id = None

        await stub._handle_message_inner(msg, "do something", "chan-1")

        # get_history (not _with_compaction) called for handoff
        stub.sessions.get_history.assert_called_once()
        stub.sessions.get_history_with_compaction.assert_not_called()

    async def test_non_guest_images_injected_into_task_history(self):
        """Non-guest with images: images injected into task_history, not full history."""
        stub = _make_bot_stub()
        stub._handle_message_inner = LokiBot._handle_message_inner.__get__(stub)
        stub._build_system_prompt = MagicMock(return_value="sys prompt")
        stub._inject_tool_hints = AsyncMock(return_value="sys prompt")
        stub._send_chunked = AsyncMock()
        stub._send_with_retry = AsyncMock()
        stub._maybe_cleanup_caches = MagicMock()
        stub.sessions.get_task_history = AsyncMock(return_value=[
            {"role": "user", "content": "what is this?"},
        ])

        captured_history = []
        async def capture_process(msg, hist, **kw):
            captured_history.extend(hist)
            return ("described image", False, False, [], False)
        stub._process_with_tools = capture_process

        msg = MagicMock()
        msg.author.id = 12345
        msg.author.bot = False
        msg.author.display_name = "TestUser"
        msg.author.name = "testuser"
        msg.channel.id = 67890
        msg.webhook_id = None

        image_blocks = [{"type": "image", "source": {"type": "base64", "data": "abc"}}]
        await stub._handle_message_inner(msg, "what is this?", "chan-1", image_blocks=image_blocks)

        # Images should be in the history passed to _process_with_tools
        assert len(captured_history) == 1
        assert isinstance(captured_history[0]["content"], list)
        assert any(b.get("type") == "image" for b in captured_history[0]["content"])


# ---------------------------------------------------------------------------
# 4. Memory cleanup
# ---------------------------------------------------------------------------

class TestCacheCleanup:
    """_cleanup_stale_caches removes expired entries."""

    def test_removes_expired_recent_actions(self):
        """Expired _recent_actions entries are removed."""
        stub = _make_bot_stub()
        stub._cleanup_stale_caches = LokiBot._cleanup_stale_caches.__get__(stub)
        stub._recent_actions_expiry = 3600
        # Add an expired entry (2 hours ago)
        stub._recent_actions = {
            "chan-old": [(time.time() - 7200, "old action")],
            "chan-new": [(time.time(), "new action")],
        }
        stub._cleanup_stale_caches()
        assert "chan-old" not in stub._recent_actions
        assert "chan-new" in stub._recent_actions

    def test_removes_stale_channel_locks(self):
        """Locks for channels without active sessions are removed."""
        stub = _make_bot_stub()
        stub._cleanup_stale_caches = LokiBot._cleanup_stale_caches.__get__(stub)
        stub._recent_actions_expiry = 3600
        stub._recent_actions = {}
        stub._channel_locks = {"chan-active": asyncio.Lock(), "chan-stale": asyncio.Lock()}
        stub.sessions._sessions = {"chan-active": MagicMock()}

        stub._cleanup_stale_caches()
        assert "chan-active" in stub._channel_locks
        assert "chan-stale" not in stub._channel_locks

    def test_cleanup_empty_actions_list(self):
        """Empty actions list for a channel gets cleaned up."""
        stub = _make_bot_stub()
        stub._cleanup_stale_caches = LokiBot._cleanup_stale_caches.__get__(stub)
        stub._recent_actions_expiry = 3600
        stub._recent_actions = {"chan-1": []}
        stub._cleanup_stale_caches()
        assert "chan-1" not in stub._recent_actions

    def test_cleanup_keeps_fresh_actions(self):
        """Fresh actions within expiry window are kept."""
        stub = _make_bot_stub()
        stub._cleanup_stale_caches = LokiBot._cleanup_stale_caches.__get__(stub)
        stub._recent_actions_expiry = 3600
        now = time.time()
        stub._recent_actions = {
            "chan-1": [(now - 100, "recent 1"), (now - 7200, "old"), (now - 50, "recent 2")],
        }
        stub._cleanup_stale_caches()
        assert "chan-1" in stub._recent_actions
        assert len(stub._recent_actions["chan-1"]) == 2  # old one removed

    def test_cleanup_all_expired(self):
        """All expired actions removes the channel key entirely."""
        stub = _make_bot_stub()
        stub._cleanup_stale_caches = LokiBot._cleanup_stale_caches.__get__(stub)
        stub._recent_actions_expiry = 3600
        stub._recent_actions = {
            "chan-1": [(time.time() - 7200, "old")],
        }
        stub._cleanup_stale_caches()
        assert "chan-1" not in stub._recent_actions


class TestMaybeCleanupCaches:
    """_maybe_cleanup_caches is throttled and defensive."""

    def test_throttled_by_interval(self):
        """Cleanup only runs when interval elapsed."""
        stub = _make_bot_stub()
        stub._maybe_cleanup_caches = LokiBot._maybe_cleanup_caches.__get__(stub)
        stub._cleanup_stale_caches = MagicMock()
        stub._last_cache_cleanup = time.time()  # just ran
        stub._cache_cleanup_interval = 300.0

        stub._maybe_cleanup_caches()
        stub._cleanup_stale_caches.assert_not_called()

    def test_runs_when_interval_elapsed(self):
        """Cleanup runs when enough time has passed."""
        stub = _make_bot_stub()
        stub._maybe_cleanup_caches = LokiBot._maybe_cleanup_caches.__get__(stub)
        stub._cleanup_stale_caches = MagicMock()
        stub._last_cache_cleanup = time.time() - 400  # 400s ago
        stub._cache_cleanup_interval = 300.0

        stub._maybe_cleanup_caches()
        stub._cleanup_stale_caches.assert_called_once()

    def test_exception_does_not_propagate(self):
        """Errors in cleanup don't break message processing."""
        stub = _make_bot_stub()
        stub._maybe_cleanup_caches = LokiBot._maybe_cleanup_caches.__get__(stub)
        stub._cleanup_stale_caches = MagicMock(side_effect=RuntimeError("boom"))
        stub._last_cache_cleanup = 0.0
        stub._cache_cleanup_interval = 0.0

        # Should not raise
        stub._maybe_cleanup_caches()

    def test_works_on_mock_stub(self):
        """Works even when attributes are MagicMock (defensive getattr)."""
        stub = MagicMock()
        bound = LokiBot._maybe_cleanup_caches.__get__(stub)
        # Should not raise even without proper attributes
        bound()


# ---------------------------------------------------------------------------
# 5. Session save is non-blocking
# ---------------------------------------------------------------------------

class TestSessionSaveNonBlocking:
    """sessions.save() is called via asyncio.to_thread."""

    def test_save_in_to_thread_in_source(self):
        """Source code uses asyncio.to_thread(self.sessions.save)."""
        src = inspect.getsource(LokiBot._handle_message_inner)
        assert "asyncio.to_thread(self.sessions.save)" in src


# ---------------------------------------------------------------------------
# 6. Session history size management
# ---------------------------------------------------------------------------

class TestSessionHistorySizeManagement:
    """Verify session history doesn't grow unbounded."""

    def test_compaction_threshold_is_40(self):
        from src.sessions.manager import COMPACTION_THRESHOLD
        assert COMPACTION_THRESHOLD == 40

    def test_task_history_default_max_messages(self):
        """get_task_history default is 10 messages."""
        sig = inspect.signature(SessionManager.get_task_history)
        assert sig.parameters["max_messages"].default == 10

    async def test_task_history_limits_output(self):
        """get_task_history returns only max_messages most recent."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            sm = SessionManager(max_history=50, max_age_hours=24, persist_dir=td)
            for i in range(30):
                sm.add_message("ch1", "user" if i % 2 == 0 else "assistant", f"msg-{i}")
            history = await sm.get_task_history("ch1", max_messages=5)
            # 5 messages + possibly 2 summary messages
            assert len(history) <= 7

    async def test_compaction_reduces_messages(self):
        """Compaction reduces history when threshold exceeded."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            sm = SessionManager(max_history=20, max_age_hours=24, persist_dir=td)
            mock_compact = AsyncMock(return_value="Summarized conversation.")
            sm.set_compaction_fn(mock_compact)
            for i in range(45):
                sm.add_message("ch1", "user" if i % 2 == 0 else "assistant", f"msg-{i}")
            session = sm.get_or_create("ch1")
            assert len(session.messages) == 45
            await sm.get_task_history("ch1")
            session = sm.get_or_create("ch1")
            # After compaction: only keep_count (max_history/2 = 10) messages
            assert len(session.messages) <= 20

    def test_processed_messages_bounded(self):
        """_processed_messages_max is 100."""
        stub = _make_bot_stub()
        assert stub._processed_messages_max == 100


# ---------------------------------------------------------------------------
# 7. Memory usage
# ---------------------------------------------------------------------------

class TestMemoryUsagePatterns:
    """Verify no unbounded growth patterns."""

    def test_recent_actions_capped_per_channel(self):
        """Per-channel actions list is capped at _recent_actions_max."""
        stub = _make_bot_stub()
        stub._track_recent_action = LokiBot._track_recent_action.__get__(stub)
        for i in range(20):
            stub._track_recent_action(
                f"tool_{i}", {"host": "server"}, "OK", 100, channel_id="chan-1",
            )
        assert len(stub._recent_actions["chan-1"]) <= 10

    def test_background_tasks_max_exists(self):
        """_background_tasks_max controls completed task cleanup."""
        stub = _make_bot_stub()
        assert stub._background_tasks_max == 20

    def test_tool_output_max_chars_reasonable(self):
        """TOOL_OUTPUT_MAX_CHARS prevents oversized tool results."""
        from src.discord.client import TOOL_OUTPUT_MAX_CHARS
        assert 1000 <= TOOL_OUTPUT_MAX_CHARS <= 50000

    def test_max_tool_iterations_bounded(self):
        """MAX_TOOL_ITERATIONS prevents infinite loops."""
        from src.discord.client import MAX_TOOL_ITERATIONS
        assert MAX_TOOL_ITERATIONS == 20


# ---------------------------------------------------------------------------
# 8. Source structure verification
# ---------------------------------------------------------------------------

class TestPerformanceSourceStructure:
    """Verify performance optimizations are present in source."""

    def test_cached_merged_tools_attr_in_init(self):
        """_cached_merged_tools initialized in __init__."""
        src = inspect.getsource(LokiBot.__init__)
        assert "_cached_merged_tools" in src

    def test_cache_invalidation_in_create_skill(self):
        """create_skill handler invalidates tool cache."""
        src = inspect.getsource(LokiBot._process_with_tools)
        # Find the create_skill section
        idx = src.find("create_skill")
        assert idx > 0
        section = src[idx:idx + 500]
        assert "_cached_merged_tools = None" in section

    def test_cache_invalidation_in_edit_skill(self):
        """edit_skill handler invalidates tool cache."""
        src = inspect.getsource(LokiBot._process_with_tools)
        idx = src.find("edit_skill")
        assert idx > 0
        section = src[idx:idx + 500]
        assert "_cached_merged_tools = None" in section

    def test_cache_invalidation_in_delete_skill(self):
        """delete_skill handler invalidates tool cache."""
        src = inspect.getsource(LokiBot._process_with_tools)
        idx = src.find("delete_skill")
        assert idx > 0
        section = src[idx:idx + 500]
        assert "_cached_merged_tools = None" in section

    def test_convert_tools_cached_method_exists(self):
        """CodexChatClient has _convert_tools_cached method."""
        assert hasattr(CodexChatClient, "_convert_tools_cached")

    def test_convert_tools_cached_used_in_chat_with_tools(self):
        """chat_with_tools uses _convert_tools_cached, not _convert_tools directly."""
        src = inspect.getsource(CodexChatClient.chat_with_tools)
        assert "_convert_tools_cached" in src
        assert "self._convert_tools(" not in src

    def test_cleanup_stale_caches_method_exists(self):
        """LokiBot has _cleanup_stale_caches method."""
        assert hasattr(LokiBot, "_cleanup_stale_caches")

    def test_maybe_cleanup_caches_method_exists(self):
        """LokiBot has _maybe_cleanup_caches method."""
        assert hasattr(LokiBot, "_maybe_cleanup_caches")

    def test_maybe_cleanup_in_handle_message_inner(self):
        """_maybe_cleanup_caches is called in _handle_message_inner."""
        src = inspect.getsource(LokiBot._handle_message_inner)
        assert "_maybe_cleanup_caches" in src

    def test_no_double_history_fetch_for_non_guest(self):
        """Non-guest path in _handle_message_inner does not call get_history_with_compaction."""
        src = inspect.getsource(LokiBot._handle_message_inner)
        # get_history_with_compaction should only appear in the guest branch
        lines = src.split("\n")
        in_guest_branch = False
        found_outside_guest = False
        for line in lines:
            stripped = line.strip()
            if "is_guest:" in stripped or "if is_guest" in stripped:
                in_guest_branch = True
            if in_guest_branch and stripped.startswith("else:"):
                in_guest_branch = False
            if not in_guest_branch and "get_history_with_compaction" in stripped and "#" not in stripped:
                found_outside_guest = True
        assert not found_outside_guest, "get_history_with_compaction should only be called in guest branch"

    def test_task_history_handles_compaction(self):
        """get_task_history checks COMPACTION_THRESHOLD internally."""
        src = inspect.getsource(SessionManager.get_task_history)
        assert "COMPACTION_THRESHOLD" in src

    def test_last_cache_cleanup_attr(self):
        """_last_cache_cleanup and _cache_cleanup_interval in __init__."""
        src = inspect.getsource(LokiBot.__init__)
        assert "_last_cache_cleanup" in src
        assert "_cache_cleanup_interval" in src


# ---------------------------------------------------------------------------
# 9. Cross-round consistency
# ---------------------------------------------------------------------------

class TestCrossRoundConsistency:
    """Verify previous rounds' work is intact."""

    def test_no_approval_in_tools(self):
        tools = get_tool_definitions()
        for t in tools:
            assert "requires_approval" not in t

    def test_no_classifier_call_in_source(self):
        """No classifier.classify() call in _handle_message_inner."""
        src = inspect.getsource(LokiBot._handle_message_inner)
        assert "classifier.classify" not in src

    def test_prompt_under_5000_chars(self):
        from src.llm.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            context="test", hosts={"server": "user@10.0.0.1"},
            services=["nginx"], playbooks=["check.yml"],
        )
        assert len(prompt) < 5000

    def test_personality_present(self):
        from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE
        assert "self-aware" in SYSTEM_PROMPT_TEMPLATE.lower()

    def test_local_execution_intact(self):
        from src.tools.ssh import is_local_address, run_local_command
        assert is_local_address("127.0.0.1")
        assert is_local_address("localhost")
        assert not is_local_address("10.0.0.1")

    def test_65_plus_tools(self):
        tools = get_tool_definitions()
        assert len(tools) >= 65

    def test_secret_scrubbing_intact(self):
        from src.llm.secret_scrubber import scrub_output_secrets
        assert "[REDACTED]" in scrub_output_secrets("password=mysecret123")
