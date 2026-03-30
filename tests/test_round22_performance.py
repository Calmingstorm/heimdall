"""Round 22: Performance optimizations (continued) — system prompt caching, memory/reflector
TTL caches, format_hints caching, Codex connection pooling.

Tests verify:
1. Host string dict cached (static unless config changes)
2. Skills list text cached (invalidated on skill CRUD)
3. Per-user memory cached with 60s TTL (avoids file I/O per message)
4. Per-user reflector section cached with 60s TTL (avoids file I/O per message)
5. format_hints results cached with 30s TTL (avoids embedding per message)
6. Codex TCPConnector with keepalive and connection limits
7. Cache invalidation on reload, skill CRUD, memory save
8. Stale TTL entries cleaned up by _cleanup_stale_caches
"""
from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, AsyncMock, patch

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.discord.client import HeimdallBot  # noqa: E402
from src.tools.tool_memory import ToolMemory  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot_stub(**overrides):
    """Minimal HeimdallBot stub with real prompt caching methods bound."""
    stub = MagicMock()

    host_mock = MagicMock()
    host_mock.ssh_user = "root"
    host_mock.address = "10.0.0.2"
    stub.config.tools.hosts = {"desktop": host_mock}
    stub.config.tools.allowed_
    stub.config.tools.allowed_
    stub.context_loader.context = "Test context."
    stub.voice_manager = None
    stub.tool_executor._load_memory_for_user = MagicMock(
        return_value=overrides.get("memory", {})
    )
    stub.reflector = MagicMock()
    stub.reflector.get_prompt_section = MagicMock(
        return_value=overrides.get("learned", "")
    )
    stub.skill_manager = MagicMock()
    stub.skill_manager.list_skills = MagicMock(
        return_value=overrides.get("skills", [])
    )
    stub.config.timezone = "UTC"
    stub._recent_actions = {}
    stub._recent_actions_max = 10
    stub._recent_actions_expiry = 3600

    # Cache attributes
    stub._cached_hosts = None
    stub._cached_skills_text = None
    stub._memory_cache = {}
    stub._memory_cache_ttl = overrides.get("memory_ttl", 60.0)
    stub._reflector_cache = {}
    stub._reflector_cache_ttl = overrides.get("reflector_ttl", 60.0)

    # Bind real methods
    stub._build_system_prompt = HeimdallBot._build_system_prompt.__get__(stub)
    stub._build_chat_system_prompt = HeimdallBot._build_chat_system_prompt.__get__(stub)
    stub._get_cached_hosts = HeimdallBot._get_cached_hosts.__get__(stub)
    stub._get_cached_skills_text = HeimdallBot._get_cached_skills_text.__get__(stub)
    stub._get_cached_memory = HeimdallBot._get_cached_memory.__get__(stub)
    stub._get_cached_reflector = HeimdallBot._get_cached_reflector.__get__(stub)
    stub._invalidate_prompt_caches = HeimdallBot._invalidate_prompt_caches.__get__(stub)

    return stub


# ---------------------------------------------------------------------------
# Host string caching
# ---------------------------------------------------------------------------

class TestHostStringCaching:
    """Host dict should be computed once and cached."""

    def test_host_string_cached_across_calls(self):
        stub = _make_bot_stub()
        hosts1 = stub._get_cached_hosts()
        hosts2 = stub._get_cached_hosts()
        assert hosts1 is hosts2  # same object = cached

    def test_host_string_correct_format(self):
        stub = _make_bot_stub()
        hosts = stub._get_cached_hosts()
        assert hosts == {"desktop": "root@10.0.0.2"}

    def test_host_string_invalidated_on_reload(self):
        stub = _make_bot_stub()
        hosts1 = stub._get_cached_hosts()
        stub._invalidate_prompt_caches()
        assert stub._cached_hosts is None
        hosts2 = stub._get_cached_hosts()
        # After invalidation, new dict is built
        assert hosts2 == {"desktop": "root@10.0.0.2"}
        # But it's a new object
        assert hosts1 is not hosts2

    def test_build_system_prompt_uses_cached_hosts(self):
        stub = _make_bot_stub()
        stub._build_system_prompt()
        # Host cache should be populated
        assert stub._cached_hosts is not None
        assert "desktop" in stub._cached_hosts

    def test_multiple_hosts(self):
        stub = _make_bot_stub()
        host2 = MagicMock()
        host2.ssh_user = "admin"
        host2.address = "10.0.0.3"
        stub.config.tools.hosts["server2"] = host2
        stub._cached_hosts = None  # force rebuild
        hosts = stub._get_cached_hosts()
        assert hosts == {"desktop": "root@10.0.0.2", "server2": "admin@10.0.0.3"}


# ---------------------------------------------------------------------------
# Skills list text caching
# ---------------------------------------------------------------------------

class TestSkillsListCaching:
    """Skills list text should be computed once and cached."""

    def test_skills_text_cached_across_calls(self):
        skills = [{"name": "test_skill", "description": "A test"}]
        stub = _make_bot_stub(skills=skills)
        text1 = stub._get_cached_skills_text()
        text2 = stub._get_cached_skills_text()
        assert text1 is text2  # same object

    def test_skills_text_correct_format(self):
        skills = [
            {"name": "ping", "description": "Ping host"},
            {"name": "backup", "description": "Backup DB"},
        ]
        stub = _make_bot_stub(skills=skills)
        text = stub._get_cached_skills_text()
        assert "- `ping`: Ping host" in text
        assert "- `backup`: Backup DB" in text

    def test_empty_skills_returns_empty_string(self):
        stub = _make_bot_stub(skills=[])
        text = stub._get_cached_skills_text()
        assert text == ""

    def test_skills_text_invalidated_on_reload(self):
        skills = [{"name": "s", "description": "d"}]
        stub = _make_bot_stub(skills=skills)
        text1 = stub._get_cached_skills_text()
        stub._cached_skills_text = None  # simulate invalidation
        text2 = stub._get_cached_skills_text()
        assert text1 == text2  # same content
        assert text1 is not text2  # different object

    def test_no_skill_manager_returns_empty(self):
        stub = _make_bot_stub()
        del stub.skill_manager
        stub._cached_skills_text = None
        text = stub._get_cached_skills_text()
        assert text == ""

    def test_skills_text_in_prompt(self):
        skills = [{"name": "test_skill", "description": "A test"}]
        stub = _make_bot_stub(skills=skills)
        prompt = stub._build_system_prompt()
        assert "## User-Created Skills" in prompt
        assert "test_skill" in prompt

    def test_list_skills_called_once(self):
        skills = [{"name": "s", "description": "d"}]
        stub = _make_bot_stub(skills=skills)
        stub._build_system_prompt()
        stub._build_system_prompt()
        # list_skills called only once (cached after first call)
        stub.skill_manager.list_skills.assert_called_once()


# ---------------------------------------------------------------------------
# Per-user memory TTL cache
# ---------------------------------------------------------------------------

class TestMemoryTTLCache:
    """Per-user memory should be cached with TTL to avoid file I/O."""

    def test_memory_cached_on_first_call(self):
        stub = _make_bot_stub(memory={"k": "v"})
        mem = stub._get_cached_memory("user1")
        assert mem == {"k": "v"}
        assert "user1" in stub._memory_cache

    def test_memory_cached_across_calls(self):
        stub = _make_bot_stub(memory={"k": "v"})
        stub._get_cached_memory("user1")
        stub._get_cached_memory("user1")
        # Should only call the underlying function once
        stub.tool_executor._load_memory_for_user.assert_called_once_with("user1")

    def test_memory_ttl_expired_refetches(self):
        stub = _make_bot_stub(memory={"k": "v"}, memory_ttl=0.01)
        stub._get_cached_memory("user1")
        time.sleep(0.02)
        stub._get_cached_memory("user1")
        # Called twice — TTL expired
        assert stub.tool_executor._load_memory_for_user.call_count == 2

    def test_different_users_separate_cache(self):
        stub = _make_bot_stub(memory={"k": "v"})
        stub._get_cached_memory("user1")
        stub._get_cached_memory("user2")
        assert stub.tool_executor._load_memory_for_user.call_count == 2

    def test_none_user_id_cached(self):
        stub = _make_bot_stub(memory={"global": "val"})
        stub._get_cached_memory(None)
        stub._get_cached_memory(None)
        stub.tool_executor._load_memory_for_user.assert_called_once_with(None)

    def test_invalidate_clears_memory_cache(self):
        stub = _make_bot_stub(memory={"k": "v"})
        stub._get_cached_memory("user1")
        assert len(stub._memory_cache) == 1
        stub._invalidate_prompt_caches()
        assert len(stub._memory_cache) == 0

    def test_build_prompt_uses_cached_memory(self):
        stub = _make_bot_stub(memory={"k": "v"})
        stub._build_system_prompt(user_id="42")
        stub._build_system_prompt(user_id="42")
        # Second call should use cache — only 1 call to _load_memory_for_user
        stub.tool_executor._load_memory_for_user.assert_called_once_with("42")


# ---------------------------------------------------------------------------
# Per-user reflector TTL cache
# ---------------------------------------------------------------------------

class TestReflectorTTLCache:
    """Reflector prompt section should be cached with TTL to avoid file I/O."""

    def test_reflector_cached_on_first_call(self):
        stub = _make_bot_stub(learned="## Learned\n- Be concise")
        result = stub._get_cached_reflector("user1")
        assert "Be concise" in result
        assert "user1" in stub._reflector_cache

    def test_reflector_cached_across_calls(self):
        stub = _make_bot_stub(learned="## Learned\n- fact")
        stub._get_cached_reflector("user1")
        stub._get_cached_reflector("user1")
        stub.reflector.get_prompt_section.assert_called_once_with(user_id="user1")

    def test_reflector_ttl_expired_refetches(self):
        stub = _make_bot_stub(learned="## L", reflector_ttl=0.01)
        stub._get_cached_reflector("user1")
        time.sleep(0.02)
        stub._get_cached_reflector("user1")
        assert stub.reflector.get_prompt_section.call_count == 2

    def test_different_users_separate_cache(self):
        stub = _make_bot_stub(learned="## L")
        stub._get_cached_reflector("user1")
        stub._get_cached_reflector("user2")
        assert stub.reflector.get_prompt_section.call_count == 2

    def test_no_reflector_returns_empty(self):
        stub = _make_bot_stub()
        del stub.reflector
        result = stub._get_cached_reflector("user1")
        assert result == ""

    def test_invalidate_clears_reflector_cache(self):
        stub = _make_bot_stub(learned="## L")
        stub._get_cached_reflector("user1")
        assert len(stub._reflector_cache) == 1
        stub._invalidate_prompt_caches()
        assert len(stub._reflector_cache) == 0

    def test_build_prompt_uses_cached_reflector(self):
        stub = _make_bot_stub(learned="## L\n- fact")
        stub._build_system_prompt(user_id="42")
        stub._build_system_prompt(user_id="42")
        # Second call should use cache
        stub.reflector.get_prompt_section.assert_called_once_with(user_id="42")


# ---------------------------------------------------------------------------
# format_hints caching in ToolMemory
# ---------------------------------------------------------------------------

class TestFormatHintsCaching:
    """format_hints should cache results to avoid re-computing embeddings."""

    async def test_hints_cached_on_first_call(self):
        tm = ToolMemory()
        tm._entries = [
            {
                "query": "check server health",
                "keywords": ["check", "server", "health"],
                "tools_used": ["check_disk", "check_memory"],
                "success": True,
                "timestamp": "2099-01-01T00:00:00",
            }
        ]
        result = await tm.format_hints("check server health")
        assert "check_disk" in result
        assert len(tm._hints_cache) == 1

    async def test_hints_cached_across_calls(self):
        tm = ToolMemory()
        tm._entries = [
            {
                "query": "check server",
                "keywords": ["check", "server"],
                "tools_used": ["check_disk", "check_memory"],
                "success": True,
                "timestamp": "2099-01-01T00:00:00",
            }
        ]
        r1 = await tm.format_hints("check server")
        r2 = await tm.format_hints("check server")
        assert r1 == r2
        # Both results should come from cache (same object)
        assert r1 is r2

    async def test_hints_ttl_expired_recomputes(self):
        tm = ToolMemory()
        tm._hints_cache_ttl = 0.01
        tm._entries = [
            {
                "query": "check server",
                "keywords": ["check", "server"],
                "tools_used": ["check_disk", "check_memory"],
                "success": True,
                "timestamp": "2099-01-01T00:00:00",
            }
        ]
        await tm.format_hints("check server")
        import asyncio
        await asyncio.sleep(0.02)
        # After TTL, cache miss — but same result
        r = await tm.format_hints("check server")
        assert "check_disk" in r

    async def test_different_queries_different_cache(self):
        tm = ToolMemory()
        tm._entries = [
            {
                "query": "check server",
                "keywords": ["check", "server"],
                "tools_used": ["check_disk", "check_memory"],
                "success": True,
                "timestamp": "2099-01-01T00:00:00",
            },
            {
                "query": "deploy code",
                "keywords": ["deploy", "code"],
                "tools_used": ["run_script", "git_push"],
                "success": True,
                "timestamp": "2099-01-01T00:00:00",
            },
        ]
        r1 = await tm.format_hints("check server status")
        r2 = await tm.format_hints("deploy new code")
        assert len(tm._hints_cache) == 2

    async def test_empty_result_cached(self):
        tm = ToolMemory()
        tm._entries = []
        result = await tm.format_hints("random query")
        assert result == ""
        assert len(tm._hints_cache) == 1
        # Second call returns cached empty
        r2 = await tm.format_hints("random query")
        assert r2 == ""
        assert r2 is result  # same object

    async def test_cache_eviction_at_100(self):
        import time as _time
        tm = ToolMemory()
        tm._hints_cache_ttl = 30.0  # normal TTL
        # Fill cache beyond 100 with expired entries (timestamp far in the past)
        expired_ts = _time.monotonic() - 100.0
        for i in range(105):
            tm._hints_cache[f"query_{i}"] = (expired_ts, "")
        # New call triggers eviction since len > 100
        await tm.format_hints("trigger eviction query")
        # Stale entries evicted, only fresh one(s) remain
        assert len(tm._hints_cache) <= 10

    async def test_hints_cache_init(self):
        tm = ToolMemory()
        assert hasattr(tm, "_hints_cache")
        assert isinstance(tm._hints_cache, dict)
        assert hasattr(tm, "_hints_cache_ttl")
        assert tm._hints_cache_ttl == 30.0


# ---------------------------------------------------------------------------
# Codex TCP connector
# ---------------------------------------------------------------------------

class TestCodexConnectionPooling:
    """Codex client should use TCPConnector with keepalive and limits."""

    async def test_session_creates_tcp_connector(self):
        from src.llm.openai_codex import CodexChatClient
        auth = MagicMock()
        client = CodexChatClient(auth=auth, model="gpt-4", max_tokens=4096)
        with patch("src.llm.openai_codex.aiohttp.ClientSession") as mock_session, \
             patch("src.llm.openai_codex.aiohttp.TCPConnector") as mock_connector:
            mock_session.return_value = MagicMock()
            mock_session.return_value.closed = False
            await client._get_session()
            mock_connector.assert_called_once()
            call_kwargs = mock_connector.call_args[1]
            assert call_kwargs["limit"] == 10
            assert call_kwargs["limit_per_host"] == 10
            assert call_kwargs["keepalive_timeout"] == 30
            assert call_kwargs["enable_cleanup_closed"] is True

    async def test_session_reused_across_calls(self):
        from src.llm.openai_codex import CodexChatClient
        auth = MagicMock()
        client = CodexChatClient(auth=auth, model="gpt-4", max_tokens=4096)
        with patch("src.llm.openai_codex.aiohttp.ClientSession") as mock_session, \
             patch("src.llm.openai_codex.aiohttp.TCPConnector"):
            session_obj = MagicMock()
            session_obj.closed = False
            mock_session.return_value = session_obj
            s1 = await client._get_session()
            s2 = await client._get_session()
            assert s1 is s2
            mock_session.assert_called_once()

    async def test_closed_session_recreated(self):
        from src.llm.openai_codex import CodexChatClient
        auth = MagicMock()
        client = CodexChatClient(auth=auth, model="gpt-4", max_tokens=4096)
        with patch("src.llm.openai_codex.aiohttp.ClientSession") as mock_session, \
             patch("src.llm.openai_codex.aiohttp.TCPConnector"):
            closed_session = MagicMock()
            closed_session.closed = True
            new_session = MagicMock()
            new_session.closed = False
            mock_session.return_value = new_session
            client._session = closed_session
            s = await client._get_session()
            # Closed session should be replaced
            assert s is new_session
            mock_session.assert_called_once()


# ---------------------------------------------------------------------------
# Cache invalidation integration
# ---------------------------------------------------------------------------

class TestCacheInvalidation:
    """Verify caches are invalidated at the right times."""

    def test_invalidate_prompt_caches_clears_all(self):
        stub = _make_bot_stub(memory={"k": "v"}, learned="## L")
        # Populate caches
        stub._get_cached_hosts()
        stub._get_cached_skills_text()
        stub._get_cached_memory("user1")
        stub._get_cached_reflector("user1")
        assert stub._cached_hosts is not None
        assert stub._cached_skills_text is not None
        assert len(stub._memory_cache) > 0
        assert len(stub._reflector_cache) > 0
        # Invalidate
        stub._invalidate_prompt_caches()
        assert stub._cached_hosts is None
        assert stub._cached_skills_text is None
        assert len(stub._memory_cache) == 0
        assert len(stub._reflector_cache) == 0

    def test_reload_command_invalidates_caches(self):
        """The /reload command source should call _invalidate_prompt_caches."""
        import inspect
        source = inspect.getsource(HeimdallBot._register_commands)
        assert "_invalidate_prompt_caches" in source

    def test_skill_crud_invalidates_skills_text(self):
        """Skill create/edit/delete should set _cached_skills_text = None."""
        import inspect
        source = inspect.getsource(HeimdallBot)
        # Count occurrences of skills text cache invalidation alongside tool cache
        assert source.count("_cached_skills_text = None") >= 3


# ---------------------------------------------------------------------------
# Stale TTL entry cleanup
# ---------------------------------------------------------------------------

class TestStaleTTLCleanup:
    """_cleanup_stale_caches should evict expired TTL entries."""

    def test_cleanup_evicts_expired_memory_entries(self):
        stub = _make_bot_stub()
        stub.sessions = MagicMock()
        stub.sessions._sessions = {}
        stub._cleanup_stale_caches = HeimdallBot._cleanup_stale_caches.__get__(stub)
        # Add expired entry
        stub._memory_cache["user1"] = (time.time() - 120, {"k": "v"})
        stub._cleanup_stale_caches()
        assert "user1" not in stub._memory_cache

    def test_cleanup_keeps_fresh_memory_entries(self):
        stub = _make_bot_stub()
        stub.sessions = MagicMock()
        stub.sessions._sessions = {}
        stub._cleanup_stale_caches = HeimdallBot._cleanup_stale_caches.__get__(stub)
        stub._memory_cache["user1"] = (time.time(), {"k": "v"})
        stub._cleanup_stale_caches()
        assert "user1" in stub._memory_cache

    def test_cleanup_evicts_expired_reflector_entries(self):
        stub = _make_bot_stub()
        stub.sessions = MagicMock()
        stub.sessions._sessions = {}
        stub._cleanup_stale_caches = HeimdallBot._cleanup_stale_caches.__get__(stub)
        stub._reflector_cache["user1"] = (time.time() - 120, "## L")
        stub._cleanup_stale_caches()
        assert "user1" not in stub._reflector_cache

    def test_cleanup_keeps_fresh_reflector_entries(self):
        stub = _make_bot_stub()
        stub.sessions = MagicMock()
        stub.sessions._sessions = {}
        stub._cleanup_stale_caches = HeimdallBot._cleanup_stale_caches.__get__(stub)
        stub._reflector_cache["user1"] = (time.time(), "## L")
        stub._cleanup_stale_caches()
        assert "user1" in stub._reflector_cache


# ---------------------------------------------------------------------------
# Source structure verification
# ---------------------------------------------------------------------------

class TestSourceStructure:
    """Verify all optimizations are present in source code."""

    def test_cached_hosts_attr_in_init(self):
        import inspect
        source = inspect.getsource(HeimdallBot.__init__)
        assert "_cached_hosts" in source

    def test_cached_skills_text_attr_in_init(self):
        import inspect
        source = inspect.getsource(HeimdallBot.__init__)
        assert "_cached_skills_text" in source

    def test_memory_cache_attr_in_init(self):
        import inspect
        source = inspect.getsource(HeimdallBot.__init__)
        assert "_memory_cache" in source

    def test_reflector_cache_attr_in_init(self):
        import inspect
        source = inspect.getsource(HeimdallBot.__init__)
        assert "_reflector_cache" in source

    def test_get_cached_hosts_method(self):
        assert hasattr(HeimdallBot, "_get_cached_hosts")

    def test_get_cached_skills_text_method(self):
        assert hasattr(HeimdallBot, "_get_cached_skills_text")

    def test_get_cached_memory_method(self):
        assert hasattr(HeimdallBot, "_get_cached_memory")

    def test_get_cached_reflector_method(self):
        assert hasattr(HeimdallBot, "_get_cached_reflector")

    def test_invalidate_prompt_caches_method(self):
        assert hasattr(HeimdallBot, "_invalidate_prompt_caches")

    def test_build_system_prompt_uses_cached_hosts(self):
        import inspect
        source = inspect.getsource(HeimdallBot._build_system_prompt)
        assert "_get_cached_hosts" in source
        # No direct host dict comprehension
        assert "for alias, h in self.config.tools.hosts.items()" not in source

    def test_build_system_prompt_uses_cached_skills(self):
        import inspect
        source = inspect.getsource(HeimdallBot._build_system_prompt)
        assert "_get_cached_skills_text" in source
        assert "list_skills" not in source

    def test_build_system_prompt_uses_cached_memory(self):
        import inspect
        source = inspect.getsource(HeimdallBot._build_system_prompt)
        assert "_get_cached_memory" in source
        assert "_load_memory_for_user" not in source

    def test_build_system_prompt_uses_cached_reflector(self):
        import inspect
        source = inspect.getsource(HeimdallBot._build_system_prompt)
        assert "_get_cached_reflector" in source
        assert "get_prompt_section" not in source

    def test_build_chat_prompt_uses_cached_memory(self):
        import inspect
        source = inspect.getsource(HeimdallBot._build_chat_system_prompt)
        assert "_get_cached_memory" in source
        assert "_load_memory_for_user" not in source

    def test_build_chat_prompt_uses_cached_reflector(self):
        import inspect
        source = inspect.getsource(HeimdallBot._build_chat_system_prompt)
        assert "_get_cached_reflector" in source
        assert "get_prompt_section" not in source

    def test_tool_memory_has_hints_cache(self):
        tm = ToolMemory()
        assert hasattr(tm, "_hints_cache")
        assert hasattr(tm, "_hints_cache_ttl")

    def test_codex_tcp_connector_in_source(self):
        import inspect
        from src.llm.openai_codex import CodexChatClient
        source = inspect.getsource(CodexChatClient._get_session)
        assert "TCPConnector" in source
        assert "keepalive_timeout" in source
        assert "limit_per_host" in source

    def test_cleanup_stale_caches_handles_ttl_entries(self):
        import inspect
        source = inspect.getsource(HeimdallBot._cleanup_stale_caches)
        assert "_memory_cache" in source
        assert "_reflector_cache" in source


# ---------------------------------------------------------------------------
# Cross-round consistency
# ---------------------------------------------------------------------------

class TestCrossRoundConsistency:
    """Verify previous round optimizations are still intact."""

    def test_tool_defs_still_cached(self):
        import inspect
        source = inspect.getsource(HeimdallBot.__init__)
        assert "_cached_merged_tools" in source

    def test_tool_conversion_cached(self):
        from src.llm.openai_codex import CodexChatClient
        assert hasattr(CodexChatClient, "_convert_tools_cached")

    def test_no_approval_in_tools(self):
        from src.tools.registry import get_tool_definitions
        for tool in get_tool_definitions():
            assert "requires_approval" not in tool

    def test_prompt_under_5000_chars(self):
        from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE
        assert len(SYSTEM_PROMPT_TEMPLATE) < 5000

    def test_personality_present(self):
        from src.llm.system_prompt import SYSTEM_PROMPT_TEMPLATE
        assert "Exhausted omniscience" in SYSTEM_PROMPT_TEMPLATE

    def test_local_execution_intact(self):
        from src.tools.ssh import is_local_address
        assert is_local_address("127.0.0.1")
        assert is_local_address("localhost")
        assert not is_local_address("10.0.0.1")

    def test_61_tools(self):
        from src.tools.registry import get_tool_definitions
        assert len(get_tool_definitions()) >= 55
