"""Round 46: Performance Optimization Tests.

Tests for:
- O(n²) → O(n) fence counting fix in combine_bot_messages
- Tool definitions caching by pack config
- Pre-computed tool→pack mapping in web API
- ZoneInfo caching in system prompt
- __slots__ on hot-path dataclasses (ToolCall, LLMResponse, Message, Session)
- Pre-compiled regex in combine_bot_messages
"""
from __future__ import annotations

import time
from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. combine_bot_messages: incremental fence counting (O(n) not O(n²))
# ---------------------------------------------------------------------------

class TestCombineBotMessagesPerformance:
    """Verify combine_bot_messages uses incremental fence counting."""

    def test_basic_join_two_text_parts(self):
        from src.discord.client import combine_bot_messages
        result = combine_bot_messages(["Hello", "World"])
        assert result == "Hello\n\nWorld"

    def test_unclosed_code_block_uses_single_newline(self):
        from src.discord.client import combine_bot_messages
        result = combine_bot_messages(["```python\ncode", "more code\n```"])
        assert "```python\ncode\nmore code\n```" in result

    def test_closed_block_uses_double_newline(self):
        from src.discord.client import combine_bot_messages
        result = combine_bot_messages(["```\ncode\n```", "Next part"])
        assert "\n\n" in result

    def test_single_part_returns_as_is(self):
        from src.discord.client import combine_bot_messages
        assert combine_bot_messages(["only one"]) == "only one"

    def test_empty_list_returns_empty(self):
        from src.discord.client import combine_bot_messages
        assert combine_bot_messages([]) == ""

    def test_many_parts_performance(self):
        """Ensure O(n) behavior — many parts should be fast."""
        from src.discord.client import combine_bot_messages
        # 100 parts with alternating code blocks
        parts = []
        for i in range(50):
            parts.append(f"```\ncode block {i}\n```")
            parts.append(f"Text part {i}")

        start = time.monotonic()
        result = combine_bot_messages(parts)
        elapsed = time.monotonic() - start

        assert "code block 0" in result
        assert "Text part 49" in result
        # Should complete in well under 100ms even on slow CI
        assert elapsed < 0.1

    def test_fence_tracking_accuracy_with_triple_backtick_in_text(self):
        """Fence counting should track ``` correctly across parts."""
        from src.discord.client import combine_bot_messages
        # Part 1: opens a code block
        # Part 2: has text (should be single-newline joined since block is open)
        # Part 3: closes the block
        result = combine_bot_messages(["```bash", "echo hello", "```"])
        # All three parts should be joined with single newline (block open through all)
        assert "```bash\necho hello\n```" in result

    def test_adjacent_fence_merging_still_works(self):
        """The regex merge for adjacent fences should still work."""
        from src.discord.client import combine_bot_messages
        result = combine_bot_messages(["```\ncode1\n```", "```bash\ncode2\n```"])
        # Adjacent fences should be merged
        assert "```\n\n```" not in result

    def test_pre_compiled_regex_used(self):
        """Verify _ADJACENT_FENCE_RE is a compiled pattern at module level."""
        from src.discord.client import _ADJACENT_FENCE_RE
        import re
        assert isinstance(_ADJACENT_FENCE_RE, re.Pattern)


# ---------------------------------------------------------------------------
# 2. Tool definitions caching by pack config
# ---------------------------------------------------------------------------

class TestToolDefinitionsCache:
    """Test that get_tool_definitions caches results by pack config."""

    def test_cache_returns_same_object(self):
        from src.tools.registry import get_tool_definitions
        result1 = get_tool_definitions()
        result2 = get_tool_definitions()
        assert result1 is result2  # Same list object — cached

    def test_cache_different_packs(self):
        from src.tools.registry import get_tool_definitions
        all_tools = get_tool_definitions()
        systemd_only = get_tool_definitions(enabled_packs=["systemd"])
        # Different results for different pack configs
        assert len(all_tools) >= len(systemd_only)

    def test_cache_same_packs_returns_same_object(self):
        from src.tools.registry import get_tool_definitions
        r1 = get_tool_definitions(enabled_packs=["systemd"])
        r2 = get_tool_definitions(enabled_packs=["systemd"])
        assert r1 is r2

    def test_invalidate_clears_cache(self):
        from src.tools.registry import get_tool_definitions, invalidate_tool_defs_cache
        r1 = get_tool_definitions()
        invalidate_tool_defs_cache()
        r2 = get_tool_definitions()
        assert r1 is not r2  # New list object after invalidation
        assert len(r1) == len(r2)  # Same content

    def test_frozenset_order_independence(self):
        """Pack order shouldn't matter for cache key."""
        from src.tools.registry import get_tool_definitions
        r1 = get_tool_definitions(enabled_packs=["systemd", "incus"])
        r2 = get_tool_definitions(enabled_packs=["incus", "systemd"])
        assert r1 is r2  # frozenset makes order irrelevant

    def test_empty_list_treated_as_no_packs(self):
        from src.tools.registry import get_tool_definitions
        r1 = get_tool_definitions()
        r2 = get_tool_definitions(enabled_packs=[])
        assert r1 is r2  # Both use frozenset() as key

    def test_cache_preserves_correct_count(self):
        from src.tools.registry import get_tool_definitions, TOOLS
        result = get_tool_definitions()
        assert len(result) == len(TOOLS)

    def test_invalidate_function_exported(self):
        from src.tools.registry import invalidate_tool_defs_cache
        assert callable(invalidate_tool_defs_cache)


# ---------------------------------------------------------------------------
# 3. Pre-computed tool→pack mapping in web API
# ---------------------------------------------------------------------------

class TestToolToPackMapping:
    """Test the _TOOL_TO_PACK pre-computed mapping."""

    def test_mapping_exists(self):
        from src.web.api import _TOOL_TO_PACK
        assert isinstance(_TOOL_TO_PACK, dict)
        assert len(_TOOL_TO_PACK) > 0

    def test_mapping_covers_all_pack_tools(self):
        from src.web.api import _TOOL_TO_PACK
        from src.tools.registry import TOOL_PACKS
        for pack_name, tool_names in TOOL_PACKS.items():
            for tool_name in tool_names:
                assert tool_name in _TOOL_TO_PACK
                assert _TOOL_TO_PACK[tool_name] == pack_name

    def test_core_tools_not_in_mapping(self):
        """Core tools (not in any pack) should not appear in _TOOL_TO_PACK."""
        from src.web.api import _TOOL_TO_PACK
        from src.tools.registry import TOOLS, _ALL_PACK_TOOLS
        for tool in TOOLS:
            if tool["name"] not in _ALL_PACK_TOOLS:
                assert tool["name"] not in _TOOL_TO_PACK

    def test_mapping_is_dict_constant_time(self):
        """Mapping lookup is O(1) — no nested loop."""
        from src.web.api import _TOOL_TO_PACK
        # Just verify it's a plain dict (O(1) lookup)
        assert type(_TOOL_TO_PACK) is dict

    def test_systemd_tools_mapped_correctly(self):
        from src.web.api import _TOOL_TO_PACK
        for tool in ["check_service", "restart_service", "check_logs"]:
            assert _TOOL_TO_PACK.get(tool) == "systemd"


# ---------------------------------------------------------------------------
# 4. ZoneInfo caching in system prompt
# ---------------------------------------------------------------------------

class TestZoneInfoCaching:
    """Test that ZoneInfo objects are cached via lru_cache."""

    def test_get_zone_returns_zoneinfo(self):
        from src.llm.system_prompt import _get_zone
        from zoneinfo import ZoneInfo
        tz = _get_zone("UTC")
        assert isinstance(tz, ZoneInfo)

    def test_get_zone_caches(self):
        from src.llm.system_prompt import _get_zone
        tz1 = _get_zone("America/New_York")
        tz2 = _get_zone("America/New_York")
        assert tz1 is tz2  # Same object — cached

    def test_get_zone_different_zones(self):
        from src.llm.system_prompt import _get_zone
        utc = _get_zone("UTC")
        eastern = _get_zone("America/New_York")
        assert utc is not eastern

    def test_format_datetime_uses_cached_zone(self):
        """_format_datetime should work correctly with cached zones."""
        from src.llm.system_prompt import _format_datetime
        result = _format_datetime("UTC")
        assert "UTC" in result

    def test_build_system_prompt_works_with_cache(self):
        from src.llm.system_prompt import build_system_prompt
        result = build_system_prompt(
            context="Test context",
            hosts={"server": "root@10.0.0.1"},
            services=["nginx"],
            playbooks=["deploy.yml"],
            tz="America/Chicago",
        )
        assert "Heimdall" in result
        assert "server" in result

    def test_build_chat_system_prompt_works_with_cache(self):
        from src.llm.system_prompt import build_chat_system_prompt
        result = build_chat_system_prompt(tz="Europe/London")
        assert "Heimdall" in result


# ---------------------------------------------------------------------------
# 5. __slots__ on hot-path dataclasses
# ---------------------------------------------------------------------------

class TestSlotsDataclasses:
    """Verify __slots__ is set on hot-path dataclasses."""

    def test_tool_call_has_slots(self):
        from src.llm.types import ToolCall
        assert hasattr(ToolCall, "__slots__")
        tc = ToolCall(id="1", name="test", input={})
        assert not hasattr(tc, "__dict__")

    def test_llm_response_has_slots(self):
        from src.llm.types import LLMResponse
        assert hasattr(LLMResponse, "__slots__")
        r = LLMResponse()
        assert not hasattr(r, "__dict__")

    def test_message_has_slots(self):
        from src.sessions.manager import Message
        assert hasattr(Message, "__slots__")
        m = Message(role="user", content="test")
        assert not hasattr(m, "__dict__")

    def test_session_has_slots(self):
        from src.sessions.manager import Session
        assert hasattr(Session, "__slots__")
        s = Session(channel_id="123")
        assert not hasattr(s, "__dict__")

    def test_tool_call_fields_accessible(self):
        from src.llm.types import ToolCall
        tc = ToolCall(id="call_1", name="run_command", input={"cmd": "ls"})
        assert tc.id == "call_1"
        assert tc.name == "run_command"
        assert tc.input == {"cmd": "ls"}

    def test_llm_response_defaults(self):
        from src.llm.types import LLMResponse
        r = LLMResponse()
        assert r.text == ""
        assert r.tool_calls == []
        assert r.stop_reason == "end_turn"

    def test_llm_response_is_tool_use(self):
        from src.llm.types import LLMResponse, ToolCall
        r = LLMResponse(
            tool_calls=[ToolCall(id="1", name="test", input={})],
            stop_reason="tool_use",
        )
        assert r.is_tool_use is True

    def test_message_defaults(self):
        from src.sessions.manager import Message
        m = Message(role="assistant", content="Hello")
        assert m.role == "assistant"
        assert m.content == "Hello"
        assert m.user_id is None
        assert m.timestamp > 0

    def test_session_defaults(self):
        from src.sessions.manager import Session
        s = Session(channel_id="ch_1")
        assert s.channel_id == "ch_1"
        assert s.messages == []
        assert s.summary == ""
        assert s.last_user_id is None

    def test_session_asdict_still_works(self):
        """dataclasses.asdict must work with slots for serialization."""
        from dataclasses import asdict
        from src.sessions.manager import Session, Message
        s = Session(
            channel_id="ch_1",
            messages=[Message(role="user", content="hello")],
            summary="test summary",
        )
        d = asdict(s)
        assert d["channel_id"] == "ch_1"
        assert len(d["messages"]) == 1
        assert d["messages"][0]["content"] == "hello"


# ---------------------------------------------------------------------------
# 6. Cross-optimization integration tests
# ---------------------------------------------------------------------------

class TestPerformanceIntegration:
    """Integration tests verifying optimizations work together."""

    def test_tool_defs_cache_with_invalidation_cycle(self):
        """Full cycle: cache → invalidate → re-cache."""
        from src.tools.registry import (
            get_tool_definitions, invalidate_tool_defs_cache, TOOLS,
        )
        # First call populates cache
        r1 = get_tool_definitions()
        assert len(r1) == len(TOOLS)

        # Same call returns cached
        r2 = get_tool_definitions()
        assert r1 is r2

        # Invalidate and re-fetch
        invalidate_tool_defs_cache()
        r3 = get_tool_definitions()
        assert r1 is not r3
        assert len(r3) == len(TOOLS)

    def test_combine_bot_messages_with_many_code_blocks(self):
        """Stress test: many alternating code blocks and text."""
        from src.discord.client import combine_bot_messages
        parts = []
        for i in range(20):
            parts.append(f"```python\ndef func_{i}():\n    pass\n```")
            parts.append(f"Function {i} defined.")
        result = combine_bot_messages(parts)
        assert "func_0" in result
        assert "func_19" in result
        assert "Function 19 defined." in result

    def test_system_prompt_under_5000_chars(self):
        """System prompt must stay under 5000 chars (existing constraint)."""
        from src.llm.system_prompt import build_system_prompt
        prompt = build_system_prompt(
            context="Test context",
            hosts={"server": "root@10.0.0.1", "backup": "root@10.0.0.2"},
            services=["nginx", "postgres", "redis"],
            playbooks=["deploy.yml", "backup.yml"],
            tz="America/New_York",
        )
        assert len(prompt) < 5000

    def test_tool_definitions_structure_preserved(self):
        """Cached tool definitions must have required fields."""
        from src.tools.registry import get_tool_definitions
        tools = get_tool_definitions()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert isinstance(tool["name"], str)
            assert isinstance(tool["description"], str)
            assert isinstance(tool["input_schema"], dict)

    def test_llm_response_memory_efficiency(self):
        """Slots-based objects should use less memory than dict-based."""
        from src.llm.types import ToolCall
        import sys
        # With __slots__, instances don't have __dict__
        tc = ToolCall(id="1", name="test", input={})
        assert not hasattr(tc, "__dict__")
        # Basic size check — slots instances are smaller
        size = sys.getsizeof(tc)
        assert size < 200  # Reasonable size for a small slotted object
