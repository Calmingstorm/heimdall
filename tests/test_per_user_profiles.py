"""Tests for per-user profiles: scoped memory, learned context, and system prompt injection.

Session 40 (R6 S2): Adds per-user profile support to Loki so that memory
notes, learned preferences, and system prompt context are scoped per user.

Tests cover:
- Memory format migration (flat dict → scoped dict)
- Per-user memory save/list/delete with scope parameter
- User context setting on ToolExecutor
- Per-user learned context filtering in reflector
- Reflector user_id tagging on preference/correction entries
- Session last_user_id tracking
- System prompt injection with per-user memory
- Registry tool definition updates
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Stub out heavy discord extension before any src.discord imports
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.tools.executor import ToolExecutor  # noqa: E402
from src.tools.registry import get_tool_definitions  # noqa: E402
from src.learning.reflector import ConversationReflector  # noqa: E402
from src.sessions.manager import SessionManager, Session, Message  # noqa: E402


# ===========================================================================
# Memory format migration
# ===========================================================================

class TestMemoryFormatMigration:
    """Old flat memory.json format should be auto-migrated to scoped format."""

    def test_flat_dict_migrated_to_global(self, tmp_path):
        """Old flat dict {key: value} becomes {"global": {key: value}}."""
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps({"owner": "TestUser", "timezone": "ET"}))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = executor._load_all_memory()
        assert "global" in result
        assert result["global"]["owner"] == "TestUser"
        assert result["global"]["timezone"] == "ET"

    def test_flat_dict_migration_persists(self, tmp_path):
        """After migration, the file should be saved in new format."""
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps({"key": "value"}))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        executor._load_all_memory()
        # Re-read the file directly
        saved = json.loads(mem_file.read_text())
        assert "global" in saved
        assert saved["global"]["key"] == "value"

    def test_already_scoped_format_not_re_migrated(self, tmp_path):
        """If file already has "global" key, don't re-wrap."""
        data = {"global": {"a": "1"}, "user_123": {"b": "2"}}
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps(data))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = executor._load_all_memory()
        assert result == data

    def test_missing_file_returns_empty_global(self, tmp_path):
        executor = ToolExecutor(MagicMock(), memory_path=str(tmp_path / "nope.json"))
        result = executor._load_all_memory()
        assert result == {"global": {}}

    def test_corrupt_json_returns_empty_global(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        mem_file.write_text("not valid json!!!")
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = executor._load_all_memory()
        assert result == {"global": {}}

    def test_non_dict_json_returns_empty_global(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps([1, 2, 3]))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = executor._load_all_memory()
        assert result == {"global": {}}

    def test_no_memory_path_returns_empty_global(self):
        executor = ToolExecutor(MagicMock(), memory_path=None)
        result = executor._load_all_memory()
        assert result == {"global": {}}


# ===========================================================================
# User context and memory scoping
# ===========================================================================

class TestUserContext:
    """ToolExecutor should track current user for memory scoping."""

    def test_set_user_context(self):
        executor = ToolExecutor(MagicMock())
        executor.set_user_context("12345")
        assert executor._current_user_id == "12345"

    def test_set_user_context_none(self):
        executor = ToolExecutor(MagicMock())
        executor.set_user_context(None)
        assert executor._current_user_id is None


class TestLoadMemoryForUser:
    """_load_memory_for_user should merge global + user-specific entries."""

    def test_global_only(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps({"global": {"infra": "5 hosts"}}))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = executor._load_memory_for_user("999")
        assert result == {"infra": "5 hosts"}

    def test_user_specific_merged(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        data = {
            "global": {"infra": "5 hosts"},
            "user_123": {"birthday": "Feb 7"},
        }
        mem_file.write_text(json.dumps(data))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = executor._load_memory_for_user("123")
        assert result == {"infra": "5 hosts", "birthday": "Feb 7"}

    def test_user_overrides_global(self, tmp_path):
        """User-specific entries should override global entries with same key."""
        mem_file = tmp_path / "memory.json"
        data = {
            "global": {"greeting": "Hello"},
            "user_123": {"greeting": "Hey dude"},
        }
        mem_file.write_text(json.dumps(data))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = executor._load_memory_for_user("123")
        assert result["greeting"] == "Hey dude"

    def test_other_user_entries_excluded(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        data = {
            "global": {"infra": "5 hosts"},
            "user_111": {"secret": "mine"},
            "user_222": {"other": "theirs"},
        }
        mem_file.write_text(json.dumps(data))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = executor._load_memory_for_user("111")
        assert "secret" in result
        assert "other" not in result

    def test_none_user_id_returns_global_only(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        data = {"global": {"a": "1"}, "user_123": {"b": "2"}}
        mem_file.write_text(json.dumps(data))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = executor._load_memory_for_user(None)
        assert result == {"a": "1"}


# ===========================================================================
# Memory manage tool — per-user scoping
# ===========================================================================

class TestMemoryManagePerUser:
    """memory_manage tool should scope saves per user by default."""

    async def test_save_personal_default(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps({"global": {}}))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = await executor._handle_memory_manage(
            {"action": "save", "key": "color", "value": "blue"}, user_id="42"
        )
        assert "personal" in result
        data = json.loads(mem_file.read_text())
        assert data["user_42"]["color"] == "blue"
        assert "color" not in data["global"]

    async def test_save_global_scope(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps({"global": {}}))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = await executor._handle_memory_manage(
            {"action": "save", "key": "dns", "value": "Pi-hole", "scope": "global"}, user_id="42"
        )
        assert "global" in result
        data = json.loads(mem_file.read_text())
        assert data["global"]["dns"] == "Pi-hole"

    async def test_save_without_user_context_goes_global(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps({"global": {}}))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = await executor._handle_memory_manage(
            {"action": "save", "key": "fact", "value": "test"}
        )
        assert "global" in result
        data = json.loads(mem_file.read_text())
        assert data["global"]["fact"] == "test"

    async def test_list_shows_global_and_personal(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        data = {
            "global": {"infra": "5 hosts"},
            "user_42": {"color": "blue"},
        }
        mem_file.write_text(json.dumps(data))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = await executor._handle_memory_manage({"action": "list"}, user_id="42")
        assert "Global notes" in result
        assert "infra" in result
        assert "Your personal notes" in result
        assert "color" in result

    async def test_list_empty(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps({"global": {}}))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = await executor._handle_memory_manage({"action": "list"}, user_id="42")
        assert "No notes saved yet" in result

    async def test_list_other_user_notes_hidden(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        data = {
            "global": {},
            "user_42": {"mine": "yes"},
            "user_99": {"theirs": "private"},
        }
        mem_file.write_text(json.dumps(data))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = await executor._handle_memory_manage({"action": "list"}, user_id="42")
        assert "mine" in result
        assert "theirs" not in result

    async def test_delete_personal_first(self, tmp_path):
        """Delete should check user section first."""
        mem_file = tmp_path / "memory.json"
        data = {
            "global": {"shared": "global_val"},
            "user_42": {"shared": "user_val"},
        }
        mem_file.write_text(json.dumps(data))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = await executor._handle_memory_manage(
            {"action": "delete", "key": "shared"}, user_id="42"
        )
        assert "personal" in result
        saved = json.loads(mem_file.read_text())
        # User entry deleted, global entry preserved
        assert "shared" not in saved["user_42"]
        assert saved["global"]["shared"] == "global_val"

    async def test_delete_falls_back_to_global(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        data = {"global": {"only_global": "val"}}
        mem_file.write_text(json.dumps(data))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = await executor._handle_memory_manage(
            {"action": "delete", "key": "only_global"}, user_id="42"
        )
        assert "global" in result
        saved = json.loads(mem_file.read_text())
        assert "only_global" not in saved["global"]

    async def test_delete_not_found(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps({"global": {}}))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = await executor._handle_memory_manage(
            {"action": "delete", "key": "nope"}, user_id="42"
        )
        assert "No note found" in result

    async def test_save_missing_key(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps({"global": {}}))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = await executor._handle_memory_manage(
            {"action": "save", "value": "no key"}
        )
        assert "required" in result

    async def test_save_missing_value(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps({"global": {}}))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = await executor._handle_memory_manage(
            {"action": "save", "key": "k"}
        )
        assert "required" in result

    async def test_unknown_action(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        mem_file.write_text(json.dumps({"global": {}}))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = await executor._handle_memory_manage({"action": "explode"})
        assert "Unknown memory action" in result


# ===========================================================================
# Backward-compatible _load_memory (for old callers)
# ===========================================================================

class TestLoadMemoryBackcompat:
    """_load_memory should return global section for backward compat."""

    def test_returns_global_entries(self, tmp_path):
        mem_file = tmp_path / "memory.json"
        data = {"global": {"k": "v"}, "user_1": {"x": "y"}}
        mem_file.write_text(json.dumps(data))
        executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
        result = executor._load_memory()
        assert result == {"k": "v"}


# ===========================================================================
# Registry tool definition
# ===========================================================================

class TestMemoryManageRegistry:
    """memory_manage tool definition should include scope parameter."""

    def test_scope_property_exists(self):
        tools = get_tool_definitions()
        mm = next(t for t in tools if t["name"] == "memory_manage")
        props = mm["input_schema"]["properties"]
        assert "scope" in props
        assert props["scope"]["enum"] == ["personal", "global"]

    def test_description_mentions_per_user(self):
        tools = get_tool_definitions()
        mm = next(t for t in tools if t["name"] == "memory_manage")
        assert "personal" in mm["description"].lower()
        assert "global" in mm["description"].lower()


# ===========================================================================
# Reflector per-user learned context
# ===========================================================================

class TestReflectorPerUser:
    """ConversationReflector should tag and filter entries by user_id."""

    def test_get_prompt_section_global_entries_always_included(self, tmp_path):
        path = tmp_path / "learned.json"
        data = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": "fact1", "category": "operational", "content": "5 hosts"},
            ],
        }
        path.write_text(json.dumps(data))
        reflector = ConversationReflector(str(path), enabled=False)
        section = reflector.get_prompt_section(user_id="42")
        assert "5 hosts" in section

    def test_get_prompt_section_user_entries_included_for_matching_user(self, tmp_path):
        path = tmp_path / "learned.json"
        data = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": "pref1", "category": "preference", "content": "likes blue",
                 "user_id": "42"},
            ],
        }
        path.write_text(json.dumps(data))
        reflector = ConversationReflector(str(path), enabled=False)
        section = reflector.get_prompt_section(user_id="42")
        assert "likes blue" in section

    def test_get_prompt_section_user_entries_excluded_for_other_user(self, tmp_path):
        path = tmp_path / "learned.json"
        data = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": "pref1", "category": "preference", "content": "likes blue",
                 "user_id": "42"},
                {"key": "fact1", "category": "fact", "content": "global fact"},
            ],
        }
        path.write_text(json.dumps(data))
        reflector = ConversationReflector(str(path), enabled=False)
        section = reflector.get_prompt_section(user_id="99")
        assert "likes blue" not in section
        assert "global fact" in section

    def test_get_prompt_section_no_user_id_returns_global_only(self, tmp_path):
        path = tmp_path / "learned.json"
        data = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": "pref1", "category": "preference", "content": "user pref",
                 "user_id": "42"},
                {"key": "fact1", "category": "fact", "content": "global fact"},
            ],
        }
        path.write_text(json.dumps(data))
        reflector = ConversationReflector(str(path), enabled=False)
        section = reflector.get_prompt_section(user_id=None)
        assert "user pref" not in section
        assert "global fact" in section

    def test_get_prompt_section_empty_when_all_filtered(self, tmp_path):
        path = tmp_path / "learned.json"
        data = {
            "version": 1,
            "last_reflection": None,
            "entries": [
                {"key": "pref1", "category": "preference", "content": "user pref",
                 "user_id": "42"},
            ],
        }
        path.write_text(json.dumps(data))
        reflector = ConversationReflector(str(path), enabled=False)
        section = reflector.get_prompt_section(user_id="99")
        assert section == ""

    async def test_reflect_tags_preferences_with_user_id(self, tmp_path):
        """When _reflect is called with a single user_id, preference entries get tagged."""
        path = tmp_path / "learned.json"
        path.write_text(json.dumps({"version": 1, "last_reflection": None, "entries": []}))

        text_fn = AsyncMock(return_value=json.dumps([
            {"key": "likes_dark_mode", "category": "preference", "content": "Prefers dark mode"},
            {"key": "server_fact", "category": "operational", "content": "Server runs Ubuntu"},
        ]))

        reflector = ConversationReflector(str(path), enabled=True)
        reflector.set_text_fn(text_fn)

        await reflector._reflect("test conversation", full=True, user_ids=["42"])

        data = json.loads(path.read_text())
        entries = {e["key"]: e for e in data["entries"]}
        # Preference should be tagged with user_id
        assert entries["likes_dark_mode"].get("user_id") == "42"
        # Operational should NOT be tagged (it's global)
        assert "user_id" not in entries["server_fact"]

    async def test_reflect_corrections_tagged_with_user_id(self, tmp_path):
        """Correction entries should also be tagged with user_id."""
        path = tmp_path / "learned.json"
        path.write_text(json.dumps({"version": 1, "last_reflection": None, "entries": []}))

        text_fn = AsyncMock(return_value=json.dumps([
            {"key": "no_emojis", "category": "correction", "content": "Don't use emojis"},
        ]))

        reflector = ConversationReflector(str(path), enabled=True)
        reflector.set_text_fn(text_fn)

        await reflector._reflect("test conversation", full=True, user_ids=["99"])

        data = json.loads(path.read_text())
        assert data["entries"][0].get("user_id") == "99"

    async def test_reflect_no_user_id_no_tagging(self, tmp_path):
        """Without user_ids, entries should not get user_id field."""
        path = tmp_path / "learned.json"
        path.write_text(json.dumps({"version": 1, "last_reflection": None, "entries": []}))

        text_fn = AsyncMock(return_value=json.dumps([
            {"key": "pref", "category": "preference", "content": "Some preference"},
        ]))

        reflector = ConversationReflector(str(path), enabled=True)
        reflector.set_text_fn(text_fn)

        await reflector._reflect("test conversation", full=True, user_ids=[])

        data = json.loads(path.read_text())
        assert "user_id" not in data["entries"][0]

    def test_reflect_on_session_passes_user_ids(self, tmp_path):
        """reflect_on_session should pass user_ids through to _reflect."""
        path = tmp_path / "learned.json"
        path.write_text(json.dumps({"version": 1, "last_reflection": None, "entries": []}))
        reflector = ConversationReflector(str(path), enabled=True)

        session = Session(channel_id="ch1")
        session.messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
            Message(role="user", content="thanks"),
        ]

        with patch.object(reflector, "_reflect", new_callable=AsyncMock) as mock_reflect:
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                reflector.reflect_on_session(session, user_ids=["42"])
            )
            mock_reflect.assert_called_once()
            assert mock_reflect.call_args.kwargs["user_ids"] == ["42"]

    def test_reflect_on_compacted_passes_user_ids(self, tmp_path):
        """reflect_on_compacted should pass user_ids through to _reflect."""
        path = tmp_path / "learned.json"
        path.write_text(json.dumps({"version": 1, "last_reflection": None, "entries": []}))
        reflector = ConversationReflector(str(path), enabled=True)

        messages = [Message(role="user", content=f"msg {i}") for i in range(6)]

        with patch.object(reflector, "_reflect", new_callable=AsyncMock) as mock_reflect:
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                reflector.reflect_on_compacted(messages, "summary", user_ids=["77"])
            )
            mock_reflect.assert_called_once()
            assert mock_reflect.call_args.kwargs["user_ids"] == ["77"]


# ===========================================================================
# Session last_user_id tracking
# ===========================================================================

class TestSessionUserTracking:
    """Sessions should track the last user who sent a message."""

    def test_add_message_tracks_user_id(self, tmp_path):
        mgr = SessionManager(
            max_history=50, max_age_hours=24,
            persist_dir=str(tmp_path / "sessions"),
        )
        mgr.add_message("ch1", "user", "hello", user_id="42")
        session = mgr.get_or_create("ch1")
        assert session.last_user_id == "42"

    def test_add_message_updates_user_id(self, tmp_path):
        mgr = SessionManager(
            max_history=50, max_age_hours=24,
            persist_dir=str(tmp_path / "sessions"),
        )
        mgr.add_message("ch1", "user", "hello", user_id="42")
        mgr.add_message("ch1", "user", "hey", user_id="99")
        session = mgr.get_or_create("ch1")
        assert session.last_user_id == "99"

    def test_assistant_message_does_not_change_user_id(self, tmp_path):
        mgr = SessionManager(
            max_history=50, max_age_hours=24,
            persist_dir=str(tmp_path / "sessions"),
        )
        mgr.add_message("ch1", "user", "hello", user_id="42")
        mgr.add_message("ch1", "assistant", "hi there")
        session = mgr.get_or_create("ch1")
        assert session.last_user_id == "42"

    def test_add_message_without_user_id(self, tmp_path):
        mgr = SessionManager(
            max_history=50, max_age_hours=24,
            persist_dir=str(tmp_path / "sessions"),
        )
        mgr.add_message("ch1", "user", "hello")
        session = mgr.get_or_create("ch1")
        assert session.last_user_id is None

    def test_session_default_last_user_id_none(self):
        session = Session(channel_id="ch1")
        assert session.last_user_id is None


# ===========================================================================
# System prompt injection with per-user memory
# ===========================================================================

class TestSystemPromptPerUserMemory:
    """_build_system_prompt should pass user_id for per-user memory loading."""

    def _make_prompt_stub(self, memory=None, learned=""):
        stub = MagicMock()
        host_mock = MagicMock()
        host_mock.ssh_user = "root"
        host_mock.address = "10.0.0.2"
        stub.config.tools.hosts = {"desktop": host_mock}
        stub.config.tools.allowed_services = ["nginx"]
        stub.config.tools.allowed_playbooks = ["update.yml"]
        stub.context_loader.context = "Context."
        stub.voice_manager = None
        stub.tool_executor._load_memory_for_user = MagicMock(
            return_value=memory or {}
        )
        stub.reflector = MagicMock()
        stub.reflector.get_prompt_section = MagicMock(return_value=learned)
        stub.skill_manager = MagicMock()
        stub.skill_manager.list_skills = MagicMock(return_value=[])
        stub.config.timezone = "UTC"
        stub._recent_actions = {}
        stub._recent_actions_max = 10
        stub._recent_actions_expiry = 3600

        # Cache attributes for prompt caching helpers
        stub._cached_hosts = None
        stub._cached_skills_text = None
        stub._memory_cache = {}
        stub._memory_cache_ttl = 60.0
        stub._reflector_cache = {}
        stub._reflector_cache_ttl = 60.0

        from src.discord.client import LokiBot
        stub._build_system_prompt = LokiBot._build_system_prompt.__get__(stub)
        stub._build_chat_system_prompt = LokiBot._build_chat_system_prompt.__get__(stub)
        stub._get_cached_hosts = LokiBot._get_cached_hosts.__get__(stub)
        stub._get_cached_skills_text = LokiBot._get_cached_skills_text.__get__(stub)
        stub._get_cached_memory = LokiBot._get_cached_memory.__get__(stub)
        stub._get_cached_reflector = LokiBot._get_cached_reflector.__get__(stub)
        return stub

    def test_full_prompt_passes_user_id_to_memory(self):
        stub = self._make_prompt_stub(memory={"k": "v"})
        stub._build_system_prompt(user_id="42")
        stub.tool_executor._load_memory_for_user.assert_called_with("42")

    def test_chat_prompt_passes_user_id_to_memory(self):
        stub = self._make_prompt_stub(memory={"k": "v"})
        stub._build_chat_system_prompt(user_id="42")
        stub.tool_executor._load_memory_for_user.assert_called_with("42")

    def test_full_prompt_passes_user_id_to_reflector(self):
        stub = self._make_prompt_stub(learned="## Learned\n- fact")
        stub._build_system_prompt(user_id="42")
        stub.reflector.get_prompt_section.assert_called_with(user_id="42")

    def test_chat_prompt_passes_user_id_to_reflector(self):
        stub = self._make_prompt_stub(learned="## Learned\n- fact")
        stub._build_chat_system_prompt(user_id="42")
        stub.reflector.get_prompt_section.assert_called_with(user_id="42")

    def test_no_user_id_passes_none(self):
        stub = self._make_prompt_stub(memory={"k": "v"})
        stub._build_system_prompt()
        stub.tool_executor._load_memory_for_user.assert_called_with(None)

    def test_prompt_content_includes_user_memory(self):
        stub = self._make_prompt_stub(memory={"birthday": "Feb 7"})
        prompt = stub._build_system_prompt(user_id="42")
        assert "birthday" in prompt
        assert "Feb 7" in prompt


# ===========================================================================
# Session manager reflection passes user_id
# ===========================================================================

class TestSessionManagerReflectionUserID:
    """SessionManager should pass last_user_id to reflector on reflect/compaction."""

    def test_safe_reflect_passes_user_ids(self, tmp_path):
        reflector = MagicMock()
        reflector.reflect_on_session = AsyncMock()
        mgr = SessionManager(
            max_history=50, max_age_hours=24,
            persist_dir=str(tmp_path / "sessions"),
            reflector=reflector,
        )
        session = Session(channel_id="ch1", last_user_id="42")
        session.messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
            Message(role="user", content="bye"),
        ]

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            mgr._safe_reflect(session, user_ids=["42"])
        )
        reflector.reflect_on_session.assert_called_once()
        assert reflector.reflect_on_session.call_args.kwargs["user_ids"] == ["42"]

    def test_safe_reflect_compacted_passes_user_ids(self, tmp_path):
        reflector = MagicMock()
        reflector.reflect_on_compacted = AsyncMock()
        mgr = SessionManager(
            max_history=50, max_age_hours=24,
            persist_dir=str(tmp_path / "sessions"),
            reflector=reflector,
        )
        messages = [Message(role="user", content=f"msg {i}") for i in range(6)]

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            mgr._safe_reflect_compacted(messages, "summary", user_ids=["77"])
        )
        reflector.reflect_on_compacted.assert_called_once()
        assert reflector.reflect_on_compacted.call_args.kwargs["user_ids"] == ["77"]
