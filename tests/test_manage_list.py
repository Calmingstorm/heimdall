"""Tests for universal list management (manage_list tool).

Session 41 (R6 S3): Replaces the grocery_list skill with a built-in universal
list system supporting any named list, per-user/shared ownership, done/undone
status, on-the-fly creation, and migration from old grocery_list.json.

Tests cover:
- Registry: tool definition, schema, no approval required, in user tier whitelist
- Storage: _lists_path, _load_lists, _save_lists, migration from grocery_list.json
- Actions: add, remove, show, clear, mark_done, mark_undone, list_all
- Ownership: shared vs personal lists, access control
- Edge cases: empty items, duplicates, missing list, missing list_name
- Formatting: _format_list with done/undone items
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("discord.ext.voice_recv", MagicMock())

import pytest  # noqa: E402

from src.tools.executor import ToolExecutor  # noqa: E402
from src.tools.registry import TOOLS  # noqa: E402
from src.permissions.manager import USER_TIER_TOOLS  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_executor(tmp_path, lists_data=None, grocery_data=None):
    """Create a ToolExecutor with a tmp memory_path and optional pre-seeded data."""
    mem_file = tmp_path / "memory.json"
    mem_file.write_text("{}")
    executor = ToolExecutor(MagicMock(), memory_path=str(mem_file))
    if lists_data is not None:
        lists_file = tmp_path / "lists.json"
        lists_file.write_text(json.dumps(lists_data))
    if grocery_data is not None:
        grocery_file = tmp_path / "grocery_list.json"
        grocery_file.write_text(json.dumps(grocery_data))
    return executor


# ===========================================================================
# Registry & permissions
# ===========================================================================

class TestManageListRegistry:
    """Tool definition exists and has correct schema."""

    def _get_tool(self):
        return next(t for t in TOOLS if t["name"] == "manage_list")

    def test_tool_exists(self):
        assert self._get_tool()["name"] == "manage_list"

    def test_no_approval_required(self):
        assert self._get_tool()["requires_approval"] is False

    def test_action_enum(self):
        schema = self._get_tool()["input_schema"]
        actions = schema["properties"]["action"]["enum"]
        assert "add" in actions
        assert "remove" in actions
        assert "show" in actions
        assert "clear" in actions
        assert "mark_done" in actions
        assert "mark_undone" in actions
        assert "list_all" in actions

    def test_required_field_is_action(self):
        schema = self._get_tool()["input_schema"]
        assert schema["required"] == ["action"]

    def test_list_name_property(self):
        schema = self._get_tool()["input_schema"]
        assert "list_name" in schema["properties"]

    def test_items_property_is_array(self):
        schema = self._get_tool()["input_schema"]
        assert schema["properties"]["items"]["type"] == "array"

    def test_owner_enum(self):
        schema = self._get_tool()["input_schema"]
        assert schema["properties"]["owner"]["enum"] == ["personal", "shared"]

    def test_in_user_tier_whitelist(self):
        assert "manage_list" in USER_TIER_TOOLS


# ===========================================================================
# Storage: paths, load, save, migration
# ===========================================================================

class TestListsStorage:
    """_lists_path, _load_lists, _save_lists, and grocery migration."""

    def test_lists_path_from_memory_path(self, tmp_path):
        executor = make_executor(tmp_path)
        assert executor._lists_path() == tmp_path / "lists.json"

    def test_lists_path_none_without_memory_path(self):
        executor = ToolExecutor(MagicMock(), memory_path=None)
        assert executor._lists_path() is None

    def test_load_empty_when_no_file(self, tmp_path):
        executor = make_executor(tmp_path)
        assert executor._load_lists() == {}

    def test_load_existing_lists(self, tmp_path):
        data = {"todo": {"owner": "shared", "items": []}}
        executor = make_executor(tmp_path, lists_data=data)
        result = executor._load_lists()
        assert "todo" in result

    def test_load_handles_corrupt_json(self, tmp_path):
        executor = make_executor(tmp_path)
        (tmp_path / "lists.json").write_text("not json{{{")
        assert executor._load_lists() == {}

    def test_load_handles_non_dict_json(self, tmp_path):
        executor = make_executor(tmp_path)
        (tmp_path / "lists.json").write_text(json.dumps([1, 2, 3]))
        assert executor._load_lists() == {}

    def test_save_lists(self, tmp_path):
        executor = make_executor(tmp_path)
        data = {"grocery": {"owner": "shared", "items": [{"name": "milk", "added_by": "", "added_at": "", "done": False}]}}
        executor._save_lists(data)
        loaded = json.loads((tmp_path / "lists.json").read_text())
        assert loaded["grocery"]["items"][0]["name"] == "milk"

    def test_save_noop_without_memory_path(self):
        executor = ToolExecutor(MagicMock(), memory_path=None)
        executor._save_lists({"test": {}})  # Should not raise

    def test_load_returns_empty_without_memory_path(self):
        executor = ToolExecutor(MagicMock(), memory_path=None)
        assert executor._load_lists() == {}


class TestGroceryMigration:
    """Old grocery_list.json should be auto-migrated to lists.json."""

    def test_migrates_grocery_items(self, tmp_path):
        old_data = {"items": [
            {"name": "Milk", "added_by": "Aaron", "added_at": "2026-03-10T12:00:00"},
            {"name": "Eggs", "added_by": "Jessica", "added_at": "2026-03-11T09:00:00"},
        ]}
        executor = make_executor(tmp_path, grocery_data=old_data)
        result = executor._load_lists()
        assert "grocery" in result
        assert result["grocery"]["owner"] == "shared"
        assert len(result["grocery"]["items"]) == 2
        assert result["grocery"]["items"][0]["name"] == "Milk"
        assert result["grocery"]["items"][1]["name"] == "Eggs"

    def test_migrated_items_have_done_false(self, tmp_path):
        old_data = {"items": [{"name": "Bread", "added_by": "", "added_at": ""}]}
        executor = make_executor(tmp_path, grocery_data=old_data)
        result = executor._load_lists()
        assert result["grocery"]["items"][0]["done"] is False

    def test_migration_persists_to_lists_json(self, tmp_path):
        old_data = {"items": [{"name": "Butter", "added_by": "", "added_at": ""}]}
        executor = make_executor(tmp_path, grocery_data=old_data)
        executor._load_lists()
        assert (tmp_path / "lists.json").exists()
        persisted = json.loads((tmp_path / "lists.json").read_text())
        assert persisted["grocery"]["items"][0]["name"] == "Butter"

    def test_no_migration_when_lists_json_exists(self, tmp_path):
        existing = {"todo": {"owner": "shared", "items": []}}
        old_data = {"items": [{"name": "Milk", "added_by": "", "added_at": ""}]}
        executor = make_executor(tmp_path, lists_data=existing, grocery_data=old_data)
        result = executor._load_lists()
        assert "grocery" not in result
        assert "todo" in result

    def test_migration_handles_empty_grocery(self, tmp_path):
        old_data = {"items": []}
        executor = make_executor(tmp_path, grocery_data=old_data)
        result = executor._load_lists()
        assert result["grocery"]["items"] == []

    def test_migration_handles_corrupt_grocery(self, tmp_path):
        executor = make_executor(tmp_path)
        (tmp_path / "grocery_list.json").write_text("broken{{{")
        assert executor._load_lists() == {}


# ===========================================================================
# Action: add
# ===========================================================================

class TestManageListAdd:
    """Adding items to lists."""

    @pytest.mark.asyncio
    async def test_add_creates_new_list(self, tmp_path):
        executor = make_executor(tmp_path)
        result = await executor._handle_manage_list({"action": "add", "list_name": "grocery", "items": ["milk"]})
        assert "Added to 'grocery'" in result
        assert "milk" in result

    @pytest.mark.asyncio
    async def test_add_to_existing_list(self, tmp_path):
        data = {"grocery": {"owner": "shared", "items": [{"name": "milk", "added_by": "", "added_at": "", "done": False}]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "add", "list_name": "grocery", "items": ["eggs"]})
        assert "Added to 'grocery'" in result
        assert "eggs" in result

    @pytest.mark.asyncio
    async def test_add_duplicate_detected(self, tmp_path):
        data = {"grocery": {"owner": "shared", "items": [{"name": "milk", "added_by": "", "added_at": "", "done": False}]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "add", "list_name": "grocery", "items": ["Milk"]})
        assert "Already on the list" in result

    @pytest.mark.asyncio
    async def test_add_no_items(self, tmp_path):
        executor = make_executor(tmp_path)
        result = await executor._handle_manage_list({"action": "add", "list_name": "grocery", "items": []})
        assert "No items specified" in result

    @pytest.mark.asyncio
    async def test_add_persists(self, tmp_path):
        executor = make_executor(tmp_path)
        await executor._handle_manage_list({"action": "add", "list_name": "todo", "items": ["fix DNS"]})
        loaded = json.loads((tmp_path / "lists.json").read_text())
        assert loaded["todo"]["items"][0]["name"] == "fix DNS"

    @pytest.mark.asyncio
    async def test_add_records_user_id(self, tmp_path):
        executor = make_executor(tmp_path)
        await executor._handle_manage_list({"action": "add", "list_name": "todo", "items": ["test"]}, user_id="441602773310767105")
        loaded = json.loads((tmp_path / "lists.json").read_text())
        assert loaded["todo"]["items"][0]["added_by"] == "441602773310767105"

    @pytest.mark.asyncio
    async def test_add_records_timestamp(self, tmp_path):
        executor = make_executor(tmp_path)
        await executor._handle_manage_list({"action": "add", "list_name": "todo", "items": ["test"]})
        loaded = json.loads((tmp_path / "lists.json").read_text())
        assert loaded["todo"]["items"][0]["added_at"] != ""

    @pytest.mark.asyncio
    async def test_add_creates_shared_by_default(self, tmp_path):
        executor = make_executor(tmp_path)
        await executor._handle_manage_list({"action": "add", "list_name": "grocery", "items": ["milk"]})
        loaded = json.loads((tmp_path / "lists.json").read_text())
        assert loaded["grocery"]["owner"] == "shared"

    @pytest.mark.asyncio
    async def test_add_creates_personal_list(self, tmp_path):
        executor = make_executor(tmp_path)
        await executor._handle_manage_list({"action": "add", "list_name": "my todo", "items": ["rest"], "owner": "personal"}, user_id="441602773310767105")
        loaded = json.loads((tmp_path / "lists.json").read_text())
        assert loaded["my todo"]["owner"] == "441602773310767105"

    @pytest.mark.asyncio
    async def test_add_normalizes_list_name(self, tmp_path):
        executor = make_executor(tmp_path)
        await executor._handle_manage_list({"action": "add", "list_name": " Grocery ", "items": ["milk"]})
        loaded = json.loads((tmp_path / "lists.json").read_text())
        assert "grocery" in loaded

    @pytest.mark.asyncio
    async def test_add_skips_empty_strings(self, tmp_path):
        executor = make_executor(tmp_path)
        result = await executor._handle_manage_list({"action": "add", "list_name": "grocery", "items": ["", "  ", "milk"]})
        assert "milk" in result
        loaded = json.loads((tmp_path / "lists.json").read_text())
        assert len(loaded["grocery"]["items"]) == 1

    @pytest.mark.asyncio
    async def test_add_multiple_items(self, tmp_path):
        executor = make_executor(tmp_path)
        result = await executor._handle_manage_list({"action": "add", "list_name": "grocery", "items": ["milk", "eggs", "bread"]})
        assert "milk" in result
        assert "eggs" in result
        assert "bread" in result
        loaded = json.loads((tmp_path / "lists.json").read_text())
        assert len(loaded["grocery"]["items"]) == 3


# ===========================================================================
# Action: remove
# ===========================================================================

class TestManageListRemove:
    """Removing items from lists."""

    @pytest.mark.asyncio
    async def test_remove_item(self, tmp_path):
        data = {"grocery": {"owner": "shared", "items": [
            {"name": "milk", "added_by": "", "added_at": "", "done": False},
            {"name": "eggs", "added_by": "", "added_at": "", "done": False},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "remove", "list_name": "grocery", "items": ["milk"]})
        assert "Removed from 'grocery'" in result
        assert "milk" in result

    @pytest.mark.asyncio
    async def test_remove_substring_match(self, tmp_path):
        data = {"grocery": {"owner": "shared", "items": [
            {"name": "whole milk", "added_by": "", "added_at": "", "done": False},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "remove", "list_name": "grocery", "items": ["milk"]})
        assert "whole milk" in result

    @pytest.mark.asyncio
    async def test_remove_not_found(self, tmp_path):
        data = {"grocery": {"owner": "shared", "items": [
            {"name": "milk", "added_by": "", "added_at": "", "done": False},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "remove", "list_name": "grocery", "items": ["bread"]})
        assert "Not found" in result

    @pytest.mark.asyncio
    async def test_remove_nonexistent_list(self, tmp_path):
        executor = make_executor(tmp_path)
        result = await executor._handle_manage_list({"action": "remove", "list_name": "grocery", "items": ["milk"]})
        assert "doesn't exist" in result

    @pytest.mark.asyncio
    async def test_remove_no_items(self, tmp_path):
        data = {"grocery": {"owner": "shared", "items": [{"name": "milk", "added_by": "", "added_at": "", "done": False}]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "remove", "list_name": "grocery", "items": []})
        assert "No items specified" in result

    @pytest.mark.asyncio
    async def test_remove_persists(self, tmp_path):
        data = {"grocery": {"owner": "shared", "items": [
            {"name": "milk", "added_by": "", "added_at": "", "done": False},
            {"name": "eggs", "added_by": "", "added_at": "", "done": False},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        await executor._handle_manage_list({"action": "remove", "list_name": "grocery", "items": ["milk"]})
        loaded = json.loads((tmp_path / "lists.json").read_text())
        assert len(loaded["grocery"]["items"]) == 1
        assert loaded["grocery"]["items"][0]["name"] == "eggs"

    @pytest.mark.asyncio
    async def test_remove_shows_empty_message(self, tmp_path):
        data = {"grocery": {"owner": "shared", "items": [{"name": "milk", "added_by": "", "added_at": "", "done": False}]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "remove", "list_name": "grocery", "items": ["milk"]})
        assert "now empty" in result


# ===========================================================================
# Action: show
# ===========================================================================

class TestManageListShow:
    """Showing list contents."""

    @pytest.mark.asyncio
    async def test_show_list(self, tmp_path):
        data = {"grocery": {"owner": "shared", "items": [
            {"name": "milk", "added_by": "Aaron", "added_at": "2026-03-10T12:00:00", "done": False},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "show", "list_name": "grocery"})
        assert "Grocery List" in result
        assert "milk" in result

    @pytest.mark.asyncio
    async def test_show_empty_list(self, tmp_path):
        data = {"grocery": {"owner": "shared", "items": []}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "show", "list_name": "grocery"})
        assert "empty" in result

    @pytest.mark.asyncio
    async def test_show_nonexistent_list(self, tmp_path):
        executor = make_executor(tmp_path)
        result = await executor._handle_manage_list({"action": "show", "list_name": "nope"})
        assert "empty" in result


# ===========================================================================
# Action: clear
# ===========================================================================

class TestManageListClear:
    """Clearing a list."""

    @pytest.mark.asyncio
    async def test_clear_list(self, tmp_path):
        data = {"grocery": {"owner": "shared", "items": [
            {"name": "milk", "added_by": "", "added_at": "", "done": False},
            {"name": "eggs", "added_by": "", "added_at": "", "done": False},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "clear", "list_name": "grocery"})
        assert "Cleared 2 item(s)" in result

    @pytest.mark.asyncio
    async def test_clear_empty_list(self, tmp_path):
        data = {"grocery": {"owner": "shared", "items": []}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "clear", "list_name": "grocery"})
        assert "already empty" in result

    @pytest.mark.asyncio
    async def test_clear_nonexistent_list(self, tmp_path):
        executor = make_executor(tmp_path)
        result = await executor._handle_manage_list({"action": "clear", "list_name": "nope"})
        assert "already empty" in result

    @pytest.mark.asyncio
    async def test_clear_persists(self, tmp_path):
        data = {"grocery": {"owner": "shared", "items": [{"name": "milk", "added_by": "", "added_at": "", "done": False}]}}
        executor = make_executor(tmp_path, lists_data=data)
        await executor._handle_manage_list({"action": "clear", "list_name": "grocery"})
        loaded = json.loads((tmp_path / "lists.json").read_text())
        assert loaded["grocery"]["items"] == []


# ===========================================================================
# Action: mark_done / mark_undone
# ===========================================================================

class TestManageListMarkDone:
    """Marking items as done."""

    @pytest.mark.asyncio
    async def test_mark_done(self, tmp_path):
        data = {"todo": {"owner": "shared", "items": [
            {"name": "fix DNS", "added_by": "", "added_at": "", "done": False},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "mark_done", "list_name": "todo", "items": ["DNS"]})
        assert "Marked done" in result
        assert "fix DNS" in result

    @pytest.mark.asyncio
    async def test_mark_done_persists(self, tmp_path):
        data = {"todo": {"owner": "shared", "items": [
            {"name": "fix DNS", "added_by": "", "added_at": "", "done": False},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        await executor._handle_manage_list({"action": "mark_done", "list_name": "todo", "items": ["DNS"]})
        loaded = json.loads((tmp_path / "lists.json").read_text())
        assert loaded["todo"]["items"][0]["done"] is True

    @pytest.mark.asyncio
    async def test_mark_done_already_done(self, tmp_path):
        data = {"todo": {"owner": "shared", "items": [
            {"name": "fix DNS", "added_by": "", "added_at": "", "done": True},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "mark_done", "list_name": "todo", "items": ["DNS"]})
        assert "already done" in result

    @pytest.mark.asyncio
    async def test_mark_done_not_found(self, tmp_path):
        data = {"todo": {"owner": "shared", "items": [
            {"name": "fix DNS", "added_by": "", "added_at": "", "done": False},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "mark_done", "list_name": "todo", "items": ["nope"]})
        assert "Not found or already done" in result

    @pytest.mark.asyncio
    async def test_mark_done_nonexistent_list(self, tmp_path):
        executor = make_executor(tmp_path)
        result = await executor._handle_manage_list({"action": "mark_done", "list_name": "todo", "items": ["x"]})
        assert "doesn't exist" in result

    @pytest.mark.asyncio
    async def test_mark_done_no_items(self, tmp_path):
        data = {"todo": {"owner": "shared", "items": [{"name": "x", "added_by": "", "added_at": "", "done": False}]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "mark_done", "list_name": "todo", "items": []})
        assert "No items specified" in result


class TestManageListMarkUndone:
    """Marking items as undone."""

    @pytest.mark.asyncio
    async def test_mark_undone(self, tmp_path):
        data = {"todo": {"owner": "shared", "items": [
            {"name": "fix DNS", "added_by": "", "added_at": "", "done": True},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "mark_undone", "list_name": "todo", "items": ["DNS"]})
        assert "Marked undone" in result

    @pytest.mark.asyncio
    async def test_mark_undone_persists(self, tmp_path):
        data = {"todo": {"owner": "shared", "items": [
            {"name": "fix DNS", "added_by": "", "added_at": "", "done": True},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        await executor._handle_manage_list({"action": "mark_undone", "list_name": "todo", "items": ["DNS"]})
        loaded = json.loads((tmp_path / "lists.json").read_text())
        assert loaded["todo"]["items"][0]["done"] is False

    @pytest.mark.asyncio
    async def test_mark_undone_not_done(self, tmp_path):
        data = {"todo": {"owner": "shared", "items": [
            {"name": "fix DNS", "added_by": "", "added_at": "", "done": False},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "mark_undone", "list_name": "todo", "items": ["DNS"]})
        assert "Not found or not done" in result

    @pytest.mark.asyncio
    async def test_mark_undone_nonexistent_list(self, tmp_path):
        executor = make_executor(tmp_path)
        result = await executor._handle_manage_list({"action": "mark_undone", "list_name": "todo", "items": ["x"]})
        assert "doesn't exist" in result

    @pytest.mark.asyncio
    async def test_mark_undone_no_items(self, tmp_path):
        data = {"todo": {"owner": "shared", "items": [{"name": "x", "added_by": "", "added_at": "", "done": True}]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "mark_undone", "list_name": "todo", "items": []})
        assert "No items specified" in result


# ===========================================================================
# Action: list_all
# ===========================================================================

class TestManageListAll:
    """Listing all available lists."""

    @pytest.mark.asyncio
    async def test_list_all_empty(self, tmp_path):
        executor = make_executor(tmp_path)
        result = await executor._handle_manage_list({"action": "list_all"})
        assert "No lists exist" in result

    @pytest.mark.asyncio
    async def test_list_all_shows_lists(self, tmp_path):
        data = {
            "grocery": {"owner": "shared", "items": [{"name": "milk", "added_by": "", "added_at": "", "done": False}]},
            "todo": {"owner": "shared", "items": []},
        }
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "list_all"})
        assert "grocery" in result
        assert "todo" in result

    @pytest.mark.asyncio
    async def test_list_all_shows_done_count(self, tmp_path):
        data = {"todo": {"owner": "shared", "items": [
            {"name": "a", "added_by": "", "added_at": "", "done": True},
            {"name": "b", "added_by": "", "added_at": "", "done": False},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "list_all"})
        assert "1 done" in result

    @pytest.mark.asyncio
    async def test_list_all_hides_other_personal(self, tmp_path):
        data = {
            "grocery": {"owner": "shared", "items": []},
            "secret": {"owner": "other_user_123", "items": []},
        }
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "list_all"}, user_id="441602773310767105")
        assert "grocery" in result
        assert "secret" not in result

    @pytest.mark.asyncio
    async def test_list_all_shows_own_personal(self, tmp_path):
        data = {
            "my todo": {"owner": "441602773310767105", "items": []},
        }
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "list_all"}, user_id="441602773310767105")
        assert "my todo" in result
        assert "personal" in result


# ===========================================================================
# Ownership & access control
# ===========================================================================

class TestListOwnership:
    """Access control for personal vs shared lists."""

    @pytest.mark.asyncio
    async def test_cannot_access_other_users_list(self, tmp_path):
        data = {"secret": {"owner": "other_user_123", "items": [{"name": "x", "added_by": "", "added_at": "", "done": False}]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "show", "list_name": "secret"}, user_id="441602773310767105")
        assert "don't have access" in result

    @pytest.mark.asyncio
    async def test_can_access_own_personal_list(self, tmp_path):
        data = {"my todo": {"owner": "441602773310767105", "items": [
            {"name": "rest", "added_by": "", "added_at": "", "done": False},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "show", "list_name": "my todo"}, user_id="441602773310767105")
        assert "rest" in result

    @pytest.mark.asyncio
    async def test_can_access_shared_list(self, tmp_path):
        data = {"grocery": {"owner": "shared", "items": [
            {"name": "milk", "added_by": "", "added_at": "", "done": False},
        ]}}
        executor = make_executor(tmp_path, lists_data=data)
        result = await executor._handle_manage_list({"action": "show", "list_name": "grocery"}, user_id="441602773310767105")
        assert "milk" in result


# ===========================================================================
# Edge cases
# ===========================================================================

class TestManageListEdgeCases:
    """Edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_missing_list_name(self, tmp_path):
        executor = make_executor(tmp_path)
        result = await executor._handle_manage_list({"action": "show"})
        assert "list_name is required" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self, tmp_path):
        executor = make_executor(tmp_path)
        result = await executor._handle_manage_list({"action": "explode", "list_name": "x"})
        assert "Unknown action" in result

    @pytest.mark.asyncio
    async def test_add_without_items_key(self, tmp_path):
        executor = make_executor(tmp_path)
        result = await executor._handle_manage_list({"action": "add", "list_name": "grocery"})
        assert "No items specified" in result


# ===========================================================================
# Formatting
# ===========================================================================

class TestFormatList:
    """Static _format_list output."""

    def test_format_empty(self):
        result = ToolExecutor._format_list("todo", {"items": []})
        assert "empty" in result

    def test_format_with_items(self):
        lst = {"items": [
            {"name": "milk", "added_by": "Aaron", "added_at": "2026-03-10T12:00:00", "done": False},
        ]}
        result = ToolExecutor._format_list("grocery", lst)
        assert "Grocery List" in result
        assert "1. milk" in result
        assert "Aaron" in result
        assert "Mar 10" in result

    def test_format_done_item(self):
        lst = {"items": [
            {"name": "fix DNS", "added_by": "", "added_at": "", "done": True},
        ]}
        result = ToolExecutor._format_list("todo", lst)
        assert "\u2705" in result
        assert "~~fix DNS~~" in result

    def test_format_mixed_done_undone(self):
        lst = {"items": [
            {"name": "task A", "added_by": "", "added_at": "", "done": True},
            {"name": "task B", "added_by": "", "added_at": "", "done": False},
        ]}
        result = ToolExecutor._format_list("todo", lst)
        assert "~~task A~~" in result
        assert "task B" in result
        assert result.count("\u2705") == 1

    def test_format_item_count(self):
        lst = {"items": [
            {"name": "a", "added_by": "", "added_at": "", "done": False},
            {"name": "b", "added_by": "", "added_at": "", "done": False},
            {"name": "c", "added_by": "", "added_at": "", "done": False},
        ]}
        result = ToolExecutor._format_list("shopping", lst)
        assert "3 items" in result
